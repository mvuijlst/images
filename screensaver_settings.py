"""Settings for the image screensaver: where you are, what the pictures are based on, which screen, timing,
image size and quality, and saving. Kept in screensaver_settings.json next to this file.

Open on its own (no GPU, no models):  uv run screensaver_settings.py   (or the "Screensaver Settings" shortcut)
or from the running screensaver with O / right-click → Settings. The screensaver imports everything here.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, time as dtime
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QTime, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QIcon, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QSizePolicy, QSpinBox, QTabWidget, QTimeEdit, QVBoxLayout,
                               QWidget)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realworld  # noqa: E402  light: requests; no models

APP_NAME = "Image Screensaver"
SETTINGS_PATH = HERE / "screensaver_settings.json"
SAVE_DIR = HERE / "outputs" / "screensaver"  # default; settings can change it
ICON_PATH = HERE / "icon.ico"
MAX_SHORT_SIDE = 2048  # no more than the 4× upscaler can genuinely add (512 × 4)
# image size: key → (menu text, times the screen's resolution)
RESOLUTIONS = {
    "1x": ("Same as the screen (fastest, smallest files)", 1.0),
    "2x": ("Twice the screen (sharp when you open a saved image)", 2.0),
    "max": ("As large as the upscaler allows (2048 px on the short side)", 99.0),
}
DEFAULTS = {
    "place": None,  # {"name", "country", "lat", "lon"} from the city lookup or typed in
    "city": "",
    "use_weather": True,
    "use_words": False,  # a few random words → scene, by a local LLM through Ollama
    "llm_words": 3,  # words per image
    "llm_model": realworld.DEFAULT_LLM,
    "ollama_url": realworld.OLLAMA_URL,
    "text_check": True,  # the local LLM looks at each image; text in it → made again (needs a vision model)
    "word_check": True,  # … and at each words draft; none of the words in it → drawn again
    "use_news": True,
    "use_history": True,
    "feeds": realworld.DEFAULT_FEEDS,
    "skip_distressing": True,
    "skip_words": realworld.DEFAULT_SKIP_WORDS,
    "monitor": None,  # QScreen.name(); None → primary
    "seconds": 60,  # each image stays up this long
    "fade_seconds": 2.5,  # crossfade between images
    "quiet_hours": False,  # between quiet_from and quiet_to: black screen, no images made, display may sleep
    "quiet_from": "23:00",
    "quiet_to": "07:00",
    "captions": True,
    "clock": True,
    "clock_24h": True,
    "model": "sd",
    "resolution": "2x",  # key of RESOLUTIONS: size images are made and saved at
    "safety": True,
    "steps": 25,
    "save_all": True,  # every image + JSON sidecar, in a folder per day
    "save_dir": None,  # None → outputs/screensaver
    "keep_days": 0,  # delete day folders older than this many days; 0 = keep everything
}


def load_settings() -> dict:
    cfg = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            cfg.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            logging.exception("could not read settings; using defaults")
    return cfg


def save_settings(cfg: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def save_root(cfg: dict) -> Path:
    return Path(cfg["save_dir"]) if cfg.get("save_dir") else SAVE_DIR


def save_size(screen_w: int, screen_h: int, resolution: str = "2x") -> tuple[int, int]:
    """Size images are made and saved at: the screen × the chosen factor, short side at most MAX_SHORT_SIDE."""
    factor = RESOLUTIONS.get(resolution, RESOLUTIONS["2x"])[1]
    scale = min(factor, MAX_SHORT_SIDE / min(screen_w, screen_h))
    return max(screen_w, round(screen_w * scale)), max(screen_h, round(screen_h * scale))


def physical_size(screen) -> tuple[int, int]:
    g, dpr = screen.geometry(), screen.devicePixelRatio()
    return round(g.width() * dpr), round(g.height() * dpr)


def _hm(text: str) -> dtime:
    h, m = (int(x) for x in str(text).split(":")[:2])
    return dtime(h, m)


def is_quiet(cfg: dict, now: datetime | None = None) -> bool:
    """True during the quiet hours (which may run past midnight, e.g. 23:00–07:00)."""
    if not cfg.get("quiet_hours"):
        return False
    try:
        start, end = _hm(cfg.get("quiet_from", "23:00")), _hm(cfg.get("quiet_to", "07:00"))
    except ValueError:
        return False
    t = (now or datetime.now()).time()
    return start <= t < end if start < end else (t >= start or t < end) if start != end else False


def clock_text(now: datetime, h24: bool) -> str:
    return f"{now:%H:%M}" if h24 else f"{now:%I:%M}".lstrip("0") + f" {now:%p}".lower()


# --------------------------------------------------------------------------- monitor map


class MonitorMap(QWidget):
    """The monitors as they are arranged in Windows, numbered; click one to choose it."""

    picked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected: str | None = None
        self.setMinimumSize(QSize(360, 150))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click a monitor to show the screensaver on it")

    def _layout(self) -> list[tuple[QRectF, object, int]]:
        screens = QGuiApplication.screens()
        if not screens:
            return []
        geos = [s.geometry() for s in screens]
        left, top = min(g.left() for g in geos), min(g.top() for g in geos)
        right, bottom = max(g.right() for g in geos), max(g.bottom() for g in geos)
        pad = 10
        scale = min((self.width() - 2 * pad) / max(1, right - left), (self.height() - 2 * pad) / max(1, bottom - top))
        ox = (self.width() - (right - left) * scale) / 2
        oy = (self.height() - (bottom - top) * scale) / 2
        out = []
        for i, (s, g) in enumerate(zip(screens, geos), 1):
            r = QRectF(ox + (g.left() - left) * scale + 3, oy + (g.top() - top) * scale + 3,
                       g.width() * scale - 6, g.height() * scale - 6)
            # a small monitor next to big ones (800 × 480 beside 4K) would be a speck: grow it around its centre
            min_w = 84.0
            if r.width() < min_w:
                c, k = r.center(), min_w / max(r.width(), 1)
                r = QRectF(0, 0, min_w, r.height() * k)
                r.moveCenter(c)
                r.moveRight(min(r.right(), self.width() - 2))
                r.moveTop(max(r.top(), 2))
            out.append((r, s, i))
        return out

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        for rect, s, i in self._layout():
            on = s.name() == self.selected
            p.setBrush(QColor("#2f6fb5") if on else pal.base().color())
            p.setPen(QPen(QColor("#4aa3ff") if on else pal.mid().color(), 2))
            p.drawRoundedRect(rect, 6, 6)
            p.setPen(QColor("white") if on else pal.text().color())
            f = QFont(self.font())
            f.setPixelSize(max(12, int(min(rect.height(), rect.width()) * 0.32)))
            f.setBold(True)
            p.setFont(f)
            p.drawText(rect.adjusted(0, -rect.height() * 0.12, 0, 0), int(Qt.AlignmentFlag.AlignCenter), str(i))
            w, h = physical_size(s)
            f.setPixelSize(max(10, int(min(rect.height(), rect.width()) * 0.11)))
            f.setBold(False)
            p.setFont(f)
            p.drawText(rect.adjusted(4, rect.height() * 0.45, -4, 0), int(Qt.AlignmentFlag.AlignCenter),
                       f"{w} × {h}")

    def mousePressEvent(self, e) -> None:
        for rect, s, _i in self._layout():
            if rect.contains(e.position()):
                self.selected = s.name()
                self.picked.emit(s.name())
                self.update()
                return


# --------------------------------------------------------------------------- dialog


def _note(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: palette(placeholder-text);")
    return lbl


class SettingsDialog(QDialog):
    """All settings in tabs. mode: "first" (first start: Start), "running" (from the screensaver: Save),
    "standalone" (the settings shortcut: Save, and Save and start)."""

    def __init__(self, cfg: dict, parent=None, mode: str = "first"):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} settings")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.cfg = dict(cfg)
        self.mode = mode
        self.start_after = False
        self.place = cfg.get("place")
        self.coords_typed = False
        self.setMinimumWidth(620)

        tabs = QTabWidget()
        tabs.addTab(self._place_tab(cfg), "Place")
        tabs.addTab(self._pictures_tab(cfg), "Pictures")
        tabs.addTab(self._screen_tab(cfg), "Screen")
        tabs.addTab(self._timing_tab(cfg), "Timing")
        tabs.addTab(self._quality_tab(cfg), "Quality")
        tabs.addTab(self._saving_tab(cfg), "Saving")

        buttons = QHBoxLayout()  # in this order on every platform: main action, Save, Cancel
        buttons.addStretch(1)
        if mode == "standalone":
            start = QPushButton("Save and start screensaver")
            start.clicked.connect(lambda: self._accept(start_after=True))
            save = QPushButton("Save")
            save.clicked.connect(lambda: self._accept(start_after=False))
            buttons.addWidget(start)
            buttons.addWidget(save)
            start.setDefault(True)
        else:
            ok = QPushButton("Start" if mode == "first" else "Save")
            ok.clicked.connect(lambda: self._accept(start_after=False))
            buttons.addWidget(ok)
            ok.setDefault(True)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(_note("Weather: Open-Meteo (only your town's coordinates are sent). Random words: Ollama on "
                               "this PC, nothing leaves it. History: Wikipedia. Names of people and organisations are "
                               "left out of image prompts; captions show the real text."))
        layout.addLayout(buttons)
        self._update_size_info()

    # ----- tabs

    def _place_tab(self, cfg: dict) -> QWidget:
        page = QWidget()
        self.city = QLineEdit(cfg.get("city") or (self.place or {}).get("name", ""))
        self.city.setPlaceholderText("Your town or city, e.g. Utrecht")
        lookup = QPushButton("Look up")
        lookup.clicked.connect(self._lookup)
        self.city.returnPressed.connect(self._lookup)
        row = QHBoxLayout()
        row.addWidget(self.city, 1)
        row.addWidget(lookup)
        self.place_label = QLabel()
        self.lat, self.lon = QDoubleSpinBox(), QDoubleSpinBox()
        for box, lim in ((self.lat, 90), (self.lon, 180)):
            box.setRange(-lim, lim)
            box.setDecimals(4)
            box.setSingleStep(0.01)
            box.valueChanged.connect(self._coords_edited)
        coords = QHBoxLayout()
        coords.addWidget(QLabel("Latitude"))
        coords.addWidget(self.lat)
        coords.addSpacing(12)
        coords.addWidget(QLabel("Longitude"))
        coords.addWidget(self.lon)
        coords.addStretch(1)
        check = QPushButton("Check the weather there now")
        check.clicked.connect(self._check_weather)
        self.weather_label = QLabel()
        self.weather_label.setWordWrap(True)
        self._show_place()

        f = QFormLayout()
        f.addRow("Town or city", row)
        f.addRow("", self.place_label)
        f.addRow("Coordinates", coords)
        f.addRow("", check)
        f.addRow("", self.weather_label)
        v = QVBoxLayout(page)
        v.addLayout(f)
        v.addWidget(_note("The weather, the time of day (from sunrise and sunset) and the season come from here, "
                          "and so does the light of the hour: night frames are dark. Look up a town, or type exact "
                          "coordinates. Without a location, pictures follow this PC's clock and the calendar. The "
                          "random-words scenes (scene_parts.txt) are written for Ghent; edit them for your town."))
        v.addStretch(1)
        return page

    def _pictures_tab(self, cfg: dict) -> QWidget:
        page = QWidget()
        self.use_weather = QCheckBox("The weather, time of day and season where you are")
        self.use_words = QCheckBox("Random words, made into a local scene by an LLM on this PC (Ollama)")
        self.use_words.setToolTip("Words from words.txt; settings, viewpoints, styles from scene_parts.txt;\n"
                                  "instructions for the LLM in llm_prompt.txt (edits apply to the next image).")
        self.llm_words = QSpinBox()
        self.llm_words.setRange(1, 5)
        self.llm_words.setSuffix(" words per image")
        self.llm_words.setValue(int(cfg.get("llm_words", 3)))
        self.llm_words.setToolTip("A small model uses two words more reliably than three (and SD fits them in better).")
        self.llm_model = QLineEdit(cfg.get("llm_model") or realworld.DEFAULT_LLM)
        self.llm_model.setToolTip("Any Ollama model name. It is pulled once if it isn't there yet.\n"
                                  "The text and word checks need one that can see images (qwen3.5 can).")
        self.ollama_url = QLineEdit(cfg.get("ollama_url") or realworld.OLLAMA_URL)
        words_row = QHBoxLayout()
        words_row.addWidget(self.llm_words)
        words_row.addWidget(QLabel("Model"))
        words_row.addWidget(self.llm_model, 1)
        self.use_news = QCheckBox("News headlines, as allegories (from the feeds below)")
        self.use_history = QCheckBox("Wikipedia: on this day in history")
        for box, key in ((self.use_weather, "use_weather"), (self.use_news, "use_news"),
                         (self.use_history, "use_history")):
            box.setChecked(cfg.get(key, True))
        self.use_words.setChecked(cfg.get("use_words", False))
        self.feeds = QPlainTextEdit("\n".join(cfg.get("feeds") or []))
        self.feeds.setFixedHeight(76)
        self.feeds.setToolTip("RSS or Atom feeds, one per line. English headlines work best.")
        self.skip = QCheckBox("Skip headlines and events containing these words:")
        self.skip.setChecked(cfg.get("skip_distressing", True))
        self.skip_words = QLineEdit(cfg.get("skip_words", realworld.DEFAULT_SKIP_WORDS))
        self.skip_words.setCursorPosition(0)
        for box, widgets in ((self.use_words, (self.llm_words, self.llm_model, self.ollama_url)),
                             (self.use_news, (self.feeds,)), (self.skip, (self.skip_words,))):
            for w in widgets:
                box.toggled.connect(w.setEnabled)
                w.setEnabled(box.isChecked())

        f = QFormLayout(page)
        f.addRow(_note("Each image is based on one of these, taken in turn."))
        f.addRow(self.use_weather)
        f.addRow(self.use_words)
        f.addRow("Words", words_row)
        f.addRow("Ollama at", self.ollama_url)
        f.addRow(self.use_news)
        f.addRow("Feeds", self.feeds)
        f.addRow(self.use_history)
        f.addRow(self.skip)
        f.addRow("", self.skip_words)
        return page

    def _screen_tab(self, cfg: dict) -> QWidget:
        page = QWidget()
        self.map = MonitorMap()
        self.monitor = QComboBox()
        for i, s in enumerate(QGuiApplication.screens(), 1):
            w, h = physical_size(s)
            primary = "  (main)" if s == QGuiApplication.primaryScreen() else ""
            self.monitor.addItem(f"{i}: {w} × {h}  {s.model() or s.name()}{primary}", s.name())
        idx = self.monitor.findData(cfg.get("monitor"))
        self.monitor.setCurrentIndex(idx if idx >= 0 else self.monitor.findData(QGuiApplication.primaryScreen().name()))
        self.map.selected = self.monitor.currentData()
        self.map.picked.connect(lambda name: self.monitor.setCurrentIndex(self.monitor.findData(name)))
        self.monitor.currentIndexChanged.connect(lambda _i: (setattr(self.map, "selected", self.monitor.currentData()),
                                                             self.map.update(), self._update_size_info()))
        identify = QPushButton("Identify monitors")
        identify.setToolTip("Shows each monitor's number on it for a few seconds")
        identify.clicked.connect(self._identify)
        mon_row = QHBoxLayout()
        mon_row.addWidget(self.monitor, 1)
        mon_row.addWidget(identify)
        self.captions = QCheckBox("Show what each image is based on (C switches it while running)")
        self.captions.setChecked(cfg.get("captions", True))
        self.clock = QCheckBox("Show a clock and the date")
        self.clock.setChecked(cfg.get("clock", True))
        self.clock_24h = QComboBox()
        self.clock_24h.addItem("24-hour (14:05)", True)
        self.clock_24h.addItem("12-hour (2:05 pm)", False)
        self.clock_24h.setCurrentIndex(0 if cfg.get("clock_24h", True) else 1)
        self.clock.toggled.connect(self.clock_24h.setEnabled)
        self.clock_24h.setEnabled(self.clock.isChecked())

        v = QVBoxLayout(page)
        v.addWidget(self.map, 1)
        f = QFormLayout()
        f.addRow("Monitor", mon_row)
        f.addRow("", self.captions)
        f.addRow("", self.clock)
        f.addRow("Clock", self.clock_24h)
        v.addLayout(f)
        return page

    def _timing_tab(self, cfg: dict) -> QWidget:
        page = QWidget()
        self.seconds = QSpinBox()
        self.seconds.setRange(10, 3600)
        self.seconds.setSuffix(" seconds")
        self.seconds.setValue(int(cfg.get("seconds", 60)))
        self.seconds.valueChanged.connect(self._update_size_info)
        self.fade = QDoubleSpinBox()
        self.fade.setRange(0, 10)
        self.fade.setDecimals(1)
        self.fade.setSingleStep(0.5)
        self.fade.setSuffix(" seconds")
        self.fade.setSpecialValueText("none (cut)")
        self.fade.setValue(float(cfg.get("fade_seconds", 2.5)))
        self.quiet = QCheckBox("Quiet hours: a black screen, no images made, and Windows may turn the display off")
        self.quiet.setChecked(cfg.get("quiet_hours", False))
        self.quiet_from, self.quiet_to = QTimeEdit(), QTimeEdit()
        for box, key, default in ((self.quiet_from, "quiet_from", "23:00"), (self.quiet_to, "quiet_to", "07:00")):
            box.setDisplayFormat("HH:mm")
            box.setTime(QTime.fromString(str(cfg.get(key, default)), "HH:mm") if QTime.fromString(
                str(cfg.get(key, default)), "HH:mm").isValid() else QTime.fromString(default, "HH:mm"))
            self.quiet.toggled.connect(box.setEnabled)
            box.setEnabled(self.quiet.isChecked())
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("from"))
        q_row.addWidget(self.quiet_from)
        q_row.addWidget(QLabel("to"))
        q_row.addWidget(self.quiet_to)
        q_row.addStretch(1)

        f = QFormLayout(page)
        f.addRow("Each image stays", self.seconds)
        f.addRow("", _note("A new image takes about 10–25 seconds to make (longer at night, when it is drawn twice "
                           "for the light of the hour), and is made while the current one shows. Shorter than that, "
                           "an image stays until the next one is ready."))
        f.addRow("Crossfade", self.fade)
        f.addRow(self.quiet)
        f.addRow("", q_row)
        f.addRow("", _note("The hours may run past midnight (23:00 to 07:00). The screensaver keeps running and "
                           "picks up again by itself."))
        return page

    def _quality_tab(self, cfg: dict) -> QWidget:
        page = QWidget()
        self.model = QComboBox()
        self.model.addItem("Stable Diffusion 1.5  (fills the screen)", "sd")
        self.model.addItem("DALL·E Mega  (square, on a blurred backdrop)", "mega")
        self.model.setCurrentIndex(max(0, self.model.findData(cfg.get("model", "sd"))))
        self.model.currentIndexChanged.connect(self._update_size_info)
        self.resolution = QComboBox()
        for key, (text, _f) in RESOLUTIONS.items():
            self.resolution.addItem(text, key)
        self.resolution.setCurrentIndex(max(0, self.resolution.findData(cfg.get("resolution", "2x"))))
        self.resolution.currentIndexChanged.connect(self._update_size_info)
        self.size_info = QLabel()
        self.size_info.setWordWrap(True)
        self.steps = QSpinBox()
        self.steps.setRange(10, 60)
        self.steps.setSuffix(" steps")
        self.steps.setValue(int(cfg.get("steps", 25)))
        self.steps.setToolTip("Stable Diffusion sampling steps: more is finer detail and slower. 20–30 is plenty.")
        self.safety = QCheckBox("Stable Diffusion safety filter (blacked-out images are skipped)")
        self.safety.setChecked(cfg.get("safety", True))
        self.text_check = QCheckBox("Make an image again when the local LLM sees text in it (watermarks, signatures)")
        self.text_check.setChecked(cfg.get("text_check", True))
        self.text_check.setToolTip("SD 1.5 adds fake watermarks and lettering that no negative prompt stops.\n"
                                   "The LLM looks at each image (~1 s); text → new seed, up to twice.")
        self.word_check = QCheckBox("Draw a random-words image again when the LLM sees none of the words in it")
        self.word_check.setChecked(cfg.get("word_check", True))
        self.word_check.setToolTip("SD 1.5 often draws the setting and drops small or odd subjects. The LLM looks at\n"
                                   "each first draft (~1 s a word); none of the words → new seed, up to twice.")

        f = QFormLayout(page)
        f.addRow("Model", self.model)
        f.addRow("Image size", self.resolution)
        f.addRow("", self.size_info)
        f.addRow("Detail", self.steps)
        f.addRow("", self.safety)
        f.addRow("", self.text_check)
        f.addRow("", self.word_check)
        return page

    def _saving_tab(self, cfg: dict) -> QWidget:
        page = QWidget()
        self.save_all = QCheckBox("Save every image as JPEG, with a .json file of its details next to it")
        self.save_all.setChecked(cfg.get("save_all", True))
        self.save_all.toggled.connect(self._update_size_info)
        self.save_dir = QLineEdit(str(save_root(cfg)))
        self.save_dir.setReadOnly(True)
        change = QPushButton("Change…")
        change.clicked.connect(self._choose_folder)
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(self._open_folder)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.save_dir, 1)
        dir_row.addWidget(change)
        dir_row.addWidget(open_btn)
        self.keep_days = QSpinBox()
        self.keep_days.setRange(0, 3650)
        self.keep_days.setSpecialValueText("never (keep everything)")
        self.keep_days.setSuffix(" days")
        self.keep_days.setValue(int(cfg.get("keep_days", 0) or 0))
        self.keep_days.setToolTip("Day folders older than this are deleted (images and their .json files).\n"
                                  "Favorites (S key) are never deleted.")
        self.estimate = QLabel()
        self.estimate.setWordWrap(True)
        for w in (self.save_dir, change, self.keep_days):
            self.save_all.toggled.connect(w.setEnabled)
            w.setEnabled(self.save_all.isChecked())

        f = QFormLayout(page)
        f.addRow(self.save_all)
        f.addRow("Folder", dir_row)
        f.addRow("Delete after", self.keep_days)
        f.addRow("", self.estimate)
        f.addRow("", _note("Images go in a folder per day. Favorites (S while it runs) are copied to "
                           "favorites\\ and never deleted. `uv run daymovie.py` makes a time-lapse of a day."))
        return page

    # ----- helpers

    def _screen(self):
        return next((s for s in QGuiApplication.screens() if s.name() == self.monitor.currentData()),
                    QGuiApplication.primaryScreen())

    def _update_size_info(self) -> None:
        if not hasattr(self, "estimate") or not hasattr(self, "size_info"):
            return  # still building the tabs
        sw, sh = physical_size(self._screen())
        w, h = save_size(sw, sh, self.resolution.currentData())
        drawn = "Stable Diffusion draws at 512 px on the short side" if self.model.currentData() == "sd" \
            else "DALL·E Mega draws 256 × 256"
        how = "and the AI upscaler (Real-ESRGAN) adds detail up to that size" if min(w, h) > 512 \
            else "and it is scaled to fit"
        self.size_info.setText(f"For this monitor ({sw} × {sh}): {w} × {h}. {drawn}, {how}.")
        if not self.save_all.isChecked():
            self.estimate.setText("Nothing is saved; press S to keep a favorite.")
            return
        mb = w * h * 0.3 / 1e6  # typical JPEG q92 size for these images
        per_day = mb * 86400 / max(self.seconds.value(), 15) / 1000
        self.estimate.setText(f"About {mb:.1f} MB an image: ≈ {per_day:.1f} GB per day of non-stop running.")

    def _show_place(self) -> None:
        p = self.place
        self.place_label.setText(f"✓ {p['name']}{', ' + p['country'] if p.get('country') else ''}" if p
                                 else "Not set yet")
        for box, key in ((self.lat, "lat"), (self.lon, "lon")):
            box.blockSignals(True)
            box.setValue(float(p[key]) if p else 0.0)
            box.blockSignals(False)

    def _coords_edited(self) -> None:
        name = (self.place or {}).get("name") or self.city.text().strip() or "Here"
        country = (self.place or {}).get("country", "")
        self.place = {"name": name, "country": country, "lat": round(self.lat.value(), 4),
                      "lon": round(self.lon.value(), 4)}
        self.coords_typed = True
        self.place_label.setText(f"✓ {name} (coordinates typed in)")
        self.weather_label.clear()

    def _lookup(self) -> bool:
        query = self.city.text().strip()
        if not query:
            self.place = None
            self._show_place()
            return True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            p = realworld.geocode(query)
        except Exception as e:
            p = None
            QMessageBox.warning(self, APP_NAME, f"Could not look up “{query}”:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
        if p is None:
            self.place_label.setText(f"“{query}” not found")
            return False
        self.place = vars(p)
        self.coords_typed = False
        self._show_place()
        self.weather_label.clear()
        return True

    def _check_weather(self) -> None:
        if not self.place:
            self.weather_label.setText("Look up a town or type coordinates first.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            p = self.place
            c = realworld.WeatherSource(realworld.Place(p["name"], p.get("country", ""), p["lat"], p["lon"])).conditions()
            desc = realworld.WMO.get(c.code, ("", ""))[0]
            sun = f" · sunrise {c.sunrise:%H:%M} · sunset {c.sunset:%H:%M}" if c.sunrise and c.sunset else ""
            self.weather_label.setText(f"Now there: {c.temp:.0f} °C, {desc} · local time {c.local:%H:%M}{sun}")
        except Exception as e:
            self.weather_label.setText(f"Couldn't get the weather: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder for saved images", self.save_dir.text())
        if folder:
            self.save_dir.setText(str(Path(folder)))

    def _open_folder(self) -> None:
        Path(self.save_dir.text()).mkdir(parents=True, exist_ok=True)
        os.startfile(self.save_dir.text())

    def _identify(self) -> None:
        labels = []
        for i, s in enumerate(QGuiApplication.screens(), 1):
            lbl = QLabel(str(i))
            lbl.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                               | Qt.WindowType.Tool)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background: #111; color: white; border: 4px solid #4aa3ff;")
            f = lbl.font()
            f.setPointSize(72)
            f.setBold(True)
            lbl.setFont(f)
            g = s.geometry()
            lbl.setGeometry(g.x() + g.width() // 2 - 110, g.y() + g.height() // 2 - 110, 220, 220)
            lbl.show()
            labels.append(lbl)
        QTimer.singleShot(2500, lambda: [l.close() for l in labels])
        self._identify_labels = labels  # keep alive

    def _accept(self, start_after: bool) -> None:
        city = self.city.text().strip()
        if city and not self.coords_typed and (not self.place or self.place.get("name", "").lower() not in city.lower()):
            if not self._lookup():
                return
        if (self.use_weather.isChecked() or self.use_words.isChecked()) and not self.place:
            if QMessageBox.question(self, APP_NAME, "No location set: weather scenes will only follow your clock "
                                    "and the calendar. Continue?") != QMessageBox.StandardButton.Yes:
                return
        if not any(b.isChecked() for b in (self.use_weather, self.use_words, self.use_news, self.use_history)):
            QMessageBox.information(self, APP_NAME, "Choose at least one thing to base the pictures on (Pictures tab).")
            return
        self.cfg.update(
            place=self.place, city=city,
            use_weather=self.use_weather.isChecked(), use_news=self.use_news.isChecked(),
            use_words=self.use_words.isChecked(), use_history=self.use_history.isChecked(),
            llm_model=self.llm_model.text().strip() or realworld.DEFAULT_LLM, llm_words=self.llm_words.value(),
            ollama_url=self.ollama_url.text().strip() or realworld.OLLAMA_URL,
            feeds=[l.strip() for l in self.feeds.toPlainText().splitlines() if l.strip()],
            skip_distressing=self.skip.isChecked(), skip_words=self.skip_words.text(),
            monitor=self.monitor.currentData(), captions=self.captions.isChecked(), clock=self.clock.isChecked(),
            clock_24h=bool(self.clock_24h.currentData()),
            seconds=self.seconds.value(), fade_seconds=round(self.fade.value(), 1),
            quiet_hours=self.quiet.isChecked(), quiet_from=self.quiet_from.time().toString("HH:mm"),
            quiet_to=self.quiet_to.time().toString("HH:mm"),
            model=self.model.currentData(), resolution=self.resolution.currentData(), steps=self.steps.value(),
            safety=self.safety.isChecked(), text_check=self.text_check.isChecked(),
            word_check=self.word_check.isChecked(),
            save_all=self.save_all.isChecked(),
            save_dir=None if Path(self.save_dir.text()) == SAVE_DIR else self.save_dir.text(),
            keep_days=self.keep_days.value(),
        )
        self.start_after = start_after
        self.accept()


# --------------------------------------------------------------------------- standalone


def start_screensaver() -> None:
    """Start screensaver.pyw without a console window, independent of this process."""
    pythonw = HERE / ".venv" / "Scripts" / "pythonw.exe"
    exe = str(pythonw) if pythonw.exists() else sys.executable
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    subprocess.Popen([exe, str(HERE / "screensaver.pyw")], cwd=HERE, creationflags=flags, close_fds=True)


def main() -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("zog.dallemega.screensaver.settings")
    app = QApplication(sys.argv)
    app.setApplicationName(f"{APP_NAME} settings")
    dlg = SettingsDialog(load_settings(), mode="standalone")
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    save_settings(dlg.cfg)
    if dlg.start_after:
        start_screensaver()
    else:
        QMessageBox.information(None, APP_NAME, "Settings saved. A screensaver that is already running uses them "
                                "after a restart (or change them there with O).")


if __name__ == "__main__":
    main()
