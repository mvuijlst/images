"""Full-screen ambient image generator: pictures based on the weather outside, three random words
turned into a scene by a local LLM (Ollama), today's news and Wikipedia's "on this day", made by
Stable Diffusion 1.5 (or DALL·E Mega) on this PC.

Launch with the "Image Screensaver" shortcut, or:  uv run screensaver.pyw [--settings] [--monitor N]
Settings: the "Screensaver Settings" shortcut, uv run screensaver_settings.py, or O while it runs.

Runs full screen until closed (Esc / Q, right-click → Exit, or Alt+F4); clicking doesn't close it.
Keys: → or Space = next image · S = favorite · C = captions · P = prompt · O = settings · H = help ·
right-click = menu. Every image is saved as JPEG + JSON sidecar in a folder per day (see settings).
"""

import argparse
import ctypes
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LOG_PATH = HERE / "screensaver.log"
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
if sys.stdout is None or sys.stderr is None:  # pythonw has no console; keep library prints from crashing
    _log_stream = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log_stream
    sys.stderr = sys.stderr or _log_stream

# torch must load its DLLs before PySide6 (PySide6 ships an older MSVC runtime/OpenMP; see app.pyw)
import torch  # noqa: F401,E402

from PIL import Image, ImageEnhance, ImageFilter
from PySide6.QtCore import QEasingCurve, QObject, QRect, Qt, QThread, QTimer, QVariantAnimation, Signal, Slot
from PySide6.QtGui import (QColor, QFont, QGuiApplication, QIcon, QImage, QLinearGradient, QPainter,
                           QPixmap)
from PySide6.QtWidgets import QApplication, QDialog, QMenu, QWidget

import hourlight  # numpy + PIL: grades frames to the light of the hour
import realworld  # light: requests + download_weights; models load in the worker
from screensaver_settings import (APP_NAME, ICON_PATH, SETTINGS_PATH, SettingsDialog, clock_text, is_quiet,  # noqa: E402
                                  load_settings, physical_size, save_root, save_settings, save_size)

JPEG_QUALITY = 92
TEXT_REDOS = 2  # an image the local LLM sees text in is made again (new seed) up to this many times
WORD_REDOS = 2  # a words draft that shows none of its words is drawn again (new seed) up to this many times
DAY_FOLDER = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HELP = ("Right-click: menu   ·   → / Space: next   ·   S: favorite   ·   C: captions   ·   P: prompt   ·   "
        "O: settings   ·   Esc: exit")


def keep_display_awake(on: bool) -> None:
    """Ask Windows not to blank the display or sleep while we run (reverts when the process ends)."""
    if sys.platform != "win32":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def pil_to_qimage(im: Image.Image) -> QImage:
    im = im.convert("RGB")
    data = im.tobytes()
    img = QImage(data, im.width, im.height, 3 * im.width, QImage.Format.Format_RGB888).copy()
    del data  # QImage(buffer) doesn't copy: keep `data` alive until .copy() returned
    return img


def fits_by_cropping(img: Image.Image, w: int, h: int) -> bool:
    return abs(img.width / img.height - w / h) / (w / h) < 0.08


def slug(text: str, n: int = 50) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _ascii(text).lower()).strip("-")[:n] or "image"


def compose(img: Image.Image, w: int, h: int) -> Image.Image:
    """Fit a generated image to the screen: cover (crop) when the shapes are close, otherwise
    (DALL·E Mega's squares) centre it on a blurred, darkened enlargement of itself."""
    if fits_by_cropping(img, w, h):
        scale = max(w / img.width, h / img.height)
        big = img.resize((max(w, round(img.width * scale)), max(h, round(img.height * scale))), Image.LANCZOS)
        x, y = (big.width - w) // 2, (big.height - h) // 2
        return big.crop((x, y, x + w, y + h))
    scale = max(w / img.width, h / img.height)
    back = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    x, y = (back.width - w) // 2, (back.height - h) // 2
    back = back.crop((x, y, x + w, y + h)).filter(ImageFilter.GaussianBlur(max(w, h) / 60))
    back = ImageEnhance.Brightness(back).enhance(0.55)
    fit = min(w / img.width, h / img.height)
    front = img.resize((round(img.width * fit), round(img.height * fit)), Image.LANCZOS)
    back.paste(front, ((w - front.width) // 2, (h - front.height) // 2))
    return back


def _ascii(text: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def write_image(folder: Path, stem: str, img: Image.Image, meta: dict) -> Path:
    """Save `stem`.jpg plus `stem`.json (all metadata). The JPEG also carries the caption in EXIF,
    so Explorer / Photos show it (Title, Subject, Comments, Tags) even without the sidecar."""
    folder.mkdir(parents=True, exist_ok=True)
    jpg, sidecar = folder / f"{stem}.jpg", folder / f"{stem}.json"
    cap = meta["caption"]
    xp = lambda s: s.encode("utf-16-le") + b"\x00\x00"  # noqa: E731  Windows XP* tags are UTF-16
    exif = Image.Exif()
    exif[0x010E] = _ascii(f"{cap['kicker']}: {cap['text']}".replace(" · ", " - "))  # ImageDescription (ASCII by spec)
    exif[0x0131] = _ascii(f"{APP_NAME} ({meta['model']})")  # Software
    stamp = datetime.fromisoformat(meta["created"]).strftime("%Y:%m:%d %H:%M:%S")
    exif[0x0132] = stamp  # DateTime
    exif.get_ifd(0x8769)[0x9003] = stamp  # DateTimeOriginal
    exif[0x9C9B] = xp(cap["text"])  # XPTitle: the real headline / event / conditions
    exif[0x9C9F] = xp(cap["kicker"])  # XPSubject
    exif[0x9C9C] = xp(f"{cap['kicker']}: {cap['text']}\n\nPrompt: {meta['prompt']}")  # XPComment
    exif[0x9C9E] = xp(";".join(t for t in ("screensaver", meta["source"], meta["model"]) if t))  # XPKeywords
    img.convert("RGB").save(jpg, "JPEG", quality=JPEG_QUALITY, optimize=True, exif=exif)
    sidecar.write_text(json.dumps({"file": jpg.name, **meta}, indent=2, ensure_ascii=False), encoding="utf-8")
    return jpg


def prune_days(root: Path, keep_days: int) -> int:
    """Delete images + sidecars in day folders older than keep_days (never the favorites)."""
    if keep_days <= 0 or not root.exists():
        return 0
    cutoff = datetime.now().date().toordinal() - keep_days
    removed = 0
    for day in root.iterdir():
        if not (day.is_dir() and DAY_FOLDER.match(day.name)):
            continue
        try:
            if datetime.strptime(day.name, "%Y-%m-%d").date().toordinal() >= cutoff:
                continue
        except ValueError:
            continue
        for f in day.iterdir():
            if f.suffix.lower() in (".jpg", ".json"):
                f.unlink()
                removed += 1
        if not any(day.iterdir()):
            day.rmdir()
    return removed


def screen_for(name: str | None):
    screens = QGuiApplication.screens()
    for s in screens:
        if s.name() == name:
            return s
    return QGuiApplication.primaryScreen()



# --------------------------------------------------------------------------- worker


class Producer(QObject):
    """Makes one finished, screen-sized frame per produce() call, in a background thread."""

    produced = Signal(dict)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, cfg: dict, size: tuple[int, int]):
        super().__init__()
        self.cfg, self.size = cfg, size
        self.cancel = threading.Event()
        self.pipeline = None
        self.sources = None
        self.core = None
        self.text_check = None
        self.word_check = None
        self._pruned = 0.0

    @Slot(dict, object)
    def reconfigure(self, cfg: dict, size: tuple) -> None:
        self.cfg, self.size = cfg, tuple(size)
        self.sources = None  # rebuilt with the new place / feeds / filters
        self.text_check = None
        self.word_check = None

    def _words_seen(self, image, words: list[str]) -> list[str] | None:
        """Which of the random words the local vision LLM sees in the image; None when off or unavailable."""
        if not words or not self.cfg.get("word_check", True):
            return None
        if self.word_check is None:
            self.word_check = realworld.WordCheck(self.cfg.get("llm_model") or realworld.DEFAULT_LLM,
                                                  self.cfg.get("ollama_url") or realworld.OLLAMA_URL)
        return self.word_check(image, words)

    def _shows_text(self, image) -> bool | None:
        """Asks the local vision LLM (Ollama) whether the image has text in it; None when off or unavailable."""
        if not self.cfg.get("text_check", True):
            return None
        if self.text_check is None:
            self.text_check = realworld.TextCheck(self.cfg.get("llm_model") or realworld.DEFAULT_LLM,
                                                  self.cfg.get("ollama_url") or realworld.OLLAMA_URL)
        return self.text_check(image)

    @Slot()
    def produce(self) -> None:
        try:
            if self.pipeline is None:
                import dalle_core

                self.core = dalle_core
                self.pipeline = dalle_core.Pipeline("float16")
            if self.sources is None:
                self.status.emit("Gathering today's weather, words, news and history…")
                self.sources = realworld.Sources(self.cfg)
            w, h = self.size  # the monitor, in physical pixels
            sw, sh = save_size(w, h, self.cfg.get("resolution", "2x"))  # made (and saved) at this size, shown fitted to the monitor
            model = self.cfg["model"]
            for _attempt in range(3):
                idea = self.sources.next()
                light = hourlight.light_for(idea.origin or {})  # blended through dawn and dusk; None by day
                t0 = time.perf_counter()
                remade = 0
                for redo in range(TEXT_REDOS + 1):  # same idea, new seed, while the image shows text
                    s = self.core.Settings(
                        prompt=idea.prompt, model=model, aspect=f"{sw}:{sh}" if model == "sd" else "1:1", grid=1,
                        size=min(sw, sh), negative_prompt=idea.negative if model == "sd" else "",
                        steps=self.cfg["steps"], safety=self.cfg["safety"],
                    )
                    r, lit, words = self._make(s, light, (idea.origin or {}).get("words") or [])
                    if r.notes:
                        break
                    text = self._shows_text(r.images[0])
                    if not text:
                        break
                    if redo == TEXT_REDOS:
                        logging.info("text in the image after %d remakes; kept it", TEXT_REDOS)
                        break
                    remade += 1
                    logging.info("text in the image (seed %s), making it again", r.settings.seed)
                if r.notes:  # blacked out by the safety filter: try another idea
                    logging.info("skipped (%s): %s", "; ".join(r.notes), idea.text.replace("\n", " · "))
                    continue
                if lit is not None:  # the last guard: a night frame is never shown in daylight
                    lit["brightness"] = round(hourlight.brightness(r.images[0]), 3)
                    if lit["brightness"] > light.ceiling:
                        r.images[0] = hourlight.grade(r.images[0], light)
                        lit["graded_after"] = True
                        logging.info("still too bright for %s (%.2f): graded", lit["light"], lit["brightness"])
                full = compose(r.images[0], sw, sh)
                shown = full if (sw, sh) == (w, h) else full.resize((w, h), Image.LANCZOS)
                meta = self._metadata(idea, r, (sw, sh), time.perf_counter() - t0)
                meta["text_check"] = {"text_seen": text, "remade_for_text": remade} \
                    if self.cfg.get("text_check", True) else None
                meta["light"] = lit
                meta["word_check"] = words
                saved = self._save(full, meta, idea) if self.cfg.get("save_all", True) else None
                logging.info("frame in %.1fs [%s] %s | %s", time.perf_counter() - t0, idea.source,
                             idea.text.replace("\n", " · "), idea.prompt)
                self.produced.emit({"image": pil_to_qimage(shown), "full": full, "meta": meta, "saved": saved,
                                    "idea": vars(idea), "settings": {**vars(r.settings), "init_image": None}})
                return
            raise RuntimeError("the safety filter blocked three images in a row")
        except Exception as e:
            if self.core is not None and isinstance(e, self.core.Cancelled):
                return
            logging.exception("producing a frame failed")
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _make(self, s, light: hourlight.Light | None, words: list[str]):
        """One image for s; how it was lit (None in full daylight); what the word check saw (None: off or no words).

        SD first draws a draft at its own size. With random words, the local vision LLM looks at the draft (still in
        daylight, where small things are easiest to see): if it shows none of the words, it is drawn again with a new
        seed, up to WORD_REDOS times. Then at night, dusk, dawn and golden hour the draft is graded to the light of the
        hour and SD draws it again from there, which turns the grade into real night or evening light (see
        hourlight.py); in full daylight the draft is just upscaled."""
        run = lambda s: self.pipeline.run(s, None, on_status=self.status.emit, cancel=self.cancel)  # noqa: E731
        sd = s.model == "sd"
        for redo in range(WORD_REDOS + 1):
            draft = run(replace(s, size=None) if sd else s)  # SD: its own size, no upscaling yet
            if draft.notes:
                return draft, None, None
            seen = self._words_seen(draft.images[0], words)
            if seen is None or seen:
                break
            if redo < WORD_REDOS:
                logging.info("none of %s in the image (seed %s), drawing it again", words, draft.settings.seed)
            else:
                logging.info("none of %s in the image after %d remakes; kept it", words, WORD_REDOS)
        checked = None if seen is None else {"words": words, "words_seen": seen, "remade_for_words": redo}
        if light is None or not sd or light.strength < hourlight.REDRAW_FROM:  # day, near day, or DALL·E Mega
            r = self._upscaled(draft, s) if sd else draft
            if light is None:
                return r, None, checked
            r.images[0] = hourlight.grade(r.images[0], light)
            return r, {"light": light.name, "graded": True}, checked
        start = hourlight.grade(draft.images[0], light)
        r = run(replace(s, seed=draft.settings.seed, init_image=start, strength=round(light.strength, 3)))
        return r, {"light": light.name, "redrawn_from_graded": True, "strength": round(light.strength, 3),
                   "brightness_draft": round(hourlight.brightness(draft.images[0]), 3),
                   "brightness_graded": round(hourlight.brightness(start), 3)}, checked

    def _upscaled(self, r, s):
        """The draft at the asked size (what Pipeline.run would have done with s.size)."""
        im = r.images[0]
        if min(im.size) == s.size:
            return r
        self.status.emit("Upscaling")
        big = self.pipeline.upscaler.resize(im, s.size, s.upscaler, s.seamless)
        torch.cuda.empty_cache()
        return replace(r, settings=replace(r.settings, size=s.size), images=[big])

    def _metadata(self, idea, r, size: tuple[int, int], seconds: float) -> dict:
        s, model = r.settings, r.settings.model
        if model == "sd":
            from sd_engine import SAMPLERS, output_size

            native = output_size(self.core.aspect_value(s.aspect))
            gen = {"seed": s.seed, "steps": s.steps, "guidance_scale": s.cfg, "sampler": SAMPLERS.get(s.sampler, s.sampler),
                   "negative_prompt": s.negative_prompt, "safety_filter": s.safety}
        else:
            native = (256, 256)
            gen = {"seed": s.seed, "temperature": s.temperature, "top_k": s.top_k, "guidance": s.supercondition}
        generated = r.images[0]
        upscaled = min(generated.size) > min(native)
        return {
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": idea.source,
            "caption": {"kicker": idea.kicker, "text": idea.text},
            "origin": idea.origin,
            "prompt": idea.prompt,
            "model": self.core.MODELS[model],
            "generation": gen,
            "image": {
                "width": size[0], "height": size[1], "format": "JPEG", "quality": JPEG_QUALITY,
                "drawn_at": list(native),
                "upscaled_with": ("Real-ESRGAN x4plus" if s.upscaler == "ai" else "Lanczos") if upscaled else None,
                "fitted": "cropped to the screen's shape" if fits_by_cropping(generated, *size)
                else "centred on a blurred backdrop",
            },
            "screen": {"monitor": self.cfg.get("monitor"), "width": self.size[0], "height": self.size[1]},
            "seconds_to_make": round(seconds, 1),
            "app": APP_NAME,
        }

    def _save(self, full: Image.Image, meta: dict, idea) -> str | None:
        root = save_root(self.cfg)
        now = datetime.fromisoformat(meta["created"])
        try:
            path = write_image(root / f"{now:%Y-%m-%d}", f"{now:%H%M%S}_{idea.source}_{slug(idea.text)}", full, meta)
        except OSError:  # disk full, folder gone…: keep showing images
            logging.exception("saving failed")
            return None
        keep = int(self.cfg.get("keep_days", 0) or 0)
        if keep > 0 and time.time() - self._pruned > 3600:
            self._pruned = time.time()
            try:
                if n := prune_days(root, keep):
                    logging.info("removed %d files older than %d days", n, keep)
            except OSError:
                logging.exception("pruning old images failed")
        return str(path)


# --------------------------------------------------------------------------- full-screen window


class Screensaver(QWidget):
    request_frame = Signal()
    request_reconfigure = Signal(dict, object)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle(APP_NAME)
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.current: QPixmap | None = None
        self.previous: QPixmap | None = None
        self.current_meta: dict | None = None
        self.previous_meta: dict | None = None
        self.pending: dict | None = None
        self.producing = False
        self.due = False  # the display time ran out while the next frame wasn't ready
        self.show_prompt = False
        self.fade = 1.0
        self.message = "Starting…"
        self.toast, self.toast_until = "", 0.0
        self.help_until = time.time() + 10
        self.quiet = False  # in the quiet hours: black screen, nothing made, the display may sleep

        self.anim = QVariantAnimation(self)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.anim.valueChanged.connect(self._on_fade)

        self.display_timer = QTimer(self)
        self.display_timer.setSingleShot(True)
        self.display_timer.timeout.connect(self._advance)
        self.tick = QTimer(self)  # clock, toasts, help
        self.tick.timeout.connect(self.update)
        self.tick.start(1000)
        self.hide_cursor = QTimer(self)  # the pointer shows while the mouse moves, then hides again
        self.hide_cursor.setSingleShot(True)
        self.hide_cursor.setInterval(3000)
        self.hide_cursor.timeout.connect(lambda: self.setCursor(Qt.CursorShape.BlankCursor))
        self.setMouseTracking(True)
        self.quiet_timer = QTimer(self)
        self.quiet_timer.timeout.connect(self._check_quiet)
        self.quiet_timer.start(20_000)

        self.worker_thread = QThread(self)
        self.producer = Producer(cfg, (1920, 1080))
        self.producer.moveToThread(self.worker_thread)
        self.request_frame.connect(self.producer.produce)
        self.request_reconfigure.connect(self.producer.reconfigure)
        self.producer.produced.connect(self.on_produced)
        self.producer.status.connect(self.on_status)
        self.producer.failed.connect(self.on_failed)
        self.worker_thread.start()

    # ----- placement

    def place_on_screen(self) -> None:
        screen = screen_for(self.cfg.get("monitor"))
        g = screen.geometry()
        # With monitors at different scaling (e.g. 150 % and 100 %), Qt can put a full-screen window
        # on the wrong one if it goes full screen straight away. First show a small window well inside
        # the target monitor so Windows assigns it there, then go full screen.
        self.showNormal()
        self.setGeometry(g.x() + g.width() // 4, g.y() + g.height() // 4, g.width() // 2, g.height() // 2)
        self.show()
        QApplication.processEvents()
        self.windowHandle().setScreen(screen)
        self.showFullScreen()
        QApplication.processEvents()
        if self.screen() != screen:  # second attempt via the native window
            self.windowHandle().setScreen(screen)
            self.windowHandle().setGeometry(g)
            self.showFullScreen()
            QApplication.processEvents()
        logging.info("placed on %s (wanted %s), full screen: %s", self.screen().name(), screen.name(), self.isFullScreen())
        self.raise_()
        self.activateWindow()
        self.setFocus()
        self.request_reconfigure.emit(self.cfg, physical_size(screen))

    def start(self) -> None:
        self.place_on_screen()
        self._check_quiet()
        self._request()

    def _check_quiet(self) -> None:
        quiet = is_quiet(self.cfg)
        if quiet == self.quiet:
            return
        self.quiet = quiet
        if quiet:
            logging.info("quiet hours: black screen until %s", self.cfg.get("quiet_to"))
            self.display_timer.stop()
            keep_display_awake(False)
        else:
            logging.info("quiet hours over")
            keep_display_awake(True)
            self.due = True
            if self.pending is not None:
                self._advance()
            else:
                self._request()
        self.update()

    # ----- frame flow

    def _request(self) -> None:
        if not self.producing and not self.quiet:
            self.producing = True
            self.request_frame.emit()

    @Slot(dict)
    def on_produced(self, frame: dict) -> None:
        self.producing = False
        if (self.current is None or self.due) and not self.quiet:
            self._show(frame)
        else:
            self.pending = frame

    def _show(self, frame: dict) -> None:
        self.due = False
        pm = QPixmap.fromImage(frame["image"])
        pm.setDevicePixelRatio(self.devicePixelRatioF())
        self.previous, self.previous_meta = self.current, self.current_meta
        self.current, self.current_meta = pm, frame
        self.message = ""
        self.anim.stop()
        fade_ms = int(float(self.cfg.get("fade_seconds", 2.5)) * 1000)
        self.fade = 0.0 if self.previous is not None and fade_ms > 0 else 1.0
        if self.fade < 1.0:
            self.anim.setDuration(fade_ms)
            self.anim.start()
        else:
            self.previous, self.previous_meta = None, None
        self.update()
        self.display_timer.start(int(self.cfg.get("seconds", 60)) * 1000)
        self._request()  # start on the next one right away

    def _advance(self) -> None:
        if self.quiet:
            return
        if self.pending is not None:
            frame, self.pending = self.pending, None
            self._show(frame)
        else:
            self.due = True
            self._request()

    def _on_fade(self, v) -> None:
        self.fade = float(v)
        if self.fade >= 1.0:
            self.previous, self.previous_meta = None, None
        self.update()

    @Slot(str)
    def on_status(self, text: str) -> None:
        if self.current is None:
            self.message = text
            self.update()

    @Slot(str)
    def on_failed(self, text: str) -> None:
        self.producing = False
        if self.current is None:
            self.message = f"Couldn't make an image ({text}). Trying again in 30 s…"
        else:
            self._toast("Couldn't make the next image; trying again in 30 s")
        self.update()
        QTimer.singleShot(30_000, self._request)

    # ----- input

    def keyPressEvent(self, e) -> None:
        k = e.key()
        if k in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_Space, Qt.Key.Key_N):
            self._next()
        elif k == Qt.Key.Key_S:
            self._favorite()
        elif k == Qt.Key.Key_C:
            self._toggle_captions()
        elif k == Qt.Key.Key_P:
            self._toggle_prompt()
        elif k in (Qt.Key.Key_H, Qt.Key.Key_F1):
            self.help_until = time.time() + 10
            self.update()
        elif k == Qt.Key.Key_O:
            self._open_settings()
        else:
            super().keyPressEvent(e)

    def _next(self) -> None:
        if self.quiet:
            self._toast(f"Quiet hours until {self.cfg.get('quiet_to')}")
            return
        if self.pending is not None:
            self.display_timer.stop()
            self._advance()
        else:
            self.due = True
            self._request()
            self._toast("Making the next image…")

    def _toggle_captions(self) -> None:
        self.cfg["captions"] = not self.cfg.get("captions", True)
        save_settings(self.cfg)
        self.update()

    def _toggle_prompt(self) -> None:
        self.show_prompt = not self.show_prompt
        self.update()

    # Clicking never closes the window: it runs until closed with Esc / Q, the menu, or Alt+F4.
    def mouseMoveEvent(self, _e) -> None:
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.hide_cursor.start()

    def mousePressEvent(self, _e) -> None:
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.hide_cursor.start()

    def contextMenuEvent(self, e) -> None:
        self.hide_cursor.stop()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        menu = QMenu(self)
        menu.addAction("Next image\t→", self._next)
        menu.addAction("Add to favorites\tS", self._favorite).setEnabled(self.current_meta is not None)
        captions = menu.addAction("Captions\tC", self._toggle_captions)
        captions.setCheckable(True)
        captions.setChecked(self.cfg.get("captions", True))
        prompt = menu.addAction("Show prompt\tP", self._toggle_prompt)
        prompt.setCheckable(True)
        prompt.setChecked(self.show_prompt)
        menu.addAction("Open saved images", self._open_folder)
        menu.addAction("Settings…\tO", self._open_settings)
        menu.addSeparator()
        menu.addAction("Exit\tEsc", self.close)
        menu.exec(e.globalPos())
        self.hide_cursor.start()

    def _open_folder(self) -> None:
        folder = save_root(self.cfg)
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def _open_settings(self) -> None:
        self.display_timer.stop()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        dlg = SettingsDialog(self.cfg, self, mode="running")
        accepted = dlg.exec() == QDialog.DialogCode.Accepted
        self.setCursor(Qt.CursorShape.BlankCursor)
        if accepted:
            self.cfg = dlg.cfg
            save_settings(self.cfg)
            self.pending = None  # made with the old settings
            self.place_on_screen()
            self.quiet = not is_quiet(self.cfg)  # force _check_quiet to apply the (new) quiet hours
            self._check_quiet()
            self._toast("Settings saved; new images follow")
        if self.current is not None and not self.quiet:
            self.display_timer.start(3000 if accepted else int(self.cfg.get("seconds", 60)) * 1000)
        self._request()

    def _favorite(self) -> None:
        """Copy the current image (+ sidecar, marked "favorite") into <save folder>/favorites."""
        m = self.current_meta
        if not m:
            return
        folder = save_root(self.cfg) / "favorites"
        meta = dict(m["meta"], favorite=True)
        created = datetime.fromisoformat(meta["created"])
        stem = f"{created:%Y-%m-%d_%H%M%S}_{meta['source']}_{slug(meta['caption']['text'])}"
        try:
            saved = Path(m["saved"]) if m.get("saved") else None
            if saved and saved.exists():  # already on disk: copy, mark the sidecar
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(saved, folder / f"{stem}.jpg")
                (folder / f"{stem}.json").write_text(json.dumps({"file": f"{stem}.jpg", **meta}, indent=2,
                                                                ensure_ascii=False), encoding="utf-8")
            else:
                write_image(folder, stem, m["full"], meta)
        except OSError as e:
            logging.exception("saving favorite failed")
            self._toast(f"Couldn't save: {e}")
            return
        self._toast("★ Added to favorites")

    def _toast(self, text: str) -> None:
        self.toast, self.toast_until = text, time.time() + 4
        self.update()

    # ----- painting

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        r = self.rect()
        if self.quiet:
            if time.time() < self.toast_until:
                self._text(p, r, self.toast, 0.016, Qt.AlignmentFlag.AlignCenter, QColor(255, 255, 255, 120))
            return
        if self.previous is not None and self.fade < 1.0:
            p.drawPixmap(r, self.previous)
        if self.current is not None:
            p.setOpacity(self.fade)
            p.drawPixmap(r, self.current)
            p.setOpacity(1.0)
        if self.cfg.get("captions", True) and self.current_meta:
            p.setOpacity(self.fade)
            self._draw_caption(p, self.current_meta["idea"])
            p.setOpacity(1.0)
        if self.cfg.get("clock", True):
            self._draw_clock(p)
        if self.message and self.current is None:
            self._text(p, r, self.message, 0.022, Qt.AlignmentFlag.AlignCenter, QColor(200, 200, 200))
        now = time.time()
        if now < self.help_until:
            self._text(p, r.adjusted(0, int(r.height() * 0.02), 0, 0), HELP, 0.014,
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, QColor(255, 255, 255, 170))
        if now < self.toast_until:
            self._text(p, r.adjusted(0, 0, -int(r.width() * 0.02), 0).adjusted(0, int(r.height() * 0.05), 0, 0),
                       self.toast, 0.016, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, QColor(255, 255, 255, 220))

    def _font(self, rel: float, bold=False, spacing=0.0) -> QFont:
        f = QFont(self.font())
        f.setPixelSize(max(9, int(self.height() * rel)))
        f.setBold(bold)
        if spacing:
            f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 100 + spacing)
        return f

    def _text(self, p, rect, text, rel, align, color) -> None:
        p.setFont(self._font(rel))
        p.setPen(QColor(0, 0, 0, 160))
        p.drawText(rect.translated(2, 2), int(align) | int(Qt.TextFlag.TextWordWrap), text)
        p.setPen(color)
        p.drawText(rect, int(align) | int(Qt.TextFlag.TextWordWrap), text)

    def _draw_caption(self, p: QPainter, idea: dict) -> None:
        r = self.rect()
        band = int(r.height() * 0.30)
        grad = QLinearGradient(0, r.height() - band, 0, r.height())
        grad.setColorAt(0, QColor(0, 0, 0, 0))
        grad.setColorAt(1, QColor(0, 0, 0, 175))
        p.fillRect(QRect(0, r.height() - band, r.width(), band), grad)

        margin = int(r.height() * 0.045)
        width = int(r.width() * 0.62)
        kicker_font = self._font(0.0155, bold=True, spacing=12)
        text_font = self._font(0.029)
        text = idea["text"]
        prompt = f"Prompt: {idea['prompt']}" if self.show_prompt else ""
        p.setFont(text_font)
        text_h = p.fontMetrics().boundingRect(QRect(0, 0, width, 10_000), int(Qt.TextFlag.TextWordWrap), text).height()
        prompt_font = self._font(0.0135)
        p.setFont(prompt_font)
        prompt_h = p.fontMetrics().boundingRect(QRect(0, 0, width, 10_000), int(Qt.TextFlag.TextWordWrap),
                                                prompt).height() if prompt else 0
        y = r.height() - margin - text_h - (prompt_h + int(margin * 0.3) if prompt else 0)
        p.setFont(kicker_font)
        kh = p.fontMetrics().height()
        p.setPen(QColor(255, 255, 255, 190))
        p.drawText(QRect(margin, y - kh - int(margin * 0.15), width, kh), int(Qt.AlignmentFlag.AlignLeft),
                   idea["kicker"].upper())
        p.setFont(text_font)
        p.setPen(QColor(0, 0, 0, 150))
        p.drawText(QRect(margin + 2, y + 2, width, text_h), int(Qt.TextFlag.TextWordWrap), text)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRect(margin, y, width, text_h), int(Qt.TextFlag.TextWordWrap), text)
        if prompt:
            p.setFont(prompt_font)
            p.setPen(QColor(255, 255, 255, 170))
            p.drawText(QRect(margin, y + text_h + int(margin * 0.3), width, prompt_h), int(Qt.TextFlag.TextWordWrap), prompt)

    def _draw_clock(self, p: QPainter) -> None:
        r = self.rect()
        margin = int(r.height() * 0.045)
        now = datetime.now()
        p.setFont(self._font(0.06))
        th = p.fontMetrics().height()
        box = QRect(r.width() - margin - int(r.width() * 0.3), r.height() - margin - th - int(r.height() * 0.03),
                    int(r.width() * 0.3), th)
        for dx, color in ((2, QColor(0, 0, 0, 150)), (0, QColor(255, 255, 255, 235))):
            p.setPen(color)
            p.drawText(box.translated(dx, dx), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom),
                       clock_text(now, self.cfg.get("clock_24h", True)))
        p.setFont(self._font(0.017))
        date_box = QRect(box.x(), box.bottom() + 2, box.width(), int(r.height() * 0.03))
        for dx, color in ((2, QColor(0, 0, 0, 150)), (0, QColor(255, 255, 255, 200))):
            p.setPen(color)
            p.drawText(date_box.translated(dx, dx), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop),
                       f"{now:%A %d %B}")

    # ----- shutdown

    def closeEvent(self, e) -> None:
        self.display_timer.stop()
        self.producer.cancel.set()
        self.worker_thread.quit()
        if not self.worker_thread.wait(3000):  # mid-step or loading: don't hang the exit
            logging.info("worker busy at exit; terminating process")
            keep_display_awake(False)
            os._exit(0)
        keep_display_awake(False)
        e.accept()


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-screen real-world image generator")
    ap.add_argument("--settings", action="store_true", help="open the settings dialog first")
    ap.add_argument("--monitor", type=int, help="monitor number as shown in settings (1, 2, …)")
    args = ap.parse_args()

    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("zog.dallemega.screensaver")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    def excepthook(etype, value, tb):
        logging.error("uncaught", exc_info=(etype, value, tb))

    sys.excepthook = excepthook

    cfg = load_settings()
    if args.monitor:
        screens = QGuiApplication.screens()
        if 1 <= args.monitor <= len(screens):
            cfg["monitor"] = screens[args.monitor - 1].name()
    if args.settings or not SETTINGS_PATH.exists():
        dlg = SettingsDialog(cfg, mode="first")
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.cfg
        save_settings(cfg)

    keep_display_awake(True)
    win = Screensaver(cfg)
    win.start()
    code = app.exec()
    keep_display_awake(False)
    sys.exit(code)


if __name__ == "__main__":
    main()
