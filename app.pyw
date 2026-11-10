"""DALL·E Mega desktop app (PySide6).

Launch with the "DALL·E Mega" shortcut, or:  uv run app.pyw
"""

import ctypes
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LOG_PATH = HERE / "app.log"
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
if sys.stdout is None or sys.stderr is None:  # pythonw has no console; keep library prints from crashing
    _log_stream = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log_stream
    sys.stderr = sys.stderr or _log_stream

# torch must load its DLLs before PySide6: PySide6 bundles an older MSVC runtime and OpenMP
# (vcomp140), and if those load first, generation crashes with heap corruption (0xC0000374).
import torch  # noqa: F401,E402

from PIL import Image
from PySide6.QtCore import QObject, QRect, QSettings, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QGuiApplication, QIcon, QImage, QImageReader, QKeySequence, QPainter, QPalette, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sd_engine import DEFAULT_CFG, DEFAULT_STEPS, SAMPLERS, output_size  # light: diffusers loads with the model

APP_NAME = "DALL·E Mega"
OUTPUTS = HERE / "outputs"
ICON_PATH = HERE / "icon.ico"
SIZES = [256, 512, 768, 1024, 1536, 2048]  # short side; keep in sync with dalle_core.SIZES
GRIDS = [(1, "1 image"), (2, "4 images"), (3, "9 images"), (4, "16 images")]
UPSCALERS = [("ai", "AI (Real-ESRGAN)"), ("lanczos", "Smooth (Lanczos)")]
MODELS = [  # keep keys in sync with dalle_core.MODELS
    ("mega", "DALL·E Mega  (square, 256 px)"),
    ("sd", "Stable Diffusion 1.5  (any shape, 512 px)"),
]
MODEL_NAMES = {"mega": "DALL·E Mega", "sd": "Stable Diffusion 1.5"}
MODEL_NATIVE = {"mega": 256, "sd": 512}  # short side
ASPECTS = [
    ("1:1", "1:1  square"), ("4:3", "4:3"), ("3:2", "3:2"), ("16:9", "16:9  widescreen"),
    ("2:1", "2:1"), ("32:9", "32:9  panorama"), ("3:4", "3:4"), ("2:3", "2:3"),
    ("9:16", "9:16  portrait"), ("1:2", "1:2"), ("9:32", "9:32  tall strip"),
]
MEGA_DEFAULTS = {"guidance": 16, "temperature": 1.0, "top_k": 256}
SD_DEFAULTS = {"steps": DEFAULT_STEPS, "cfg": DEFAULT_CFG, "sampler": "dpmpp_2m_karras"}


def aspect_value(aspect: str) -> float:
    w, h = aspect.split(":")
    return float(w) / float(h)


def output_dims(model: str, aspect: str, size: int) -> tuple[int, int]:
    if model == "mega":
        return size, size
    w, h = output_size(aspect_value(aspect))
    scale = size / min(w, h)
    return round(w * scale), round(h * scale)


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds >= 60 else f"{seconds} s"
DISPLAY_MAX = 1024  # pixmaps kept in memory are capped at this size; files stay full size
THUMB = 88


def array_to_qimage(a) -> QImage:
    """HxWx3 uint8 array → QImage that owns its pixels.
    QImage(buffer, ...) does not copy, so `data` must stay referenced until .copy() returns."""
    h, w = a.shape[:2]
    data = a.tobytes()
    img = QImage(data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    del data
    return img


def pil_to_qimage(im: Image.Image) -> QImage:
    im = im.convert("RGB")
    data = im.tobytes()
    img = QImage(data, im.width, im.height, 3 * im.width, QImage.Format.Format_RGB888).copy()
    del data
    return img


def load_pixmap(path: Path, max_side: int = DISPLAY_MAX) -> QPixmap:
    reader = QImageReader(str(path))
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > max_side:
        reader.setScaledSize(size.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio))
    return QPixmap.fromImage(reader.read())


def show_in_explorer(path: Path) -> None:
    subprocess.Popen(["explorer", f"/select,{path}"])


# --------------------------------------------------------------------------- worker


class Worker(QObject):
    """Owns the models; lives in a background QThread so the UI never blocks."""

    loaded = Signal(str)  # status-bar text
    load_failed = Signal(str)
    status = Signal(str)
    progress = Signal(int, int)
    preview = Signal(QImage)
    finished = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.core = None
        self.cancel = threading.Event()

    def _status_bar_text(self) -> str:
        return f"Loaded: {', '.join(self.pipeline.loaded()) or 'nothing'}  ·  {self.pipeline.device_name}"

    @Slot(str)
    def load(self, model: str):
        try:
            import dalle_core

            self.core = dalle_core
            self.pipeline = dalle_core.Pipeline("float16")
            t0 = time.perf_counter()
            self.pipeline.load(model, on_status=self.status.emit)
            logging.info("%s loaded on %s in %.1fs", model, self.pipeline.device_name, time.perf_counter() - t0)
            self.loaded.emit(self._status_bar_text())
        except Exception:
            logging.exception("model load failed")
            self.load_failed.emit(traceback.format_exc())
            return
        try:  # while the user types: a Generate click just queues behind this
            self.pipeline.upscaler.warmup()
        except Exception:
            logging.exception("upscaler warm-up failed (will retry on first use)")

    @Slot(dict)
    def run(self, params: dict):
        self.cancel.clear()
        try:
            if self.pipeline is None:  # initial load failed: try again now
                import dalle_core

                self.core = dalle_core
                self.pipeline = dalle_core.Pipeline("float16")
            s = self.core.Settings(**params)
            result = self.pipeline.run(
                s,
                OUTPUTS,
                on_step=self.progress.emit,
                on_preview=lambda a: self.preview.emit(array_to_qimage(a)),
                on_status=self.status.emit,
                cancel=self.cancel,
            )
            logging.info("generated %d file(s) in %.1fs: %s", len(result.paths), result.seconds, vars(result.settings))
            s = result.settings
            # Mega: keep the generated grid layout; SD (any shape): let the view pick the best fit
            cols = (1 if s.seamless else s.grid) if s.model == "mega" else 0
            display = []
            for im in result.images:
                if max(im.size) > DISPLAY_MAX:
                    im = im.copy()
                    im.thumbnail((DISPLAY_MAX, DISPLAY_MAX), Image.LANCZOS)
                display.append(pil_to_qimage(im))
            if s.model == "mega":
                thumb = self.core.make_grid(result.images, cols, max(THUMB // cols, 16))
            else:
                thumb = result.images[0].copy()
                thumb.thumbnail((THUMB, THUMB), Image.LANCZOS)
            self.finished.emit(
                {
                    "settings": vars(s),
                    "paths": [str(p) for p in result.paths],
                    "seconds": result.seconds,
                    "cols": cols,
                    "dims": list(result.images[0].size),
                    "images": display,
                    "thumb": pil_to_qimage(thumb),
                    "status_bar": self._status_bar_text(),
                    "notes": result.notes,
                }
            )
        except Exception as e:
            if self.core is not None and isinstance(e, self.core.Cancelled):
                logging.info("generation cancelled")
                self.cancelled.emit()
            else:
                logging.exception("generation failed")
                self.failed.emit(traceback.format_exc())


# --------------------------------------------------------------------------- image view


class BatchView(QWidget):
    """Shows the current batch as a grid, scaled to fit. Double-click opens an image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.pixmaps: list[QPixmap] = []
        self.paths: list[Path] = []
        self.cols = 1
        self.is_preview = False
        self.tiled = False  # draw a seamless texture 2×2 to show it tiles
        self.seamless = False
        self.message = "Your images will appear here"
        self.hover = -1
        self.rects: list[QRect] = []

    def show_message(self, text: str):
        self.pixmaps, self.paths, self.message, self.hover = [], [], text, -1
        self.update()

    def set_preview(self, pm: QPixmap):
        self.pixmaps, self.paths, self.cols, self.is_preview, self.hover = [pm], [], 1, True, -1
        self.update()

    def set_batch(self, pixmaps: list[QPixmap], paths: list[Path], cols: int, seamless: bool):
        """cols=0: pick the column count that shows the images largest."""
        self.pixmaps, self.paths, self.cols, self.is_preview = pixmaps, paths, cols, False
        self.seamless, self.hover = seamless, -1
        self.tiled = self.tiled and seamless
        self.update()

    def _layout(self) -> list[QRect]:
        n = len(self.pixmaps)
        if not n:
            return []
        ratio = self.pixmaps[0].width() / max(1, self.pixmaps[0].height())  # w / h, same for the whole batch
        gap = 0 if self.is_preview or n == 1 else 10
        margin = 16
        avail_w, avail_h = self.width() - 2 * margin, self.height() - 2 * margin

        def cell_width(cols: int) -> float:
            rows = -(-n // cols)
            return min((avail_w - (cols - 1) * gap) / cols, (avail_h - (rows - 1) * gap) / rows * ratio)

        cols = self.cols or max(range(1, n + 1), key=lambda c: (cell_width(c), -c))
        rows = -(-n // cols)
        cw = max(16, int(cell_width(cols)))
        ch = max(16, int(cw / ratio))
        total_w, total_h = cols * cw + (cols - 1) * gap, rows * ch + (rows - 1) * gap
        x0, y0 = (self.width() - total_w) // 2, (self.height() - total_h) // 2
        return [QRect(x0 + (i % cols) * (cw + gap), y0 + (i // cols) * (ch + gap), cw, ch) for i in range(n)]

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), self.palette().base())
        if not self.pixmaps:
            p.setPen(self.palette().placeholderText().color())
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        self.rects = self._layout()
        for i, (pm, r) in enumerate(zip(self.pixmaps, self.rects)):
            if self.tiled and self.seamless and not self.is_preview:
                half = r.width() // 2
                for dy in (0, 1):
                    for dx in (0, 1):
                        p.drawPixmap(QRect(r.x() + dx * half, r.y() + dy * half, half, half), pm)
            else:
                p.drawPixmap(r, pm)
            if i == self.hover and self.paths:
                p.setPen(QPen(self.palette().highlight().color(), 3))
                p.drawRect(r.adjusted(1, 1, -2, -2))

    def index_at(self, pos) -> int:
        for i, r in enumerate(self.rects):
            if r.contains(pos):
                return i
        return -1

    def mouseMoveEvent(self, e):
        i = self.index_at(e.position().toPoint()) if self.paths else -1
        if i != self.hover:
            self.hover = i
            self.setCursor(Qt.CursorShape.PointingHandCursor if i >= 0 else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, _e):
        if self.hover != -1:
            self.hover = -1
            self.update()

    def mouseDoubleClickEvent(self, e):
        i = self.index_at(e.position().toPoint())
        if 0 <= i < len(self.paths):
            os.startfile(self.paths[i])

    def contextMenuEvent(self, e):
        i = self.index_at(e.pos())
        if not (0 <= i < len(self.paths)):
            return
        path = self.paths[i]
        menu = QMenu(self)
        menu.addAction("Open", lambda: os.startfile(path))
        menu.addAction("Copy image", lambda: QGuiApplication.clipboard().setImage(QImage(str(path))))
        menu.addAction("Save as…", lambda: self._save_as(path))
        menu.addAction("Show in Explorer", lambda: show_in_explorer(path))
        if self.seamless:
            menu.addSeparator()
            act = menu.addAction("Show tiled (2×2)")
            act.setCheckable(True)
            act.setChecked(self.tiled)
            act.toggled.connect(self._set_tiled)
        menu.exec(e.globalPos())

    def _set_tiled(self, on: bool):
        self.tiled = on
        self.update()

    def _save_as(self, path: Path):
        dest, _ = QFileDialog.getSaveFileName(self, "Save image", str(Path.home() / "Pictures" / path.name), "PNG image (*.png)")
        if dest:
            shutil.copyfile(path, dest)


# --------------------------------------------------------------------------- main window


class PromptEdit(QPlainTextEdit):
    """Wrapping prompt box where Enter generates (prompts are single-line to the model)."""

    submitted = Signal()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.submitted.emit()
            return
        super().keyPressEvent(e)


class MainWindow(QMainWindow):
    request_load = Signal(str)
    request_run = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.settings = QSettings("zog", "DALL-E Mega")
        self.busy = False
        self.ready = False
        self.current_item: QListWidgetItem | None = None
        self.sample_start: float | None = None
        self._build_ui()
        self._restore()
        self._update_model_ui()
        self._set_busy(False)  # Generate stays disabled until the first model has loaded

        self.worker_thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.worker_thread)
        self.request_load.connect(self.worker.load)
        self.request_run.connect(self.worker.run)
        self.worker.loaded.connect(self.on_loaded)
        self.worker.load_failed.connect(self.on_load_failed)
        self.worker.status.connect(self.status.setText)
        self.worker.progress.connect(self.on_progress)
        self.worker.preview.connect(self.on_preview)  # bound slot → queued to the GUI thread (a lambda would run in the worker)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker_thread.start()
        self.progress.setRange(0, 0)  # indeterminate while loading
        self.status.setText("Starting…")
        self.request_load.emit(self.model.currentData())

    # ----- UI construction

    def _build_ui(self):
        # controls
        self.prompt = PromptEdit()
        self.prompt.setPlaceholderText("Describe the image…  (Enter to generate)")
        self.prompt.setFixedHeight(96)
        self.prompt.submitted.connect(self.generate)
        self.negative_label = QLabel("Negative prompt")
        self.negative = QLineEdit()
        self.negative.setPlaceholderText("what to avoid, e.g. blurry, text, watermark")
        self.negative.setClearButtonEnabled(True)
        self.negative.setToolTip("Stable Diffusion steers away from what you list here.")
        self.negative.returnPressed.connect(self.generate)

        self.model = QComboBox()
        for key, label in MODELS:
            self.model.addItem(label, key)
        self.model.setToolTip("DALL·E Mega (2022): square images, its own primitive look.\n"
                              "Stable Diffusion 1.5 (2022): any shape, sharper and more controllable.\n"
                              "A model loads (and downloads, once) when you generate with it; only one is\n"
                              "kept in GPU memory, so switching models takes a few seconds to ~25 s.")

        self.aspect = QComboBox()
        for key, label in ASPECTS:
            self.aspect.addItem(label, key)

        self.grid = QComboBox()
        for n, label in GRIDS:
            self.grid.addItem(label, n)
        self.grid.setToolTip("How many variations to make in one batch (a batch is faster per image).\n"
                             "DALL·E Mega with Seamless on: one tileable texture made of N×N tiles.")

        self.size_combo = QComboBox()
        for s in SIZES:
            self.size_combo.addItem(str(s), s)
        self.size_combo.setToolTip("Output size. DALL·E Mega draws 256 px, Stable Diffusion 512 px on the short side;\n"
                                   "larger sizes are upscaled.")
        self.size_hint = self._hint_label()

        self.upscaler = QComboBox()
        for key, label in UPSCALERS:
            self.upscaler.addItem(label, key)
        self.upscaler.setToolTip("AI adds plausible detail (sharper); Smooth only interpolates (softer).")

        self.seed = QSpinBox()
        self.seed.setRange(1, 2**31 - 1)
        self.seed.setToolTip("Same prompt + seed + settings gives the same image.")
        self.random_seed = QCheckBox("Random")
        self.random_seed.toggled.connect(lambda on: self.seed.setEnabled(not on))
        seed_row = QHBoxLayout()
        seed_row.addWidget(self.seed, 1)
        seed_row.addWidget(self.random_seed)

        form = QFormLayout()
        form.addRow("Model", self.model)
        form.addRow("Shape", self.aspect)
        form.addRow("Images", self.grid)
        form.addRow("Size", self.size_combo)
        form.addRow("", self.size_hint)
        form.addRow("Upscaler", self.upscaler)
        form.addRow("Seed", seed_row)

        self.adv_toggle = QToolButton()
        self.adv_toggle.setText("Advanced")
        self.adv_toggle.setCheckable(True)
        self.adv_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.adv_toggle.setStyleSheet("QToolButton { border: none; }")
        self.adv_toggle.toggled.connect(self._toggle_advanced)

        # DALL·E Mega settings
        self.guidance = QSpinBox()
        self.guidance.setRange(1, 64)
        self.guidance.setToolTip("Supercondition factor: how strongly to follow the prompt (default 16).")
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.1, 3.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setToolTip("Lower = safer and more uniform, higher = more varied (default 1.0).")
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 16384)
        self.top_k.setToolTip("Sample from the k most likely tokens at each step (default 256).")
        self.seamless = QCheckBox("Tileable texture")
        self.seamless.setToolTip("Make one texture whose edges wrap around seamlessly.")
        self.mega_adv = QWidget()
        mega_form = QFormLayout(self.mega_adv)
        mega_form.setContentsMargins(0, 0, 0, 0)
        mega_form.addRow("Guidance", self.guidance)
        mega_form.addRow("Temperature", self.temperature)
        mega_form.addRow("Top-k", self.top_k)
        mega_form.addRow("Seamless", self.seamless)

        # Stable Diffusion settings
        self.sd_steps = QSpinBox()
        self.sd_steps.setRange(5, 150)
        self.sd_steps.setToolTip(f"Denoising steps: more is slower, rarely much better past ~30 (default {DEFAULT_STEPS}).")
        self.sd_cfg = QDoubleSpinBox()
        self.sd_cfg.setRange(1.0, 20.0)
        self.sd_cfg.setSingleStep(0.5)
        self.sd_cfg.setDecimals(1)
        self.sd_cfg.setToolTip(f"Guidance scale: how strictly to follow the prompt (default {DEFAULT_CFG}).\n"
                               "Too high looks overcooked.")
        self.sd_sampler = QComboBox()
        for key, label in SAMPLERS.items():
            self.sd_sampler.addItem(label, key)
        self.sd_sampler.setToolTip("Sampling method. DPM++ 2M Karras is good at 20–30 steps;\n"
                                   "Euler a gives more varied, softer results.")
        self.sd_safety = QCheckBox("Safety filter")
        self.sd_safety.setToolTip("Stable Diffusion's NSFW checker: blacks out images it flags.\n"
                                  "It also has false positives; turn off if harmless images come out black.")
        self.sd_adv = QWidget()
        sd_form = QFormLayout(self.sd_adv)
        sd_form.setContentsMargins(0, 0, 0, 0)
        sd_form.addRow("Steps", self.sd_steps)
        sd_form.addRow("Guidance", self.sd_cfg)
        sd_form.addRow("Sampler", self.sd_sampler)
        sd_form.addRow("", self.sd_safety)

        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self._reset_advanced)
        self.adv_box = QWidget()
        adv_layout = QVBoxLayout(self.adv_box)
        adv_layout.setContentsMargins(12, 0, 0, 0)
        adv_layout.addWidget(self.mega_adv)
        adv_layout.addWidget(self.sd_adv)
        adv_layout.addWidget(reset)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setDefault(True)
        self.generate_btn.clicked.connect(self.generate)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.generate_btn, 2)
        btn_row.addWidget(self.cancel_btn, 1)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.status = QLabel()
        self.status.setWordWrap(True)
        open_dir = QPushButton("Open outputs folder")
        open_dir.clicked.connect(lambda: (OUTPUTS.mkdir(exist_ok=True), os.startfile(OUTPUTS)))

        left = QWidget()
        left.setFixedWidth(340)
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("<b>Prompt</b>"))
        lv.addWidget(self.prompt)
        lv.addWidget(self.negative_label)
        lv.addWidget(self.negative)
        lv.addLayout(form)
        lv.addWidget(self.adv_toggle)
        lv.addWidget(self.adv_box)
        lv.addStretch(1)
        lv.addLayout(btn_row)
        lv.addWidget(self.progress)
        lv.addWidget(self.status)
        lv.addWidget(open_dir)

        # results
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.view = BatchView()
        self.history = QListWidget()
        self.history.setViewMode(QListView.ViewMode.IconMode)
        self.history.setFlow(QListView.Flow.LeftToRight)
        self.history.setWrapping(False)
        self.history.setMovement(QListView.Movement.Static)
        self.history.setIconSize(QSize(THUMB, THUMB))
        self.history.setSpacing(4)
        self.history.setFixedHeight(THUMB + 30)
        self.history.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.history.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history.customContextMenuRequested.connect(self._history_menu)
        self.history.itemClicked.connect(self._show_history_item)
        self.history.setToolTip("This session's batches. Right-click to reuse settings.")

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(self.info)
        rv.addWidget(self.view, 1)
        rv.addWidget(self.history)

        self.splitter = QSplitter()
        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)

        for w in (self.model, self.aspect, self.size_combo, self.grid, self.upscaler):
            w.currentIndexChanged.connect(self._update_model_ui)
        self.seamless.toggled.connect(self._update_model_ui)

        QShortcut(QKeySequence("Ctrl+Return"), self, self.generate)
        QShortcut(QKeySequence("Escape"), self, self.cancel)
        self.statusBar().showMessage("Loading model…")

    @staticmethod
    def _hint_label() -> QLabel:
        label = QLabel()
        label.setWordWrap(True)
        pal = label.palette()
        pal.setColor(QPalette.ColorRole.WindowText, pal.color(QPalette.ColorRole.PlaceholderText))
        label.setPalette(pal)
        font = label.font()
        font.setPointSizeF(font.pointSizeF() * 0.9)
        label.setFont(font)
        return label

    def _toggle_advanced(self, on: bool):
        self.adv_box.setVisible(on)
        self.adv_toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

    def _reset_advanced(self):
        self.guidance.setValue(MEGA_DEFAULTS["guidance"])
        self.temperature.setValue(MEGA_DEFAULTS["temperature"])
        self.top_k.setValue(MEGA_DEFAULTS["top_k"])
        self.seamless.setChecked(False)
        self.sd_steps.setValue(SD_DEFAULTS["steps"])
        self.sd_cfg.setValue(SD_DEFAULTS["cfg"])
        self.sd_sampler.setCurrentIndex(self.sd_sampler.findData(SD_DEFAULTS["sampler"]))
        self.sd_safety.setChecked(True)

    def _update_model_ui(self):
        """Enable what applies to the chosen model and spell out the output size."""
        model = self.model.currentData()
        mega = model == "mega"
        aspect = "1:1" if mega else self.aspect.currentData()
        self.aspect.setEnabled(not mega)
        self.aspect.setToolTip("DALL·E Mega only makes squares; choose Stable Diffusion for other shapes."
                               if mega else "Width : height, drawn directly at 512 px on the short side.\n"
                               "Beyond about 2:1, SD 1.5 tends to repeat the subject across the image.")
        self.mega_adv.setVisible(mega)
        self.sd_adv.setVisible(not mega)
        self.negative.setVisible(not mega)
        self.negative_label.setVisible(not mega)

        seamless = mega and self.seamless.isChecked()
        for i, s in enumerate(SIZES):
            w, h = output_dims(model, aspect, s)
            self.size_combo.setItemText(i, f"{w} × {h}")

        size = self.size_combo.currentData()
        native = MODEL_NATIVE[model] * (self.grid.currentData() if seamless else 1)
        ai = self.upscaler.currentData() == "ai"
        self.upscaler.setEnabled(size > native)
        if size == native:
            text = "Native resolution."
        elif size < native:
            text = f"Downscaled from {native} px."
        elif not ai:
            text = f"Interpolated from {native} px (soft)."
        elif size <= 4 * native:
            text = f"AI-upscaled from {native} px."
        else:
            text = f"AI-upscaled to {4 * native} px, then interpolated (no extra detail beyond that)."
        if not mega:
            nw, nh = output_size(aspect_value(aspect))
            text = text.replace(f"{native} px", f"{nw} × {nh} px")
        self.size_hint.setText(text)

    # ----- settings persistence

    def _restore(self):
        s = self.settings
        self.prompt.setPlainText(s.value("prompt", ""))
        self.model.setCurrentIndex(max(0, self.model.findData(s.value("model", "mega"))))
        self.aspect.setCurrentIndex(max(0, self.aspect.findData(s.value("aspect", "3:2"))))
        self.negative.setText(s.value("negative", ""))
        self.sd_steps.setValue(int(s.value("sd_steps", SD_DEFAULTS["steps"])))
        self.sd_cfg.setValue(float(s.value("sd_cfg", SD_DEFAULTS["cfg"])))
        self.sd_sampler.setCurrentIndex(max(0, self.sd_sampler.findData(s.value("sd_sampler", SD_DEFAULTS["sampler"]))))
        self.sd_safety.setChecked(s.value("sd_safety", "true") == "true")
        self.grid.setCurrentIndex(max(0, self.grid.findData(int(s.value("grid", 2)))))
        self.size_combo.setCurrentIndex(max(0, self.size_combo.findData(int(s.value("size", 1024)))))
        self.upscaler.setCurrentIndex(max(0, self.upscaler.findData(s.value("upscaler", "ai"))))
        self.seed.setValue(int(s.value("seed", 42)))
        self.random_seed.setChecked(s.value("random_seed", "true") == "true")
        self.seed.setEnabled(not self.random_seed.isChecked())
        self.guidance.setValue(int(s.value("guidance", 16)))
        self.temperature.setValue(float(s.value("temperature", 1.0)))
        self.top_k.setValue(int(s.value("top_k", 256)))
        self.seamless.setChecked(s.value("seamless", "false") == "true")
        adv = s.value("advanced_open", "false") == "true"
        self.adv_toggle.setChecked(adv)
        self._toggle_advanced(adv)
        if geo := s.value("geometry"):
            self.restoreGeometry(geo)
        else:
            self.resize(1180, 820)
        if split := s.value("splitter"):
            self.splitter.restoreState(split)

    def _save_settings(self):
        s = self.settings
        s.setValue("prompt", self.prompt.toPlainText())
        s.setValue("model", self.model.currentData())
        s.setValue("aspect", self.aspect.currentData())
        s.setValue("negative", self.negative.text())
        s.setValue("sd_steps", self.sd_steps.value())
        s.setValue("sd_cfg", self.sd_cfg.value())
        s.setValue("sd_sampler", self.sd_sampler.currentData())
        s.setValue("sd_safety", "true" if self.sd_safety.isChecked() else "false")
        s.setValue("grid", self.grid.currentData())
        s.setValue("size", self.size_combo.currentData())
        s.setValue("upscaler", self.upscaler.currentData())
        s.setValue("seed", self.seed.value())
        s.setValue("random_seed", "true" if self.random_seed.isChecked() else "false")
        s.setValue("guidance", self.guidance.value())
        s.setValue("temperature", self.temperature.value())
        s.setValue("top_k", self.top_k.value())
        s.setValue("seamless", "true" if self.seamless.isChecked() else "false")
        s.setValue("advanced_open", "true" if self.adv_toggle.isChecked() else "false")
        s.setValue("geometry", self.saveGeometry())
        s.setValue("splitter", self.splitter.saveState())

    # ----- actions

    def _set_busy(self, busy: bool):
        self.busy = busy
        self.generate_btn.setEnabled(self.ready and not busy)
        self.cancel_btn.setEnabled(busy)

    def generate(self):
        if self.busy or not self.ready:
            return
        prompt = " ".join(self.prompt.toPlainText().split())
        if not prompt:
            self.status.setText("Type a prompt first.")
            self.prompt.setFocus()
            return
        model = self.model.currentData()
        mega = model == "mega"
        params = {
            "prompt": prompt,
            "model": model,
            "aspect": "1:1" if mega else self.aspect.currentData(),
            "grid": self.grid.currentData(),
            "seed": None if self.random_seed.isChecked() else self.seed.value(),
            "size": self.size_combo.currentData(),
            "upscaler": self.upscaler.currentData(),
            "temperature": self.temperature.value(),
            "top_k": self.top_k.value(),
            "supercondition": self.guidance.value(),
            "seamless": mega and self.seamless.isChecked(),
            "negative_prompt": " ".join(self.negative.text().split()),
            "steps": self.sd_steps.value(),
            "cfg": self.sd_cfg.value(),
            "sampler": self.sd_sampler.currentData(),
            "safety": self.sd_safety.isChecked(),
        }
        self._set_busy(True)
        self.progress.setRange(0, 0)  # indeterminate until sampling starts (e.g. while a model loads)
        self.sample_start = None
        self.view.show_message("Working…")
        self.info.setText(f"<i>{escape(prompt)}</i>")
        self.request_run.emit(params)

    def cancel(self):
        if self.busy:
            self.worker.cancel.set()
            self.status.setText("Cancelling…")

    # ----- worker callbacks

    @Slot(str)
    def on_loaded(self, status_bar: str):
        self.ready = True
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status.setText("Ready.")
        self.statusBar().showMessage(status_bar)
        self._set_busy(False)
        self.prompt.setFocus()

    @Slot(str)
    def on_load_failed(self, tb: str):
        self.ready = True  # let Generate retry the load
        self._set_busy(False)
        self.progress.setRange(0, 1)
        self.status.setText("Could not load the model.")
        self.statusBar().showMessage("Model not loaded")
        QMessageBox.critical(self, APP_NAME, f"Could not load the model:\n\n{tb.strip().splitlines()[-1]}\n\nDetails in {LOG_PATH}")

    @Slot(QImage)
    def on_preview(self, img: QImage):
        self.view.set_preview(QPixmap.fromImage(img))

    @Slot(int, int)
    def on_progress(self, i: int, n: int):
        if self.progress.maximum() != n:
            self.progress.setRange(0, n)
        self.progress.setValue(i)
        now = time.perf_counter()
        if self.sample_start is None or i <= 1:
            self.sample_start = now
        if i < n:
            text = f"Sampling image tokens… {i:,}/{n:,}"
            elapsed = now - self.sample_start
            if i >= 32 and elapsed > 2:
                text += f"  ·  about {fmt_duration(elapsed / i * (n - i))} left"
            self.status.setText(text)

    @Slot(dict)
    def on_finished(self, r: dict):
        self._set_busy(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        s = r["settings"]
        paths = [Path(p) for p in r["paths"]]
        pixmaps = [QPixmap.fromImage(img) for img in r["images"]]
        self.view.set_batch(pixmaps, paths, r["cols"], s["seamless"])
        self.seed.setValue(s["seed"])
        n = len(paths)
        self.status.setText(f"Done in {fmt_duration(r['seconds'])}. Saved {n} file{'s' * (n > 1)} to outputs."
                            + "".join(f"\n{note[0].upper()}{note[1:]}." for note in r["notes"]))
        self.statusBar().showMessage(r["status_bar"])
        self.info.setText(self._describe(s, n, r["dims"]))

        item = QListWidgetItem(QIcon(QPixmap.fromImage(r["thumb"])), "")
        item.setData(Qt.ItemDataRole.UserRole,
                     {"settings": s, "paths": r["paths"], "cols": r["cols"], "dims": r["dims"]})
        w, h = r["dims"]
        item.setToolTip(f"{s['prompt']}\n{MODEL_NAMES[s['model']]} · {w} × {h} · seed {s['seed']}")
        self.history.insertItem(0, item)
        self.history.setCurrentItem(item)
        self.current_item = item

    @Slot(str)
    def on_failed(self, tb: str):
        self._set_busy(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        last = tb.strip().splitlines()[-1]
        self.status.setText("Generation failed.")
        self.view.show_message("Generation failed")
        QMessageBox.warning(self, APP_NAME, f"Generation failed:\n\n{last}\n\nDetails in {LOG_PATH}")

    @Slot()
    def on_cancelled(self):
        self._set_busy(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status.setText("Cancelled.")
        if self.current_item is not None:
            self._show_history_item(self.current_item)
        else:
            self.view.show_message("Cancelled")
            self.info.clear()

    # ----- history

    @staticmethod
    def _describe(s: dict, n: int, dims) -> str:
        w, h = dims
        what = "seamless texture" if s["seamless"] else f"{n} image{'s' * (n > 1)}"
        native = MODEL_NATIVE[s["model"]] * (s["grid"] if s["seamless"] else 1)
        how = "" if min(w, h) <= native else (" · AI upscaled" if s["upscaler"] == "ai" else " · smooth resize")
        line = f"{MODEL_NAMES[s['model']]} · {what} · {w} × {h} px · seed {s['seed']}{how}"
        text = f"<i>{escape(s['prompt'])}</i>"
        if s["model"] == "sd" and s.get("negative_prompt"):
            text += f"<br><small>not: {escape(s['negative_prompt'])}</small>"
        return f"{text}<br><small>{line}</small>"

    def _show_history_item(self, item: QListWidgetItem):
        d = item.data(Qt.ItemDataRole.UserRole)
        paths = [Path(p) for p in d["paths"] if Path(p).exists()]
        if not paths:
            self.view.show_message("Files no longer exist")
            return
        self.view.set_batch([load_pixmap(p) for p in paths], paths, d["cols"], d["settings"]["seamless"])
        self.info.setText(self._describe(d["settings"], len(paths), d["dims"]))
        self.current_item = item

    def _history_menu(self, pos):
        item = self.history.itemAt(pos)
        if item is None:
            return
        d = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.addAction("Reuse prompt and settings", lambda: self._apply(d["settings"], keep_seed=True))
        menu.addAction("Reuse prompt with a new seed", lambda: self._apply(d["settings"], keep_seed=False))
        menu.addAction("Show in Explorer", lambda: show_in_explorer(Path(d["paths"][0])))
        menu.exec(self.history.mapToGlobal(pos))

    def _apply(self, s: dict, keep_seed: bool):
        self.prompt.setPlainText(s["prompt"])
        self.model.setCurrentIndex(self.model.findData(s["model"]))
        if s["model"] == "sd":
            self.aspect.setCurrentIndex(max(0, self.aspect.findData(s["aspect"])))
            self.negative.setText(s["negative_prompt"])
            self.sd_steps.setValue(s["steps"])
            self.sd_cfg.setValue(s["cfg"])
            self.sd_sampler.setCurrentIndex(max(0, self.sd_sampler.findData(s["sampler"])))
            self.sd_safety.setChecked(s["safety"])
        else:
            self.guidance.setValue(s["supercondition"])
            self.temperature.setValue(s["temperature"])
            self.top_k.setValue(s["top_k"])
            self.seamless.setChecked(s["seamless"])
        self.grid.setCurrentIndex(self.grid.findData(s["grid"]))
        self.size_combo.setCurrentIndex(self.size_combo.findData(s["size"]))
        self.upscaler.setCurrentIndex(self.upscaler.findData(s["upscaler"]))
        self.random_seed.setChecked(not keep_seed)
        if keep_seed:
            self.seed.setValue(s["seed"])

    # ----- shutdown

    def closeEvent(self, e):
        self._save_settings()
        self.worker.cancel.set()
        self.worker_thread.quit()
        if not self.worker_thread.wait(5000):  # still loading or mid-step: don't hang the close
            logging.info("worker busy at exit; terminating process")
            os._exit(0)
        e.accept()


def main():
    if sys.platform == "win32":  # own taskbar identity/icon instead of python's
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("zog.dallemega.app")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    def excepthook(etype, value, tb):
        logging.error("uncaught", exc_info=(etype, value, tb))
        QMessageBox.critical(None, APP_NAME, f"Unexpected error:\n\n{value}\n\nDetails in {LOG_PATH}")

    sys.excepthook = excepthook
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
