"""Image Movie: a movie from a folder of screensaver images and their .json sidecars, with smooth transitions
and the weather and time drawn the way the screensaver draws them.

  uv run moviemaker.py                                 opens the window
  uv run moviemaker.py outputs/screensaver/2026-11-09  makes the movie from the command line
  uv run moviemaker.py FOLDER --length 1:30 --size 1920x1080 --transition slide --show weather,clock,caption

Choosing the images: a whole day is ~1300 images, which in 3 minutes is 0.13 s each: too fast for any
transition. So by default as many images are taken as fit with about --seconds (2) each, spread evenly over
the folder (one every ~16 minutes of a day), each one dissolving into the next while slowly zooming and
panning. --every-image keeps them all, as a time-lapse.

Overlays (--show): weather (bottom left, as the screensaver's caption), clock (time and date the image was
made, bottom right), caption (the screensaver's own caption: the words, the headline…), prompt.
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QEasingCurve, QObject, QPointF, QRect, QRectF, Qt, QThread, QTime, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QImage, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QRadioButton,
                               QTimeEdit, QVBoxLayout, QWidget)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from screensaver_settings import ICON_PATH, clock_text, load_settings, save_root  # noqa: E402

APP_NAME = "Image Movie"
TRANSITIONS = {"dissolve": "Dissolve", "black": "Fade through black", "slide": "Slide", "wipe": "Soft wipe",
               "cut": "Cut"}
SHOWS = ("weather", "clock", "caption", "prompt")
ZOOM = 1.08  # slow zoom and pan: between 100 % and this much
SIZES = [("1280 × 720 (720p)", (1280, 720)), ("1920 × 1080 (1080p)", (1920, 1080)),
         ("2560 × 1440", (2560, 1440)), ("3840 × 2160 (4K)", (3840, 2160))]


@dataclass
class Options:
    folder: Path
    out: Path | None = None  # default: <folder name>.mp4 next to the folder
    size: tuple[int, int] | None = None  # None: the images' own size
    length: float = 180.0  # seconds
    every_image: bool = False  # False: as many as fit with `seconds` each, spread evenly
    seconds: float = 2.0  # time per image when choosing
    transition: str = "dissolve"
    transition_seconds: float = 0.8
    zoom: bool = True  # slow zoom and pan on every image
    show: tuple[str, ...] = ("weather", "clock")
    clock_24h: bool = True
    fps: int = 30
    crf: int = 18
    ffmpeg: str | None = None  # None: from the PATH
    recursive: bool = False
    hold_caption: bool = True  # a caption the same as the one before doesn't fade out and in


# --------------------------------------------------------------------------- images


def find_ffmpeg(given: str | None = None) -> str | None:
    if given:
        p = Path(given)
        if p.is_dir():
            p = p / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        return str(p) if p.is_file() else None
    return shutil.which("ffmpeg")


def load_images(folder: Path, recursive: bool = False) -> list[dict]:
    """The images with their sidecars, in the order they were made."""
    items = []
    for sidecar in (folder.rglob("*.json") if recursive else folder.glob("*.json")):
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(meta["created"])
        except (OSError, ValueError, KeyError, TypeError):  # e.g. being written by the screensaver right now
            continue
        jpg = sidecar.with_name(meta.get("file") or f"{sidecar.stem}.jpg")
        if jpg.exists():
            items.append({"jpg": jpg, "meta": meta, "created": created})
    items.sort(key=lambda it: (it["created"], it["jpg"].name))
    # the same image twice (a favorite next to its day's copy, with --recursive): keep one
    seen, unique = set(), []
    for it in items:
        key = (it["meta"].get("created"), it["meta"].get("prompt"))
        if key not in seen:
            seen.add(key)
            unique.append(it)
    return unique


def sizes(items: list[dict]) -> tuple[tuple[int, int], tuple[int, int]]:
    """The screen the images were shown on and the size they were saved at (the most common pair)."""
    pairs = Counter()
    for it in items:
        s, im = it["meta"].get("screen") or {}, it["meta"].get("image") or {}
        if s.get("width") and s.get("height") and im.get("width") and im.get("height"):
            pairs[(s["width"], s["height"]), (im["width"], im["height"])] += 1
    if pairs:
        return pairs.most_common(1)[0][0]
    with Image.open(items[0]["jpg"]) as im:
        return (im.width // 2, im.height // 2), im.size


@dataclass
class Plan:
    items: list[dict]
    per_image: float  # seconds each image is the current one
    fade: float  # seconds of transition into each image
    screen: tuple[int, int]
    size: tuple[int, int]

    @property
    def length(self) -> float:
        return len(self.items) * self.per_image


def plan(all_items: list[dict], o: Options) -> Plan:
    if not all_items:
        raise ValueError("no images with .json files in this folder")
    screen, saved = sizes(all_items)
    size = o.size or saved
    if o.every_image:
        chosen = all_items
        per = o.length / len(chosen)
        fade = 0.0 if o.transition == "cut" else min(o.transition_seconds, per / 4)
    else:
        count = max(1, min(len(all_items), math.floor(o.length / max(o.seconds, 0.2))))
        if count == 1:
            chosen = all_items[:1]
        else:  # evenly spread, first and last included
            chosen = [all_items[round(i * (len(all_items) - 1) / (count - 1))] for i in range(count)]
        per = o.length / count
        fade = 0.0 if o.transition == "cut" else min(o.transition_seconds, per * 0.45)
    return Plan(chosen, per, fade, screen, (size[0] - size[0] % 2, size[1] - size[1] % 2))


def fill(im: Image.Image, w: int, h: int) -> Image.Image:
    """Resize to fill w × h (Lanczos), cropping what sticks out."""
    if im.size == (w, h):
        return im
    scale = max(w / im.width, h / im.height)
    im = im.resize((max(w, round(im.width * scale)), max(h, round(im.height * scale))), Image.LANCZOS)
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


def weather_line(meta: dict) -> str:
    """"5 °C · overcast · afternoon", as the screensaver's caption has it; "" when the image has no weather."""
    o = meta.get("origin") or {}
    parts = []
    if o.get("temperature_c") is not None:
        parts.append(f"{o['temperature_c']:.0f} °C")
    if o.get("conditions"):
        parts.append(o["conditions"])
    if o.get("time_of_day") and parts:
        parts.append(o["time_of_day"])
    return " · ".join(parts)


def place_name(meta: dict) -> str:
    o = meta.get("origin") or {}
    return (o.get("location") or (o.get("place") or "").split(",")[0]).strip()


# --------------------------------------------------------------------------- drawing


class Frames:
    """Draws frames like the screensaver: overlays laid out on the screen's size (the widget's pixels), rendered
    at the movie's size, as on a monitor scaled to that many pixels; pictures drawn in full."""

    def __init__(self, p: Plan, o: Options):
        self.o, self.size = o, p.size
        self.dpr = p.size[1] / p.screen[1]  # 2.0 for an 800 x 480 screen in a 1600 x 960 movie
        self.w, self.h = p.size[0] / self.dpr, p.screen[1]  # layout size (the movie's shape)
        self.base_font = QFont(QApplication.font())
        self.ease = QEasingCurve(QEasingCurve.Type.InOutSine).valueForProgress

    # ----- pictures

    def picture(self, path: Path) -> QImage:
        w, h = self.size
        if self.o.zoom:  # room to zoom and pan into, without upscaling while drawing
            w, h = round(w * ZOOM), round(h * ZOOM)
        with Image.open(path) as im:
            im = fill(im.convert("RGB"), w, h)
        data = im.tobytes()
        return QImage(data, im.width, im.height, 3 * im.width, QImage.Format.Format_RGB888).convertToFormat(
            QImage.Format.Format_RGB32)  # converting copies, so `data` may go

    def motion(self, index: int) -> tuple[float, float, float, float, float, float]:
        """Each image's slow move: zoom from z0 to z1, centre from (x0, y0) to (x1, y1), as fractions of the
        room there is. Alternates zooming in and out; the direction is random but the same on every run."""
        rnd = random.Random(index * 7919 + 17)
        zin = index % 2 == 0
        z0, z1 = (1.0, ZOOM) if zin else (ZOOM, 1.0)
        a = rnd.uniform(0, 2 * math.pi)
        x0, y0 = 0.5 * math.cos(a), 0.5 * math.sin(a)
        return z0, z1, x0, y0, -x0, -y0

    def target(self, index: int, t: float) -> QRectF:
        """Where the picture goes at t (0 → 1 over the time it is visible), in layout coordinates."""
        if not self.o.zoom:
            return QRectF(0, 0, self.w, self.h)
        z0, z1, x0, y0, x1, y1 = self.motion(index)
        e = self.ease(min(max(t, 0.0), 1.0))
        z = z0 + (z1 - z0) * e
        w, h = self.w * z, self.h * z
        room_x, room_y = (w - self.w) / 2, (h - self.h) / 2
        cx = self.w / 2 + room_x * (x0 + (x1 - x0) * e)
        cy = self.h / 2 + room_y * (y0 + (y1 - y0) * e)
        return QRectF(cx - w / 2, cy - h / 2, w, h)

    # ----- one frame

    def frame(self, cur: dict, prev: dict | None, f: float) -> bytes:
        """cur / prev: {"image", "index", "t", "overlay"}; f: how far the transition into cur is (1 = done)."""
        out = QImage(*self.size, QImage.Format.Format_RGB32)
        out.setDevicePixelRatio(self.dpr)
        out.fill(Qt.GlobalColor.black)
        p = QPainter(out)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        e = self.ease(f) if f < 1.0 else 1.0
        tr = self.o.transition if prev is not None and f < 1.0 else "none"
        draw = lambda it, opacity=1.0, dx=0.0: self._draw_picture(p, it, opacity, dx)  # noqa: E731
        if tr in ("none", "cut"):
            draw(cur)
        elif tr == "dissolve":
            draw(prev)
            draw(cur, e)
        elif tr == "black":
            if e < 0.5:
                draw(prev, 1 - 2 * e)
            else:
                draw(cur, 2 * e - 1)
        elif tr == "slide":
            draw(prev, 1.0, -e * self.w)
            draw(cur, 1.0, (1 - e) * self.w)
        elif tr == "wipe":
            draw(prev)
            self._wipe(p, cur, e)
        # overlays: text that stays the same stays put; other text fades out, then the new text fades in (two
        # texts on top of each other, as a crossfade would show them, are unreadable)
        for part in ("caption", "clock"):
            if prev is not None and f < 1.0 and tr != "cut":
                if self.o.hold_caption and self._part(prev["overlay"], part) == self._part(cur["overlay"], part):
                    self._overlay(p, cur["overlay"], part)
                else:
                    if part == "caption" and any(ov["text"] or ov["prompt"] for ov in (prev["overlay"], cur["overlay"])):
                        self._draw_band(p)  # the dark band stays; only the text changes
                    p.setOpacity(max(0.0, 1 - 2 * e))
                    self._overlay(p, prev["overlay"], part, band=False)
                    p.setOpacity(max(0.0, 2 * e - 1))
                    self._overlay(p, cur["overlay"], part, band=False)
                    p.setOpacity(1.0)
            else:
                self._overlay(p, cur["overlay"], part)
        p.end()
        return bytes(out.constBits())

    def _draw_picture(self, p: QPainter, it: dict, opacity: float, dx: float) -> None:
        if opacity <= 0:
            return
        p.setOpacity(opacity)
        p.drawImage(self.target(it["index"], it["t"]).translated(dx, 0), it["image"])
        p.setOpacity(1.0)

    def _wipe(self, p: QPainter, it: dict, e: float) -> None:
        """The new picture revealed from left to right behind a soft, slightly slanted edge."""
        layer = QImage(*self.size, QImage.Format.Format_ARGB32_Premultiplied)
        layer.setDevicePixelRatio(self.dpr)
        layer.fill(Qt.GlobalColor.transparent)
        lp = QPainter(layer)
        lp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        lp.drawImage(self.target(it["index"], it["t"]), it["image"])
        soft = self.w * 0.25
        x = -soft + (self.w + 2 * soft) * e
        grad = QLinearGradient(QPointF(x - soft, 0), QPointF(x + soft, self.h * 0.35))
        grad.setColorAt(0, QColor(0, 0, 0, 255))
        grad.setColorAt(1, QColor(0, 0, 0, 0))
        lp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        lp.fillRect(QRectF(0, 0, self.w, self.h), grad)
        lp.end()
        p.drawImage(QRectF(0, 0, self.w, self.h), layer)

    # ----- overlays, as the screensaver draws them

    def overlay_for(self, meta: dict, created: datetime) -> dict:
        show = set(self.o.show)
        cap = meta.get("caption") or {}
        kicker, text = "", ""
        weather = weather_line(meta) if "weather" in show else ""
        if "caption" in show:
            kicker, text = cap.get("kicker", ""), cap.get("text", "")
            if weather and weather not in text:
                text = f"{text}\n{weather}" if text else weather
        elif weather:
            place = place_name(meta)
            kicker, text = (f"In {place}" if place else "Weather"), weather
        clock = (clock_text(created, self.o.clock_24h), f"{created:%A %d %B}") if "clock" in show else None
        prompt = meta.get("prompt", "") if "prompt" in show else ""
        return {"kicker": kicker, "text": text, "prompt": prompt, "clock": clock}

    @staticmethod
    def _part(ov: dict, part: str):
        return (ov["kicker"], ov["text"], ov["prompt"]) if part == "caption" else ov["clock"]

    def _overlay(self, p: QPainter, ov: dict, part: str, band: bool = True) -> None:
        if part == "caption" and (ov["text"] or ov["prompt"]):
            self._draw_caption(p, ov, band)
        elif part == "clock" and ov["clock"]:
            self._draw_clock(p, *ov["clock"])

    def _font(self, rel: float, bold=False, spacing=0.0) -> QFont:
        f = QFont(self.base_font)
        f.setPixelSize(max(9, int(self.h * rel)))
        f.setBold(bold)
        if spacing:
            f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 100 + spacing)
        return f

    def _draw_band(self, p: QPainter) -> None:
        r = QRect(0, 0, int(self.w), int(self.h))
        band = int(r.height() * 0.30)
        grad = QLinearGradient(0, r.height() - band, 0, r.height())
        grad.setColorAt(0, QColor(0, 0, 0, 0))
        grad.setColorAt(1, QColor(0, 0, 0, 175))
        p.fillRect(QRect(0, r.height() - band, r.width(), band), grad)

    def _draw_caption(self, p: QPainter, ov: dict, band: bool = True) -> None:
        if band:
            self._draw_band(p)
        r = QRect(0, 0, int(self.w), int(self.h))
        margin = int(r.height() * 0.045)
        width = int(r.width() * 0.62)
        text = ov["text"]
        prompt = f"Prompt: {ov['prompt']}" if ov["prompt"] else ""
        text_font, prompt_font = self._font(0.029), self._font(0.0135)
        p.setFont(text_font)
        text_h = p.fontMetrics().boundingRect(QRect(0, 0, width, 10_000), int(Qt.TextFlag.TextWordWrap),
                                              text).height() if text else 0
        p.setFont(prompt_font)
        prompt_h = p.fontMetrics().boundingRect(QRect(0, 0, width, 10_000), int(Qt.TextFlag.TextWordWrap),
                                                prompt).height() if prompt else 0
        y = r.height() - margin - text_h - (prompt_h + int(margin * 0.3) if prompt else 0)
        if ov["kicker"]:
            p.setFont(self._font(0.0155, bold=True, spacing=12))
            kh = p.fontMetrics().height()
            p.setPen(QColor(255, 255, 255, 190))
            p.drawText(QRect(margin, y - kh - int(margin * 0.15), width, kh), int(Qt.AlignmentFlag.AlignLeft),
                       ov["kicker"].upper())
        if text:
            p.setFont(text_font)
            p.setPen(QColor(0, 0, 0, 150))
            p.drawText(QRect(margin + 2, y + 2, width, text_h), int(Qt.TextFlag.TextWordWrap), text)
            p.setPen(QColor(255, 255, 255))
            p.drawText(QRect(margin, y, width, text_h), int(Qt.TextFlag.TextWordWrap), text)
        if prompt:
            p.setFont(prompt_font)
            p.setPen(QColor(255, 255, 255, 170))
            p.drawText(QRect(margin, y + text_h + int(margin * 0.3), width, prompt_h),
                       int(Qt.TextFlag.TextWordWrap), prompt)

    def _draw_clock(self, p: QPainter, time_text: str, date_text: str) -> None:
        r = QRect(0, 0, int(self.w), int(self.h))
        margin = int(r.height() * 0.045)
        p.setFont(self._font(0.06))
        th = p.fontMetrics().height()
        box = QRect(r.width() - margin - int(r.width() * 0.3), r.height() - margin - th - int(r.height() * 0.03),
                    int(r.width() * 0.3), th)
        for dx, color in ((2, QColor(0, 0, 0, 150)), (0, QColor(255, 255, 255, 235))):
            p.setPen(color)
            p.drawText(box.translated(dx, dx), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom),
                       time_text)
        p.setFont(self._font(0.017))
        date_box = QRect(box.x(), box.bottom() + 2, box.width(), int(r.height() * 0.03))
        for dx, color in ((2, QColor(0, 0, 0, 150)), (0, QColor(255, 255, 255, 200))):
            p.setPen(color)
            p.drawText(date_box.translated(dx, dx), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop),
                       date_text)


# --------------------------------------------------------------------------- rendering


class Cancelled(Exception):
    pass


def frame_schedule(pl: Plan, fps: int) -> list[tuple[int, int, int]]:
    """Per image: (first frame, number of frames, frames of transition into it)."""
    out = []
    fade_frames = round(pl.fade * fps)
    for i in range(len(pl.items)):
        a, b = round(i * pl.per_image * fps), round((i + 1) * pl.per_image * fps)
        out.append((a, b - a, min(fade_frames, b - a) if i else 0))
    return out


def render(o: Options, on_progress=None, cancel: threading.Event | None = None, on_log=print) -> Path:
    """Makes the movie; on_progress(done_images, total_images). Raises Cancelled, or RuntimeError / ValueError."""
    ffmpeg = find_ffmpeg(o.ffmpeg)
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found: install it (winget install Gyan.FFmpeg) or give its path")
    all_items = load_images(o.folder, o.recursive)
    pl = plan(all_items, o)
    out = Path(o.out) if o.out else o.folder.parent / f"{o.folder.name}.mp4"
    frames = Frames(pl, o)
    w, h = pl.size
    sched = frame_schedule(pl, o.fps)
    total_frames = sum(n for _, n, _ in sched)
    on_log(f"{len(pl.items)} of {len(all_items)} images, {pl.per_image:.2f} s each, {pl.fade:.2f} s transitions, "
           f"{w} × {h} at {o.fps} fps: {hms(total_frames / o.fps)} → {out}")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{w}x{h}", "-r", str(o.fps), "-i", "-",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(o.crf),
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-movflags", "+faststart", str(out)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # no console window from a windowed app
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    errors: list[bytes] = []
    reader = threading.Thread(target=lambda: errors.append(proc.stderr.read()), daemon=True)
    reader.start()

    def item(i: int) -> dict | None:
        it = pl.items[i]
        try:
            image = frames.picture(it["jpg"])
        except OSError as e:  # damaged or half-written JPEG
            on_log(f"skipped {it['jpg'].name}: {e}")
            return None
        return {"image": image, "index": i, "t": 0.0, "overlay": frames.overlay_for(it["meta"], it["created"])}

    prev, prev_start = None, 0
    try:
        for i, (start, n, fading) in enumerate(sched):
            cur = item(i)
            if cur is None:
                cur = prev  # keep showing the previous image for this one's time
                if cur is None:
                    continue
            visible = n + (sched[i + 1][2] if i + 1 < len(sched) else 0)  # until the next one has faded in
            for k in range(n):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                cur["t"] = k / max(visible, 1)
                if k < fading and prev is not None and prev is not cur:
                    prev["t"] = (start + k - prev_start) / max(prev["visible"], 1)
                    proc.stdin.write(frames.frame(cur, prev, (k + 1) / (fading + 1)))
                else:
                    proc.stdin.write(frames.frame(cur, None, 1.0))
            cur["visible"] = visible
            prev, prev_start = cur, start
            if on_progress:
                on_progress(i + 1, len(sched))
        proc.stdin.close()
    except (Cancelled, KeyboardInterrupt):
        proc.kill()
        proc.wait()
        out.unlink(missing_ok=True)
        raise
    except OSError:  # ffmpeg quit: its message follows
        pass
    code = proc.wait()
    reader.join(5)
    if code != 0:
        msg = b"".join(errors).decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed (exit code {code}): {msg[-500:]}")
    return out


def preview(o: Options, at: float = 0.5) -> QImage:
    """One frame halfway through the transition into the second chosen image (or of the only one)."""
    pl = plan(load_images(o.folder, o.recursive), o)
    frames = Frames(pl, o)
    its = [{"image": frames.picture(it["jpg"]), "index": i, "t": 0.0,
            "overlay": frames.overlay_for(it["meta"], it["created"])} for i, it in enumerate(pl.items[:2])]
    if len(its) == 1:
        data = frames.frame(its[0], None, 1.0)
    else:
        its[0]["t"], its[1]["t"] = 0.92, 0.05
        data = frames.frame(its[1], its[0], at if pl.fade > 0 else 1.0)
    img = QImage(data, *pl.size, QImage.Format.Format_RGB32)
    return img.copy()


def hms(seconds: float) -> str:
    m, s = divmod(round(seconds), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"


def duration(text: str) -> float:
    """Seconds, from "180" or "3:00"."""
    minutes, _, seconds = str(text).rpartition(":")
    value = int(minutes or 0) * 60 + float(seconds)
    if value <= 0:
        raise ValueError(text)
    return value


def size_arg(text: str) -> tuple[int, int] | None:
    if text.lower() in ("", "input", "same"):
        return None
    named = {"720p": (1280, 720), "1080p": (1920, 1080), "1440p": (2560, 1440), "4k": (3840, 2160)}
    if text.lower() in named:
        return named[text.lower()]
    w, _, h = text.lower().partition("x")
    return int(w), int(h)


# --------------------------------------------------------------------------- window


class Worker(QObject):
    progress = Signal(int, int)
    log = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, o: Options):
        super().__init__()
        self.o, self.cancel = o, threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            out = render(self.o, lambda d, t: self.progress.emit(d, t), self.cancel, self.log.emit)
            self.done.emit(str(out))
        except Cancelled:
            self.failed.emit("")
        except Exception as e:  # shown in the window
            self.failed.emit(f"{type(e).__name__}: {e}")


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.cfg = load_settings()
        self.items: list[dict] = []
        self.thread: QThread | None = None
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self._refresh)
        self.setMinimumWidth(760)

        root = save_root(self.cfg)
        days = sorted(d for d in root.glob("????-??-??") if d.is_dir()) if root.exists() else []
        self.folder = QLineEdit(str(days[-1] if days else root))
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._choose_folder)
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setToolTip("For a folder of day folders: the days one after the other")
        frow = QHBoxLayout()
        frow.addWidget(self.folder, 1)
        frow.addWidget(browse)
        frow.addWidget(self.recursive)
        self.found = QLabel()

        self.length = QTimeEdit()
        self.length.setDisplayFormat("m:ss")
        self.length.setTime(QTime(0, 3, 0))
        self.pick = QRadioButton("As many as fit with smooth transitions, each about")
        self.every = QRadioButton("Every image (a time-lapse; fast with many images)")
        self.pick.setChecked(True)
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(0.5, 30)
        self.seconds.setSingleStep(0.5)
        self.seconds.setValue(2.0)
        self.seconds.setSuffix(" s")
        prow = QHBoxLayout()
        prow.addWidget(self.pick)
        prow.addWidget(self.seconds)
        prow.addStretch(1)
        self.choice = QLabel()

        self.transition = QComboBox()
        for k, v in TRANSITIONS.items():
            self.transition.addItem(v, k)
        self.trans_s = QDoubleSpinBox()
        self.trans_s.setRange(0.1, 5)
        self.trans_s.setSingleStep(0.1)
        self.trans_s.setValue(0.8)
        self.trans_s.setSuffix(" s")
        self.zoom = QCheckBox("Slow zoom and pan on every image")
        self.zoom.setChecked(True)
        trow = QHBoxLayout()
        trow.addWidget(self.transition)
        trow.addWidget(self.trans_s)
        trow.addWidget(self.zoom)
        trow.addStretch(1)

        self.size = QComboBox()
        self.size.addItem("Same as the images", None)
        for text, wh in SIZES:
            self.size.addItem(text, wh)
        self.fps = QComboBox()
        for v in (24, 25, 30, 60):
            self.fps.addItem(f"{v} fps", v)
        self.fps.setCurrentIndex(2)
        srow = QHBoxLayout()
        srow.addWidget(self.size, 1)
        srow.addWidget(self.fps)

        self.shows = {k: QCheckBox(t) for k, t in (("weather", "Weather"), ("clock", "Time and date"),
                                                    ("caption", "Caption (the words, headline…)"),
                                                    ("prompt", "Prompt"))}
        self.shows["weather"].setChecked(True)
        self.shows["clock"].setChecked(True)
        shrow = QHBoxLayout()
        for box in self.shows.values():
            shrow.addWidget(box)
        shrow.addStretch(1)

        self.ffmpeg = QLineEdit(find_ffmpeg() or "")
        self.ffmpeg.setPlaceholderText("Path to ffmpeg.exe (not found on the PATH)")
        fbrowse = QPushButton("Choose…")
        fbrowse.clicked.connect(self._choose_ffmpeg)
        self.ffmpeg_ok = QLabel()
        ffrow = QHBoxLayout()
        ffrow.addWidget(self.ffmpeg, 1)
        ffrow.addWidget(fbrowse)
        ffrow.addWidget(self.ffmpeg_ok)

        self.out = QLineEdit()
        obrowse = QPushButton("Choose…")
        obrowse.clicked.connect(self._choose_out)
        orow = QHBoxLayout()
        orow.addWidget(self.out, 1)
        orow.addWidget(obrowse)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(300)
        self.preview.setStyleSheet("background: #111; color: #888;")
        self.make = QPushButton("Make movie")
        self.make.setDefault(True)
        self.make.clicked.connect(self._make)
        self.stop = QPushButton("Cancel")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self._cancel)
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.open_btn = QPushButton("Play")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self._out_path())))
        self.show_btn = QPushButton("Show in folder")
        self.show_btn.setEnabled(False)
        self.show_btn.clicked.connect(lambda: subprocess.Popen(["explorer", f"/select,{self._out_path()}"]))
        brow = QHBoxLayout()
        brow.addWidget(self.bar, 1)
        brow.addWidget(self.make)
        brow.addWidget(self.stop)
        brow.addWidget(self.open_btn)
        brow.addWidget(self.show_btn)

        form = QFormLayout()
        form.addRow("Images from", frow)
        form.addRow("", self.found)
        form.addRow("Length", self.length)
        form.addRow("Images", prow)
        form.addRow("", self.every)
        form.addRow("", self.choice)
        form.addRow("Transition", trow)
        form.addRow("Size", srow)
        form.addRow("Show", shrow)
        form.addRow("ffmpeg", ffrow)
        form.addRow("Save as", orow)
        v = QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(self.preview, 1)
        v.addLayout(brow)
        v.addWidget(self.status)

        self.folder.editingFinished.connect(self._load)
        self.recursive.toggled.connect(self._load)
        for sig in (self.length.timeChanged, self.seconds.valueChanged, self.pick.toggled, self.trans_s.valueChanged,
                    self.transition.currentIndexChanged, self.zoom.toggled, self.size.currentIndexChanged,
                    *(b.toggled for b in self.shows.values())):
            sig.connect(self._refresh_soon)
        self.ffmpeg.textChanged.connect(self._check_ffmpeg)
        self._check_ffmpeg()
        self._load()

    # ----- state

    def options(self) -> Options:
        t = self.length.time()
        return Options(
            folder=Path(self.folder.text().strip()), out=Path(self.out.text()) if self.out.text().strip() else None,
            size=self.size.currentData(), length=max(1, t.minute() * 60 + t.second()),
            every_image=self.every.isChecked(), seconds=self.seconds.value(),
            transition=self.transition.currentData(), transition_seconds=self.trans_s.value(),
            zoom=self.zoom.isChecked(), show=tuple(k for k, b in self.shows.items() if b.isChecked()),
            clock_24h=self.cfg.get("clock_24h", True), fps=self.fps.currentData(),
            ffmpeg=self.ffmpeg.text().strip() or None, recursive=self.recursive.isChecked())

    def _out_path(self) -> str:
        o = self.options()
        return str(o.out or o.folder.parent / f"{o.folder.name}.mp4")

    def _load(self) -> None:
        folder = Path(self.folder.text().strip())
        self.items = load_images(folder, self.recursive.isChecked()) if folder.is_dir() else []
        if self.items:
            a, b = self.items[0]["created"], self.items[-1]["created"]
            span = f"{a:%Y-%m-%d %H:%M} – {b:%H:%M}" if a.date() == b.date() else f"{a:%Y-%m-%d} – {b:%Y-%m-%d}"
            screen, saved = sizes(self.items)
            self.size.setItemText(0, f"Same as the images ({saved[0]} × {saved[1]})")
            self.found.setText(f"{len(self.items):,} images, {span}")
        else:
            self.found.setText("No screensaver images (.jpg with .json) in this folder")
        self.out.setText(str(folder.parent / f"{folder.name}.mp4") if folder.name else "")
        self._refresh()

    def _refresh(self) -> None:
        self.seconds.setEnabled(self.pick.isChecked())
        self.trans_s.setEnabled(self.transition.currentData() != "cut")
        if not self.items:
            self.choice.clear()
            self.preview.setText("No images")
            self.make.setEnabled(False)
            return
        o = self.options()
        pl = plan(self.items, o)
        n = len(pl.items)
        if not o.every_image and n < len(self.items):
            span = (self.items[-1]["created"] - self.items[0]["created"]).total_seconds() / max(n - 1, 1)
            every = (f"one every {span:.0f} seconds" if span < 90 else f"one every {span / 60:.0f} minutes"
                     if span < 5400 else f"one every {span / 3600:.1f} hours")
            self.choice.setText(f"{n} of {len(self.items):,} images, {every} of the time they cover; "
                                f"{pl.per_image:.1f} s each with {pl.fade:.1f} s transitions")
        else:
            self.choice.setText(f"All {n:,} images, {pl.per_image:.2f} s each"
                                + (f", {pl.fade:.2f} s transitions" if pl.fade else ", no time for transitions"))
        try:
            img = preview(o)
            pm = QPixmap.fromImage(img).scaled(self.preview.width() - 8, max(200, self.preview.height() - 8),
                                               Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
            self.preview.setPixmap(pm)
            self.preview.setToolTip("Halfway through the first transition")
        except Exception as e:
            self.preview.setText(f"No preview: {e}")
        self.make.setEnabled(self.thread is None)

    def _refresh_soon(self) -> None:
        self.refresh_timer.start(150)  # one refresh after a burst of changes (typing, resizing)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self.items and self.thread is None:
            self._refresh_soon()

    def _check_ffmpeg(self) -> None:
        ok = find_ffmpeg(self.ffmpeg.text().strip() or None) is not None
        self.ffmpeg_ok.setText("✓" if ok else "✗ not found")
        self.ffmpeg_ok.setStyleSheet("color: #3c3;" if ok else "color: #e55;")

    def _choose_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder of screensaver images", self.folder.text())
        if d:
            self.folder.setText(str(Path(d)))
            self._load()

    def _choose_ffmpeg(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "ffmpeg", self.ffmpeg.text(), "ffmpeg (ffmpeg.exe ffmpeg);;All (*)")
        if f:
            self.ffmpeg.setText(str(Path(f)))

    def _choose_out(self) -> None:
        f, _ = QFileDialog.getSaveFileName(self, "Save movie as", self._out_path(), "MP4 movie (*.mp4)")
        if f:
            self.out.setText(str(Path(f)))

    # ----- making

    def _make(self) -> None:
        o = self.options()
        if not find_ffmpeg(o.ffmpeg):
            QMessageBox.warning(self, APP_NAME, "ffmpeg wasn't found. Install it (winget install Gyan.FFmpeg) "
                                "or choose ffmpeg.exe.")
            return
        out = Path(self._out_path())
        if out.exists() and QMessageBox.question(self, APP_NAME, f"{out.name} exists. Replace it?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.bar.setValue(0)
        self.status.setText("Making the movie…")
        self.make.setEnabled(False)
        self.stop.setEnabled(True)
        self.open_btn.setEnabled(False)
        self.show_btn.setEnabled(False)
        self.started = time.time()
        self.thread = QThread(self)
        self.worker = Worker(o)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.log.connect(self.status.setText)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.thread.start()

    def _progress(self, done: int, total: int) -> None:
        self.bar.setMaximum(total)
        self.bar.setValue(done)
        el = time.time() - self.started
        self.bar.setFormat(f"%p%  ·  {done}/{total} images  ·  {hms(el / done * (total - done))} left")

    def _finish(self) -> None:
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.make.setEnabled(bool(self.items))
        self.stop.setEnabled(False)

    def _done(self, path: str) -> None:
        self._finish()
        size = Path(path).stat().st_size / 1e6
        self.bar.setFormat("Done")
        self.status.setText(f"Done in {hms(time.time() - self.started)}: {path} ({size:.0f} MB)")
        self.open_btn.setEnabled(True)
        self.show_btn.setEnabled(True)

    def _failed(self, msg: str) -> None:
        self._finish()
        self.bar.setFormat("Stopped" if not msg else "Failed")
        self.status.setText("Stopped; no movie written." if not msg else msg)

    def _cancel(self) -> None:
        if self.thread is not None:
            self.worker.cancel.set()

    def closeEvent(self, e) -> None:
        if self.thread is not None:
            self.worker.cancel.set()
            self.thread.quit()
            self.thread.wait(5000)
        e.accept()


# --------------------------------------------------------------------------- command line


def progress_bar(done: int, total: int, started: float) -> None:
    el = time.time() - started
    left = hms(el / done * (total - done)) if done else "?"
    tail = f" {done / total:4.0%}  {done}/{total} images  {hms(el)} elapsed, {left} left "
    width = max(10, min(40, shutil.get_terminal_size().columns - len(tail) - 1))
    filled = round(width * done / total)
    print(f"\r{'█' * filled}{'░' * (width - filled)}{tail}", end="", file=sys.stderr, flush=True)


def parser(cfg: dict) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="A movie from a folder of screensaver images (.jpg + .json). "
                                             "Without a folder, opens the window.")
    ap.add_argument("folder", nargs="?", type=Path, help="folder of images; none: open the window")
    ap.add_argument("--out", type=Path, help="movie file (default: <folder>.mp4 next to the folder)")
    ap.add_argument("--size", type=size_arg, default=None,
                    help="WxH, 720p, 1080p, 1440p or 4k (default: the images' own size)")
    ap.add_argument("--length", type=duration, default=180.0, help="seconds or m:ss (default 3:00)")
    ap.add_argument("--every-image", action="store_true",
                    help="use every image (time-lapse) instead of as many as fit with --seconds each")
    ap.add_argument("--seconds", type=float, default=2.0, help="time per image when choosing (default 2)")
    ap.add_argument("--transition", choices=list(TRANSITIONS), default="dissolve", help="default: dissolve")
    ap.add_argument("--transition-seconds", type=float, default=0.8, help="default 0.8")
    ap.add_argument("--no-zoom", action="store_true", help="no slow zoom and pan")
    ap.add_argument("--show", default="weather,clock",
                    help=f"comma-separated, from {', '.join(SHOWS)}; 'none' for a clean movie (default weather,clock)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=18, help="x264 quality, lower = better and bigger (default 18)")
    ap.add_argument("--ffmpeg", help="path to ffmpeg(.exe) or its folder (default: from the PATH)")
    ap.add_argument("--recursive", action="store_true", help="include subfolders (several days)")
    return ap


def main() -> None:
    cfg = load_settings()
    args = parser(cfg).parse_args()
    if args.folder is None:
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("zog.dallemega.moviemaker")
        app = QApplication(sys.argv[:1])
        app.setApplicationName(APP_NAME)
        w = Window()
        w.resize(820, 860)
        w.show()
        sys.exit(app.exec())
    show = tuple(s for s in (x.strip() for x in args.show.split(",")) if s and s != "none")
    if bad := [s for s in show if s not in SHOWS]:
        sys.exit(f"--show: unknown {', '.join(bad)} (use {', '.join(SHOWS)})")
    o = Options(folder=args.folder, out=args.out, size=args.size, length=args.length, every_image=args.every_image,
                seconds=args.seconds, transition=args.transition, transition_seconds=args.transition_seconds,
                zoom=not args.no_zoom, show=show, clock_24h=cfg.get("clock_24h", True), fps=args.fps, crf=args.crf,
                ffmpeg=args.ffmpeg, recursive=args.recursive)
    run_cli(o)


def run_cli(o: Options) -> None:
    app = QApplication.instance() or QApplication(sys.argv[:1])  # noqa: F841  fonts as in the screensaver
    started = time.time()
    try:
        out = render(o, lambda d, t: progress_bar(d, t, started))
    except KeyboardInterrupt:
        sys.exit("\nStopped; no movie written.")
    except (RuntimeError, ValueError) as e:
        sys.exit(f"\n{e}")
    print(file=sys.stderr)
    print(f"Done in {hms(time.time() - started)}: {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
