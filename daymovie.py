"""Turn one day of the Image Screensaver into a movie: every image saved that day, in order, each frame
drawn the way the screensaver showed it (caption band, clock) with its crossfade from one image to the
next. The movie has the saved images' own resolution (1600 x 960 for an 800 x 480 monitor); captions and
clock keep their size relative to the screen, drawn sharp at that resolution. Needs ffmpeg on the PATH.

  uv run daymovie.py                        the latest day
  uv run daymovie.py 2026-11-06             a date, or the path of a day folder
  uv run daymovie.py 2026-11-06 --length 1:30

A whole day fits in --length (default 2:55): each image gets length / number of images, at most --seconds
(2), so a full day of ~1300 images runs at ~0.13 s per image. The crossfade takes at most a quarter of an
image's time. Unlike on screen, a caption that is the same as the one before stays put during the fade
instead of fading out and in again (at time-lapse speed that would flicker).

The movie goes next to the day folders (outputs/screensaver/2026-11-06.mp4) unless --out says otherwise.
Captions and clock follow the screensaver's settings unless --captions / --no-captions, --clock /
--no-clock are given. The clock shows when each image was made (on screen it was up to a minute later).

The drawing mirrors Screensaver.paintEvent, _draw_caption and _draw_clock in screensaver.pyw: if the
screensaver's look changes, change it here too.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QEasingCurve, QPointF, QRect, Qt
from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter
from PySide6.QtWidgets import QApplication

HERE = Path(__file__).resolve().parent
SETTINGS_PATH = HERE / "screensaver_settings.json"
SAVE_DIR = HERE / "outputs" / "screensaver"  # default; the screensaver's settings can change it
DAY_FOLDER = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def find_day(arg: str | None, root: Path) -> Path:
    if arg:
        for p in (Path(arg), root / arg):
            if p.is_dir():
                return p
        sys.exit(f"No day folder {arg!r} (looked in {root})")
    days = sorted(d for d in root.glob("*") if d.is_dir() and DAY_FOLDER.match(d.name) and any(d.glob("*.json")))
    if not days:
        sys.exit(f"No day folders in {root}")
    return days[-1]


def load_images(day: Path) -> list[dict]:
    """The day's images with their sidecars, in the order they were made."""
    items = []
    for sidecar in day.glob("*.json"):
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(meta["created"])
        except (OSError, ValueError, KeyError):  # e.g. being written by the running screensaver right now
            print(f"skipped {sidecar.name}: unreadable sidecar", file=sys.stderr)
            continue
        jpg = sidecar.with_name(meta.get("file") or f"{sidecar.stem}.jpg")
        if jpg.exists():
            items.append({"jpg": jpg, "meta": meta, "created": created})
    return sorted(items, key=lambda it: (it["created"], it["jpg"].name))


def sizes(items: list[dict]) -> tuple[tuple[int, int], tuple[int, int]]:
    """The monitor the day's images were shown on and the size they were saved at (the most common
    pair, if the monitor changed during the day)."""
    pairs = Counter()
    for it in items:
        s, im = it["meta"].get("screen") or {}, it["meta"].get("image") or {}
        if s.get("width") and s.get("height") and im.get("width") and im.get("height"):
            pairs[(s["width"], s["height"]), (im["width"], im["height"])] += 1
    if pairs:
        return pairs.most_common(1)[0][0]
    with Image.open(items[0]["jpg"]) as im:  # saved at twice the screen's size
        return (im.width // 2, im.height // 2), im.size


def fit(im: Image.Image, w: int, h: int) -> Image.Image:
    """Resize to w × h as the screensaver does (Lanczos); crop to fill if the shape differs."""
    if im.size == (w, h):
        return im
    scale = max(w / im.width, h / im.height)
    im = im.resize((max(w, round(im.width * scale)), max(h, round(im.height * scale))), Image.LANCZOS)
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


class Frames:
    """Draws frames like Screensaver.paintEvent: laid out on the screen's w × h (the widget's pixels),
    rendered at the saved images' size, as on a monitor scaled to that many physical pixels."""

    def __init__(self, screen: tuple[int, int], size: tuple[int, int], captions: bool, clock: bool, prompt: bool):
        (self.w, self.h), self.size = screen, size
        self.dpr = size[1] / screen[1]  # 2.0 for 800 x 480 saved at 1600 x 960
        self.captions, self.clock, self.show_prompt = captions, clock, prompt
        self.clock_24h = True  # as the screensaver's clock_24h setting
        self.base_font = QFont(QApplication.font())  # the widget's font: Qt's default
        self.ease = QEasingCurve(QEasingCurve.Type.InOutSine).valueForProgress  # the screensaver's fade

    def picture(self, path: Path) -> QImage:
        with Image.open(path) as im:
            im = fit(im.convert("RGB"), *self.size)
        data = im.tobytes()
        q = QImage(data, im.width, im.height, 3 * im.width, QImage.Format.Format_RGB888).convertToFormat(
            QImage.Format.Format_RGB32)  # converting copies, so `data` may go
        q.setDevicePixelRatio(self.dpr)
        return q

    def frame(self, current: QImage, idea: dict, when: datetime, previous: QImage | None = None,
              fade: float = 1.0, caption_fade: float | None = None) -> bytes:
        """One frame as raw BGRA bytes (QImage RGB32 on little-endian). The caption fades in with the
        picture unless caption_fade says otherwise."""
        out = QImage(*self.size, QImage.Format.Format_RGB32)
        out.setDevicePixelRatio(self.dpr)
        out.fill(Qt.GlobalColor.black)
        p = QPainter(out)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if previous is not None and fade < 1.0:  # pictures are drawn pixel for pixel
            p.drawImage(QPointF(0, 0), previous)
        p.setOpacity(fade)
        p.drawImage(QPointF(0, 0), current)
        p.setOpacity(1.0)
        if self.captions:  # the previous image's caption is gone as soon as the fade starts, as on screen
            p.setOpacity(fade if caption_fade is None else caption_fade)
            self._draw_caption(p, idea)
            p.setOpacity(1.0)
        if self.clock:
            self._draw_clock(p, when)
        p.end()
        return bytes(out.constBits())

    def _font(self, rel: float, bold=False, spacing=0.0) -> QFont:
        f = QFont(self.base_font)
        f.setPixelSize(max(9, int(self.h * rel)))
        f.setBold(bold)
        if spacing:
            f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 100 + spacing)
        return f

    def _draw_caption(self, p: QPainter, idea: dict) -> None:
        r = QRect(0, 0, self.w, self.h)
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

    def _draw_clock(self, p: QPainter, now: datetime) -> None:
        r = QRect(0, 0, self.w, self.h)
        margin = int(r.height() * 0.045)
        p.setFont(self._font(0.06))
        th = p.fontMetrics().height()
        box = QRect(r.width() - margin - int(r.width() * 0.3), r.height() - margin - th - int(r.height() * 0.03),
                    int(r.width() * 0.3), th)
        for dx, color in ((2, QColor(0, 0, 0, 150)), (0, QColor(255, 255, 255, 235))):
            p.setPen(color)
            p.drawText(box.translated(dx, dx), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom),
                       f"{now:%H:%M}" if self.clock_24h else f"{now:%I:%M}".lstrip("0") + f" {now:%p}".lower())
        p.setFont(self._font(0.017))
        date_box = QRect(box.x(), box.bottom() + 2, box.width(), int(r.height() * 0.03))
        for dx, color in ((2, QColor(0, 0, 0, 150)), (0, QColor(255, 255, 255, 200))):
            p.setPen(color)
            p.drawText(date_box.translated(dx, dx), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop),
                       f"{now:%A %d %B}")


def duration(text: str) -> float:
    """Seconds, from "175" or "2:55"."""
    minutes, _, seconds = text.rpartition(":")
    value = int(minutes or 0) * 60 + float(seconds)
    if value <= 0:
        raise ValueError(text)
    return value


def hms(seconds: float) -> str:
    m, s = divmod(round(seconds), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"


def progress(done: int, total: int, elapsed: float) -> None:
    """A progress bar on one line of the terminal, redrawn in place."""
    left = hms(elapsed / done * (total - done)) if done else "?"
    tail = f" {done / total:4.0%}  {done}/{total} images  {hms(elapsed)} elapsed, {left} left "
    width = max(10, min(40, shutil.get_terminal_size().columns - len(tail) - 1))  # one line, no wrapping
    filled = round(width * done / total)
    print(f"\r{'█' * filled}{'░' * (width - filled)}{tail}", end="", file=sys.stderr, flush=True)


def main() -> None:
    cfg = load_settings()
    ap = argparse.ArgumentParser(description="Make a movie of one day of the Image Screensaver's images, "
                                             "each frame as it looked on screen.")
    ap.add_argument("day", nargs="?", help="a date (2026-11-06) or a day folder; default: the latest day")
    ap.add_argument("--length", type=duration, default=175.0,
                    help="longest the movie may be, in seconds or m:ss (default 2:55); "
                         "the more images, the less time each gets")
    ap.add_argument("--seconds", type=float, default=2.0, help="most time per image, fade included (default 2)")
    ap.add_argument("--fade", type=float, default=0.5,
                    help="crossfade into each image, at most this many seconds and a quarter of the image's time; "
                         "0 = cut (default 0.5; the screensaver takes 2.5)")
    ap.add_argument("--fps", type=int, default=30, help="frames per second (default 30)")
    ap.add_argument("--captions", action=argparse.BooleanOptionalAction, default=cfg.get("captions", True),
                    help="caption band (default: as in the screensaver's settings)")
    ap.add_argument("--clock", action=argparse.BooleanOptionalAction, default=cfg.get("clock", True),
                    help="clock and date (default: as in the screensaver's settings)")
    ap.add_argument("--prompt", action="store_true", help="also show each image's prompt, as the P key does")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality, lower = better and bigger (default 18)")
    ap.add_argument("--out", type=Path, help="movie file (default: <day>.mp4 next to the day folders)")
    args = ap.parse_args()
    if args.seconds <= 0 or args.fade < 0 or args.fps <= 0:
        ap.error("--seconds and --fps must be positive, --fade 0 or more")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found on the PATH")

    root = Path(cfg["save_dir"]) if cfg.get("save_dir") else SAVE_DIR
    day = find_day(args.day, root)
    items = load_images(day)
    if not items:
        sys.exit(f"No images in {day}")
    screen, (pw, ph) = sizes(items)
    out = args.out or day.parent / f"{day.name}.mp4"

    app = QApplication(sys.argv[:1])  # noqa: F841  fonts and text rendering as in the screensaver
    frames = Frames(screen, (pw, ph), args.captions, args.clock, args.prompt)
    frames.clock_24h = cfg.get("clock_24h", True)
    seconds = min(args.seconds, args.length / len(items))  # per image
    fade = min(args.fade, seconds / 4)
    per_image, fade_frames = seconds * args.fps, round(fade * args.fps)
    print(f"{day.name}: {len(items)} images -> {out}\n"
          f"{pw} x {ph}, {seconds:.3g} s per image ({per_image:.3g} frames, {fade_frames} of them fading), "
          f"{args.fps} fps: {hms(len(items) * seconds)} long")
    if per_image < 1:
        print("Some images get no frame at all; raise --fps or --length to show every one.", file=sys.stderr)

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{pw}x{ph}", "-r", str(args.fps), "-i", "-",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    previous, previous_idea, t0 = None, None, time.time()
    progress(0, len(items), 0)
    try:
        for i, it in enumerate(items):
            meta = it["meta"]
            idea = {"kicker": meta["caption"]["kicker"], "text": meta["caption"]["text"], "prompt": meta.get("prompt", "")}
            try:
                current = frames.picture(it["jpg"])
            except OSError as e:  # damaged or half-written JPEG
                print(f"\nskipped {it['jpg'].name}: {e}", file=sys.stderr)
                continue
            n = round((i + 1) * per_image) - round(i * per_image)
            fading = min(fade_frames, n) if previous is not None else 0  # the first image just appears
            same = (previous_idea is not None and not args.prompt
                    and (previous_idea["kicker"], previous_idea["text"]) == (idea["kicker"], idea["text"]))
            for k in range(fading):  # every fading frame a real blend: no frame of 0 % (only the old picture)
                f = frames.ease((k + 1) / (fade_frames + 1))
                proc.stdin.write(frames.frame(current, idea, it["created"], previous, f, 1.0 if same else None))
            still = frames.frame(current, idea, it["created"])
            for _ in range(n - fading):
                proc.stdin.write(still)
            previous, previous_idea = current, idea
            progress(i + 1, len(items), time.time() - t0)
        proc.stdin.close()
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        out.unlink(missing_ok=True)
        sys.exit("\nStopped; no movie written.")
    except OSError:  # writing failed: ffmpeg quit (its own message is above)
        pass
    print(file=sys.stderr)
    if proc.wait() != 0:
        sys.exit(f"ffmpeg failed (exit code {proc.returncode})")
    print(f"Done in {hms(time.time() - t0)}: {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
