"""One day of the Image Screensaver as a time-lapse: every image saved that day, in order, drawn the way the
screensaver showed it (its caption band and clock). Kept for its short command line; moviemaker.py (Image Movie)
does the work and has more choices: picking images for smooth transitions, slow zoom, size, what to show.

  uv run daymovie.py                        the latest day
  uv run daymovie.py 2026-11-06             a date, or the path of a day folder
  uv run daymovie.py 2026-11-06 --length 1:30

A whole day fits in --length (default 2:55): each image gets length / number of images, at most --seconds (2),
so a full day of ~1300 images runs at ~0.13 s per image, crossfading for at most a quarter of that. The movie
goes next to the day folders (outputs/screensaver/2026-11-06.mp4) unless --out says otherwise. Captions and
clock follow the screensaver's settings unless --captions / --no-captions, --clock / --no-clock are given.
"""

import argparse
import re
import sys
from pathlib import Path

from moviemaker import Options, duration, load_images, run_cli
from screensaver_settings import load_settings, save_root

DAY_FOLDER = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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


def main() -> None:
    cfg = load_settings()
    ap = argparse.ArgumentParser(description="Make a time-lapse of one day of the Image Screensaver's images, "
                                             "each frame as it looked on screen. (moviemaker.py has more choices.)")
    ap.add_argument("day", nargs="?", help="a date (2026-11-06) or a day folder; default: the latest day")
    ap.add_argument("--length", type=duration, default=175.0, help="longest the movie may be, seconds or m:ss "
                                                                   "(default 2:55)")
    ap.add_argument("--seconds", type=float, default=2.0, help="most time per image (default 2)")
    ap.add_argument("--fade", type=float, default=0.5, help="crossfade, at most this and a quarter of an image's "
                                                            "time; 0 = cut (default 0.5)")
    ap.add_argument("--fps", type=int, default=30, help="frames per second (default 30)")
    ap.add_argument("--captions", action=argparse.BooleanOptionalAction, default=cfg.get("captions", True))
    ap.add_argument("--clock", action=argparse.BooleanOptionalAction, default=cfg.get("clock", True))
    ap.add_argument("--prompt", action="store_true", help="also show each image's prompt")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality, lower = better and bigger (default 18)")
    ap.add_argument("--out", type=Path, help="movie file (default: <day>.mp4 next to the day folders)")
    args = ap.parse_args()
    if args.seconds <= 0 or args.fade < 0 or args.fps <= 0:
        ap.error("--seconds and --fps must be positive, --fade 0 or more")
    day = find_day(args.day, save_root(cfg))
    n = len(load_images(day))
    if not n:
        sys.exit(f"No images in {day}")
    show = tuple(k for k, on in (("caption", args.captions), ("clock", args.clock), ("prompt", args.prompt)) if on)
    run_cli(Options(folder=day, out=args.out or day.parent / f"{day.name}.mp4", length=min(args.length, n * args.seconds),
                    every_image=True, transition="dissolve" if args.fade else "cut",
                    transition_seconds=args.fade or 0.1, zoom=False, show=show,
                    clock_24h=cfg.get("clock_24h", True), fps=args.fps, crf=args.crf))


if __name__ == "__main__":
    main()
