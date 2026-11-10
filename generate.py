"""Command-line DALL·E Mega / Stable Diffusion 1.5. For the desktop app, run app.pyw instead.

One-shot:
    uv run generate.py "a watercolor painting of a lighthouse at dusk"
    uv run generate.py "an astronaut riding a horse" --grid 3 --seed 42 --size 1024
    uv run generate.py "a gothic quarter at night" --model sd --aspect 16:9 --negative "blurry"

Interactive (the model stays loaded between prompts):
    uv run generate.py
    > a red fox in the snow --grid 3 --size 512
    > a mountain lake at dawn --model sd --aspect 3:2
    > (empty line repeats the last prompt with a new seed)
"""

import argparse
import os
import re
import shlex
import time
from pathlib import Path

import dalle_core as core

QUIT = {"q", "quit", "exit"}
GEN_KEYS = ("model", "aspect", "grid", "seed", "size", "upscaler", "temperature", "top_k", "supercondition",
            "seamless", "negative_prompt", "steps", "cfg", "sampler", "safety")
SAMPLERS = ("dpmpp_2m_karras", "euler_a", "ddim", "pndm")  # keys of sd_engine.SAMPLERS


def add_generation_args(p: argparse.ArgumentParser) -> None:
    """Options that can change per prompt (also usable inline in interactive mode)."""
    p.add_argument("--model", choices=core.MODELS, default="mega",
                   help="mega = DALL·E Mega (square, 256 px); sd = Stable Diffusion 1.5 (any shape, 512 px)")
    p.add_argument("--aspect", default="1:1", help="w:h, e.g. 16:9 or 9:16 (SD only)")
    p.add_argument("--grid", type=int, default=1, help="N gives N×N images (seamless: one texture of N×N tiles)")
    p.add_argument("--seed", type=int, default=None, help="random seed (default: random, printed)")
    p.add_argument("--size", type=int, default=None,
                   help="output short side in px (default: native, 256 Mega / 512 SD; larger is upscaled)")
    p.add_argument("--upscaler", choices=core.UPSCALERS, default="ai",
                   help="ai = Real-ESRGAN x4 (sharp), lanczos = plain resize (soft)")
    p.add_argument("--temperature", type=float, default=1.0, help="DALL·E Mega")
    p.add_argument("--top-k", type=int, default=256, help="DALL·E Mega")
    p.add_argument("--supercondition", type=int, default=16, help="DALL·E Mega guidance strength")
    p.add_argument("--seamless", action="store_true", help="make one tileable texture (DALL·E Mega)")
    p.add_argument("--negative", dest="negative_prompt", default="", help="SD: what to avoid, e.g. 'blurry, text'")
    p.add_argument("--steps", type=int, default=25, help="SD denoising steps")
    p.add_argument("--cfg", type=float, default=7.5, help="SD guidance scale")
    p.add_argument("--sampler", choices=SAMPLERS, default="dpmpp_2m_karras", help="SD sampler")
    p.add_argument("--no-safety", dest="safety", action="store_false", help="SD: turn off the NSFW safety filter")
    p.add_argument("--out", type=Path, default=core.OUTPUTS)
    p.add_argument("--open", action="store_true", help="open the result in the default image viewer")


def generate(pipeline: core.Pipeline, prompt: str, opts: argparse.Namespace) -> None:
    s = core.Settings(prompt=prompt, **{k: getattr(opts, k) for k in GEN_KEYS})
    r = pipeline.run(s, opts.out, on_status=print)
    for note in r.notes:
        print(f"note: {note}")
    paths = r.paths
    if len(r.images) > 1:  # also save a contact sheet of the batch
        w, h = r.images[0].size
        cols = s.grid if w <= h * 1.5 else 1  # wide images stack vertically
        grid_path = paths[0].with_name(paths[0].stem.rsplit("_", 1)[0] + "_grid.png")
        core.make_grid(r.images, cols, min(w, 512 if cols > 1 else 1024)).save(grid_path)
        paths = [grid_path, *paths]
    print(f"done in {r.seconds:.1f}s (seed {r.settings.seed})")
    for path in paths:
        print(f"saved {path}")
    if opts.open and hasattr(os, "startfile"):
        os.startfile(paths[0])


def split_line(line: str) -> tuple[str, list[str]]:
    """'a dog's toy --grid 3' -> ("a dog's toy", ["--grid", "3"]).
    The prompt is everything before the first ' --', so apostrophes in it are fine."""
    m = re.search(r"(^|\s)--", line)
    if not m:
        return line.strip(), []
    # posix=False keeps Windows backslashes (e.g. --out M:\pics); strip quotes ourselves
    tokens = shlex.split(line[m.start():], posix=False)
    tokens = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in tokens]
    return line[: m.start()].strip(), tokens


def interactive(pipeline: core.Pipeline, defaults: argparse.Namespace) -> None:
    line_parser = argparse.ArgumentParser(prog=">", add_help=False, exit_on_error=False)
    add_generation_args(line_parser)
    line_parser.set_defaults(**{k: v for k, v in vars(defaults).items() if k not in ("prompt", "dtype", "interactive")})

    print(
        "\nType a prompt, optionally followed by options (e.g. `a red fox in the snow --grid 3 --size 1024`).\n"
        "Options: " + " ".join(a.option_strings[0] for a in line_parser._actions) + "\n"
        "Empty line = repeat last prompt (new seed unless one was given). q or Ctrl+C to quit.\n"
    )
    last = None
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.lower() in QUIT:
            return
        if not line:
            if last is None:
                continue
            line = last
            print(f"> {line}")
        try:
            prompt, rest = split_line(line)
            if not prompt:
                print("error: no prompt text before the options")
                continue
            opts = line_parser.parse_args(rest)
        except (argparse.ArgumentError, ValueError, SystemExit) as e:
            print(f"error: {e}")
            continue
        last = line
        try:
            generate(pipeline, prompt, opts)
        except KeyboardInterrupt:
            print("\ncancelled")
        except Exception as e:  # keep the session (and the loaded model) alive
            print(f"error: {type(e).__name__}: {e}")
        print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="DALL·E Mega / Stable Diffusion 1.5 text-to-image. Without a prompt, starts interactive mode."
    )
    p.add_argument("prompt", nargs="?", help="text prompt (omit for interactive mode)")
    p.add_argument("-i", "--interactive", action="store_true", help="stay open for more prompts after this one")
    p.add_argument("--dtype", choices=core.DTYPES, default="float16", help="DALL·E Mega weights")
    add_generation_args(p)
    args = p.parse_args()

    if not core.torch.cuda.is_available():
        print("warning: CUDA not available, running on CPU (very slow)")
    pipeline = core.Pipeline(args.dtype)
    t0 = time.perf_counter()
    pipeline.load(args.model, on_status=print)
    print(f"{core.MODELS[args.model]} loaded in {time.perf_counter() - t0:.1f}s on {pipeline.device_name}")
    if args.prompt:
        generate(pipeline, args.prompt, args)
    if args.interactive or not args.prompt:
        interactive(pipeline, args)


if __name__ == "__main__":
    main()
