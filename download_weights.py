"""Download model weights into ./pretrained.

    uv run download_weights.py            # DALL·E Mega + upscaler
    uv run download_weights.py --all      # also Stable Diffusion 1.5

- DALL·E Mega (min-dalle PyTorch conversion, HF kuprel/min-dalle), ~5.1 GB
- Real-ESRGAN x4plus upscaler (official GitHub release), ~64 MB
- Stable Diffusion 1.5 (stable-diffusion-v1-5/stable-diffusion-v1-5, CreativeML OpenRAIL-M), fp16, ~2.6 GB

Streams each file to disk, resumes partial downloads, and verifies every file: SHA-256 for
LFS files and pinned files, git blob SHA-1 for small Hugging Face files.
The apps also download a missing model the first time it is used.
"""

import argparse
import hashlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable

import requests

HERE = Path(__file__).parent
ROOT = HERE / "pretrained"

MEGA_REPO = "kuprel/min-dalle"
MEGA_FILES = {  # remote file -> local path (relative to ROOT)
    "vocab.json": "dalle_bart_mega/vocab.json",
    "merges.txt": "dalle_bart_mega/merges.txt",
    "encoder.pt": "dalle_bart_mega/encoder.pt",
    "decoder.pt": "dalle_bart_mega/decoder.pt",
    "detoker.pt": "vqgan/detoker.pt",
}

UPSCALER_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
UPSCALER_PATH = ROOT / "upscaler" / "RealESRGAN_x4plus.pth"
# GitHub publishes no digest for this asset; pinned value matches the HF mirror lllyasviel/Annotators.
UPSCALER_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"

SD_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD_DIR = ROOT / "stable-diffusion-v1-5"
SD_FILES = [  # diffusers layout, fp16 weights only
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
    "tokenizer/merges.txt", "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
    "feature_extractor/preprocessor_config.json",
    "safety_checker/config.json", "safety_checker/model.fp16.safetensors",
]

# Named-entity recognition, used by the screensaver to keep real people's names out of prompts
NER_REPO = "dslim/distilbert-NER"  # Apache-2.0, ~250 MB
NER_DIR = ROOT / "distilbert-NER"
NER_FILES = ["config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json",
             "tokenizer_config.json", "vocab.txt"]

Progress = Callable[[str, int, int], None]  # (label, done_bytes, total_bytes)


def _print_progress(label: str, done: int, total: int) -> None:
    if total:
        print(f"\r  {label}: {done / 2**20:,.0f} / {total / 2**20:,.0f} MB", end="", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download(url: str, dest: Path, progress: Progress = _print_progress) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if have and r.status_code != 206:  # server ignored the range: restart
            have = 0
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + have
        done = have
        last = -1
        with part.open("ab" if have else "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done >> 20 != last:  # report once per MB
                    last = done >> 20
                    progress(dest.name, done, total)
    if progress is _print_progress:
        print()
    part.replace(dest)


def _fetch_verified(url: str, dest: Path, check: Callable[[Path], bool] | None, progress: Progress) -> None:
    if dest.exists() and (check is None or check(dest)):
        return
    download(url, dest, progress)
    if check is not None and not check(dest):
        dest.unlink()
        raise RuntimeError(f"{dest.name}: checksum mismatch, file deleted")


@lru_cache(maxsize=None)
def _hf_tree(repo: str) -> dict[str, dict]:
    r = requests.get(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true", timeout=30)
    r.raise_for_status()
    return {e["path"]: e for e in r.json() if e.get("type") == "file"}


def ensure_hf(repo: str, filename: str, dest: Path, progress: Progress = _print_progress) -> Path:
    if dest.exists() and not dest.with_suffix(dest.suffix + ".part").exists():
        try:
            entry = _hf_tree(repo)[filename]
        except requests.RequestException:  # offline: trust what was verified when downloaded
            return dest
    else:
        entry = _hf_tree(repo)[filename]
    if entry.get("lfs"):
        expected = entry["lfs"]["oid"]
        check = lambda p: sha256(p) == expected  # noqa: E731
    else:
        expected = entry["oid"]
        check = lambda p: git_blob_sha1(p) == expected  # noqa: E731
    _fetch_verified(f"https://huggingface.co/{repo}/resolve/main/{filename}", dest, check, progress)
    return dest


def ensure_mega(progress: Progress = _print_progress) -> None:
    for name, rel in MEGA_FILES.items():
        ensure_hf(MEGA_REPO, name, ROOT / rel, progress)


def ensure_upscaler(progress: Progress = _print_progress) -> Path:
    _fetch_verified(UPSCALER_URL, UPSCALER_PATH, lambda p: sha256(p) == UPSCALER_SHA256, progress)
    return UPSCALER_PATH


def ensure_sd(progress: Progress = _print_progress) -> Path:
    for filename in SD_FILES:
        ensure_hf(SD_REPO, filename, SD_DIR / filename, progress)
    return SD_DIR


def ensure_ner(progress: Progress = _print_progress) -> Path:
    for filename in NER_FILES:
        ensure_hf(NER_REPO, filename, NER_DIR / filename, progress)
    return NER_DIR


def mega_present() -> bool:
    return all((ROOT / rel).exists() for rel in MEGA_FILES.values())


def sd_present() -> bool:
    return all((SD_DIR / f).exists() for f in SD_FILES)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--all", action="store_true", help="also Stable Diffusion 1.5 and the screensaver's name model")
    args = p.parse_args()
    try:
        print("DALL·E Mega weights")
        ensure_mega()
        print("Real-ESRGAN x4plus upscaler")
        ensure_upscaler()
        if args.all:
            print("Stable Diffusion 1.5")
            ensure_sd()
            print("Name recognition (screensaver)")
            ensure_ner()
    except Exception as e:
        print(f"FAIL {e}", file=sys.stderr)
        return 1
    print(f"All requested weights verified in {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
