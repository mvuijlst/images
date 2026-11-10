"""Image generation shared by generate.py (CLI) and app.pyw (GUI).

Models:
- DALL·E Mega (min-dalle PyTorch port): square, 256 × 256.
- Stable Diffusion 1.5 (sd_engine.py): any aspect ratio, 512 px on the short side.

Larger output sizes come from Real-ESRGAN x4 (real added detail up to 4×), then a Lanczos resize
to the exact size; beyond 4× the extra pixels are interpolated.
"""

import gc
import json
import os
import random
import re
import threading
import time
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

warnings.filterwarnings("ignore", category=FutureWarning, module="min_dalle")  # torch.cuda.amp.autocast deprecation

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from min_dalle import MinDalle
from min_dalle.text_tokenizer import TextTokenizer

import download_weights

HERE = Path(__file__).parent
OUTPUTS = HERE / "outputs"
NATIVE = 256
TOKENS = 256  # DALL·E Mega image tokens per image
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
SIZES = [256, 512, 768, 1024, 1536, 2048]  # short side
UPSCALERS = {"ai": "AI (Real-ESRGAN)", "lanczos": "Smooth (Lanczos)"}
MODELS = {"mega": "DALL·E Mega", "sd": "Stable Diffusion 1.5"}
MODEL_NATIVE = {"mega": 256, "sd": 512}  # short side, px
ASPECTS = ["1:1", "4:3", "3:2", "16:9", "2:1", "32:9", "3:4", "2:3", "9:16", "1:2", "9:32"]


class Cancelled(Exception):
    pass


def aspect_value(aspect: str) -> float:
    w, h = aspect.split(":")
    return float(w) / float(h)


@dataclass
class Settings:
    prompt: str
    model: str = "mega"  # key of MODELS
    aspect: str = "1:1"  # w:h; SD only (DALL·E Mega is square)
    grid: int = 1  # N → N×N images (Mega seamless: one texture of N×N tiles)
    seed: int | None = None  # None → random; SD batches use seed, seed+1, …
    size: int | None = None  # output short side in pixels; None → the model's native size
    upscaler: str = "ai"  # key of UPSCALERS
    temperature: float = 1.0  # Mega
    top_k: int = 256  # Mega
    supercondition: int = 16  # Mega guidance
    seamless: bool = False  # Mega
    negative_prompt: str = ""  # SD
    steps: int = 25  # SD
    cfg: float = 7.5  # SD guidance scale
    sampler: str = "dpmpp_2m_karras"  # SD, key of sd_engine.SAMPLERS
    safety: bool = True  # SD safety checker (blacks out images it flags as NSFW)
    init_image: Image.Image | None = field(default=None, repr=False)  # SD: start from this image (img2img)
    strength: float = 1.0  # SD img2img: how much of init_image is redrawn (1 = all of it)


@dataclass
class Result:
    settings: Settings  # seed and size filled in
    images: list[Image.Image]
    paths: list[Path]
    seconds: float
    notes: list[str] = field(default_factory=list)  # e.g. safety-filter hits


def slugify(text: str, n: int = 50) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "image"


def make_grid(images: list[Image.Image], cols: int, tile: int, gap: int = 0) -> Image.Image:
    """Contact sheet; tile is each cell's width (height follows the images' aspect ratio)."""
    tw, th = tile, max(1, round(tile * images[0].height / images[0].width))
    rows = -(-len(images) // cols)
    grid = Image.new("RGB", (cols * tw + (cols - 1) * gap, rows * th + (rows - 1) * gap), "white")
    for i, im in enumerate(images):
        if im.size != (tw, th):
            im = im.resize((tw, th), Image.LANCZOS)
        grid.paste(im, ((i % cols) * (tw + gap), (i // cols) * (th + gap)))
    return grid


class LocalMinDalle(MinDalle):
    """MinDalle that skips the Hugging Face 'download counter' request on every start,
    so generation works offline once the weights are present."""

    def init_tokenizer(self):
        if not (os.path.exists(self.vocab_path) and os.path.exists(self.merges_path)):
            self.download_tokenizer()
        with open(self.vocab_path, encoding="utf8") as f:
            vocab = json.load(f)
        with open(self.merges_path, encoding="utf8") as f:
            merges = f.read().split("\n")[1:-1]
        self.tokenizer = TextTokenizer(vocab, merges)


# --------------------------------------------------------------------------- DALL·E Mega


class MegaEngine:
    default_top_k = 256

    def __init__(self, dtype: str = "float16"):
        self.model = LocalMinDalle(
            models_root=str(HERE / "pretrained"),
            dtype=DTYPES[dtype],
            is_mega=True,
            is_reusable=True,
            is_verbose=False,
        )
        self.device = self.model.device

    def generate(
        self,
        s: Settings,
        on_step: Callable[[int, int], None] | None = None,
        on_preview: Callable[[np.ndarray], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> list[Image.Image]:
        """Native 256-px images (one image of grid*256 px when seamless). s.seed must be set."""
        decoder = self.model.decoder
        sample_tokens = decoder.sample_tokens
        step = 0

        def hooked(*args, **kwargs):  # per-token progress + cancellation
            nonlocal step
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            out = sample_tokens(*args, **kwargs)
            step += 1
            if on_step:
                on_step(step, TOKENS)
            return out

        decoder.sample_tokens = hooked
        try:
            raw = None
            for raw in self.model.generate_raw_image_stream(
                text=s.prompt,
                seed=s.seed,
                grid_size=s.grid,
                progressive_outputs=on_preview is not None,
                is_seamless=s.seamless,
                temperature=s.temperature,
                top_k=s.top_k,
                supercondition_factor=s.supercondition,
            ):
                if on_preview:
                    on_preview(raw.to(torch.uint8).cpu().numpy())
            raw = raw.to(torch.uint8)
        finally:
            del decoder.sample_tokens  # drop the instance attribute → class method again
            torch.cuda.empty_cache()

        if s.seamless:
            return [Image.fromarray(raw.cpu().numpy())]
        g = s.grid
        tiles = raw.view(g * NATIVE, g, NATIVE, 3).transpose(1, 0).reshape(g * g, NATIVE, NATIVE, 3)
        return [Image.fromarray(t) for t in tiles.cpu().numpy()]


# --------------------------------------------------------------------------- upscaling


class Upscaler:
    def __init__(self, device: str):
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from spandrel import ImageModelDescriptor, ModelLoader

            desc = ModelLoader().load_from_file(download_weights.ensure_upscaler())
            assert isinstance(desc, ImageModelDescriptor)
            desc.to(self.device).eval()
            if self.device == "cuda" and desc.supports_half:
                desc.half()
            self._model = desc
        return self._model

    def esrgan(self, img: Image.Image, seamless: bool = False) -> Image.Image:
        """Real-ESRGAN x4, tiled so VRAM stays bounded for large inputs."""
        model = self._load()
        scale = model.scale
        dtype = next(model.model.parameters()).dtype
        x = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1)[None].to(self.device, dtype) / 255
        pad = 16
        # circular padding keeps seamless textures seamless; reflect avoids dark borders otherwise
        x = F.pad(x, (pad,) * 4, mode="circular" if seamless else "reflect")
        _, _, h, w = x.shape
        tile = 384
        if h <= tile + 2 * pad and w <= tile + 2 * pad:
            y = model(x)
        else:
            y = torch.empty(1, 3, h * scale, w * scale, device=self.device, dtype=dtype)
            for y0 in range(0, h, tile):
                for x0 in range(0, w, tile):
                    y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                    ya, xa, yb, xb = max(y0 - pad, 0), max(x0 - pad, 0), min(y1 + pad, h), min(x1 + pad, w)
                    t = model(x[:, :, ya:yb, xa:xb])
                    oy, ox = (y0 - ya) * scale, (x0 - xa) * scale
                    y[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = t[
                        :, :, oy : oy + (y1 - y0) * scale, ox : ox + (x1 - x0) * scale
                    ]
        p = pad * scale
        y = y[:, :, p:-p, p:-p].clamp(0, 1).mul(255).round().to(torch.uint8)
        return Image.fromarray(y[0].permute(1, 2, 0).cpu().numpy())

    def warmup(self) -> None:
        """Load Real-ESRGAN and run it once so the first real upscale isn't ~6 s slower."""
        with torch.inference_mode():
            self.esrgan(Image.new("RGB", (NATIVE, NATIVE), "gray"))
        torch.cuda.empty_cache()

    def resize(self, img: Image.Image, size: int, method: str, seamless: bool = False) -> Image.Image:
        """Scale so the short side becomes `size`, keeping the aspect ratio."""
        short = min(img.size)
        if short == size:
            return img
        scale = size / short
        target = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        if scale > 1 and method == "ai":
            img = self.esrgan(img, seamless)
        if img.size != target:
            img = img.resize(target, Image.LANCZOS)
        return img


def output_dims(s: Settings) -> tuple[int, int]:
    """Final (w, h) in pixels for these settings."""
    size = s.size or MODEL_NATIVE[s.model]
    if s.model == "mega":
        return size, size
    from sd_engine import output_size

    w, h = output_size(aspect_value(s.aspect))
    scale = size / min(w, h)
    return round(w * scale), round(h * scale)


# --------------------------------------------------------------------------- pipeline


class Pipeline:
    """Loads models on demand, generates, upscales and saves.

    Only one image model is kept in VRAM: what a 16 GB card has left beside the desktop (~10 GB
    here) is not always enough for two, and when Windows runs short it pages GPU memory to system
    RAM, which slows sampling ~10×. Switching models costs a reload (~5–25 s) instead."""

    def __init__(self, dtype: str = "float16"):
        self.dtype = dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
        self.engines: dict[str, object] = {}
        self.upscaler = Upscaler(self.device)

    def loaded(self) -> list[str]:
        return [MODELS[k] for k in self.engines]

    def load(self, model: str, on_status: Callable[[str], None] | None = None):
        if model in self.engines:
            return self.engines[model]
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r} (choose from {', '.join(MODELS)})")
        status = on_status or (lambda _m: None)
        if self.engines:  # one image model in VRAM at a time (see class docstring)
            self.engines.clear()
            gc.collect()
            torch.cuda.empty_cache()
        if model == "mega":
            if not download_weights.mega_present():
                status("Downloading DALL·E Mega weights (≈5 GB, one time)…")
                download_weights.ensure_mega(
                    progress=lambda f, done, total: status(f"Downloading {f}: {done >> 20:,} / {total >> 20:,} MB")
                )
            status("Loading DALL·E Mega…")
            engine = MegaEngine(self.dtype)
        else:
            from sd_engine import SDEngine

            engine = SDEngine(self.device, on_status=status)
        self.engines[model] = engine
        return engine

    def run(
        self,
        s: Settings,
        out_dir: Path | None = OUTPUTS,
        on_step: Callable[[int, int], None] | None = None,
        on_preview: Callable[[np.ndarray], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> Result:
        if s.model not in MODELS:
            raise ValueError(f"unknown model {s.model!r} (choose from {', '.join(MODELS)})")
        if s.grid < 1:
            raise ValueError("grid must be at least 1")
        if s.size is not None and s.size < 16:
            raise ValueError("size must be at least 16 px")
        aspect_value(s.aspect)  # raises on malformed input; ratios outside ASPECTS are allowed
        status = on_status or (lambda _msg: None)
        t0 = time.perf_counter()

        engine = self.load(s.model, status)
        s = replace(
            s,
            seed=s.seed if s.seed is not None else random.randint(1, 2**31 - 1),
            size=s.size or MODEL_NATIVE[s.model],
        )
        if s.model == "mega":
            s = replace(s, aspect="1:1")
        else:
            s = replace(s, seamless=False)

        with torch.inference_mode():
            n = 1 if s.seamless else s.grid**2
            status(f"Generating {n} image{'s' * (n > 1)}, seed {s.seed}")
            images = engine.generate(s, on_step, on_preview, cancel)
            notes = list(getattr(engine, "last_notes", []))
            short = min(images[0].size)
            if s.size != short:
                verb = "Upscaling" if s.size > short else "Resizing"
                for i, im in enumerate(images):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    status(f"{verb} {i + 1}/{len(images)}")
                    images[i] = self.upscaler.resize(im, s.size, s.upscaler, s.seamless)
                torch.cuda.empty_cache()

        paths = save(images, s, out_dir) if out_dir is not None else []  # None: caller decides (screensaver)
        return Result(s, images, paths, time.perf_counter() - t0, notes)


def save(images: list[Image.Image], s: Settings, out_dir: Path = OUTPUTS) -> list[Path]:
    """Save each image as PNG with the generation settings embedded as text metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}_{slugify(s.prompt)}"
    if s.model != "mega":
        stem += f"_{s.model}"
    if s.aspect != "1:1":
        stem += "_" + s.aspect.replace(":", "x")
    stem += f"_s{s.seed}"
    w, h = images[0].size
    if (w, h) != (NATIVE, NATIVE) or s.seamless:
        stem += f"_{w}x{h}"
    if s.seamless:
        stem += "_seamless"
    meta = PngInfo()
    meta.add_text("prompt", s.prompt)
    if s.model == "sd" and s.negative_prompt:
        meta.add_text("negative_prompt", s.negative_prompt)
    meta.add_text("parameters", json.dumps({k: v for k, v in vars(s).items() if k != "init_image"}))
    meta.add_text("Software", MODELS[s.model])
    paths = []
    for i, im in enumerate(images, 1):
        path = out_dir / (f"{stem}.png" if len(images) == 1 else f"{stem}_{i}.png")
        im.save(path, pnginfo=meta)
        paths.append(path)
    return paths
