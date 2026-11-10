"""Stable Diffusion 1.5 (CreativeML OpenRAIL-M) via diffusers, fp16.

SD 1.5 was trained at 512 × 512 and works directly at other shapes: the short side stays 512 and
the long side is rounded to a multiple of 64. Very wide or tall shapes (2:1 and beyond) tend to
repeat the subject, a known SD 1.x trait.
"""

import logging
import math
import threading
import warnings
from typing import Callable

import numpy as np
import torch
from PIL import Image

import download_weights as dw

NATIVE = 512
CHUNK = 4  # images per diffusion batch; larger batches run in chunks to bound VRAM
DEFAULT_STEPS = 25
DEFAULT_CFG = 7.5
SAMPLERS = {
    "dpmpp_2m_karras": "DPM++ 2M Karras",
    "euler_a": "Euler a",
    "ddim": "DDIM",
    "pndm": "PNDM (original)",
}
PREVIEW_EVERY = 4  # steps

# Linear latent → RGB approximation for SD 1.x (the widely used "latent2rgb" factors): a cheap,
# blurry live preview without running the VAE decoder.
_LATENT_RGB = torch.tensor(
    [[0.298, 0.207, 0.208], [0.187, 0.286, 0.173], [-0.158, 0.189, 0.264], [-0.184, -0.271, -0.473]]
)


def output_size(aspect: float) -> tuple[int, int]:
    """Native (w, h): short side 512, long side rounded to a multiple of 64."""
    if aspect >= 1:
        return int(round(NATIVE * aspect / 64)) * 64, NATIVE
    return NATIVE, int(round(NATIVE / aspect / 64)) * 64


def _latents_to_rgb(latents: torch.Tensor) -> np.ndarray:
    """(b, 4, h, w) latents → (b, h, w, 3) uint8, ×8 smaller than the final image."""
    rgb = torch.einsum("bchw,cr->bhwr", latents.float(), _LATENT_RGB.to(latents.device))
    return ((rgb + 1) / 2).clamp(0, 1).mul(255).byte().cpu().numpy()


def _grid(tiles: np.ndarray, cols: int) -> np.ndarray:
    n, h, w, c = tiles.shape
    rows = math.ceil(n / cols)
    out = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i, t in enumerate(tiles):
        out[(i // cols) * h : (i // cols + 1) * h, (i % cols) * w : (i % cols + 1) * w] = t
    return out


class SDEngine:
    def __init__(self, device: str = "cuda", on_status: Callable[[str], None] | None = None):
        status = on_status or (lambda _m: None)
        if not dw.sd_present():
            status("Downloading Stable Diffusion 1.5 (≈2.6 GB, one time)…")
        path = dw.ensure_sd(
            progress=lambda f, done, total: status(f"Downloading {f}: {done >> 20:,} / {total >> 20:,} MB")
        )
        status("Loading Stable Diffusion 1.5…")
        # importing diffusers/accelerate pulls in torch.distributed, which logs a Windows-only notice
        logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import diffusers
            import transformers
            from diffusers import StableDiffusionPipeline

            diffusers.utils.logging.disable_progress_bar()
            transformers.utils.logging.disable_progress_bar()
            diffusers.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            pipe = StableDiffusionPipeline.from_pretrained(
                str(path), torch_dtype=torch.float16, variant="fp16", use_safetensors=True, local_files_only=True
            )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.device = device
        self._base_scheduler_config = pipe.scheduler.config
        self._sampler = None
        self._safety_checker = pipe.safety_checker
        self._img2img = None
        self.last_notes: list[str] = []

    @property
    def img2img(self):
        """The image-to-image pipeline; shares the loaded weights and scheduler, so it costs no extra VRAM."""
        if self._img2img is None:
            from diffusers import StableDiffusionImg2ImgPipeline

            self._img2img = StableDiffusionImg2ImgPipeline(**self.pipe.components)
            self._img2img.set_progress_bar_config(disable=True)
        self._img2img.scheduler = self.pipe.scheduler
        self._img2img.safety_checker = self.pipe.safety_checker
        return self._img2img

    def _use_sampler(self, name: str) -> None:
        if name == self._sampler:
            return
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, PNDMScheduler

        cfg = self._base_scheduler_config
        self.pipe.scheduler = {
            "dpmpp_2m_karras": lambda: DPMSolverMultistepScheduler.from_config(cfg, use_karras_sigmas=True),
            "euler_a": lambda: EulerAncestralDiscreteScheduler.from_config(cfg),
            "ddim": lambda: DDIMScheduler.from_config(cfg),
            "pndm": lambda: PNDMScheduler.from_config(cfg),
        }[name]()
        self._sampler = name

    def generate(self, s, on_step=None, on_preview=None, cancel: threading.Event | None = None) -> list[Image.Image]:
        """s: dalle_core.Settings (prompt, aspect, grid, seed, steps, cfg, sampler, negative_prompt, safety,
        and for img2img init_image + strength: the image is redrawn from partway, keeping its layout and light)."""
        from dalle_core import Cancelled, aspect_value

        w, h = output_size(aspect_value(s.aspect))
        n = s.grid**2
        chunks = [list(range(i, min(i + CHUNK, n))) for i in range(0, n, CHUNK)]
        img2img = s.init_image is not None and s.strength < 1
        # img2img runs only the last `strength` part of the schedule: ask for more steps so about s.steps run
        sched_steps = max(s.steps, round(s.steps / s.strength)) if img2img else s.steps
        run_steps = int(sched_steps * s.strength) if img2img else s.steps
        total = run_steps * len(chunks)
        cols = s.grid if w <= h * 1.5 else 1
        self._use_sampler(s.sampler)
        self.pipe.safety_checker = self._safety_checker if s.safety else None
        images: list[Image.Image] = []
        flagged = 0

        for c, idx in enumerate(chunks):
            def callback(pipe, i, t, kwargs, c=c):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                if on_step:
                    on_step(c * run_steps + i + 1, total)
                if on_preview and (i + 1) % PREVIEW_EVERY == 0 and i + 1 < run_steps:
                    # DPM++ keeps its estimate of the finished image (x0); that previews far better
                    # than the still-noisy latents. Other samplers fall back to the latents.
                    outputs = getattr(pipe.scheduler, "model_outputs", None)
                    x0 = outputs[-1] if outputs and outputs[-1] is not None else None
                    latents = x0 if x0 is not None and x0.shape == kwargs["latents"].shape else kwargs["latents"]
                    on_preview(_grid(_latents_to_rgb(latents), min(cols, len(idx))))
                return kwargs

            generators = [torch.Generator(self.device).manual_seed(s.seed + i) for i in idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                common = dict(
                    prompt=s.prompt,
                    negative_prompt=s.negative_prompt or None,
                    num_inference_steps=sched_steps,
                    guidance_scale=s.cfg,
                    num_images_per_prompt=len(idx),
                    generator=generators,
                    callback_on_step_end=callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
                if img2img:
                    start = s.init_image.convert("RGB")
                    if start.size != (w, h):
                        start = start.resize((w, h), Image.LANCZOS)
                    out = self.img2img(image=start, strength=s.strength, **common)
                else:
                    out = self.pipe(width=w, height=h, **common)
            images += out.images
            flagged += sum(bool(x) for x in (out.nsfw_content_detected or []))
            torch.cuda.empty_cache()

        self.last_notes = (
            [f"{flagged} image{'s' * (flagged > 1)} blacked out by the safety filter"] if flagged else []
        )
        return images
