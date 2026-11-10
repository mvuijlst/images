"""The light of the hour: makes a finished image look like the time of day it was made for.

SD 1.5 pulls every image towards mid-grey brightness (its noise schedule never lets it start from a truly
dark or bright image), so "at night" in a prompt barely shows: night frames averaged 0.43 brightness
against 0.46 by day, and only ~5 % came out dark. Negative prompts ("daylight, sunny") don't change that.

What works: draw the image as usual, grade it to the hour (darker, cooler, warmer), then let SD draw it
again from the graded version (img2img, strength ~0.6). The redraw keeps the subject and layout and turns
the grade into real light: lit windows, street lamps, a moon. Tested on the same prompts and seeds: night
frames went from 0.41 to 0.11–0.16 brightness, all clearly night. A final check grades any frame that still
comes out too bright, so a night frame is never shown in daylight.

The light follows the real sunrise and sunset and changes gradually: between the key moments below
(full night, dawn, day, golden hour, dusk) it is blended, so a frame a minute later is never a different
world (no dark 6:59 next to a bright 7:00).
"""

from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np
from PIL import Image

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


@dataclass(frozen=True)
class Light:
    tint: tuple[float, float, float]  # r, g, b multipliers
    desaturate: float  # 0–1, towards grey before the tint
    target: float | None  # mean brightness (0–1) the grade brings the image down to; None: tint only
    strength: float  # how much SD redraws the graded image (img2img); 0: grade only
    ceiling: float  # a finished frame brighter than this is graded down to target once more
    negative: str  # added to the negative prompt (terms that contain one of the image's words are left out)
    # a soft pool of light in the middle, where SD usually puts the subject: 0.4 → the centre 1.4× as bright,
    # the edges ~0.8×, the corners ~0.65× (same mean brightness). In plain darkness small subjects got lost:
    # the vision LLM saw a quarter of the words in night frames, a third once the same frames were brightened.
    pool: float = 0.0
    name: str = ""


LIGHTS = {
    "night": Light((0.75, 0.9, 1.25), 0.5, 0.14, 0.62, 0.24,
                   "daylight, daytime, sunlight, sunshine, sunny, blue sky, bright, overexposed", pool=0.4),
    "dusk": Light((0.82, 0.9, 1.18), 0.3, 0.23, 0.6, 0.32, "daylight, midday, harsh sunlight, bright blue sky"),
    "dawn": Light((0.98, 0.88, 1.06), 0.3, 0.28, 0.6, 0.36, "midday, harsh sunlight"),
    "golden hour": Light((1.2, 0.96, 0.7), 0.0, 0.38, 0.5, 1.0, "overcast, cold blue light"),
    # full daylight is SD's own: nothing is done (a target of 0.6 only matters while blending towards it,
    # as frames average ~0.46 by day)
    "day": Light((1.0, 1.0, 1.0), 0.0, 0.6, 0.0, 1.0, ""),
}
LIGHTS = {k: replace(v, name=k) for k, v in LIGHTS.items()}
# The key moments: (minutes from sunrise, light) in the morning and (minutes from sunset, light) in the evening.
# Between two of them the light is blended; before the first and after the last it stays as it is.
# Ghent in November (sunrise ~7:55, sunset ~16:55): night until 6:35, dawn 7:30–8:20, day from 9:35;
# day until 14:25, golden hour 15:55–16:55, dusk 17:20–17:45, night from 18:25.
MORNING = [(-80, "night"), (-25, "dawn"), (25, "dawn"), (100, "day")]
EVENING = [(-150, "day"), (-60, "golden hour"), (0, "golden hour"), (25, "dusk"), (50, "dusk"), (90, "night")]
REDRAW_FROM = 0.3  # with less redraw strength than this (close to day) the grade alone is enough


def _blend(a: Light, b: Light, f: float) -> Light:
    lerp = lambda x, y: x + (y - x) * f  # noqa: E731
    return Light(tuple(lerp(x, y) for x, y in zip(a.tint, b.tint)), lerp(a.desaturate, b.desaturate),
                 lerp(a.target, b.target), lerp(a.strength, b.strength), lerp(a.ceiling, b.ceiling),
                 (a if f < 0.5 else b).negative, lerp(a.pool, b.pool), f"{a.name} -> {b.name} {f:.0%}")  # ASCII: the log file is cp1252


def light_at(now: datetime, sunrise: datetime | None = None, sunset: datetime | None = None) -> Light | None:
    """The light at this moment; None in full daylight. Without sunrise/sunset: 7:00 and 19:00."""
    if sunrise is None or sunset is None:
        sunrise = now.replace(hour=7, minute=0, second=0, microsecond=0)
        sunset = now.replace(hour=19, minute=0, second=0, microsecond=0)
    keys, ref = (MORNING, sunrise) if now < sunrise + (sunset - sunrise) / 2 else (EVENING, sunset)
    m = (now - ref).total_seconds() / 60
    light = LIGHTS[keys[0][1]] if m <= keys[0][0] else LIGHTS[keys[-1][1]]
    for (t0, a), (t1, b) in zip(keys, keys[1:]):
        if t0 <= m <= t1:
            light = LIGHTS[a] if a == b else _blend(LIGHTS[a], LIGHTS[b], (m - t0) / (t1 - t0))
            break
    return None if light is LIGHTS["day"] else light


def light_for(origin: dict) -> Light | None:
    """The light for an image from its origin (local_time, and sunrise/sunset when known, as ISO strings);
    None in full daylight or when it has no time (news, history)."""
    if not origin.get("local_time"):
        return None
    parse = lambda k: datetime.fromisoformat(origin[k]) if origin.get(k) else None  # noqa: E731
    return light_at(parse("local_time"), parse("sunrise"), parse("sunset"))


def brightness(image: Image.Image) -> float:
    a = np.asarray(image.convert("RGB").resize((200, 120)), np.float32) / 255
    return float((a @ _LUMA).mean())


def _curve(a: np.ndarray, t: float) -> np.ndarray:
    """t = 0: unchanged … t = 1: very dark. Gamma lowers the midtones more than the highlights, so lamps and
    lit windows stay bright while the rest falls into shadow."""
    return a ** (1 + 1.6 * t) * (1 - 0.8 * t)


def _pool(h: int, w: int, pool: float) -> np.ndarray:
    """(h, w, 1) brightness factors: 1 + pool in the centre, falling off smoothly towards the corners."""
    y, x = np.linspace(-1, 1, h, dtype=np.float32)[:, None], np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    g = np.exp(-(x * x + y * y) / 2 / 0.35)
    return (1 + pool * (2 * g - 1))[..., None]


def grade(image: Image.Image, light: Light) -> Image.Image:
    """Desaturate, tint, and (if it is brighter) darken to the light's target brightness, with the light's
    pool of brightness in the middle."""
    a = np.asarray(image.convert("RGB"), np.float32) / 255
    if light.desaturate:
        a = (a @ _LUMA)[..., None] * light.desaturate + a * (1 - light.desaturate)
    a = a * np.array(light.tint, np.float32)
    if light.target is not None:
        small = np.asarray(Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8)).resize((200, 120)),
                           np.float32) / 255
        lit_small = _pool(120, 200, light.pool)
        mean = lambda t: float(((_curve(small, t) * lit_small).clip(0, 1) @ _LUMA).mean())  # noqa: E731
        if mean(0) > light.target:
            lo, hi = 0.0, 1.0
            for _ in range(14):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if mean(mid) > light.target else (lo, mid)
            a = _curve(a.clip(0, 1), hi) * _pool(*a.shape[:2], light.pool)
    return Image.fromarray((a.clip(0, 1) * 255 + 0.5).astype(np.uint8))


def add_negative(negative: str, light: Light | None, words: list[str] = (), stem=lambda w: w) -> str:
    """The light's negative terms after the base ones, without terms that contain one of the words."""
    if light is None:
        return negative
    extra = [t for t in light.negative.split(", ") if not any(stem(w) in t for w in words)]
    return ", ".join(t for t in (negative, ", ".join(extra)) if t)
