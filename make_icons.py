"""Icons for the four programs, all from icon.png (the palette DALL·E Mega drew): the same palette and brush
on a different background colour, with a small white sign in the corner.

  Images                icon.png / icon.ico            purple (the original)
  Image Screensaver     icon_screensaver.png / .ico    night blue, a crescent moon
  Image Movie           icon_movie.png / .ico          warm red, a film frame
  Screensaver Settings  icon_settings.png / .ico       slate teal, a gear

  uv run make_icons.py      (writes the files next to it; run make_shortcuts.py afterwards)
"""

import colorsys
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
ICO_SIZES = [(16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64), (128, 128), (256, 256)]
# name → (hue 0–1, saturation factor, brightness factor) for the background, and the corner sign
VARIANTS = {
    "screensaver": ((0.62, 1.0, 0.78), "moon"),
    "movie": ((0.99, 1.05, 1.0), "film"),
    "settings": ((0.50, 0.55, 0.85), "gear"),
}


def background_mask(im: Image.Image) -> np.ndarray:
    """0–1: how much each pixel is the purple-to-pink background (not the palette, the brush or the paint)."""
    a = np.asarray(im.convert("RGBA"), np.float32) / 255
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a[..., :3].max(-1), a[..., :3].min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    # purple and its shadows: blue and red above green, clearly coloured
    purple = ((b > g + 0.08) & (r > g + 0.05) & (sat > 0.25)) \
        | ((r > g + 0.15) & (np.abs(g - b) < 0.1) & (sat > 0.3))  # and the warm pink towards the bottom
    # keep only the purple that reaches the edge (the pink paint blob on the palette does not)
    seed = Image.fromarray((purple * 255).astype(np.uint8)).copy()  # an own copy: fromarray images may be read-only views
    w, h = seed.size
    for xy in ((w // 2, 6), (6, h // 2), (w - 7, h // 2), (w // 2, h - 7)):
        if seed.getpixel(xy) == 255:
            ImageDraw.floodfill(seed, xy, 128)
    region = np.asarray(seed) == 128
    soft = Image.fromarray((region * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)).filter(
        ImageFilter.GaussianBlur(1.5))
    return (np.asarray(soft, np.float32) / 255) * (a[..., 3] > 0)


def recolour(im: Image.Image, mask: np.ndarray, hue: float, sat_k: float, val_k: float) -> Image.Image:
    a = np.asarray(im.convert("RGBA"), np.float32) / 255
    rgb = a[..., :3]
    out = rgb.copy()
    ys, xs = np.nonzero(mask > 0.01)
    for y, x in zip(ys, xs):  # ~700k pixels: fine for a one-off script
        _h, s, v = colorsys.rgb_to_hsv(*rgb[y, x])
        out[y, x] = colorsys.hsv_to_rgb(hue, min(1.0, s * sat_k), min(1.0, v * val_k))
    m = mask[..., None]
    a[..., :3] = rgb * (1 - m) + out * m
    return Image.fromarray((a.clip(0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")


def sign(kind: str, size: int = 1024) -> Image.Image:
    """A white sign on a dark disc, for the bottom-right corner, with a soft shadow."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy, rad = int(size * 0.78), int(size * 0.78), int(size * 0.17)
    disc = (cx - rad, cy - rad, cx + rad, cy + rad)
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse((disc[0] + 10, disc[1] + 14, disc[2] + 10, disc[3] + 14), fill=(0, 0, 0, 120))
    layer = Image.alpha_composite(layer, shadow.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(layer)
    d.ellipse(disc, fill=(20, 20, 32, 235), outline=(255, 255, 255, 255), width=int(size * 0.012))
    white = (255, 255, 255, 255)
    k = rad * 0.62
    if kind == "moon":
        d.ellipse((cx - k, cy - k, cx + k, cy + k), fill=white)
        o = k * 0.55
        d.ellipse((cx - k + o, cy - k - o * 0.35, cx + k + o, cy + k - o * 0.35), fill=(20, 20, 32, 235))
    elif kind == "film":
        fw, fh = k * 1.7, k * 1.25
        d.rounded_rectangle((cx - fw / 2, cy - fh / 2, cx + fw / 2, cy + fh / 2), radius=k * 0.12, fill=white)
        hole = k * 0.14
        for i in range(4):  # sprocket holes along the top and bottom
            x = cx - fw / 2 + fw * (i + 0.5) / 4
            for y in (cy - fh / 2 + hole * 1.4, cy + fh / 2 - hole * 1.4):
                d.rectangle((x - hole / 2, y - hole / 2, x + hole / 2, y + hole / 2), fill=(20, 20, 32, 255))
        d.polygon([(cx - k * 0.22, cy - k * 0.32), (cx - k * 0.22, cy + k * 0.32), (cx + k * 0.34, cy)],
                  fill=(20, 20, 32, 255))  # a play triangle in the frame
    elif kind == "gear":
        teeth, r_out, r_in = 8, k * 1.0, k * 0.78
        pts = []
        for i in range(teeth * 4):
            ang = 2 * math.pi * i / (teeth * 4)
            r = r_out if i % 4 in (1, 2) else r_in
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        d.polygon(pts, fill=white)
        hole = k * 0.34
        d.ellipse((cx - hole, cy - hole, cx + hole, cy + hole), fill=(20, 20, 32, 255))
    return layer


def save(im: Image.Image, stem: str) -> None:
    im.save(HERE / f"{stem}.png")
    im.save(HERE / f"{stem}.ico", sizes=ICO_SIZES)
    print(f"wrote {stem}.png, {stem}.ico")


def main() -> None:
    base = Image.open(HERE / "icon.png").convert("RGBA")
    mask = background_mask(base)
    for name, ((hue, sk, vk), kind) in VARIANTS.items():
        im = recolour(base, mask, hue, sk, vk)
        save(Image.alpha_composite(im, sign(kind, base.width)), f"icon_{name}")


if __name__ == "__main__":
    main()
