"""Deterministic, provider-independent character rig extraction.

Current action art is flattened, so the local rig uses a small manifest for known
hero assets and color segmentation inside those regions. Unknown assets fall back
to coarse blue-object detection. The result is a reusable head-performance proxy
rather than a claim of full skeletal separation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

W, H = 1080, 1920


@dataclass(frozen=True)
class CharacterRig:
    character: str
    source: str
    bbox: tuple[int, int, int, int]
    head_bbox: tuple[int, int, int, int]
    head_layer: Image.Image
    cover_plate: Image.Image
    cover_mask: Image.Image
    center: tuple[float, float]


# Fractions of the 1080x1920 covered frame, tuned to the supplied PIP action art.
# These prevent blue kitchen/window pixels from being mistaken for the character.
PIP_ACTION_REGIONS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "pip_press_button.png": {
        "body": (0.10, 0.22, 0.66, 0.76),
        "head": (0.17, 0.27, 0.60, 0.54),
    },
    "pip_shock.png": {
        "body": (0.46, 0.37, 0.995, 0.77),
        "head": (0.52, 0.40, 0.99, 0.63),
    },
    "pip_overwhelmed.png": {
        "body": (0.03, 0.49, 0.62, 0.95),
        "head": (0.08, 0.51, 0.59, 0.78),
    },
    "pip_deadpan.png": {
        "body": (0.31, 0.37, 0.78, 0.75),
        "head": (0.36, 0.39, 0.73, 0.61),
    },
}


def _cover(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    scale = max(W / image.width, H / image.height)
    image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    x = max(0, (image.width - W) // 2)
    y = max(0, (image.height - H) // 2)
    return image.crop((x, y, x + W, y + H))


def _blue_mask(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    ok = (b >= r + 24) & (b >= g + 5) & ((b - r) + (b - g) >= 34) & ((arr.max(axis=2) - arr.min(axis=2)) >= 38)
    return Image.fromarray((ok.astype(np.uint8) * 255), mode="L")


def _orange_mask(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    ok = (r >= b + 28) & (r >= g + 8) & (r > 115) & (g > 45)
    return Image.fromarray((ok.astype(np.uint8) * 220), mode="L")


def _region_pixels(mask: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    out = Image.new("L", mask.size, 0)
    out.paste(mask.crop((x0, y0, x1, y1)), (x0, y0))
    return out


def _fraction_bbox(region: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = region
    return (round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H))


def extract_rig(path: Path, character: str = "PIP") -> CharacterRig:
    base = _cover(Image.open(path))
    manifest = PIP_ACTION_REGIONS.get(path.name) if character == "PIP" else None

    if manifest:
        bbox = _fraction_bbox(manifest["body"])
        head_bbox = _fraction_bbox(manifest["head"])
    else:
        blue = _blue_mask(base)
        data = np.asarray(blue) > 150
        ys, xs = np.where(data)
        if len(xs) < 50:
            raise ValueError(f"Could not detect {character} subject in {path}")
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        bx0, by0, bx1, by1 = bbox
        head_bbox = (bx0, by0, bx1, by0 + round((by1 - by0) * 0.52))

    # For known PIP poses, use a feathered elliptical head mask rather than
    # trusting scene colors. The generated kitchen contains many cyan/blue
    # objects, so pure color segmentation creates visible rectangular seams.
    head_mask = Image.new("L", (W, H), 0)
    hx0, hy0, hx1, hy1 = head_bbox
    mx = round((hx1 - hx0) * 0.08)
    my = round((hy1 - hy0) * 0.06)
    draw = ImageDraw.Draw(head_mask)
    draw.ellipse((hx0 + mx, hy0 + my, hx1 - mx, hy1 - my), fill=245)
    head_mask = head_mask.filter(ImageFilter.GaussianBlur(22))

    head = base.convert("RGBA")
    head.putalpha(head_mask)

    cover_mask = head_mask.filter(ImageFilter.GaussianBlur(12))
    blurred = base.filter(ImageFilter.GaussianBlur(11)).convert("RGBA")
    cover = Image.composite(blurred, base.convert("RGBA"), cover_mask)

    x0, y0, x1, y1 = head_bbox
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return CharacterRig(character, str(path), bbox, head_bbox, head, cover, cover_mask, center)


@lru_cache(maxsize=32)
def get_rig(path: str, character: str = "PIP") -> CharacterRig:
    return extract_rig(Path(path), character)


def rig_summary(rig: CharacterRig) -> dict:
    return {
        "character": rig.character,
        "source": rig.source,
        "bbox": list(rig.bbox),
        "head_bbox": list(rig.head_bbox),
        "center": [round(rig.center[0], 2), round(rig.center[1], 2)],
        "layers": ["head", "face_proxy", "body_proxy", "backpack_proxy"],
        "mode": "flattened-art-performance-proxy",
        "manifest": Path(rig.source).name in PIP_ACTION_REGIONS,
    }
