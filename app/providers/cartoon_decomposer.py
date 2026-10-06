"""Local PIP asset decomposition for flattened action images.

This creates reusable RGBA layers from existing action renders. It intentionally
does not invent missing pixels: GrabCut provides the character silhouette and
known action manifests define conservative semantic masks for head/arms/backpack.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter

W, H = 1080, 1920


@dataclass(frozen=True)
class ActionRegions:
    body: tuple[float, float, float, float]
    head: tuple[float, float, float, float]
    left_arm: tuple[float, float, float, float]
    right_arm: tuple[float, float, float, float]
    backpack: tuple[float, float, float, float]


REGIONS = {
    "pip_press_button.png": ActionRegions(
        (0.10, 0.22, 0.90, 0.78), (0.23, 0.27, 0.67, 0.48),
        (0.22, 0.42, 0.49, 0.59), (0.63, 0.29, 0.83, 0.51),
        (0.11, 0.45, 0.31, 0.71),
    ),
    "pip_shock.png": ActionRegions(
        (0.46, 0.37, 0.995, 0.78), (0.58, 0.45, 0.985, 0.65),
        (0.48, 0.55, 0.67, 0.73), (0.86, 0.50, 0.995, 0.69),
        (0.83, 0.57, 0.995, 0.78),
    ),
    "pip_overwhelmed.png": ActionRegions(
        (0.02, 0.48, 0.63, 0.95), (0.19, 0.57, 0.60, 0.78),
        (0.10, 0.64, 0.34, 0.81), (0.45, 0.62, 0.60, 0.79),
        (0.02, 0.58, 0.20, 0.93),
    ),
    "pip_deadpan.png": ActionRegions(
        (0.30, 0.37, 0.80, 0.76), (0.38, 0.42, 0.73, 0.64),
        (0.29, 0.55, 0.55, 0.74), (0.64, 0.52, 0.83, 0.73),
        (0.29, 0.49, 0.45, 0.74),
    ),
}


def _cover(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    scale = max(W / image.width, H / image.height)
    image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    x = max(0, (image.width - W) // 2)
    y = max(0, (image.height - H) // 2)
    return image.crop((x, y, x + W, y + H))


def _seed_masks(rgb: np.ndarray, body: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    blue = (
        (hsv[..., 0] >= 85) & (hsv[..., 0] <= 125) &
        (hsv[..., 1] >= 35) & (hsv[..., 2] >= 60)
    )
    orange = (
        (hsv[..., 0] <= 28) & (hsv[..., 0] >= 3) &
        (hsv[..., 1] >= 45) & (hsv[..., 2] >= 70)
    )
    h, w = rgb.shape[:2]
    inside = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = body
    inside[y0:y1, x0:x1] = True
    sure_fg = (blue | orange) & inside
    # Restrict background certainty to a margin outside the character box.
    sure_bg = ~inside
    return sure_fg, sure_bg


def _grabcut_alpha(rgb: np.ndarray, body: tuple[int, int, int, int]) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = 0.5
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    body_small = tuple(int(v * scale) for v in body)
    fg, bg = _seed_masks(small, body_small)

    mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    mask[bg] = cv2.GC_BGD
    mask[fg] = cv2.GC_FGD
    # Tighten sure background around the outer image edges.
    mask[: max(2, sh // 30), :] = cv2.GC_BGD
    mask[-max(2, sh // 30):, :] = cv2.GC_BGD
    mask[:, : max(2, sw // 30)] = cv2.GC_BGD
    mask[:, -max(2, sw // 30):] = cv2.GC_BGD

    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)

    alpha = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.1)
    return cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)


def _box(region: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(round(v * s) for v, s in zip(region, (W, H, W, H)))


def _ellipse_mask(bbox: tuple[int, int, int, int], feather: int = 8) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, ((x0 + x1) // 2, (y0 + y1) // 2),
                (max(2, (x1 - x0) // 2), max(2, (y1 - y0) // 2)), 0, 0, 360, 255, -1)
    if feather:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask


def _layer(src: Image.Image, alpha: np.ndarray, selector: np.ndarray) -> Image.Image:
    combined = np.minimum(alpha, selector).astype(np.uint8)
    rgba = np.dstack([np.asarray(src), combined])
    return Image.fromarray(rgba, "RGBA")


def _semantic_color_mask(src: Image.Image, region: tuple[float, float, float, float], kind: str) -> np.ndarray:
    rgb = np.asarray(src)
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    x0, y0, x1, y1 = _box(region)
    out = np.zeros((H, W), np.uint8)
    if kind == "blue":
        m = (hsv[..., 0] >= 85) & (hsv[..., 0] <= 125) & (hsv[..., 1] >= 30) & (hsv[..., 2] >= 55)
    else:
        m = (hsv[..., 0] <= 30) & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 65)
    crop = (m[y0:y1, x0:x1].astype(np.uint8) * 255)
    crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    out[y0:y1, x0:x1] = crop
    return cv2.GaussianBlur(out, (0, 0), 2.0)


def _body_plate(src: Image.Image, alpha: np.ndarray, masks: list[np.ndarray]) -> Image.Image:
    rgb = np.asarray(src)
    remove = np.zeros((H, W), np.uint8)
    for m in masks:
        remove = np.maximum(remove, (m > 60).astype(np.uint8) * 255)
    # Remove only movable limbs/backpack. Keeping the head in the plate makes the
    # first milestone robust; localized head motion remains handled by the rig.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    plate = cv2.inpaint(bgr, remove, 5, cv2.INPAINT_TELEA)
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
    # Outside the source character silhouette, retain the original scene exactly.
    # The plate therefore remains a normal full-frame background with arm pixels
    # removed only where motion will be overlaid.
    return Image.fromarray(plate, "RGB")


def decompose_action(path: Path, output_dir: Path, character: str = "PIP") -> dict:
    if character != "PIP":
        raise ValueError("Only PIP action decomposition is currently supported")
    regions = REGIONS.get(path.name)
    if not regions:
        raise ValueError(f"No decomposition manifest for {path.name}")

    src = _cover(Image.open(path))
    rgb = np.asarray(src)
    body_box = _box(regions.body)
    alpha = _grabcut_alpha(rgb, body_box)

    head = np.minimum(_ellipse_mask(_box(regions.head), 9), alpha)
    left_arm = np.minimum(
        _ellipse_mask(_box(regions.left_arm), 7),
        np.maximum(_semantic_color_mask(src, regions.left_arm, "blue"), _semantic_color_mask(src, regions.left_arm, "orange")),
    )
    right_arm = np.minimum(
        _ellipse_mask(_box(regions.right_arm), 7),
        np.maximum(_semantic_color_mask(src, regions.right_arm, "blue"), _semantic_color_mask(src, regions.right_arm, "orange")),
    )
    backpack = np.minimum(
        _ellipse_mask(_box(regions.backpack), 7),
        _semantic_color_mask(src, regions.backpack, "orange"),
    )
    left_arm = np.minimum(left_arm, alpha)
    right_arm = np.minimum(right_arm, alpha)
    backpack = np.minimum(backpack, alpha)

    # Body remains the complete cutout in this first decomposition milestone.
    # The separate semantic layers are additive overlays; this avoids destructive
    # inpainting artifacts on the flattened backgrounds.
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    _layer(src, alpha, np.full((H, W), 255, np.uint8)).save(output_dir / f"{stem}_full.png")
    _layer(src, alpha, np.full((H, W), 255, np.uint8)).save(output_dir / f"{stem}_body.png")
    _body_plate(src, alpha, [left_arm, right_arm, backpack]).save(output_dir / f"{stem}_body_plate.jpg", quality=92)
    _layer(src, alpha, head).save(output_dir / f"{stem}_head.png")
    _layer(src, alpha, left_arm).save(output_dir / f"{stem}_left_arm.png")
    _layer(src, alpha, right_arm).save(output_dir / f"{stem}_right_arm.png")
    _layer(src, alpha, backpack).save(output_dir / f"{stem}_backpack.png")

    metadata = {
        "character": character,
        "body_plate": f"{stem}_body_plate.jpg",
        "source": str(path),
        "mode": "local-grabcut-semantic-proxy",
        "resolution": [W, H],
        "layers": ["full", "body", "head", "left_arm", "right_arm", "backpack"],
        "regions": regions.__dict__,
        "alpha_coverage": round(float((alpha > 128).mean()), 4),
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def decompose_pip_actions(action_dir: Path, output_dir: Path) -> list[dict]:
    results = []
    for path in sorted(action_dir.glob("pip_*.png")):
        if path.name in REGIONS:
            results.append(decompose_action(path, output_dir))
    return results


def ensure_action_decomposed(path: Path, output_dir: Path) -> Path:
    stem = path.stem
    required = output_dir / f"{stem}_right_arm.png"
    if not required.exists():
        decompose_action(path, output_dir)
    return output_dir


@lru_cache(maxsize=32)
def load_action_layers(path: Path, output_dir: Path) -> dict[str, Image.Image]:
    ensure_action_decomposed(path, output_dir)
    stem = path.stem
    names = ("body_plate", "head", "left_arm", "right_arm", "backpack")
    result: dict[str, Image.Image] = {}
    for name in names:
        ext = "jpg" if name == "body_plate" else "png"
        result[name] = Image.open(output_dir / f"{stem}_{name}.{ext}").convert("RGBA" if ext == "png" else "RGB")
    return result
