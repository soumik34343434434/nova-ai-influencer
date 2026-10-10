"""Build MOMO/BUBU action images from their canonical character sheets.

Each action in ``config/cartoon_action_packs.json`` that declares a ``crop`` is
cut out of its source sheet with GrabCut, composited onto a clean studio
backdrop at the shared 768x1376 action canvas, and saved under
``assets/cartoon/actions``. The rig regions (body/head/face, as fractions of the
1080x1920 covered frame) are written back into the registry so the renderer can
animate the head and face without guessing from colours.

The output is deterministic for a given sheet and registry, so it can be
re-run safely after a sheet or crop changes:

    python3 tools/cartoon/build_action_packs.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "config" / "cartoon_action_packs.json"
ASSETS = ROOT / "assets" / "cartoon"
ACTION_DIR = ASSETS / "actions"
COVER_W, COVER_H = 1080, 1920

# Share of the canvas the character occupies, by placement type.
FULL_BODY_HEIGHT = 0.42
FULL_BODY_MAX_UPSCALE = 2.3
FOOT_LINE = 0.80
CLOSE_UP_WIDTH = 0.66
CLOSE_UP_CENTER_Y = 0.46


def _grabcut_alpha(sheet: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return a feathered alpha for the pose inside ``box`` and the padded crop box."""
    h, w = sheet.shape[:2]
    x0, y0, x1, y1 = box
    pad = 18
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
    crop = np.ascontiguousarray(sheet[cy0:cy1, cx0:cx1])
    mask = np.zeros(crop.shape[:2], np.uint8)
    rect = (x0 - cx0 - 4, y0 - cy0 - 4, (x1 - x0) + 8, (y1 - y0) + 8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Keep the pose itself, not stray neighbouring fragments of the sheet.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = np.zeros_like(alpha)
        for index in range(1, count):
            if index == largest or stats[index, cv2.CC_STAT_AREA] >= 0.08 * stats[largest, cv2.CC_STAT_AREA]:
                keep[labels == index] = 255
        alpha = keep
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # Fill interior holes (eye whites, chest fur) that GrabCut can drop.
    flood = alpha.copy()
    ff_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    alpha = alpha | cv2.bitwise_not(flood)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
    return alpha, (cx0, cy0, cx1, cy1)


def _backdrop(size: tuple[int, int], spec: dict) -> Image.Image:
    w, h = size
    top = np.array(spec.get("top", [240, 238, 232]), dtype=np.float32)
    floor = np.array(spec.get("floor", [226, 220, 208]), dtype=np.float32)
    horizon = float(spec.get("horizon", 0.62))
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    # Soft wall-to-floor blend around the horizon line.
    blend = np.clip((ys - (horizon - 0.06)) / 0.12, 0.0, 1.0)
    rows = top[None, :] * (1.0 - blend) + floor[None, :] * blend
    image = np.repeat(rows[:, None, :], w, axis=1)
    # Gentle vignette keeps attention on the character.
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    vignette = 1.0 - 0.07 * np.clip(xs ** 2 + yy ** 2, 0.0, 1.6)
    image = image * vignette[..., None]
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB")


def _cover_fraction(box: tuple[float, float, float, float], canvas: tuple[int, int]) -> list[float]:
    """Convert a canvas-pixel box to fractions of the renderer's 1080x1920 cover frame."""
    cw, ch = canvas
    scale = max(COVER_W / cw, COVER_H / ch)
    off_x = (cw * scale - COVER_W) / 2.0
    off_y = (ch * scale - COVER_H) / 2.0
    x0, y0, x1, y1 = box
    out = [
        (x0 * scale - off_x) / COVER_W,
        (y0 * scale - off_y) / COVER_H,
        (x1 * scale - off_x) / COVER_W,
        (y1 * scale - off_y) / COVER_H,
    ]
    return [round(min(1.0, max(0.0, value)), 4) for value in out]


def _sub_box(body: tuple[float, float, float, float], rel: list[float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = body
    bw, bh = x1 - x0, y1 - y0
    return (x0 + rel[0] * bw, y0 + rel[1] * bh, x0 + rel[2] * bw, y0 + rel[3] * bh)


def build_action(sheet_path: Path, spec: dict, backdrop: dict, canvas: tuple[int, int]) -> tuple[Image.Image, dict]:
    sheet = np.asarray(Image.open(sheet_path).convert("RGB"))
    alpha, (cx0, cy0, cx1, cy1) = _grabcut_alpha(sheet, tuple(spec["crop"]))
    rgba = np.dstack([sheet[cy0:cy1, cx0:cx1], alpha])
    ys, xs = np.where(alpha > 24)
    if len(xs) < 200:
        raise ValueError(f"Pose cut-out is empty for {spec.get('file')}")
    tight = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    pose = Image.fromarray(rgba, "RGBA").crop(tight)

    w, h = canvas
    if spec.get("placement") == "close_up":
        scale = (CLOSE_UP_WIDTH * w) / pose.width
    else:
        scale = min((FULL_BODY_HEIGHT * h) / pose.height, FULL_BODY_MAX_UPSCALE)
    size = (max(1, round(pose.width * scale)), max(1, round(pose.height * scale)))
    pose = pose.resize(size, Image.Resampling.LANCZOS)

    left = round((w - pose.width) / 2)
    if spec.get("placement") == "close_up":
        top = round(CLOSE_UP_CENTER_Y * h - pose.height / 2)
    else:
        top = round(FOOT_LINE * h - pose.height)
    frame = _backdrop(canvas, backdrop)
    if spec.get("placement") != "close_up":
        shadow = Image.new("L", canvas, 0)
        ImageDraw.Draw(shadow).ellipse(
            (left + pose.width * 0.12, FOOT_LINE * h - 16, left + pose.width * 0.88, FOOT_LINE * h + 18), fill=90
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(14))
        frame.paste((120, 112, 100), (0, 0), shadow)
    frame.paste(pose, (left, top), pose)

    body = (left, top, left + pose.width, top + pose.height)
    regions = {
        "body": _cover_fraction(body, canvas),
        "head": _cover_fraction(_sub_box(body, spec["head_rel"]), canvas),
        "face": _cover_fraction(_sub_box(body, spec["face_rel"]), canvas),
    }
    return frame, regions


def build_all(registry_path: Path = REGISTRY, output_dir: Path = ACTION_DIR, write_registry: bool = True) -> list[Path]:
    registry = json.loads(registry_path.read_text())
    canvas = tuple(registry.get("canvas", [768, 1376]))
    written: list[Path] = []
    for character, entry in registry["characters"].items():
        source = entry.get("source")
        if not source:
            continue
        sheet = ASSETS / source
        for action, spec in entry["actions"].items():
            if "crop" not in spec:
                continue
            image, regions = build_action(sheet, spec, entry.get("backdrop", {}), canvas)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / spec["file"]
            image.save(path, optimize=True)
            spec["regions"] = regions
            written.append(path)
    if write_registry:
        registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ACTION_DIR)
    parser.add_argument("--no-write-registry", action="store_true")
    args = parser.parse_args()
    for path in build_all(output_dir=args.output_dir, write_registry=not args.no_write_registry):
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
