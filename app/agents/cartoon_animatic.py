"""Cheap pre-production animatic used before the expensive final render."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 540, 960


def _font(size: int):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default()


def _cover(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    scale = max(W / img.width, H / img.height)
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    left = max(0, (img.width - W) // 2)
    top = max(0, (img.height - H) // 2)
    return img.crop((left, top, left + W, top + H))


def _frame(path: Path, caption: str, progress: float, index: int) -> Image.Image:
    base = _cover(path)
    zoom = 1.02 + 0.08 * progress
    crop_w = max(1, int(base.width / zoom))
    crop_h = max(1, int(base.height / zoom))
    x = int((base.width - crop_w) * (0.46 + 0.08 * progress))
    y = int((base.height - crop_h) * (0.50 + 0.03 * ((index % 2) - 0.5)))
    frame = base.crop((x, y, x + crop_w, y + crop_h)).resize((W, H), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(frame, "RGBA")
    font = _font(34)
    text = " ".join(caption.split())
    box = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    tw = box[2] - box[0]
    draw.text(((W - tw) / 2, H - 150), text, font=font, fill=(255, 255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0, 235))
    draw.text((18, 18), f"SHOT {index + 1}", font=_font(22), fill=(255, 255, 255, 210))
    return frame


def render_animatic(images: list[Path], captions: list[str], output: Path, seconds_per_shot: float = 0.7, fps: int = 6) -> Path:
    if not images or len(images) != len(captions):
        raise ValueError("images and captions must have equal non-zero lengths")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / (output.stem + "_frames")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    index = 0
    frames_per_shot = max(2, round(seconds_per_shot * fps))
    for shot, (image, caption) in enumerate(zip(images, captions)):
        for i in range(frames_per_shot):
            progress = i / max(1, frames_per_shot - 1)
            _frame(image, caption, progress, shot).save(work / f"frame_{index:04d}.jpg", quality=82)
            index += 1
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i", str(work / "frame_%04d.jpg"),
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(output)],
        check=True, capture_output=True, text=True,
    )
    return output
