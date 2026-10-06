"""Zero-cost photo-mix renderer for original cartoon comedy shorts."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 1080, 1920


def _font(size: int):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _fit_cover(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    scale = max(W / img.width, H / img.height)
    size = (int(img.width * scale), int(img.height * scale))
    img = img.resize(size, Image.Resampling.LANCZOS)
    x = max(0, (img.width - W) // 2)
    y = max(0, (img.height - H) // 2)
    return img.crop((x, y, x + W, y + H))


def _caption_lines(text: str, max_width: int = 900) -> list[str]:
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    font = _font(60)
    lines: list[str] = []
    current = ""
    for word in " ".join(text.split()).split():
        candidate = (current + " " + word).strip()
        if draw.textbbox((0, 0), candidate, font=font, stroke_width=1)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def render_frame(image: Path, caption: str, progress: float, zoom: float) -> Image.Image:
    base = _fit_cover(Image.open(image))
    crop_w = max(1, int(base.width / zoom))
    crop_h = max(1, int(base.height / zoom))
    phase = progress * math.tau
    x = int((base.width - crop_w) * (0.5 + 0.18 * math.sin(phase)))
    y = int((base.height - crop_h) * (0.5 + 0.08 * math.cos(phase)))
    x = max(0, min(x, base.width - crop_w))
    y = max(0, min(y, base.height - crop_h))
    frame = base.crop((x, y, x + crop_w, y + crop_h)).resize((W, H), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(frame, "RGBA")
    lines = _caption_lines(caption)
    font = _font(60)
    line_h = 68
    y0 = H - 360
    for i, line in enumerate(lines):
        box = draw.textbbox((0, 0), line, font=font, stroke_width=2)
        tw = box[2] - box[0]
        draw.text(
            ((W - tw) / 2, y0 + i * line_h),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=6,
            stroke_fill=(0, 0, 0, 235),
        )

    disclosure = "AI-generated • fictional cartoon"
    df = _font(30)
    db = draw.textbbox((0, 0), disclosure, font=df)
    draw.text((W - (db[2] - db[0]) - 42, H - 56), disclosure, font=df, fill=(245, 245, 245, 230))
    return frame


def render_photo_mix(images: list[Path], captions: list[str], durations: list[float], output: Path) -> Path:
    if not images or len(images) != len(captions) or len(images) != len(durations):
        raise ValueError("images, captions and durations must have equal non-zero lengths")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")

    work = output.parent / (output.stem + "_frames")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    fps = 8
    frame_count = 0
    concat = []
    for shot, (img, caption, duration) in enumerate(zip(images, captions, durations)):
        n = max(1, round(duration * fps))
        for i in range(n):
            progress = i / max(1, n - 1)
            frame = render_frame(img, caption, progress, 1.04 + 0.08 * (shot % 3))
            path = work / f"frame_{frame_count:05d}.jpg"
            frame.save(path, format="JPEG", quality=88, optimize=True)
            frame_count += 1

    video_noaudio = output.with_suffix(".silent.mp4")
    subprocess.run(
        [
            ffmpeg, "-y", "-framerate", str(fps), "-i", str(work / "frame_%05d.jpg"),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-vf", "scale=1080:1920,format=yuv420p", str(video_noaudio),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    final = output
    voice = output.with_suffix(".wav")
    text = " ".join(captions)
    say = shutil.which("say")
    if say:
        aiff = output.with_suffix(".aiff")
        subprocess.run([say, "-v", "en-US", "-r", "180", "-o", str(aiff), text], check=True)
        subprocess.run(
            [ffmpeg, "-y", "-i", str(aiff), "-ar", "44100", "-ac", "1", str(voice)],
            check=True, capture_output=True, text=True
        )
        aiff.unlink(missing_ok=True)
        subprocess.run(
            [ffmpeg, "-y", "-i", str(video_noaudio), "-i", str(voice), "-shortest",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", str(final)],
            check=True, capture_output=True, text=True
        )
        voice.unlink(missing_ok=True)
    else:
        video_noaudio.replace(final)

    video_noaudio.unlink(missing_ok=True)
    return final
