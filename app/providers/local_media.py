"""Local zero-cost media pipeline using macOS say or Linux espeak-ng + FFmpeg.

The renderer deliberately avoids FFmpeg drawtext because some macOS FFmpeg builds
are compiled without libfreetype. Captions/disclosure are rendered into a PNG with
Pillow, then FFmpeg handles only video/audio work.
"""

from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont


class LocalMediaError(RuntimeError):
    pass


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise LocalMediaError(f"Required local binary not found: {binary}")
    return path


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise LocalMediaError(result.stderr[-4000:] or "Media command failed")


def _font(size: int):
    candidates = []
    if sys.platform == "darwin":
        candidates.append("/System/Library/Fonts/Supplemental/Arial.ttf")
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ])
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int = 54):
    for size in range(start_size, 20, -2):
        f = _font(size)
        if draw.textbbox((0, 0), text, font=f)[2] <= max_width:
            return f
    return _font(20)


def make_caption_frame(source: Path, output: Path, headline: str) -> Path:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)

    overlay = (50, 70, image.width - 50, 265)
    draw.rounded_rectangle(overlay, radius=28, fill=(15, 17, 22))

    headline = " ".join(headline.split())
    font = _fit_text(draw, headline, image.width - 180, 56)
    draw.multiline_text(
        (90, 105),
        headline,
        font=font,
        fill=(255, 255, 255),
        spacing=8,
        align="left",
    )

    disclosure_font = _font(30)
    disclosure = "AI-generated fictional creator"
    bbox = draw.textbbox((0, 0), disclosure, font=disclosure_font)
    draw.text(
        (image.width - bbox[2] - 70, image.height - 90),
        disclosure,
        font=disclosure_font,
        fill=(225, 228, 236),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    return output


def make_voiceover(text: str, output_wav: Path, voice: str = "en-US") -> Path:
    """Create speech with OS tools; no paid API."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    say = shutil.which("say")
    if sys.platform == "darwin" and say:
        aiff = output_wav.with_suffix(".aiff")
        _run([say, "-v", voice, "-r", "170", "-o", str(aiff), text])
        _run([
            _require("ffmpeg"), "-y", "-i", str(aiff),
            "-ar", "44100", "-ac", "1", str(output_wav)
        ])
        aiff.unlink(missing_ok=True)
        return output_wav

    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak:
        _run([espeak, "-v", voice.lower(), "-s", "155", "-w", str(output_wav), text])
        return output_wav

    raise LocalMediaError(
        "No local speech engine found. macOS uses built-in 'say'; Linux uses espeak-ng."
    )


def make_vertical_video(
    image: Path,
    audio: Path,
    output: Path,
    duration_seconds: int = 12,
) -> Path:
    ffmpeg = _require("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)

    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "zoompan=z='min(zoom+0.0008,1.08)':x='iw/2-(iw/zoom/2)':"
        "y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:fps=30"
    )

    _run([
        ffmpeg, "-y", "-loop", "1", "-i", str(image), "-i", str(audio),
        "-t", str(duration_seconds), "-vf", vf, "-r", "30",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest", str(output),
    ])
    return output


def build_video_from_draft(image: Path, draft: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = "_".join(
        "".join(ch.lower() if ch.isalnum() else "_" for ch in draft["title"]).split()
    )[:60]

    frame = output_dir / f"{slug}_frame.png"
    audio = output_dir / f"{slug}.wav"
    video = output_dir / f"{slug}.mp4"

    make_caption_frame(image, frame, draft["hook"])
    make_voiceover(draft["script"], audio)
    duration = max(8, min(30, 5 + len(draft["script"]) // 13))
    return make_vertical_video(frame, audio, video, duration)
