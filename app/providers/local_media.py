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

from app.agents.storyboard import Storyboard


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
    """Build a multi-shot, voice-synced Reel from the draft."""
    from app.agents.storyboard import build_storyboard

    storyboard: Storyboard = build_storyboard(draft)
    output_dir.mkdir(parents=True, exist_ok=True)
    return build_video_from_storyboard(image, storyboard, output_dir)


# --- Storyboard / shot renderer -------------------------------------------------

def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_shot_frame(source: Path, output: Path, shot) -> Path:
    source_img = Image.open(source).convert("RGB")
    source_ratio = source_img.width / source_img.height
    target_ratio = 1080 / 1920
    if source_ratio > target_ratio:
        crop_h = source_img.height
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = source_img.width
        crop_h = int(crop_w / target_ratio)

    camera = shot.camera
    zoom = {
        "hook_close_up": 1.18,
        "reaction_close_up": 1.22,
        "over_shoulder": 1.08,
        "medium": 1.0,
        "wide_desk": 0.92,
    }.get(camera, 1.0)
    crop_w = max(1, int(crop_w / zoom))
    crop_h = max(1, int(crop_h / zoom))
    max_x = max(0, source_img.width - crop_w)
    max_y = max(0, source_img.height - crop_h)
    offset = [0, 0.18, 0.38, 0.62, 0.82][shot.index % 5]
    x = int(max_x * offset)
    y = int(max_y * (0.5 + (0.08 if shot.index % 2 else -0.04)))
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))
    frame = source_img.crop((x, y, x + crop_w, y + crop_h)).resize((1080, 1920), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(frame, "RGBA")
    # Lightweight social-native UI and a distinct visual beat per shot.
    draw.rounded_rectangle((44, 44, 350, 120), radius=28, fill=(10, 12, 18, 205))
    draw.text((78, 66), f"NOVA • {shot.index + 1}/{shot.index + 1}", font=_font(28), fill=(255, 255, 255, 255))

    dialogue_font = _font(46)
    bubble_w = 930
    lines = _wrap_lines(draw, shot.dialogue, dialogue_font, 840)
    line_h = 58
    bubble_h = min(500, 48 + line_h * len(lines))
    bubble_y = 1290 if shot.camera == "reaction_close_up" else 1360
    draw.rounded_rectangle((60, bubble_y, bubble_w, bubble_y + bubble_h), radius=36, fill=(8, 10, 15, 232))
    for idx, line in enumerate(lines[:7]):
        draw.text((100, bubble_y + 28 + idx * line_h), line, font=dialogue_font, fill=(255, 255, 255, 255))

    action_font = _font(27)
    action = shot.action.replace("_", " ").upper()
    draw.rounded_rectangle((62, bubble_y - 74, 440, bubble_y - 20), radius=22, fill=(255, 255, 255, 35), outline=(255, 255, 255, 100), width=2)
    draw.text((82, bubble_y - 61), action, font=action_font, fill=(240, 242, 247, 255))
    disclosure = "AI-generated fictional creator"
    disc_font = _font(28)
    bbox = draw.textbbox((0, 0), disclosure, font=disc_font)
    draw.rounded_rectangle((1080 - bbox[2] - 74, 1812, 1070, 1880), radius=20, fill=(8, 10, 15, 210))
    draw.text((1080 - bbox[2] - 54, 1832), disclosure, font=disc_font, fill=(228, 230, 235, 255))

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.save(output, format="PNG", optimize=True)
    return output


def build_video_from_storyboard(image: Path, storyboard, output_dir: Path) -> Path:
    """Render each storyboard shot to its own segment, with its own synced voiceover."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shot_dir = output_dir / "shots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []

    for shot in storyboard.shots:
        slug = f"shot_{shot.index:02d}"
        frame = shot_dir / f"{slug}.png"
        audio = shot_dir / f"{slug}.wav"
        segment = shot_dir / f"{slug}.mp4"
        make_shot_frame(image, frame, shot)
        make_voiceover(shot.dialogue, audio)
        duration = max(2.4, shot.duration)
        ffmpeg = _require("ffmpeg")
        vf = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "zoompan=z='min(zoom+0.0012,1.06)':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            "d=1:s=1080x1920:fps=30"
        )
        _run([
            ffmpeg, "-y", "-loop", "1", "-i", str(frame), "-i", str(audio),
            "-t", str(duration), "-vf", vf, "-r", "30",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest", str(segment),
        ])
        segments.append(segment)

    concat_file = shot_dir / "concat.txt"
    concat_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in segments))
    slug = "_".join("".join(ch.lower() if ch.isalnum() else "_" for ch in storyboard.hook).split())[:60] or "nova_story"
    output = output_dir / f"{slug}_storyboard.mp4"
    ffmpeg = _require("ffmpeg")
    _run([
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy", str(output),
    ])
    return output
