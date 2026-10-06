"""Local zero-cost media pipeline using macOS say or Linux espeak-ng + FFmpeg."""

from pathlib import Path
import shutil
import subprocess
import sys


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


def make_voiceover(text: str, output_wav: Path, voice: str = "en-US") -> Path:
    """Create speech with only OS tools; no paid API."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    say = shutil.which("say")
    if sys.platform == "darwin" and say:
        aiff = output_wav.with_suffix(".aiff")
        _run([say, "-v", voice, "-r", "170", "-o", str(aiff), text])
        _run([_require("ffmpeg"), "-y", "-i", str(aiff), "-ar", "44100", "-ac", "1", str(output_wav)])
        aiff.unlink(missing_ok=True)
        return output_wav

    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if espeak:
        _run([espeak, "-v", voice.lower(), "-s", "155", "-w", str(output_wav), text])
        return output_wav

    raise LocalMediaError("Install no package on macOS: built-in 'say' is expected. On Linux CI install espeak-ng.")


def make_vertical_video(
    image: Path,
    audio: Path,
    output: Path,
    headline: str,
    duration_seconds: int = 12,
) -> Path:
    ffmpeg = _require("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)

    safe = headline.replace("\", "\\").replace(":", "\:").replace("'", "\'")
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "zoompan=z='min(zoom+0.0008,1.08)':x='iw/2-(iw/zoom/2)':"
        "y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:fps=30,"
        "drawbox=x=50:y=80:w=980:h=150:color=black@0.55:t=fill,"
        f"drawtext=text='{safe}':fontcolor=white:fontsize=54:x=90:y=125,"
        "drawtext=text='AI-generated fictional creator':fontcolor=white@0.85:fontsize=28:x=90:y=1750"
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
    audio = output_dir / f"{slug}.wav"
    video = output_dir / f"{slug}.mp4"
    make_voiceover(draft["script"], audio)
    duration = max(8, min(30, 5 + len(draft["script"]) // 13))
    return make_vertical_video(image, audio, video, draft["hook"], duration)
