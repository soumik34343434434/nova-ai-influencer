"""Local zero-cost media pipeline using FFmpeg + eSpeak."""

from pathlib import Path
import shutil
import subprocess

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

def make_voiceover(text: str, output_wav: Path, voice: str = "en-us") -> Path:
    espeak = _require("espeak")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    _run([espeak, "-v", voice, "-s", "155", "-w", str(output_wav), text])
    return output_wav

def make_vertical_video(image: Path, audio: Path, output: Path, headline: str, duration_seconds: int = 12) -> Path:
    ffmpeg = _require("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)

    safe = headline.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
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
