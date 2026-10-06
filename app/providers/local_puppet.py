"""Zero-cost 2D puppet animation for the NOVA fallback character.

This is the local motion path: it adds deterministic head sway, blinking,
audio-driven mouth movement, breathing, and small gesture changes without
requiring a paid video model or GPU.
"""

from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

from PIL import Image, ImageDraw


W, H = 540, 960
BG = (21, 23, 28)
SKIN = (215, 169, 130)
DARK = (22, 24, 29)
SUIT = (37, 43, 58)


def _audio_envelope(audio: Path, fps: int, frames: int) -> list[float]:
    try:
        with wave.open(str(audio), "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            raw = wav.readframes(wav.getnframes())
        if width != 2:
            return [0.0] * frames
        import array

        samples = array.array("h")
        samples.frombytes(raw)
        if channels > 1:
            mono = []
            for i in range(0, len(samples), channels):
                mono.append(sum(abs(samples[i + c]) for c in range(channels)) / channels)
        else:
            mono = [abs(x) for x in samples]
        chunk = max(1, int(rate / fps))
        peak = max(mono) or 1.0
        env = []
        for frame in range(frames):
            start = frame * chunk
            block = mono[start:start + chunk]
            value = sum(block) / max(1, len(block))
            env.append(min(1.0, (value / peak) * 2.4))
        return env
    except Exception:
        return [0.15] * frames


def _fit_cover(image: Image.Image) -> Image.Image:
    src = image.convert("RGB")
    ratio = max(W / src.width, H / src.height)
    size = (int(src.width * ratio), int(src.height * ratio))
    src = src.resize(size, Image.Resampling.LANCZOS)
    left = (src.width - W) // 2
    top = (src.height - H) // 2
    return src.crop((left, top, left + W, top + H))


def _rig_frame(base: Image.Image, amp: float, t: float, shot_index: int, performance: dict | None = None) -> Image.Image:
    performance = performance or {}
    expression = performance.get("expression", "neutral")
    gesture = performance.get("gesture", "talk")
    attention = performance.get("attention", "scene")
    movement_scale = float(performance.get("movement_scale", 0.8))
    if "head" in performance:
        movement_scale = max(movement_scale, float(performance["head"]))

    # Performance metadata controls how much the whole shot moves.
    expression_scale = {
        "deadpan": 0.45,
        "tired": 0.60,
        "skeptical": 0.75,
        "neutral": 0.80,
        "surprised": 1.00,
        "overloaded": 1.10,
        "confident": 0.70,
        "curious": 0.90,
        "annoyed": 0.75,
    }.get(expression, 0.80)
    gesture_scale = {
        "freeze": 0.25,
        "phone_glance": 1.00,
        "point_calendar": 0.90,
        "coffee_reach": 0.75,
        "tab_scan": 1.10,
        "lean": 0.90,
        "talk": 0.95,
    }.get(gesture, 0.95)

    angle = 1.25 * math.sin(t * 1.85 + shot_index * 0.7) * movement_scale * expression_scale * gesture_scale
    if attention == "camera":
        angle *= 0.65
    sx = 1.0 + (0.012 * math.sin(t * 1.45) + amp * 0.006) * movement_scale
    sy = 1.0 + (0.010 * math.sin(t * 1.45 + 1.0) + amp * 0.004) * movement_scale
    frame = base.resize((int(W * sx), int(H * sy)), Image.Resampling.BICUBIC)
    frame = frame.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=BG)
    frame = _fit_cover(frame)

    # Preserve the generated character pixels. Performance is expressed through
    # camera sway, breathing, push-in and attention rather than repainting features.
    pan_strength = {
        "phone_glance": 0.012,
        "point_calendar": 0.010,
        "coffee_reach": 0.008,
        "tab_scan": 0.015,
        "lean": 0.012,
        "freeze": 0.003,
    }.get(gesture, 0.008) * movement_scale
    if attention == "camera":
        pan_strength *= 0.55

    dx = int(math.sin(t * 0.95 + shot_index) * W * pan_strength)
    dy = int(math.cos(t * 0.8 + shot_index * 0.4) * H * pan_strength * 0.45)
    if dx or dy:
        frame = _fit_cover(frame.transform(frame.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy)))

    draw = ImageDraw.Draw(frame, "RGBA")
    # Minimal visual cue per shot; avoids every shot becoming identical.
    label = {0: "HOOK", 1: "REACTION", 2: "REVEAL", 3: "PUNCHLINE", 4: "LOOP"}.get(shot_index, "NOVA")
    draw.rounded_rectangle((24, 24, 154, 62), radius=14, fill=(10, 12, 18))
    draw.text((42, 34), label, fill=(235, 238, 244))

    return frame


def make_puppet_video(image: Path, audio: Path, output: Path, duration: float, shot_index: int = 0, performance: dict | None = None) -> Path:
    """Stream lightweight animated frames directly into FFmpeg."""
    fps = 15
    frames = max(1, int(math.ceil(duration * fps)))
    envelope = _audio_envelope(audio, fps, frames)
    base = _fit_cover(Image.open(image))

    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / f"{output.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(frames):
        frame = _rig_frame(base, envelope[idx], idx / fps, shot_index, performance)
        frame.save(frame_dir / f"frame_{idx:04d}.png", format="PNG", optimize=True)

    ffmpeg = "ffmpeg"
    result = subprocess.run(
        [
            ffmpeg, "-y",
            "-framerate", str(fps), "-i", str(frame_dir / "frame_%04d.png"),
            "-i", str(audio),
            "-t", f"{duration:.3f}",
            "-vf", "scale=1080:1920:flags=lanczos,format=yuv420p",
            "-r", "30",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(output),
        ],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:] or "puppet video encoding failed")
    return output
