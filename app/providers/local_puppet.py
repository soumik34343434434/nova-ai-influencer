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


def _rig_frame(base: Image.Image, amp: float, t: float, shot_index: int) -> Image.Image:
    # A light whole-character sway gives the static source a living pose.
    angle = 1.5 * math.sin(t * 2.1 + shot_index * 0.7)
    sx = 1.0 + 0.015 * math.sin(t * 1.5)
    sy = 1.0 + 0.012 * math.sin(t * 1.5 + 1.0)
    frame = base.resize((int(W * sx), int(H * sy)), Image.Resampling.BICUBIC)
    frame = frame.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=BG)
    frame = _fit_cover(frame)

    draw = ImageDraw.Draw(frame)
    # NOVA fallback face rig coordinates, scaled from the original 1080x1920 art.
    eye_y = 353
    left_eye = 212
    right_eye = 328
    mouth_x = 270
    mouth_y = 414

    # Cover only the simple drawn facial features before redrawing animated ones.
    draw.ellipse((left_eye - 27, eye_y - 18, left_eye + 27, eye_y + 18), fill=SKIN)
    draw.ellipse((right_eye - 27, eye_y - 18, right_eye + 27, eye_y + 18), fill=SKIN)
    draw.rectangle((mouth_x - 48, mouth_y - 17, mouth_x + 48, mouth_y + 21), fill=SKIN)

    blink_period = 3.7
    blink_phase = (t + shot_index * 0.23) % blink_period
    blinking = 0.0 < blink_phase < 0.13

    eye_h = 3 if blinking else 10 + int(2 * math.sin(t * 1.7))
    for x in (left_eye, right_eye):
        draw.ellipse((x - 16, eye_y - eye_h, x + 16, eye_y + eye_h), fill=DARK)

    mouth_w = int(16 + amp * 26)
    mouth_h = int(3 + amp * 18)
    draw.ellipse(
        (mouth_x - mouth_w, mouth_y - mouth_h, mouth_x + mouth_w, mouth_y + mouth_h),
        fill=DARK,
    )

    # Small animated gestures around the torso make each shot read differently.
    gesture = math.sin(t * 2.3 + shot_index * 1.4)
    body_left, body_right = 178, 362
    shoulder_y = 540
    hand_y = int(690 - gesture * 20)
    draw.line((body_left, shoulder_y, 130, hand_y), fill=(140, 148, 169), width=8)
    draw.line((body_right, shoulder_y, 410, 675 + int(gesture * 18)), fill=(140, 148, 169), width=8)
    draw.ellipse((121, hand_y - 8, 139, hand_y + 10), fill=SKIN)
    draw.ellipse((402, 667 + int(gesture * 18), 420, 685 + int(gesture * 18)), fill=SKIN)

    # Minimal visual cue per shot; avoids every shot becoming identical.
    label = {0: "HOOK", 1: "REACTION", 2: "REVEAL", 3: "PUNCHLINE", 4: "LOOP"}.get(shot_index, "NOVA")
    draw.rounded_rectangle((24, 24, 154, 62), radius=14, fill=(10, 12, 18))
    draw.text((42, 34), label, fill=(235, 238, 244))

    return frame


def make_puppet_video(image: Path, audio: Path, output: Path, duration: float, shot_index: int = 0) -> Path:
    """Stream lightweight animated frames directly into FFmpeg."""
    fps = 15
    frames = max(1, int(math.ceil(duration * fps)))
    envelope = _audio_envelope(audio, fps, frames)
    base = _fit_cover(Image.open(image))

    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / f"{output.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(frames):
        frame = _rig_frame(base, envelope[idx], idx / fps, shot_index)
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
