from __future__ import annotations

import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path


RATE = 44100


def _wav(path: Path, samples: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(
            b"".join(
                struct.pack("<h", max(-32767, min(32767, int(s * 32767))))
                for s in samples
            )
        )
    return path


def _tone(freq: float, seconds: float, volume: float = 0.22, sweep: float = 0.0) -> list[float]:
    n = max(1, int(seconds * RATE))
    samples: list[float] = []
    attack = max(1, int(0.01 * RATE))
    release = max(1, int(0.08 * RATE))
    for i in range(n):
        t = i / RATE
        f = freq + sweep * (i / max(1, n - 1))
        env = 1.0
        if i < attack:
            env = i / attack
        elif i >= n - release:
            env = (n - i) / release
        samples.append(math.sin(2 * math.pi * f * t) * volume * max(0.0, env))
    return samples


def make_sfx(kind: str, duration: float, output: Path) -> Path:
    n = max(1, int(duration * RATE))
    if kind == "button":
        samples = _tone(520, min(duration, 0.12), 0.22) + _tone(780, min(max(duration - 0.12, 0), 0.18), 0.16)
    elif kind == "chaos":
        samples = _tone(620, min(duration, 0.30), 0.20, 520)
    elif kind == "overwhelmed":
        part = max(duration / 3, 0.08)
        samples = _tone(620, part, 0.15) + _tone(500, part, 0.15) + _tone(390, part, 0.15)
    elif kind == "punchline":
        samples = _tone(260, min(duration, 0.18), 0.20, -120)
    else:
        samples = [0.0] * n
    return _wav(output, (samples + [0.0] * n)[:n])


def make_voice(text: str, output: Path, voice: str = "Samantha", rate: int = 195) -> Path | None:
    say = shutil.which("say")
    if not say or not text.strip():
        return None
    aiff = output.with_suffix(".aiff")
    subprocess.run([say, "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(aiff), "-ar", str(RATE), "-ac", "1", str(output)],
        check=True, capture_output=True, text=True,
    )
    aiff.unlink(missing_ok=True)
    return output
