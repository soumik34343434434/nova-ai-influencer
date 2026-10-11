from __future__ import annotations

import json
import math
import random
import shutil
import struct
import subprocess
import wave
from pathlib import Path


RATE = 44100
# Platform loudness target for shorts (YouTube, TikTok and Instagram normalise to about -14 LUFS
# and do not raise quiet uploads) and a true-peak ceiling with headroom for codecs.
LOUDNESS_TARGET_LUFS = -14.0
LOUDNESS_TRUE_PEAK_DBTP = -1.5
LOUDNESS_RANGE_LU = 11.0


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


def _sweep(
    f0: float,
    f1: float,
    seconds: float,
    volume: float,
    *,
    harmonics: tuple[float, ...] = (1.0, 0.5, 0.25),
    decay: float = 0.0,
) -> list[float]:
    """Pitch-gliding tone with overtones, a fast attack and an exponential decay."""
    n = max(1, int(seconds * RATE))
    attack = max(1, int(0.004 * RATE))
    release = max(1, int(0.02 * RATE))
    norm = sum(harmonics) or 1.0
    out = [0.0] * n
    phase = 0.0
    for i in range(n):
        freq = f0 + (f1 - f0) * (i / max(1, n - 1))
        phase += 2.0 * math.pi * freq / RATE
        tone = sum(h * math.sin(k * phase) for k, h in enumerate(harmonics, start=1)) / norm
        env = min(1.0, i / attack) * min(1.0, (n - i) / release) * math.exp(-decay * i / RATE)
        out[i] = volume * tone * env
    return out


def _noise(seconds: float, volume: float, low: float, high: float, *, decay: float = 0.0, seed: str = "") -> list[float]:
    """Band-limited noise between ``low`` and ``high`` Hz (a difference of two one-pole low-passes)."""
    n = max(1, int(seconds * RATE))
    rng = random.Random(seed or f"{low}-{high}-{n}")
    raw = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    hi = _lowpass(raw, high)
    lo = _lowpass(raw, low)
    band = [a - b for a, b in zip(hi, lo)]
    peak = max((abs(v) for v in band), default=0.0) or 1.0
    return [
        volume * (v / peak) * math.exp(-decay * i / RATE) * min(1.0, (n - i) / max(1, int(0.01 * RATE)))
        for i, v in enumerate(band)
    ]


def _lowpass(samples: list[float], cutoff: float) -> list[float]:
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff / RATE)
    state = 0.0
    out = []
    for s in samples:
        state += alpha * (s - state)
        out.append(state)
    return out


def _delay(samples: list[float], seconds: float) -> list[float]:
    return [0.0] * int(seconds * RATE) + samples


def _mix(*layers: list[float]) -> list[float]:
    out = [0.0] * max((len(layer) for layer in layers), default=0)
    for layer in layers:
        for i, s in enumerate(layer):
            out[i] += s
    return out


def make_sfx(kind: str, duration: float, output: Path) -> Path:
    """Layered cartoon sound effect: tone overtones plus band-limited noise, energy mostly 400 Hz to 4 kHz."""
    n = max(1, int(duration * RATE))
    d = duration
    if kind == "button":
        layers = [
            _sweep(520, 520, min(d, 0.12), 0.30, decay=20),
            _delay(_sweep(780, 780, min(d, 0.16), 0.26, decay=24), 0.09),
            _noise(min(d, 0.02), 0.12, 1200, 4000, decay=200, seed=kind),
        ]
    elif kind == "chaos":
        layers = [
            _sweep(620, 1100, min(d, 0.30), 0.26, harmonics=(1.0, 0.6, 0.4), decay=6),
            _noise(min(d, 0.30), 0.14, 600, 3000, decay=10, seed=kind),
        ]
    elif kind == "overwhelmed":
        part = max(d / 3, 0.08)
        layers = [
            _sweep(620, 620, part, 0.22, harmonics=(1.0, 0.5, 0.3), decay=10),
            _delay(_sweep(500, 500, part, 0.22, harmonics=(1.0, 0.5, 0.3), decay=10), part),
            _delay(_sweep(390, 390, part, 0.22, harmonics=(1.0, 0.5, 0.3), decay=10), 2 * part),
        ]
    elif kind == "punchline":
        # Falling "wah" with overtones: the deadpan sting after a punchline.
        layers = [
            _sweep(480, 240, min(d, 0.24), 0.30, harmonics=(1.0, 0.6, 0.4, 0.25), decay=5),
            _noise(min(d, 0.06), 0.10, 2500, 4000, decay=60, seed=kind),
        ]
    elif kind == "notification":
        layers = [
            _sweep(880, 880, min(d, 0.08), 0.26, harmonics=(1.0, 0.4), decay=20),
            _delay(_sweep(1175, 1175, min(d, 0.10), 0.26, harmonics=(1.0, 0.4), decay=18), 0.13),
        ]
    elif kind in {"impact", "reaction_hit"}:
        # A thump whose overtones and mid-band crack carry on phone speakers, which cannot reproduce
        # the sub-200 Hz fundamental of a bare sine.
        pitch = (260, 110) if kind == "impact" else (300, 140)
        layers = [
            _sweep(pitch[0], pitch[1], min(d, 0.22), 0.24, harmonics=(1.0, 0.7, 0.5, 0.35), decay=14),
            _noise(min(d, 0.12), 0.70, 1000, 4000, decay=40, seed=kind),
        ]
    elif kind == "whoosh":
        layers = [
            _noise(min(d, 0.28), 0.22, 400, 4000, decay=4, seed=kind),
            _sweep(300, 700, min(d, 0.28), 0.08, harmonics=(1.0, 0.3), decay=4),
        ]
    elif kind in {"pop", "cute", "spark"}:
        layers = [
            _sweep(800, 1400, min(d, 0.09), 0.24, harmonics=(1.0, 0.35, 0.15), decay=25),
            _noise(min(d, 0.02), 0.08, 1500, 4500, decay=150, seed=kind),
        ]
    elif kind in {"countdown", "tap"}:
        layers = [
            _sweep(1100, 1100, min(d, 0.05), 0.26, harmonics=(1.0, 0.5), decay=60),
            _delay(_sweep(1100, 1100, min(d, 0.05), 0.24, harmonics=(1.0, 0.5), decay=60), 0.08),
            _noise(min(d, 0.03), 0.20, 1800, 5000, decay=120, seed=kind),
        ]
    elif kind == "beat":
        kick = _sweep(120, 45, min(d, 0.14), 0.45, harmonics=(1.0,), decay=16)
        layers = [kick, _delay(kick, 0.25)]
    else:
        layers = []
    samples = _mix(*layers) if layers else []
    return _wav(output, (samples + [0.0] * n)[:n])


def make_music_bed(duration: float, output: Path, *, level_dbfs: float = -38.0) -> Path:
    """Soft procedural pad under the dialog: a slow Am-F-C-G progression with a quiet bass pulse.

    The bed sits about 15 to 20 LU below the dialog before ducking, so speech always stays on top.
    """
    n = max(1, int(duration * RATE))
    chords = (
        (220.00, 261.63, 329.63),  # Am
        (174.61, 220.00, 261.63),  # F
        (261.63, 329.63, 392.00),  # C
        (196.00, 246.94, 293.66),  # G
    )
    bass = (110.00, 87.31, 130.81, 98.00)
    samples = [0.0] * n
    segment = n / len(chords)
    for c, notes in enumerate(chords):
        start = int(c * segment)
        end = min(n, int((c + 1) * segment) + int(0.5 * RATE))
        for i in range(start, end):
            t = i / RATE
            local = (i - start) / RATE
            fade = min(1.0, local / 0.6) * min(1.0, max(0.0, (end - i) / (0.6 * RATE)))
            pad = sum(math.sin(2 * math.pi * f * t) + 0.3 * math.sin(4 * math.pi * f * t) for f in notes)
            pulse = (0.5 + 0.5 * math.cos(2 * math.pi * 1.0 * t)) ** 2
            low = math.sin(2 * math.pi * bass[c] * t) * pulse * 0.6
            samples[i] += fade * (pad / (len(notes) * 1.3) * 0.5 + low * 0.4)
    rms = math.sqrt(sum(s * s for s in samples) / n) or 1.0
    gain = (10 ** (level_dbfs / 20.0)) / rms
    return _wav(output, [s * gain for s in samples])


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


def measure_loudness(path: Path) -> dict[str, float]:
    """EBU R128 integrated loudness, true peak and loudness range (ffmpeg loudnorm, first pass)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for loudness measurement")
    done = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"loudnorm=I={LOUDNESS_TARGET_LUFS}:TP={LOUDNESS_TRUE_PEAK_DBTP}:LRA={LOUDNESS_RANGE_LU}:print_format=json",
         "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )
    block = done.stderr[done.stderr.rfind("{"): done.stderr.rfind("}") + 1]
    data = json.loads(block)
    return {
        "integrated_lufs": float(data["input_i"]),
        "true_peak_dbtp": float(data["input_tp"]),
        "loudness_range_lu": float(data["input_lra"]),
        "threshold_lufs": float(data["input_thresh"]),
        "target_offset_lu": float(data["target_offset"]),
    }


def loudness_normalize(source: Path, output: Path) -> dict[str, float]:
    """Bring a WAV to -14 LUFS integrated with true peak <= -1.5 dBTP (measure, then apply linearly).

    Two passes are needed because a single-pass loudnorm on a 10-second clip often falls back
    to dynamic mode. Silent input is copied through unchanged.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for loudness normalization")
    measured = measure_loudness(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not math.isfinite(measured["integrated_lufs"]):
        shutil.copyfile(source, output)
        return {**measured, "applied": False}
    af = (
        f"loudnorm=I={LOUDNESS_TARGET_LUFS}:TP={LOUDNESS_TRUE_PEAK_DBTP}:LRA={LOUDNESS_RANGE_LU}"
        f":measured_I={measured['integrated_lufs']}:measured_TP={measured['true_peak_dbtp']}"
        f":measured_LRA={measured['loudness_range_lu']}:measured_thresh={measured['threshold_lufs']}"
        f":offset={measured['target_offset_lu']}:linear=true:print_format=summary"
    )
    subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-af", af, "-ar", str(RATE), "-ac", "1", str(output)],
        check=True, capture_output=True, text=True,
    )
    return {**measured, "applied": True}
