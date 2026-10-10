"""Local-first character voice performance for cartoon shorts.

The goal is not merely to synthesize words.  Each line gets a tiny acting
profile, natural pauses, a character voice, and timing telemetry that the
animation engine can consume on the same performance clock.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, replace
from pathlib import Path

from app.providers.runway_voice import RunwayVoiceProvider, render_runway_script

RATE = 44100


@dataclass(frozen=True)
class VoiceProfile:
    voice: str
    rate: int
    pitch: float
    gain: float
    pre_pause: float
    post_pause: float
    espeak_voice: str = "en-us"


BEAT_VOICE_PROFILES: dict[str, VoiceProfile] = {
    "setup": VoiceProfile("Eddy (English (US))", 172, 1.01, 0.97, 0.035, 0.070),
    "escalation": VoiceProfile("Eddy (English (US))", 192, 1.055, 1.00, 0.025, 0.095),
    "reaction": VoiceProfile("Eddy (English (US))", 164, 0.995, 0.99, 0.035, 0.085),
    "punchline": VoiceProfile("Eddy (English (US))", 148, 0.955, 0.98, 0.060, 0.180),
}


VOICES_PATH = Path(__file__).resolve().parents[2] / "config" / "character_voices.json"


def character_voice_config(character: str | None) -> dict:
    """Per-character voice identity from config/character_voices.json ({} if unknown)."""
    if not character:
        return {}
    try:
        data = json.loads(VOICES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entry = data.get("characters", {}).get(str(character).upper(), {})
    return entry if isinstance(entry, dict) else {}


def voice_profile(beat: str, character: str | None = None) -> VoiceProfile:
    """Beat delivery profile shaped by the character's voice identity."""
    base = BEAT_VOICE_PROFILES.get(beat, BEAT_VOICE_PROFILES["setup"])
    config = character_voice_config(character)
    if not config:
        return base
    return replace(
        base,
        voice=str(config.get("local_voice", base.voice)),
        rate=int(round(base.rate * float(config.get("rate_scale", 1.0)))),
        pitch=round(base.pitch * float(config.get("pitch_scale", 1.0)), 4),
        espeak_voice=str(config.get("espeak_voice", base.espeak_voice)),
    )


def _available_say_voices() -> set[str]:
    say = shutil.which("say")
    if not say:
        return set()
    try:
        result = subprocess.run([say, "-v", "?"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return set()
    names: set[str] = set()
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        names.add(line.rsplit("  ", 1)[0].strip() if "  " in line else line.split()[0])
    return names


def _resolve_voice(preferred: str) -> str:
    available = _available_say_voices()
    if preferred in available:
        return preferred
    for fallback in ("Samantha", "Tara", "Aman", "Daniel"):
        if fallback in available:
            return fallback
    return preferred


def performance_text(text: str, beat: str) -> str:
    """Add punctuation that guides natural TTS timing without changing words."""
    clean = " ".join(text.split())
    if not clean:
        return clean
    if beat == "setup":
        return clean.rstrip(".!?") + "."
    if beat == "escalation":
        if len(clean.split()) <= 4:
            return clean.rstrip(".!?") + "?!"
        # Create a brief anticipation pause before the stressed ending.
        words = clean.rstrip(".!?").split()
        if len(words) >= 3 and not any(p in clean for p in ("...", "…")):
            return " ".join(words[:-1]) + "... " + words[-1] + "?!"
        return clean.rstrip(".") + "?!"
    if beat == "reaction":
        clean = clean.replace("?", "")
        if "," in clean:
            clean = clean.replace(",", "...", 1)
        return clean.rstrip(".!?") + "?"
    # Deadpan: a slightly longer lead-in pause improves the final look-to-camera beat.
    if "," in clean:
        clean = clean.replace(",", "…", 1)
    return clean.rstrip(".!?") + "."


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / max(1, wav.getframerate())


def _rms_activity(path: Path, points: int = 96) -> tuple[list[float], float, float]:
    with wave.open(str(path), "rb") as wav:
        data = wav.readframes(wav.getnframes())
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
    if width != 2 or not data:
        return [0.0] * points, 0.0, 0.0
    import array
    samples = array.array("h")
    samples.frombytes(data)
    values = [abs(v) / 32768.0 for v in samples]
    if channels > 1:
        values = [sum(values[i:i + channels]) / channels for i in range(0, len(values), channels)]
    n = len(values)
    if n == 0:
        return [0.0] * points, 0.0, 0.0
    window = max(1, n // points)
    raw: list[float] = []
    for i in range(points):
        start = i * window
        end = min(n, (i + 1) * window)
        chunk = values[start:end] or [0.0]
        rms = math.sqrt(sum(v * v for v in chunk) / len(chunk))
        raw.append(rms)
    peak = max(raw) or 1.0
    norm = [min(1.0, v / peak) for v in raw]
    # Smooth the activity curve so animation follows speech energy without jitter.
    smooth: list[float] = []
    for i in range(points):
        a = max(0, i - 2)
        b = min(points, i + 3)
        smooth.append(sum(norm[a:b]) / max(1, b - a))
    threshold = max(0.018, peak * 0.10)
    onset_index = next((i for i, v in enumerate(raw) if v >= threshold), 0)
    peak_index = max(range(points), key=lambda i: raw[i])
    return smooth, onset_index / max(1, points - 1), peak_index / max(1, points - 1)


def _postprocess_voice(source: Path, path: Path, profile: VoiceProfile) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for local voice performance")
    pitch = profile.pitch
    pitch_filter = ""
    if abs(pitch - 1.0) > 0.005:
        pitch_filter = f"asetrate={RATE * pitch:.2f},aresample={RATE},atempo={1.0 / pitch:.6f},"
    af = (
        f"{pitch_filter}highpass=f=70,lowpass=f=12500,"
        "acompressor=threshold=-20dB:ratio=2.2:attack=6:release=95:makeup=1.2dB,"
        f"volume={profile.gain:.3f}"
    )
    subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-af", af, "-ar", str(RATE), "-ac", "1", str(path)],
        check=True, capture_output=True, text=True,
    )


def _render_say(text: str, path: Path, profile: VoiceProfile, rate_override: int | None = None) -> None:
    say = shutil.which("say")
    if not say:
        raise RuntimeError("macOS say is not available")
    aiff = path.with_suffix(".aiff")
    rate = int(rate_override or profile.rate)
    voice = _resolve_voice(profile.voice)
    subprocess.run([say, "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True)
    _postprocess_voice(aiff, path, profile)
    aiff.unlink(missing_ok=True)


def _render_espeak(text: str, path: Path, profile: VoiceProfile, rate_override: int | None = None) -> None:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise RuntimeError("Neither macOS say nor espeak-ng/espeak is available")
    rate = int(rate_override or profile.rate)
    # eSpeak pitch is 0-99 with 50 approximately neutral; map our relative
    # character pitch around that midpoint.
    pitch = max(20, min(80, round(50 + (profile.pitch - 1.0) * 240)))
    amplitude = max(20, min(100, round(profile.gain * 100)))
    # eSpeak silently writes nothing (exit 0) when the -w path is ~200+ chars,
    # which long idea slugs reach. Synthesize to a short temp path, then move.
    with tempfile.TemporaryDirectory(prefix="nova-tts-") as tmp:
        raw = Path(tmp) / "voice.wav"
        subprocess.run(
            [exe, "-v", profile.espeak_voice or "en-us", "-s", str(rate), "-p", str(pitch), "-a", str(amplitude), "-w", str(raw), text],
            check=True, capture_output=True, text=True,
        )
        if not raw.exists() or raw.stat().st_size <= 44:
            raise RuntimeError(f"espeak produced no audio for line: {text[:60]!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(raw), str(path))
    processed = path.with_suffix(".processed.wav")
    _postprocess_voice(path, processed, profile)
    processed.replace(path)


def _render_voice(text: str, path: Path, profile: VoiceProfile, rate_override: int | None = None,
                  beat: str | None = None, runway_voice: dict | None = None) -> None:
    backend = os.getenv("NOVA_VOICE_BACKEND", "local").strip().lower()
    if backend == "runway":
        RunwayVoiceProvider().synthesize(text, beat or "setup", path, voice=runway_voice)
        return
    if shutil.which("say"):
        _render_say(text, path, profile, rate_override)
    else:
        _render_espeak(text, path, profile, rate_override)


def _add_padding(path: Path, output: Path, pre: float, post: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    delay_ms = round(max(0.0, pre) * 1000)
    pad = max(0.0, post)
    af = f"adelay={delay_ms}|{delay_ms},apad=pad_dur={pad:.3f},atrim=0:999"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-af", af, "-ar", str(RATE), "-ac", "1", str(output)],
        check=True, capture_output=True, text=True,
    )


def prepare_voice_performance(
    work: Path,
    voice_lines: list[str],
    beats: list[str],
    target_durations: list[float] | None = None,
    character: str | None = None,
) -> list[dict]:
    """Generate once, measure once, then expose timing telemetry to animation/audio."""
    if len(voice_lines) != len(beats):
        raise ValueError("voice_lines and beats must match")
    work.mkdir(parents=True, exist_ok=True)
    target_durations = target_durations or [None] * len(beats)
    results: list[dict] = []
    for i, (line, beat, target) in enumerate(zip(voice_lines, beats, target_durations)):
        profile = voice_profile(beat, character)
        runway_voice = character_voice_config(character).get("runway_voice")
        text = performance_text(line, beat)
        raw = work / f"voice_{i:02d}_raw.wav"
        final = work / f"voice_{i:02d}.wav"
        _render_voice(text, raw, profile, beat=beat, runway_voice=runway_voice)
        natural_duration = _duration(raw)
        available = None
        rate_used = profile.rate
        if target:
            available = max(0.35, float(target) - profile.pre_pause - profile.post_pause - 0.035)
            if natural_duration > available * 1.05:
                adjusted = round(profile.rate * natural_duration / available)
                adjusted = max(135, min(235, adjusted))
                _render_voice(text, raw, profile, rate_override=adjusted, beat=beat, runway_voice=runway_voice)
                natural_duration = _duration(raw)
                rate_used = adjusted
        _add_padding(raw, final, profile.pre_pause, profile.post_pause)
        duration = _duration(final)
        activity, onset, peak = _rms_activity(final)
        performance_peak = peak
        if beat == "escalation":
            performance_peak = max(0.20, min(0.40, peak + 0.10))
        elif beat == "reaction":
            performance_peak = max(0.28, min(0.58, peak))
        elif beat == "punchline":
            performance_peak = max(0.48, min(0.78, peak))
        results.append({
            "shot": i,
            "beat": beat,
            "text": text,
            "voice": _resolve_voice(profile.voice),
            "rate": rate_used,
            "pitch": profile.pitch,
            "gain": profile.gain,
            "pre_pause": profile.pre_pause,
            "post_pause": profile.post_pause,
            "natural_duration": round(natural_duration, 4),
            "duration": round(duration, 4),
            "target_duration": target,
            "onset_fraction": round(onset, 4),
            "peak_fraction": round(peak, 4),
            "performance_peak_fraction": round(performance_peak, 4),
            "activity": [round(v, 4) for v in activity],
            "path": str(final),
        })
        raw.unlink(missing_ok=True)
    (work / "voice_performance.json").write_text(
        __import__("json").dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results
