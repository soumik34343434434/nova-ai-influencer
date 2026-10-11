"""Local-first character voice performance for cartoon shorts.

The goal is not merely to synthesize words.  Each line gets a tiny acting
profile, natural pauses, a character voice, and timing telemetry that the
animation engine can consume on the same performance clock.

Voice backends (``NOVA_VOICE_BACKEND``): ``auto`` (default) uses Kokoro-82M when
its package and models are ready, else macOS ``say``, else eSpeak NG. ``kokoro``,
``espeak``, ``say`` and ``runway`` force one backend. ``runway`` is paid and runs
only through the same execution gate and credit ledger as Runway video.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from app.providers.runway_voice import RunwayVoiceProvider, render_runway_script

RATE = 44100
VOICE_BACKENDS = ("auto", "kokoro", "espeak", "say", "runway")
RUNWAY_VOICE_CREDITS_ENV = "NOVA_RUNWAY_VOICE_CREDITS_PER_LINE"
RUNWAY_VOICE_CREDITS_DEFAULT = 5.0


@dataclass(frozen=True)
class VoiceProfile:
    voice: str
    rate: int
    pitch: float
    gain: float
    pre_pause: float
    post_pause: float
    espeak_voice: str = "en-us"
    kokoro_voice: str = "af_heart"


BEAT_VOICE_PROFILES: dict[str, VoiceProfile] = {
    "setup": VoiceProfile("Eddy (English (US))", 172, 1.01, 0.97, 0.035, 0.070),
    "escalation": VoiceProfile("Eddy (English (US))", 192, 1.055, 1.00, 0.025, 0.095),
    "reaction": VoiceProfile("Eddy (English (US))", 164, 0.995, 0.99, 0.035, 0.085),
    "punchline": VoiceProfile("Eddy (English (US))", 148, 0.955, 0.98, 0.060, 0.180),
}


VOICES_PATH = Path(__file__).resolve().parents[2] / "config" / "character_voices.json"
# Words that take little stress; a pre-stress pause never lands before them.
STOPWORDS = frozenset(
    "a an and as at by for from in into it its of off on once onto or the them then to up with now".split()
)
_kokoro_failure: str | None = None


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
        kokoro_voice=str(config.get("kokoro_voice", base.kokoro_voice)),
    )


def spoken_text(text: str) -> str:
    """Spell character names as words for the TTS engine (PIP is said as 'Pip'); captions keep the brand spelling."""
    out = str(text)
    try:
        data = json.loads(VOICES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    for name, entry in data.get("characters", {}).items():
        spoken = str((entry or {}).get("spoken_name", "")).strip()
        if spoken and name.isupper():
            out = re.sub(rf"\b{re.escape(name)}\b", spoken, out)
    return out


def requested_backend() -> str:
    raw = os.getenv("NOVA_VOICE_BACKEND", "auto").strip().lower() or "auto"
    if raw == "local":  # legacy name for the default local voices
        raw = "auto"
    if raw not in VOICE_BACKENDS:
        raise ValueError(f"NOVA_VOICE_BACKEND must be one of {', '.join(VOICE_BACKENDS)}; got {raw!r}")
    return raw


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
    """Add punctuation that guides natural TTS timing without changing words or their meaning."""
    clean = " ".join(text.split())
    if not clean:
        return clean
    body = clean.rstrip(".!?")
    if beat == "setup":
        # Hooks keep their exclamation or question; everything else is a plain statement.
        return body + (clean[-1] if clean[-1] in "!?" else ".")
    if beat == "escalation":
        ending = "?!" if clean.rstrip(".!").endswith("?") else "!"
        words = body.split()
        if len(words) <= 4 or any(p in clean for p in ("...", "…")):
            return body + ending
        # Anticipation pause before the last content word, never before a function word.
        for index in range(len(words) - 1, 0, -1):
            if words[index].strip("'\",").lower() not in STOPWORDS:
                return " ".join(words[:index]) + "... " + " ".join(words[index:]) + ending
        return body + ending
    if beat == "reaction":
        # Keep the author's own questions and reveals; pause at the first comma of the first sentence only.
        ending = clean[-1] if clean[-1] in "?!" else "."
        match = re.search(r"[.!?]", body)
        head_end = match.start() if match else len(body)
        head = body[:head_end].replace(",", "...", 1)
        return head + body[head_end:] + ending
    # Deadpan: a slightly longer lead-in pause improves the final look-to-camera beat.
    if "," in body:
        body = body.replace(",", "…", 1)
    return body + "."


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / max(1, wav.getframerate())


def _sample_rate(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        return wav.getframerate()


def _rms_activity(path: Path, points: int = 96) -> tuple[list[float], float, float]:
    with wave.open(str(path), "rb") as wav:
        data = wav.readframes(wav.getnframes())
        channels = wav.getnchannels()
        width = wav.getsampwidth()
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


def _postprocess_voice(source: Path, path: Path, profile: VoiceProfile, pitch: float | None = None) -> None:
    """Resample to RATE and apply a pitch shift of ``pitch`` (default: the profile's) at constant duration.

    The input is resampled to RATE *before* the speed/pitch relabel, so the shift
    is correct for any source rate (eSpeak 22.05 kHz, Kokoro 24 kHz, AIFF from say).
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for local voice performance")
    shift = profile.pitch if pitch is None else pitch
    chain = [f"aresample={RATE}"]
    if abs(shift - 1.0) > 0.005:
        chain += [f"asetrate={RATE * shift:.2f}", f"aresample={RATE}", f"atempo={1.0 / shift:.6f}"]
    chain += [
        "highpass=f=70",
        "lowpass=f=12500",
        "acompressor=threshold=-20dB:ratio=2.2:attack=6:release=95:makeup=1.2dB",
        f"volume={profile.gain:.3f}",
    ]
    subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-af", ",".join(chain), "-ar", str(RATE), "-ac", "1", str(path)],
        check=True, capture_output=True, text=True,
    )


def _render_say(text: str, path: Path, profile: VoiceProfile, rate_override: int | None = None) -> dict[str, Any]:
    say = shutil.which("say")
    if not say:
        raise RuntimeError("macOS say is not available")
    aiff = path.with_suffix(".aiff")
    rate = int(rate_override or profile.rate)
    voice = _resolve_voice(profile.voice)
    subprocess.run([say, "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True)
    _postprocess_voice(aiff, path, profile)
    aiff.unlink(missing_ok=True)
    return {"backend": "say", "voice": voice}


def _render_espeak(text: str, path: Path, profile: VoiceProfile, rate_override: int | None = None) -> dict[str, Any]:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise RuntimeError("Neither macOS say nor espeak-ng/espeak is available")
    rate = int(rate_override or profile.rate)
    # eSpeak pitch is 0-99 with 50 approximately neutral. The character pitch is applied
    # here, by eSpeak itself (measured: BUBU -p 74 raises median F0 about 30%), and not
    # again by a resample shift, which would double it.
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
    _postprocess_voice(path, processed, profile, pitch=1.0)
    processed.replace(path)
    return {"backend": "espeak", "voice": profile.espeak_voice or "en-us"}


def _render_kokoro(text: str, path: Path, profile: VoiceProfile, rate_override: int | None = None) -> dict[str, Any]:
    from app.providers import kokoro_voice

    rate = int(rate_override or profile.rate)
    # Speed follows the beat's words-per-minute only; the character rate_scale is already in it.
    speed = round(rate / kokoro_voice.NEUTRAL_RATE, 3)
    with tempfile.TemporaryDirectory(prefix="nova-kokoro-") as tmp:
        raw = Path(tmp) / "kokoro.wav"
        kokoro_voice.synthesize(text, raw, voice=profile.kokoro_voice, speed=speed)
        # Kokoro's own prosody is the performance; the character pitch shift is left at 1.0 unless auditioned.
        _postprocess_voice(raw, path, profile, pitch=1.0)
    return {"backend": "kokoro", "voice": profile.kokoro_voice}


def _runway_voice_credits() -> float:
    try:
        return max(0.0, float(os.getenv(RUNWAY_VOICE_CREDITS_ENV, "") or RUNWAY_VOICE_CREDITS_DEFAULT))
    except ValueError:
        return RUNWAY_VOICE_CREDITS_DEFAULT


def _render_runway(text: str, path: Path, profile: VoiceProfile, beat: str, runway_voice: dict | None,
                   run_id: str, shot: int) -> dict[str, Any]:
    from app.providers.runway_video import RunwayExecutionBlocked, execution_gate, record_spend

    # Same opt-in gate and ledger as Runway video: NOVA_RUNWAY_EXECUTE, key, episode and daily caps.
    gate = execution_gate({"shots": []}, [], run_id=run_id)
    estimate = _runway_voice_credits()
    if gate["episode_spent"] + estimate > gate["budget"]:
        raise RunwayExecutionBlocked(
            f"Runway voice needs about {estimate:.0f} credits; this episode already spent "
            f"{gate['episode_spent']:.0f} of {gate['budget']:.0f}"
        )
    if gate["today_spent"] + estimate > gate["daily_cap"]:
        raise RunwayExecutionBlocked(
            f"Runway voice needs about {estimate:.0f} credits; {gate['today_spent']:.0f} already spent today "
            f"of {gate['daily_cap']:.0f}"
        )

    def _record(task_id: str) -> None:
        record_spend(run_id=run_id, shot=shot, task_id=task_id, credits=estimate, kind="voice", model="runway-speech")

    source = path.with_name(path.stem + "_runway_src")
    RunwayVoiceProvider().synthesize(text, beat or "setup", source, voice=runway_voice, on_submit=_record)
    _postprocess_voice(source, path, profile, pitch=1.0)
    source.unlink(missing_ok=True)
    preset = (runway_voice or {}).get("presetId", "runway")
    return {"backend": "runway", "voice": str(preset)}


def _auto_chain() -> list[str]:
    from app.providers import kokoro_voice

    chain = ["kokoro"] if _kokoro_failure is None and kokoro_voice.ready() else []
    if shutil.which("say"):
        chain.append("say")
    chain.append("espeak")
    return chain


def _render_with(backend: str, text: str, path: Path, profile: VoiceProfile, rate_override: int | None,
                 beat: str | None, runway_voice: dict | None, run_id: str, shot: int) -> dict[str, Any]:
    if backend == "kokoro":
        return _render_kokoro(text, path, profile, rate_override)
    if backend == "runway":
        return _render_runway(text, path, profile, beat or "setup", runway_voice, run_id, shot)
    if backend == "say":
        return _render_say(text, path, profile, rate_override)
    return _render_espeak(text, path, profile, rate_override)


def _render_voice(text: str, path: Path, profile: VoiceProfile, rate_override: int | None = None,
                  beat: str | None = None, runway_voice: dict | None = None, *,
                  run_id: str = "", shot: int = 0) -> dict[str, Any]:
    """Render one line to ``path`` and return ``{"backend", "voice"}`` for what actually ran.

    In ``auto`` mode a failing backend falls through to the next one; Kokoro is
    disabled for the rest of the process after its first failure. Explicit backends
    never fall back, so a forced choice cannot silently change.
    """
    global _kokoro_failure
    choice = requested_backend()
    if choice != "auto":
        return _render_with(choice, text, path, profile, rate_override, beat, runway_voice, run_id, shot)
    chain = _auto_chain()
    for position, backend in enumerate(chain):
        try:
            return _render_with(backend, text, path, profile, rate_override, beat, runway_voice, run_id, shot)
        except Exception as exc:
            if backend == "kokoro":
                _kokoro_failure = str(exc)[:300]
            if position == len(chain) - 1:
                raise
    raise RuntimeError("no voice backend available")  # pragma: no cover - chain always ends in espeak


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
    run_id: str = "",
) -> list[dict]:
    """Generate once, measure once, then expose timing telemetry to animation/audio.

    ``run_id`` attributes paid Runway voice spend to the episode in the credit ledger.
    """
    if len(voice_lines) != len(beats):
        raise ValueError("voice_lines and beats must match")
    work.mkdir(parents=True, exist_ok=True)
    target_durations = target_durations or [None] * len(beats)
    results: list[dict] = []
    for i, (line, beat, target) in enumerate(zip(voice_lines, beats, target_durations)):
        profile = voice_profile(beat, character)
        runway_voice = character_voice_config(character).get("runway_voice")
        spoken = spoken_text(line)
        text = performance_text(spoken, beat)
        raw = work / f"voice_{i:02d}_raw.wav"
        final = work / f"voice_{i:02d}.wav"
        meta = _render_voice(text, raw, profile, beat=beat, runway_voice=runway_voice, run_id=run_id, shot=i)
        natural_duration = _duration(raw)
        available = None
        rate_used = profile.rate
        if target:
            available = max(0.35, float(target) - profile.pre_pause - profile.post_pause - 0.035)
            # Paid Runway audio cannot change speed on a second request, so only free backends retry.
            if natural_duration > available * 1.05 and meta["backend"] != "runway":
                adjusted = round(profile.rate * natural_duration / available)
                adjusted = max(135, min(210, adjusted))
                meta = _render_voice(text, raw, profile, rate_override=adjusted, beat=beat,
                                     runway_voice=runway_voice, run_id=run_id, shot=i)
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
        backend = meta["backend"]
        words = len(spoken.split())
        spoken_seconds = max(0.1, natural_duration)
        results.append({
            "shot": i,
            "beat": beat,
            "text": text,
            "voice": meta["voice"],
            "voice_backend": backend,
            "voice_id": f"{backend}:{meta['voice']}",
            "sample_rate": _sample_rate(final),
            "rate": rate_used,
            "pitch": profile.pitch if backend in ("say", "espeak") else 1.0,
            "gain": profile.gain,
            "pre_pause": profile.pre_pause,
            "post_pause": profile.post_pause,
            "natural_duration": round(natural_duration, 4),
            "duration": round(duration, 4),
            "target_duration": target,
            "words": words,
            "speech_wpm": round(words / spoken_seconds * 60.0, 1),
            "onset_fraction": round(onset, 4),
            "peak_fraction": round(peak, 4),
            "performance_peak_fraction": round(performance_peak, 4),
            "activity": [round(v, 4) for v in activity],
            "path": str(final),
        })
        raw.unlink(missing_ok=True)
    (work / "voice_performance.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results
