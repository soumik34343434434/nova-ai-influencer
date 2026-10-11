import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from app.providers.cartoon_audio import (
    LOUDNESS_TRUE_PEAK_DBTP,
    loudness_normalize,
    make_music_bed,
    make_sfx,
    measure_loudness,
)

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")


def _read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768.0, wav.getframerate()


def _write(path: Path, samples: np.ndarray, rate: int = 44100) -> Path:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _band_fraction(samples: np.ndarray, rate: int, low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1.0 / rate)
    total = spectrum[1:].sum() or 1.0
    return float(spectrum[(freqs >= low) & (freqs <= high)].sum() / total)


@pytest.mark.parametrize("kind", ["impact", "reaction_hit", "tap", "punchline", "chaos", "pop", "whoosh"])
def test_sound_effects_are_layered_and_audible_in_the_midrange(tmp_path, kind):
    samples, rate = _read(make_sfx(kind, 0.3, tmp_path / f"{kind}.wav"))
    assert rate == 44100
    assert float(np.max(np.abs(samples))) > 0.1
    # Phone speakers reproduce 400 Hz to 4 kHz; the old pure tones put most energy below that or in one bin.
    assert _band_fraction(samples, rate, 400, 4000) >= 0.35
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    assert spectrum.max() / spectrum.sum() < 0.5  # not a single sine


def test_music_bed_is_soft_steady_and_unclipped(tmp_path):
    samples, rate = _read(make_music_bed(4.0, tmp_path / "bed.wav"))
    assert len(samples) == pytest.approx(4.0 * rate, abs=2)
    rms_db = 20 * np.log10(np.sqrt(np.mean(samples ** 2)))
    assert -39.5 < rms_db < -36.5
    assert float(np.max(np.abs(samples))) < 0.5


@needs_ffmpeg
def test_loudness_normalize_reaches_minus_14_lufs_under_the_true_peak_ceiling(tmp_path):
    t = np.arange(int(6 * 44100)) / 44100
    tone = 0.02 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.sin(2 * np.pi * 1200 * t)
    source = _write(tmp_path / "quiet.wav", tone * (1 + 0.5 * np.sin(2 * np.pi * 0.7 * t)))
    before = measure_loudness(source)["integrated_lufs"]
    out = tmp_path / "loud.wav"
    report = loudness_normalize(source, out)
    after = measure_loudness(out)
    assert report["applied"] is True
    assert before < -30
    assert -14.6 < after["integrated_lufs"] < -13.4
    assert after["true_peak_dbtp"] <= LOUDNESS_TRUE_PEAK_DBTP + 0.1


@needs_ffmpeg
def test_silence_passes_through_loudness_normalization_without_failing(tmp_path):
    silent = _write(tmp_path / "silent.wav", np.zeros(44100 * 2))
    out = tmp_path / "out.wav"
    report = loudness_normalize(silent, out)
    assert out.exists()
    assert report["applied"] is False


@needs_ffmpeg
def test_hosted_reel_voice_track_is_brought_to_the_loudness_target(tmp_path):
    from app.agents.edit_decision import build_edit_plan
    from app.providers.reel_editor import assemble_reel

    beats = [
        {"beat": "hook", "story_role": "hook", "voice": "PIP pressed one button."},
        {"beat": "escalation", "story_role": "escalation", "voice": "Then twelve sandwiches appear!"},
        {"beat": "reaction", "story_role": "reaction", "voice": "Wait, why?"},
        {"beat": "punchline", "story_role": "payoff", "voice": "Still one job."},
    ]
    durations = [1.8, 2.4, 2.5, 2.2]
    clips = []
    for i in range(4):
        clip = tmp_path / f"c{i}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=720x1280:rate=24:duration=5",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
        clips.append(clip)
    t = np.arange(int(9 * 44100)) / 44100
    quiet_voice = _write(tmp_path / "voice.wav", 0.01 * np.sin(2 * np.pi * 330 * t))
    reel = assemble_reel(clips, build_edit_plan(beats, durations), tmp_path / "reel.mp4", audio=quiet_voice)
    assert reel["passed"], reel["reasons"]
    assert -15.5 < measure_loudness(tmp_path / "reel.mp4")["integrated_lufs"] < -12.5


@needs_ffmpeg
def test_episode_mix_meets_loudness_and_music_bed_targets(tmp_path, monkeypatch):
    if not (shutil.which("espeak-ng") or shutil.which("espeak")):
        pytest.skip("espeak-ng is required to render dialog")
    from app.providers.cartoon_media import _audio_for_shots
    from app.providers.cartoon_voice import prepare_voice_performance

    monkeypatch.setenv("NOVA_VOICE_BACKEND", "espeak")
    lines = ["BUBU is trying to help. Then it happened.", "Then twelve sandwiches appear!",
             "Wait, why? Turns out the help was the problem.", "Still counts as helping. Apparently."]
    beats = ["setup", "escalation", "reaction", "punchline"]
    durations = [2.6, 2.6, 2.6, 2.6]
    perf = prepare_voice_performance(tmp_path / "voice", lines, beats, durations, character="BUBU")
    audio = _audio_for_shots(tmp_path / "audio", lines, durations, ["tap", "impact", "reaction_hit", "punchline"],
                             shutil.which("ffmpeg"), [0.3, 0.4, 0.4, 0.6], voice_performance=perf)
    final = measure_loudness(audio)
    assert -15.5 < final["integrated_lufs"] < -12.5
    assert final["true_peak_dbtp"] <= -1.0

    # Dialog alone (the voice clips laid end to end) against the bed alone, both measured.
    dialog = tmp_path / "dialog.wav"
    frames = []
    for row in perf:
        samples, rate = _read(Path(row["path"]))
        frames.append(samples)
    _write(dialog, np.concatenate(frames), 44100)
    bed = make_music_bed(sum(durations), tmp_path / "bed.wav")
    assert measure_loudness(dialog)["integrated_lufs"] - measure_loudness(bed)["integrated_lufs"] >= 15.0
