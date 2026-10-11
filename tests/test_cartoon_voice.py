import json
import re
import shutil
import subprocess
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.providers.cartoon_voice import BEAT_VOICE_PROFILES, performance_text, spoken_text
from app.providers.cartoon_animation import _performance_durations


@pytest.fixture(autouse=True)
def _espeak_backend(monkeypatch):
    # These tests pin the free eSpeak path; Kokoro has its own tests and must not depend on the operator's cache.
    monkeypatch.setenv("NOVA_VOICE_BACKEND", "espeak")


def _read_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        raw = wav.readframes(wav.getnframes())
        return np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0, wav.getframerate()


def _median_f0(samples: np.ndarray, rate: int) -> float:
    """Median pitch of the voiced 40 ms frames, by autocorrelation (the spectral peak is a formant, not F0)."""
    win, hop = int(0.04 * rate), int(0.02 * rate)
    lo, hi = int(rate / 400), int(rate / 70)
    values = []
    for start in range(0, len(samples) - win, hop):
        seg = samples[start:start + win]
        if np.sqrt(np.mean(seg ** 2)) < 0.02:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[win - 1:]
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[0] > 0 and ac[k] / ac[0] > 0.5:
            values.append(rate / k)
    return float(np.median(values))


def test_voice_performance_profiles_are_distinct():
    assert BEAT_VOICE_PROFILES["setup"].rate < BEAT_VOICE_PROFILES["escalation"].rate
    assert BEAT_VOICE_PROFILES["punchline"].rate < BEAT_VOICE_PROFILES["reaction"].rate


def test_performance_text_adds_comedic_prosody():
    assert performance_text("TWELVE?!", "escalation").endswith("?!")
    assert "..." in performance_text("The kitchen made twelve sandwiches.", "escalation")
    assert "…" in performance_text("Apparently, the kitchen misunderstood one.", "punchline")


def test_reaction_keeps_the_author_question_and_pauses_at_the_first_comma():
    assert performance_text("Wait, why? Turns out the help was the problem.", "reaction") == (
        "Wait... why? Turns out the help was the problem."
    )
    # A declarative reveal keeps its own full stop instead of gaining a rising question.
    assert performance_text("Turns out it was the slow version.", "reaction").endswith(".")


def test_escalation_pause_never_lands_before_a_function_word():
    text = performance_text("Then twelve sandwiches appear at once", "escalation")
    assert "... sandwiches" in text or "... appear" in text
    assert not re.search(r"\.\.\. (at|of|up|it|them|once|now)\b", text)


def test_setup_keeps_hook_exclamations():
    assert performance_text("Then it happened!", "setup") == "Then it happened!"
    assert performance_text("Then it happened", "setup") == "Then it happened."


def test_every_format_recipe_line_keeps_its_meaning():
    from app.agents.idea_generator_v2 import FORMAT_RECIPES

    for name, recipe in FORMAT_RECIPES.items():
        for line in recipe.get("twists", ()):
            spoken = line.replace("{C}", "PIP").replace("{N}", "Twelve").replace("{P}", "kitchen")
            out = performance_text(spoken, "reaction")
            assert out.count("?") == spoken.count("?"), (name, spoken, out)
            assert "...?" not in out, (name, out)
        for line in recipe.get("escalations", ()):
            spoken = line.replace("{C}", "PIP").replace("{N}", "Twelve").replace("{P}", "kitchen")
            out = performance_text(spoken, "escalation")
            assert not re.search(r"\.\.\. (at|of|up|it|them|once|now)\b", out), (name, out)


def test_character_names_are_spoken_as_words_but_captions_keep_the_brand():
    assert spoken_text("PIP is here. MOMO and BUBU too.") == "Pip is here. Momo and Bubu too."
    assert spoken_text("pipe and Pip") == "pipe and Pip"


def test_shot_duration_is_allowed_to_follow_voice_performance():
    perf = [
        {"duration": 1.0},
        {"duration": 0.55},
        {"duration": 1.0},
        {"duration": 1.35},
    ]
    out = _performance_durations(perf, [2.3, 2.3, 2.3, 2.8], ["setup", "escalation", "reaction", "punchline"])
    # Tails after each line are short now (dead air was 38% of the runtime), so
    # shots follow the speech closely; the punchline keeps its deadpan hold.
    assert out == [1.55, 1.2, 1.5, 2.21]
    assert sum(out) < 8.0


def test_voice_renderer_has_cross_platform_backend():
    from app.providers import cartoon_voice
    assert callable(cartoon_voice._render_voice)
    assert callable(cartoon_voice._render_espeak)


def test_short_exclamation_gets_a_visible_animation_event():
    # Short RMS peaks can occur almost immediately; visual performance gets a
    # deliberately readable center point instead of snapping on frame one.
    assert 0.20 <= max(0.20, min(0.40, 0.116 + 0.10)) <= 0.40


def test_espeak_writes_audio_even_for_very_long_output_paths(tmp_path):
    """espeak-ng silently writes nothing for -w paths of ~200+ characters."""
    from app.providers.cartoon_voice import _render_espeak

    if not (shutil.which("espeak-ng") or shutil.which("espeak")) or not shutil.which("ffmpeg"):
        pytest.skip("espeak-ng and ffmpeg are required")
    deep = tmp_path / ("idea-" + "x" * 90 + "_factory_frames") / ("y" * 60)
    target = deep / "voice_00_raw.wav"
    assert len(str(target)) > 220
    _render_espeak("One tiny test line.", target, BEAT_VOICE_PROFILES["setup"])
    assert target.exists() and target.stat().st_size > 1000


def test_postprocess_pitch_shift_keeps_duration_for_a_22khz_source(tmp_path):
    """Regression: eSpeak writes 22.05 kHz; relabelling it at 44.1 kHz doubled speed and pitch."""
    from app.providers.cartoon_voice import _postprocess_voice

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required")
    rate = 22050
    t = np.arange(rate) / rate
    source = tmp_path / "tone_22k.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((np.sin(2 * np.pi * 200 * t) * 0.5 * 32767).astype("<i2").tobytes())
    out = tmp_path / "shifted.wav"
    profile = replace(BEAT_VOICE_PROFILES["setup"], gain=1.0)
    _postprocess_voice(source, out, replace(profile, pitch=1.1))
    samples, sr = _read_wave(out)
    assert sr == 44100
    assert abs(len(samples) / sr - 1.0) < 0.03
    assert abs(_median_f0(samples, sr) - 220.0) < 6.0


def test_espeak_line_keeps_its_natural_duration(tmp_path):
    """Regression: the shipped Linux path played eSpeak lines about 2x fast."""
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe or not shutil.which("ffmpeg"):
        pytest.skip("espeak-ng and ffmpeg are required")
    from app.providers.cartoon_voice import _render_espeak

    text = "BUBU is trying to help. Then it happened."
    raw = tmp_path / "raw.wav"
    subprocess.run([exe, "-v", "en-us+f4", "-s", "182", "-w", str(raw), text], check=True, capture_output=True)
    profile = replace(BEAT_VOICE_PROFILES["setup"], espeak_voice="en-us+f4")
    out = tmp_path / "processed.wav"
    _render_espeak(text, out, replace(profile, rate=182, pitch=1.0))
    raw_seconds = _read_wave(raw)[0].size / 22050
    processed_seconds = _read_wave(out)[0].size / 44100
    assert abs(processed_seconds - raw_seconds) / raw_seconds < 0.05


def test_espeak_pitch_is_set_by_the_character_not_shifted_twice(tmp_path):
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe or not shutil.which("ffmpeg"):
        pytest.skip("espeak-ng and ffmpeg are required")
    from app.providers.cartoon_voice import _render_espeak

    text = "Then it happened again and again."
    low, high = tmp_path / "low.wav", tmp_path / "high.wav"
    base = BEAT_VOICE_PROFILES["setup"]
    _render_espeak(text, low, replace(base, espeak_voice="en-us+f4", pitch=0.94))
    _render_espeak(text, high, replace(base, espeak_voice="en-us+f4", pitch=1.10))
    low_hz = _median_f0(_read_wave(low)[0], 44100)
    high_hz = _median_f0(_read_wave(high)[0], 44100)
    assert high_hz > low_hz * 1.1


def test_each_character_has_a_distinct_voice_identity():
    from app.providers.cartoon_voice import character_voice_config, voice_profile

    profiles = {c: voice_profile("setup", c) for c in ("PIP", "MOMO", "BUBU")}
    assert len({(p.pitch, p.rate, p.espeak_voice) for p in profiles.values()}) == 3
    assert profiles["MOMO"].rate < profiles["BUBU"].rate  # the sleepy cat speaks slower
    assert voice_profile("setup", None) == BEAT_VOICE_PROFILES["setup"]
    assert character_voice_config("PIP")["runway_voice"] == {"type": "runway-preset", "presetId": "Brodie"}
    assert {profiles[c].kokoro_voice for c in profiles} == {"am_puck", "bm_george", "af_bella"}


def test_voice_telemetry_records_the_backend_that_actually_ran(tmp_path):
    if not (shutil.which("espeak-ng") or shutil.which("espeak")) or not shutil.which("ffmpeg"):
        pytest.skip("espeak-ng and ffmpeg are required")
    from app.providers.cartoon_voice import prepare_voice_performance

    rows = prepare_voice_performance(tmp_path, ["BUBU is trying to help. Then it happened."], ["setup"], character="BUBU")
    row = rows[0]
    assert row["voice_backend"] == "espeak"
    assert row["voice_id"].startswith("espeak:en-us+f4")
    assert row["sample_rate"] == 44100
    # Speech rate must sit in the contract band; the 2x bug measured about 330 wpm.
    assert 110 <= row["speech_wpm"] <= 210, row["speech_wpm"]
    saved = json.loads((tmp_path / "voice_performance.json").read_text())
    assert saved[0]["voice_backend"] == "espeak"


def test_backend_selection_is_validated(monkeypatch):
    from app.providers.cartoon_voice import requested_backend

    monkeypatch.setenv("NOVA_VOICE_BACKEND", "local")
    assert requested_backend() == "auto"
    monkeypatch.setenv("NOVA_VOICE_BACKEND", "bogus")
    with pytest.raises(ValueError, match="NOVA_VOICE_BACKEND"):
        requested_backend()


def test_auto_falls_back_to_espeak_when_kokoro_fails(monkeypatch, tmp_path):
    if not (shutil.which("espeak-ng") or shutil.which("espeak")) or not shutil.which("ffmpeg"):
        pytest.skip("espeak-ng and ffmpeg are required")
    import app.providers.cartoon_voice as voice
    from app.providers import kokoro_voice

    def broken(*args, **kwargs):
        raise kokoro_voice.KokoroUnavailable("worker crashed")

    monkeypatch.setenv("NOVA_VOICE_BACKEND", "auto")
    monkeypatch.setattr(voice, "_kokoro_failure", None)
    monkeypatch.setattr(kokoro_voice, "ready", lambda: True)
    monkeypatch.setattr(kokoro_voice, "synthesize", broken)
    rows = voice.prepare_voice_performance(tmp_path, ["Hello there friend."], ["setup"], character="PIP")
    assert rows[0]["voice_backend"] == "espeak"
    assert "worker crashed" in voice._kokoro_failure
    assert "kokoro" not in voice._auto_chain()  # a broken Kokoro is not retried for every line


def test_runway_voice_is_blocked_without_the_execution_gate(monkeypatch, tmp_path):
    import app.providers.cartoon_voice as voice
    from app.providers import runway_video

    seen = {}

    class FakeRunway:
        def synthesize(self, text, beat, path, voice=None, on_submit=None):
            seen["called"] = True

    monkeypatch.setenv("NOVA_VOICE_BACKEND", "runway")
    monkeypatch.delenv("NOVA_RUNWAY_EXECUTE", raising=False)
    monkeypatch.setattr(voice, "RunwayVoiceProvider", FakeRunway)
    monkeypatch.setattr(runway_video, "LEDGER_FILE", tmp_path / "ledger.json")
    with pytest.raises(runway_video.RunwayExecutionBlocked, match="NOVA_RUNWAY_EXECUTE"):
        voice.prepare_voice_performance(tmp_path, ["Hello there."], ["setup"], character="MOMO")
    assert "called" not in seen


def test_runway_voice_receives_the_character_voice_and_records_spend(monkeypatch, tmp_path):
    import app.providers.cartoon_voice as voice
    from app.providers import runway_video

    seen = {}

    class FakeRunway:
        def synthesize(self, text, beat, path, voice=None, on_submit=None):
            seen["voice"] = voice
            on_submit("task-123")
            raise RuntimeError("stop after capturing the request")

    monkeypatch.setenv("NOVA_VOICE_BACKEND", "runway")
    monkeypatch.setenv("NOVA_RUNWAY_EXECUTE", "true")
    monkeypatch.setenv("RUNWAYML_API_SECRET", "test-secret")
    monkeypatch.setenv("NOVA_RUNWAY_MAX_CREDITS", "50")
    monkeypatch.setenv("NOVA_RUNWAY_VOICE_CREDITS_PER_LINE", "5")
    monkeypatch.setattr(voice, "RunwayVoiceProvider", FakeRunway)
    monkeypatch.setattr(runway_video, "LEDGER_FILE", tmp_path / "ledger.json")
    with pytest.raises(RuntimeError, match="stop after"):
        voice.prepare_voice_performance(tmp_path, ["Hello there."], ["setup"], character="MOMO", run_id="run-1")
    assert seen["voice"] == {"type": "runway-preset", "presetId": "Brodie"}
    rows = json.loads((tmp_path / "ledger.json").read_text())
    assert rows == [{**rows[0], "run_id": "run-1", "task_id": "task-123", "credits": 5.0, "kind": "voice"}]


def test_runway_voice_respects_the_episode_cap(monkeypatch, tmp_path):
    import app.providers.cartoon_voice as voice
    from app.providers import runway_video

    monkeypatch.setenv("NOVA_VOICE_BACKEND", "runway")
    monkeypatch.setenv("NOVA_RUNWAY_EXECUTE", "true")
    monkeypatch.setenv("RUNWAYML_API_SECRET", "test-secret")
    monkeypatch.setenv("NOVA_RUNWAY_MAX_CREDITS", "3")
    monkeypatch.setattr(runway_video, "LEDGER_FILE", tmp_path / "ledger.json")
    with pytest.raises(runway_video.RunwayExecutionBlocked, match="voice needs"):
        voice.prepare_voice_performance(tmp_path, ["Hello there."], ["setup"], run_id="run-2")
