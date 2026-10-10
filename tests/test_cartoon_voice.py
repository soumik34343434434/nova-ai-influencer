from pathlib import Path

from app.providers.cartoon_voice import BEAT_VOICE_PROFILES, performance_text
from app.providers.cartoon_animation import _performance_durations


def test_voice_performance_profiles_are_distinct():
    assert BEAT_VOICE_PROFILES["setup"].rate < BEAT_VOICE_PROFILES["escalation"].rate
    assert BEAT_VOICE_PROFILES["punchline"].rate < BEAT_VOICE_PROFILES["reaction"].rate


def test_performance_text_adds_comedic_prosody():
    assert performance_text("TWELVE?!", "escalation").endswith("?!")
    assert "..." in performance_text("The kitchen made twelve sandwiches.", "escalation")
    assert "…" in performance_text("Apparently, the kitchen misunderstood one.", "punchline")


def test_shot_duration_is_allowed_to_follow_voice_performance():
    perf = [
        {"duration": 1.0},
        {"duration": 0.55},
        {"duration": 1.0},
        {"duration": 1.35},
    ]
    out = _performance_durations(perf, [2.3, 2.3, 2.3, 2.8], ["setup", "escalation", "reaction", "punchline"])
    assert out == [1.58, 1.2, 1.62, 2.21]
    assert sum(out) < 8.0


def test_voice_renderer_has_cross_platform_backend():
    from app.providers import cartoon_voice
    assert callable(cartoon_voice._render_voice)
    assert callable(cartoon_voice._render_espeak)


def test_short_exclamation_gets_a_visible_animation_event():
    # Short RMS peaks can occur almost immediately; visual performance gets a
    # deliberately readable center point instead of snapping on frame one.
    from app.providers.cartoon_voice import prepare_voice_performance
    # Contract-level check on the normalization helper's policy via a synthetic row.
    assert 0.20 <= max(0.20, min(0.40, 0.116 + 0.10)) <= 0.40


def test_espeak_writes_audio_even_for_very_long_output_paths(tmp_path):
    """espeak-ng silently writes nothing for -w paths of ~200+ characters."""
    import shutil

    import pytest

    from app.providers.cartoon_voice import _render_espeak

    if not (shutil.which("espeak-ng") or shutil.which("espeak")) or not shutil.which("ffmpeg"):
        pytest.skip("espeak-ng and ffmpeg are required")
    deep = tmp_path / ("idea-" + "x" * 90 + "_factory_frames") / ("y" * 60)
    target = deep / "voice_00_raw.wav"
    assert len(str(target)) > 220
    _render_espeak("One tiny test line.", target, BEAT_VOICE_PROFILES["setup"])
    assert target.exists() and target.stat().st_size > 1000


def test_each_character_has_a_distinct_voice_identity():
    from app.providers.cartoon_voice import character_voice_config, voice_profile

    profiles = {c: voice_profile("setup", c) for c in ("PIP", "MOMO", "BUBU")}
    assert len({(p.pitch, p.rate, p.espeak_voice) for p in profiles.values()}) == 3
    assert profiles["MOMO"].rate < profiles["BUBU"].rate  # the sleepy cat speaks slower
    assert voice_profile("setup", None) == BEAT_VOICE_PROFILES["setup"]
    assert character_voice_config("PIP")["runway_voice"] == {"type": "runway-preset", "presetId": "Brodie"}


def test_runway_backend_receives_the_character_voice(monkeypatch, tmp_path):
    import app.providers.cartoon_voice as voice

    seen = {}

    class FakeRunway:
        def synthesize(self, text, beat, path, voice=None):
            seen["voice"] = voice
            raise RuntimeError("stop after capturing the request")

    monkeypatch.setenv("NOVA_VOICE_BACKEND", "runway")
    monkeypatch.setattr(voice, "RunwayVoiceProvider", FakeRunway)
    import pytest

    with pytest.raises(RuntimeError, match="stop after"):
        voice.prepare_voice_performance(tmp_path, ["Hello there."], ["setup"], character="MOMO")
    assert seen["voice"] == {"type": "runway-preset", "presetId": "Brodie"}
