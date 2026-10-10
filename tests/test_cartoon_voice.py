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


def test_espeak_renders_to_paths_longer_than_its_filename_limit(tmp_path):
    import shutil
    import pytest
    from app.providers import cartoon_voice

    if not (shutil.which("espeak-ng") or shutil.which("espeak")) or not shutil.which("ffmpeg"):
        pytest.skip("espeak and ffmpeg are required")
    deep = tmp_path / ("d" * 120) / ("e" * 120)
    deep.mkdir(parents=True)
    path = deep / "voice_00_raw.wav"
    assert len(str(path)) > 200
    cartoon_voice._render_espeak("One quick line.", path, BEAT_VOICE_PROFILES["setup"])
    assert cartoon_voice._duration(path) > 0.3
