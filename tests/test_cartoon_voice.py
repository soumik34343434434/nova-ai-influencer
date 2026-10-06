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
