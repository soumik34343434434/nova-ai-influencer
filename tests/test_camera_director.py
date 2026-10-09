from app.agents.camera_director import CameraDirective, plan_camera


def test_camera_directive_keeps_actor_focused_and_bounded():
    for beat in ("setup", "escalation", "reaction", "punchline"):
        d = plan_camera(beat, 0.5, 0.8, 0.2)
        assert isinstance(d, CameraDirective)
        assert 0.0 <= d.focus_weight <= 1.2
        assert -0.20 <= d.zoom_delta <= 0.15
        assert -0.10 <= d.lateral_bias <= 0.10
        assert 0.02 <= d.lead_seconds <= 0.10
        assert 0.0 <= d.hold_strength <= 1.0


def test_escalation_camera_hits_hardest_push_at_impact():
    pre = plan_camera("escalation", 0.42, 0.2, 0.0)
    hit = plan_camera("escalation", 0.58, 1.0, 0.0)
    assert hit.zoom_delta < pre.zoom_delta
    assert hit.focus_weight > pre.focus_weight
    assert hit.hold_strength > pre.hold_strength


def test_punchline_prefers_hold_over_camera_competition():
    early = plan_camera("punchline", 0.30, 0.8, 0.1)
    late = plan_camera("punchline", 0.90, 0.0, -0.4)
    assert late.hold_strength >= 0.5
    assert abs(late.lateral_bias) < abs(early.lateral_bias)
    assert abs(late.zoom_delta) < abs(early.zoom_delta)
