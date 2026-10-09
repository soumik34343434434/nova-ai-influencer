from pathlib import Path

from PIL import Image

from app.providers.cartoon_performance import PLANS, head_motion, performance_plan
from app.providers.cartoon_rig import PIP_ACTION_REGIONS, get_rig, rig_summary


def test_all_pip_action_assets_have_rig_regions():
    root = Path("assets/cartoon/actions")
    assets = sorted(root.glob("pip_*.png"))
    assert assets
    assert {p.name for p in assets} <= set(PIP_ACTION_REGIONS)


def test_pip_rig_extracts_head_layer():
    path = Path("assets/cartoon/actions/pip_press_button.png")
    rig = get_rig(str(path))
    assert rig.head_layer.size == (1080, 1920)
    assert rig.cover_plate.size == (1080, 1920)
    assert rig.head_bbox[0] < rig.head_bbox[2]
    assert rig.head_bbox[1] < rig.head_bbox[3]
    assert rig_summary(rig)["mode"] == "flattened-art-performance-proxy"


def test_performance_plans_have_comedy_phases():
    assert [p.name for p in PLANS["escalation"].phases] == [
        "anticipate", "action", "impact", "recoil", "settle"
    ]
    assert performance_plan("punchline").freeze_after == 0.72


def test_head_motion_is_beat_specific():
    escalation = head_motion(0.45, "escalation")
    reaction = head_motion(0.45, "reaction")
    punchline = head_motion(0.80, "punchline")
    assert escalation != reaction
    assert punchline[2] != 0.0


def test_unknown_flat_image_can_skip_rig():
    path = Path("/tmp/non_character.png")
    Image.new("RGB", (1080, 1920), "white").save(path)
    try:
        get_rig(str(path))
    except ValueError:
        assert True
    else:
        raise AssertionError("unknown image should not be classified as a character rig")
    path.unlink(missing_ok=True)


def test_pip_rig_extracts_face_performance_plate():
    path = Path("assets/cartoon/actions/pip_press_button.png")
    rig = get_rig(str(path))
    assert rig.face_layer.size == (1080, 1920)
    assert rig.face_bbox[0] < rig.face_bbox[2]
    assert rig.face_bbox[1] < rig.face_bbox[3]


def test_face_motion_has_beat_specific_response():
    from app.providers.cartoon_animation import _face_motion
    setup = _face_motion(0.45, "setup")
    escalation = _face_motion(0.45, "escalation")
    reaction = _face_motion(0.45, "reaction")
    assert setup != escalation
    assert escalation != reaction
    assert all(v == v for v in escalation)


def test_face_motion_has_explicit_blink_window():
    from app.providers.cartoon_animation import _face_motion
    assert _face_motion(0.425, "reaction")[4] > 0.99
    assert _face_motion(0.57, "reaction")[4] > 0.35
    assert _face_motion(0.90, "reaction")[4] < 1e-6


def test_contact_blinks_follow_director_event_timing():
    from app.providers.cartoon_animation import _face_motion

    performance = {"shot_spec": {"phases": [
        {"name": "anticipate", "start": 0.0, "end": 0.60},
        {"name": "impact", "start": 0.70, "end": 0.74},
    ]}}
    contact_blink = _face_motion(0.745, "reaction", performance)[4]
    old_fixed_window = _face_motion(0.22, "reaction", performance)[4]
    recovery_blink = _face_motion(0.89, "reaction", performance)[4]

    assert contact_blink > 0.99
    assert old_fixed_window < 1e-6
    assert 0.30 < recovery_blink < contact_blink


def test_gaze_dart_leads_event_and_releases_afterward():
    from app.providers.cartoon_animation import _gaze_dart

    event = 0.40
    lead = _gaze_dart(event - 0.055, event, 1.0)
    release = _gaze_dart(event + 0.105, event, 1.0)
    before = _gaze_dart(0.05, event, 1.0)
    settled = _gaze_dart(0.90, event, 1.0)

    assert lead > 2.5
    assert release < 0.0
    assert before < 0.01
    assert settled < 0.01


def test_setup_gaze_leads_and_releases_toward_left_attention_target():
    from app.providers.cartoon_animation import _face_motion, _gaze_dart
    from app.providers.cartoon_performance import kinematic_state

    event = kinematic_state(0.0, "setup")["event_fraction"]
    lead = _gaze_dart(event - 0.055, event, 1.0, direction=-1.0)
    release = _gaze_dart(event + 0.105, event, 1.0, direction=-1.0)
    attention_lead = _face_motion(event - 0.055, "setup")[0]
    attention_release = _face_motion(event + 0.105, "setup")[0]

    assert lead < -2.5
    assert release > 0.0
    assert attention_lead < attention_release


def test_vertical_gaze_leads_and_releases_toward_attention_target():
    from dataclasses import replace
    from app.providers import cartoon_animation as animation
    from app.providers.cartoon_animation import _face_motion
    from app.providers.cartoon_performance import kinematic_state

    for beat, expected_direction in (("escalation", -1), ("reaction", 1)):
        event = kinematic_state(0.0, beat)["event_fraction"]
        original = animation.kinematic_state

        def without_vertical_attention(progress, current_beat, performance=None):
            state = original(progress, current_beat, performance)
            state["attention_y"] = 0.0
            state["face_target_dy"] = 0.0
            return state

        animation.kinematic_state = without_vertical_attention
        baseline_lead = _face_motion(event - 0.055, beat)[1]
        baseline_release = _face_motion(event + 0.105, beat)[1]
        animation.kinematic_state = original

        lead = _face_motion(event - 0.055, beat)[1] - baseline_lead
        release = _face_motion(event + 0.105, beat)[1] - baseline_release
        assert lead * expected_direction > 0
        assert release * expected_direction < 0


def test_camera_push_anticipates_action_and_releases_after_impact():
    from app.providers.cartoon_performance import _camera_event_push

    event = 0.42
    anticipation = _camera_event_push(event - 0.07, event, "escalation")
    impact = _camera_event_push(event, event, "escalation")
    release = _camera_event_push(event + 0.24, event, "escalation")
    unrelated = _camera_event_push(event, event, "setup")

    assert 0.0 < anticipation < impact
    assert release < impact
    assert unrelated == 0.0


def test_impact_camera_shake_follows_custom_event_timing():
    from app.providers.cartoon_animation import _impact_shake_envelope

    event = 0.73
    assert _impact_shake_envelope(event, event, "escalation") == 1.0
    assert _impact_shake_envelope(event - 0.10, event, "escalation") == 0.0
    assert _impact_shake_envelope(event + 0.10, event, "escalation") == 0.0
    assert _impact_shake_envelope(0.43, event, "escalation") == 0.0
    assert _impact_shake_envelope(event, event, "setup") == 0.0


def test_escalation_flash_follows_director_impact_timing():
    from app.providers.cartoon_animation import _impact_flash_envelope
    from app.providers.cartoon_performance import kinematic_state

    performance = {"shot_spec": {"phases": [
        {"name": "anticipate", "start": 0.0, "end": 0.60},
        {"name": "impact", "start": 0.70, "end": 0.74},
    ]}}
    event = kinematic_state(0.0, "escalation", performance)["event_fraction"]
    assert event == 0.72
    assert _impact_flash_envelope(event - 0.04, event, "escalation") > 0.99
    assert _impact_flash_envelope(0.16, event, "escalation") == 0.0
    assert _impact_flash_envelope(event, event, "reaction") == 0.0


def test_impact_blink_has_staggered_eyes_and_shared_recovery():
    from app.providers.cartoon_animation import _asymmetric_blink

    event = 0.40
    first_lead = _asymmetric_blink(event + 0.018, event)
    second_lead = _asymmetric_blink(event + 0.036, event)
    recovery = _asymmetric_blink(event + 0.175, event)

    assert first_lead[0] > first_lead[1]
    assert second_lead[1] > second_lead[0]
    assert recovery[0] > 0.25 and recovery[1] > 0.25
    assert abs(recovery[0] - recovery[1]) < 0.15


def test_kinematic_state_shares_root_motion_across_character_parts():
    from app.providers.cartoon_performance import kinematic_state, arm_motion, head_motion
    state = kinematic_state(0.5, "reaction")
    left = arm_motion(0.5, "reaction", "left")
    head = head_motion(0.5, "reaction")
    assert state["root_dx"] != 0
    assert left[0] != 0
    assert head[0] != 0
    assert abs(state["camera_dx"]) < abs(state["root_dx"])


def test_hands_lag_impact_then_damped_follow_through():
    from app.providers.cartoon_performance import kinematic_state

    event = 0.40
    before_lag = kinematic_state(event + 0.045, "reaction")
    follow_through = kinematic_state(event + 0.12, "reaction")
    late_settle = kinematic_state(event + 0.50, "reaction")

    # The delayed hand response adds motion after the body impact, then decays.
    assert follow_through["left_dx"] != before_lag["left_dx"]
    assert follow_through["right_dx"] != before_lag["right_dx"]
    assert abs(late_settle["left_dx"]) < abs(follow_through["left_dx"])
    assert abs(late_settle["right_dx"]) < abs(follow_through["right_dx"])


def test_hand_follow_through_is_staggered_between_sides():
    from app.providers.cartoon_performance import kinematic_state

    event = kinematic_state(0.0, "reaction")["event_fraction"]
    early = kinematic_state(event + 0.045, "reaction")
    later = kinematic_state(event + 0.075, "reaction")

    # The left hand starts its local follow-through first; the right hand
    # remains at its impact pose until its separately delayed response starts.
    assert early["left_hand_lag"] != 0.0
    assert early["right_hand_lag"] == 0.0
    assert later["right_hand_lag"] != 0.0


def test_escalation_arms_draw_back_before_custom_timed_contact():
    from app.providers.cartoon_performance import kinematic_state

    performance = {"shot_spec": {"phases": [
        {"name": "anticipate", "start": 0.0, "end": 0.58},
        {"name": "impact", "start": 0.70, "end": 0.74},
    ]}}
    event = kinematic_state(0.0, "escalation", performance)["event_fraction"]
    anticipation = kinematic_state(event - 0.09, "escalation", performance)
    contact = kinematic_state(event, "escalation", performance)
    follow_through = kinematic_state(event + 0.12, "escalation", performance)

    # Hands prepare in opposite directions, extend on contact, and retain a
    # delayed local response after the torso's impact peak.
    assert anticipation["left_dx"] > 0
    assert anticipation["right_dx"] < 0
    assert contact["left_dx"] < anticipation["left_dx"]
    assert contact["right_dx"] > anticipation["right_dx"]
    assert follow_through["left_dx"] != contact["left_dx"]
    assert follow_through["right_dx"] != contact["right_dx"]


def test_head_lags_torso_impact_then_damped_follow_through():
    from app.providers.cartoon_performance import kinematic_state

    event = 0.40
    at_impact = kinematic_state(event, "reaction")
    follow_through = kinematic_state(event + 0.12, "reaction")
    late_settle = kinematic_state(event + 0.50, "reaction")

    # The head's local offset continues after contact, then returns toward rest.
    impact_offset = at_impact["head_dx"] - at_impact["root_dx"]
    follow_offset = follow_through["head_dx"] - follow_through["root_dx"]
    settle_offset = late_settle["head_dx"] - late_settle["root_dx"]
    assert follow_offset != impact_offset
    assert abs(settle_offset) < abs(follow_offset)


def test_speech_activity_adds_subtle_local_head_bob():
    from app.providers.cartoon_performance import kinematic_state

    silent = kinematic_state(0.5, "setup")
    voiced = kinematic_state(0.5, "setup", {"activity": [0.0, 0.9, 0.9]})

    assert voiced["root_dy"] == silent["root_dy"]
    assert voiced["head_dy"] - silent["head_dy"] == 1.6
    assert voiced["head_rot"] - silent["head_rot"] == 0.45
    assert voiced["speech_pulse"] == 1.0


def test_camera_reserves_look_space_in_face_attention_direction():
    from app.providers.cartoon_performance import kinematic_state

    event = kinematic_state(0.40, "reaction")
    settled = kinematic_state(0.90, "reaction")
    assert event["camera_focus_x"] * event["attention_x"] > 0
    assert abs(event["camera_focus_x"]) > abs(settled["camera_focus_x"])


def test_camera_holds_action_focus_through_early_recovery():
    from app.providers.cartoon_performance import _camera_attention_envelope

    event = 0.40
    contact = _camera_attention_envelope(event, event, "escalation")
    early_recovery = _camera_attention_envelope(event + 0.08, event, "escalation")
    settled = _camera_attention_envelope(event + 0.36, event, "escalation")

    assert early_recovery > 0.70
    assert early_recovery > settled
    assert settled < 0.20
    assert contact == 1.0


def test_camera_vertical_look_space_tracks_above_and_below_action_targets():
    from app.providers.cartoon_performance import kinematic_state

    for beat in ("escalation", "reaction"):
        event = kinematic_state(0.40, beat)
        settled = kinematic_state(0.90, beat)
        assert event["camera_focus_y"] * event["attention_y"] > 0
        assert abs(event["camera_focus_y"]) > abs(settled["camera_focus_y"])
    assert kinematic_state(0.40, "escalation")["camera_focus_y"] < 0
    assert kinematic_state(0.40, "reaction")["camera_focus_y"] > 0


def test_punchline_shared_state_becomes_stable_after_hold():
    from app.providers.cartoon_performance import kinematic_state
    a = kinematic_state(0.72, "punchline")
    b = kinematic_state(0.96, "punchline")
    assert a == b


def test_flattened_rig_composes_shared_root_motion_once(monkeypatch):
    from types import SimpleNamespace
    from app.providers import cartoon_animation as animation

    frame = Image.new("RGB", (24, 24), "white")
    layer = Image.new("RGBA", (24, 24), (0, 0, 0, 0))
    rig = SimpleNamespace(
        source="missing-rig-source.png",
        head_layer=layer,
        cover_mask=Image.new("L", (24, 24), 0),
    )
    captured = {}
    root = {"root_dx": 4.0, "root_dy": -3.0, "root_scale": 1.1, "root_rot": 0.2}
    monkeypatch.setattr(animation, "head_motion", lambda *_: (2.0, 5.0, 1.2, -0.1))
    monkeypatch.setattr(animation, "kinematic_state", lambda *_: root)
    monkeypatch.setattr(animation, "_speech_pulse", lambda *_: 0.0)
    monkeypatch.setattr(animation, "_transform_part", lambda _layer, *args: captured.update(args=args) or layer)
    monkeypatch.setattr(animation, "_apply_face_performance", lambda result, *_args, **_kwargs: result)

    animation._apply_rig_performance(frame, rig, 0.25, "reaction")

    dx, dy, scale, rotation = captured["args"][:4]
    assert (dx, dy, scale, rotation) == (6.0, 2.0, 1.32, 0.1)
