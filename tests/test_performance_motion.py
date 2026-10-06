from PIL import Image
from app.providers.local_puppet import _rig_frame


def test_performance_changes_motion_profile_without_face_repainting():
    img = Image.new("RGB", (1080, 1920), (40, 40, 40))
    deadpan = {"expression": "deadpan", "gesture": "freeze", "attention": "camera", "head": 0.15}
    surprised = {"expression": "surprised", "gesture": "phone_glance", "attention": "scene", "head": 0.65}

    a = _rig_frame(img, 0.2, 0.4, 0, deadpan)
    b = _rig_frame(img, 0.8, 1.2, 0, surprised)

    assert a.size == (540, 960)
    assert b.size == (540, 960)
    assert a.tobytes() != b.tobytes()


def test_shared_kinematic_state_has_physical_attachment_offsets():
    from app.providers.cartoon_performance import kinematic_state
    state = kinematic_state(0.50, "escalation")
    assert state["root_dx"] != 0
    assert abs(state["camera_dx"]) < abs(state["root_dx"])
    assert state["left_rot"] != 0
    assert state["right_rot"] != 0


def test_root_motion_is_coupled_but_camera_is_damped():
    from app.providers.cartoon_performance import kinematic_state
    state = kinematic_state(0.50, "reaction")
    assert abs(state["camera_dx"]) < abs(state["root_dx"])
    assert abs(state["camera_dy"]) < abs(state["root_dy"])
    assert abs(state["camera_rot"]) < abs(state["root_rot"])


def test_attention_target_changes_by_story_beat():
    from app.providers.cartoon_performance import kinematic_state
    setup = kinematic_state(0.7, "setup")
    punch = kinematic_state(0.7, "punchline")
    assert setup["attention_x"] < 0
    assert punch["attention_x"] == 0
    assert setup["face_target_dx"] < 0


def test_settled_character_has_bounded_life_motion():
    from app.providers.cartoon_performance import kinematic_state
    state_a = kinematic_state(0.82, "reaction")
    state_b = kinematic_state(0.94, "reaction")
    assert abs(state_a["root_dx"]) < 8
    assert abs(state_b["root_dx"]) < 8
    assert abs(state_a["head_rot"]) < 8
