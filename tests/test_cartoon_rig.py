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
    assert _face_motion(0.22, "reaction")[4] > 0.0
    assert _face_motion(0.50, "reaction")[4] < 1e-6
