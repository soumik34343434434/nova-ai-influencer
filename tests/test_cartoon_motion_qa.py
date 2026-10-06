from pathlib import Path

from app.agents.cartoon_motion_qa import CartoonMotionQA


def test_caption_collision_gate_flags_large_overlap():
    metadata = {
        "caption_zone": [40, 1480, 1040, 1885],
        "character_bboxes": [
            [100, 900, 900, 1200],
            [100, 1400, 900, 1800],
        ],
    }
    reasons = CartoonMotionQA._caption_collisions(metadata)
    assert reasons == ["caption_character_collision_shot_1"]


def test_shot_boundary_parser_uses_durations():
    metadata = {
        "shots": [
            {"duration": 2.3, "beat": "setup"},
            {"duration": 2.3, "beat": "escalation"},
            {"duration": 2.8, "beat": "punchline"},
        ]
    }
    rows = CartoonMotionQA._shot_boundaries(metadata)
    assert [x[2] for x in rows] == ["setup", "escalation", "punchline"]
    assert [round(x[1], 2) for x in rows] == [2.3, 4.6, 7.4]


def test_missing_render_is_rejected(tmp_path: Path):
    result = CartoonMotionQA().inspect(tmp_path / "missing.mp4")
    assert not result["passed"]
    assert "missing_or_too_small" in result["reasons"]


def test_repair_plan_amplifies_dead_motion_without_changing_content():
    plan = CartoonMotionQA.plan_repair(["insufficient_character_motion"])
    assert plan["strategy"] == "targeted-single-rerender"
    assert plan["profile"]["motion_gain"] > 1.0
    assert plan["profile"]["camera_gain"] > 1.0
    assert plan["profile"]["transition_seconds"] == 0.20


def test_repair_plan_softens_pose_pop_and_transition_spike():
    plan = CartoonMotionQA.plan_repair(["possible_pose_popping", "excessive_transition_spike"])
    assert plan["profile"]["motion_gain"] < 1.0
    assert plan["profile"]["shake_gain"] < 1.0
    assert plan["profile"]["transition_seconds"] == 0.14


def test_repair_plan_moves_caption_when_collision_is_reported():
    plan = CartoonMotionQA.plan_repair(["caption_character_collision_shot_2"])
    assert plan["profile"]["caption_y"] < 1515
    assert plan["profile"]["caption_zone"][1] < 1480
