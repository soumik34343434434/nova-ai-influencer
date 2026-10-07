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
    assert plan["strategy"] == "targeted-shot-rerender"
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


def test_audio_sync_check_rejects_voice_overrun():
    from app.agents.cartoon_motion_qa import CartoonMotionQA
    metadata = {
        "audio_sync": {
            "effective_shot_durations": [1.0],
            "voice_performance": [{"duration": 1.2, "peak_fraction": 0.5}],
        }
    }
    score, reasons = CartoonMotionQA._audio_sync_check(metadata, 1.0)
    assert score < 1.0
    assert "voice_overrun_shot_0" in reasons


def test_audio_sync_check_accepts_measured_performance():
    from app.agents.cartoon_motion_qa import CartoonMotionQA
    metadata = {
        "audio_sync": {
            "effective_shot_durations": [1.7, 1.2],
            "voice_performance": [
                {"duration": 1.1, "peak_fraction": 0.59},
                {"duration": 0.55, "peak_fraction": 0.12},
            ],
        }
    }
    score, reasons = CartoonMotionQA._audio_sync_check(metadata, 2.9)
    assert score > 0.98
    assert reasons == []


def test_subject_exit_frame_flags_bbox_at_frame_edge():
    metadata = {
        "width": 1080,
        "height": 1920,
        "character_frame_bboxes": [[100, 200, 1000, 900], [0, 240, 1000, 900], [80, 240, 1070, 900]],
    }
    reasons = CartoonMotionQA._subject_exit_frame(metadata)
    assert reasons == ["subject_exit_frame_1", "subject_exit_frame_2"]


def test_subject_exit_frame_is_silent_without_frame_bboxes():
    assert CartoonMotionQA._subject_exit_frame({"character_bboxes": [[0, 0, 1080, 900]]}) == []
