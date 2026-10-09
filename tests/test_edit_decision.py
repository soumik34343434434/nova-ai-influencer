from app.agents.edit_decision import build_edit_plan


def _beats():
    return [
        {"story_role": "hook", "voice": "The toaster sent a calendar invite."},
        {"story_role": "escalation", "voice": "It booked the whole week."},
        {"story_role": "reaction", "voice": "PIP stares at the toaster."},
        {"story_role": "payoff", "voice": "Declined. It rescheduled."},
    ]


def test_edit_plan_is_deterministic_and_respects_shot_pacing():
    plan = build_edit_plan(_beats(), [1.2, 1.5, 0.8, 1.0])
    assert plan == build_edit_plan(_beats(), [1.2, 1.5, 0.8, 1.0])
    assert all(shot["duration_seconds"] >= plan["pacing_constraints"]["min_shot_seconds"] for shot in plan["shots"])
    assert plan["cut_points_seconds"] == [1.2, 2.7, 3.5]


def test_hook_starts_immediately_and_lands_inside_first_second():
    plan = build_edit_plan(_beats(), [1.2, 1.5, 0.8, 1.0])
    assert plan["hook"] == {"start_seconds": 0.0, "end_seconds": 1.0, "shot_index": 0}


def test_payoff_and_reaction_have_explicit_timing():
    plan = build_edit_plan(_beats(), [1.2, 1.5, 0.8, 1.0])
    assert plan["payoff"] == {"start_seconds": 3.5, "end_seconds": 4.5, "shot_index": 3}
    assert plan["reaction_holds"] == [{"shot_index": 2, "start_seconds": 2.7, "end_seconds": 3.5, "hold_seconds": 0.8}]


def test_caption_windows_stay_within_shot_and_vertical_safe_bounds():
    plan = build_edit_plan(_beats(), [1.2, 1.5, 0.8, 1.0])
    for shot in plan["shots"]:
        caption = shot["caption"]
        bounds = caption["safe_bounds"]
        assert shot["start_seconds"] <= caption["start_seconds"] < caption["end_seconds"] <= shot["end_seconds"]
        assert bounds["x_min"] >= 0 and bounds["x_max"] <= 1
        assert bounds["y_min"] >= 0 and bounds["y_max"] <= 1
        assert bounds["y_min"] < caption["anchor"]["y"] < bounds["y_max"]


def test_ending_loops_to_opening_with_restrained_transitions():
    plan = build_edit_plan(_beats(), [1.2, 1.5, 0.8, 1.0])
    assert plan["loop"] == {"enabled": True, "from_seconds": 4.5, "to_shot_index": 0, "match_opening": True}
    assert plan["transitions"] == {"default": "hard_cut", "effects": []}
