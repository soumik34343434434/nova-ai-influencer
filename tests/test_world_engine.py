from app.agents.storyboard import build_storyboard
from app.agents.world_engine import WorldEngine


def test_world_engine_selects_scene_and_prop():
    engine = WorldEngine()
    office = engine.plan_shot("Manager: can we schedule another meeting?", 0)
    metro = engine.plan_shot("NOVA checks a phone notification on the commute.", 1)
    home = engine.plan_shot("NOVA has seventeen tabs open.", 2)

    assert office["scene"] == "office"
    assert office["prop"] in {"calendar", "phone", "coffee", "laptop"}
    assert metro["scene"] == "metro"
    assert metro["prop"] == "phone"
    assert home["scene"] == "home"
    assert home["prop"] == "laptop"


def test_storyboard_contains_world_plan_for_every_shot():
    draft = {
        "hook": "AI meetings are multiplying.",
        "script": "Manager: Another meeting?\n\nNOVA checks the calendar.\n\nNOVA stares at the camera."
    }
    board = build_storyboard(draft)
    assert board.shots
    assert all(shot.world["scene"] for shot in board.shots)
    assert all(shot.world["wardrobe"] for shot in board.shots)
    assert all(shot.world["prop"] for shot in board.shots)
