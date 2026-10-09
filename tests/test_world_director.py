from app.agents.world_director import plan_world_depth
from app.agents.world_engine import WorldEngine


def test_world_director_creates_three_depth_layers_and_constrained_parallax():
    plan = plan_world_depth("office", "phone", "checks_phone").to_dict()
    assert plan["back_layer"]
    assert plan["mid_layer"]
    assert plan["foreground_layer"]
    assert len(plan["parallax"]) == 3
    assert plan["parallax"][0] < plan["parallax"][1] < plan["parallax"][2]
    assert "brand" in " ".join(plan["continuity_lock"])


def test_world_engine_exposes_depth_plan_to_storyboard():
    shot = WorldEngine().plan_shot("I checked my phone and got 47 new tasks", 0)
    assert shot["scene"] == "office"
    assert shot["depth_plan"]["prop_action"]
    assert shot["depth_plan"]["parallax"] == [0.08, 0.35, 0.75]
