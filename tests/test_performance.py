from app.agents.performance import plan_performance


def test_performance_maps_action_to_expression_and_gesture():
    result = plan_performance("checks_phone", "The phone just notified NOVA!", 1)
    assert result["expression"] == "surprised"
    assert result["gesture"] == "phone_glance"
    assert result["attention"] == "scene"
    assert result["speaking_style"] == "measured_deadpan"


def test_deadpan_reaction_faces_camera():
    result = plan_performance("deadpan_camera_reaction", "NOVA stares at the camera.", 0)
    assert result["expression"] == "deadpan"
    assert result["gesture"] == "freeze"
    assert result["attention"] == "camera"
