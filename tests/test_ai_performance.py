from app.providers.ai_performance import (
    PerformanceRequirement,
    RunwayPerformanceProvider,
    build_runway_repair_request,
    choose_cheapest_available_model,
    choose_cheapest_model,
    requirement_from_performance,
)


def test_cheapest_viable_model_is_selected():
    req = PerformanceRequirement(duration=5.0, acting=True, facial_expression=True, hands=True)
    selected = choose_cheapest_model(req)
    assert selected.model == "gen-4-turbo"
    assert selected.estimated_credits == 25.0


def test_routing_can_require_continuity():
    req = PerformanceRequirement(duration=5.0, acting=True, continuity_critical=True)
    selected = choose_cheapest_model(req)
    assert selected.model == "gen-4-turbo"


def test_director_intent_becomes_routing_requirements():
    req = requirement_from_performance(
        5.0,
        {"expression": "surprised", "gesture": "recoil", "camera": "push_in"},
    )
    assert req.acting is True
    assert req.facial_expression is True
    assert req.hands is True
    assert req.camera_motion is True


def test_runway_provider_never_silently_uses_an_expensive_model(tmp_path):
    provider = RunwayPerformanceProvider()
    plan = provider.plan(PerformanceRequirement(duration=5.0, acting=True, facial_expression=True))
    assert plan.model == "gen-4-turbo"
    assert plan.estimated_credits == 25.0


def test_runway_request_uses_storyboard_shot_spec(tmp_path):
    from app.agents.shot_performance import build_shot_performance_spec

    provider = RunwayPerformanceProvider()
    spec = build_shot_performance_spec(
        "deadpan_camera_reaction",
        "MOMO sees the message.",
        duration=5.0,
        camera="stable camera with subtle push-in",
    )
    request = provider.build_request(
        tmp_path / "momo.png",
        duration=5.0,
        shot_index=2,
        performance={"shot_spec": spec},
    )
    assert request["model"] == "gen-4-turbo"
    assert request["estimated_credits"] == 25.0
    assert request["resolution"] == "720p"
    assert "anticipation" in request["prompt"]
    assert "Action chain v2" in request["prompt"]
    assert "eye_lead" in request["prompt"]
    assert "Secondary motion" in request["prompt"]
    assert "no morphing" in request["prompt"]
    assert request["requirements"]["continuity_critical"] is True
    assert request["requirements"]["lip_sync"] is False


def test_cheapest_available_model_respects_verified_workspace_inventory():
    req = PerformanceRequirement(duration=5.0, acting=True, facial_expression=True)
    selected = choose_cheapest_available_model(
        req,
        {"gen-4.5", "kling-3-standard"},
    )
    assert selected.model == "kling-3-standard"


def test_runway_repair_request_is_zero_execution_and_targeted():
    request = {
        "model": "gen-4-turbo",
        "prompt": "Animate MOMO with the authored performance.",
        "shot_index": 0,
    }
    repaired = build_runway_repair_request(
        request,
        {
            "reasons": ["insufficient_character_motion", "hand_deformation"],
            "repair_shots": [0],
        },
    )
    assert repaired["model"] == "gen-4-turbo"
    assert repaired["repair"]["required"] is True
    assert repaired["repair"]["strategy"] == "targeted-rerender"
    assert repaired["repair"]["target_shots"] == [0]
    assert "Increase readable character performance" in repaired["prompt"]
    assert "hands/paws" in repaired["prompt"]


def test_runway_request_blocks_missing_required_reference_prop(tmp_path):
    from app.agents.shot_performance import build_shot_performance_spec

    provider = RunwayPerformanceProvider()
    spec = build_shot_performance_spec(
        "checks_phone",
        "MOMO checks the phone.",
        duration=5.0,
    )
    try:
        provider.build_request(
            tmp_path / "momo.png",
            duration=5.0,
            shot_index=0,
            performance={
                "shot_spec": spec,
                "reference_manifest": {
                    "character": "MOMO",
                    "scene": "office",
                    "props": ["paper"],
                    "verified": True,
                },
            },
        )
    except ValueError as exc:
        assert "reference_missing_required_prop_phone" in str(exc)
    else:
        raise AssertionError("Expected reference preflight to block missing phone")
