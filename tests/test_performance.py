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


def test_storyboard_performance_contains_temporal_shot_spec():
    from app.agents.storyboard import build_storyboard
    board = build_storyboard({
        "hook": "MOMO sees the phone.",
        "script": "MOMO checks the phone!",
    })
    assert board.shots
    spec = board.shots[0].performance["shot_spec"]
    assert spec["emotion_start"]
    assert spec["emotion_peak"]
    assert spec["emotion_end"]
    assert [p["name"] for p in spec["phases"]] == [
        "anticipation",
        "eye_lead",
        "head_follow",
        "action",
        "impact",
        "overshoot",
        "recoil",
        "secondary_motion",
        "settle",
    ]
    assert spec["phases"][0]["start"] == 0.0
    assert spec["phases"][-1]["end"] == spec["duration"]
    assert spec["action_chain_version"] == 2.3
    assert len(spec["micro_beats"]) == 9
    assert "eyes move toward the phone before the head moves" in spec["micro_beats"][1]
    assert "scarf lags the torso" in spec["secondary_motion"]
    assert len(spec["facial_performance"]) >= 4
    assert "pupils/gaze shift toward the phone" in spec["facial_performance"][0]
    assert len(spec["pose_accents"]) == 3
    assert "front-facing" in spec["orientation_lock"]
    assert "upper-body action" in spec["motion_priority"]


def test_performance_prompt_carries_facial_and_pose_accents():
    from app.agents.shot_performance import build_shot_performance_spec, performance_prompt
    spec = build_shot_performance_spec("checks_phone", "Oh no, the phone again!", 5.0)
    prompt = performance_prompt(spec)
    assert "Facial performance:" in prompt
    assert "Pose accents:" in prompt
    assert "mouth makes a tiny surprised opening" in prompt
    assert "feet/root remain fixed" in prompt
    assert "Orientation lock:" in prompt
    assert "no large yaw" in prompt
