from app.agents.story_engine import QUALITY_FLOOR, StoryEngine


def _strong_story():
    return {
        "title": "PIP vs the Smart Kitchen",
        "hook": "PIP asked for one sandwich. Big mistake!",
        "character": "PIP",
        "angle": "reverse_expectation",
        "duration_target": 9.0,
        "beats": [
            {"beat": "setup", "asset_action": "press_button", "voice": "One sandwich, please.", "visual": "PIP presses the cooking button."},
            {"beat": "escalation", "asset_action": "shock", "voice": "TWELVE?! The kitchen made twelve.", "visual": "A huge stack of sandwiches appears."},
            {"beat": "reaction", "asset_action": "overwhelmed", "voice": "PIP stares at the giant pile.", "visual": "PIP looks overwhelmed."},
            {"beat": "punchline", "asset_action": "deadpan", "voice": "PIP: The kitchen misunderstood one.", "visual": "PIP gives a deadpan look to camera."},
        ],
    }


def test_story_engine_passes_strong_four_beat_comedy():
    result = StoryEngine().evaluate(_strong_story())
    assert result.passed, result.to_dict()
    assert result.score >= QUALITY_FLOOR
    assert min(d.score for d in result.dimensions) >= QUALITY_FLOOR

    contract = StoryEngine().build(_strong_story()).to_dict()
    assert contract["quality"]["all_dimensions_above_floor"] is True
    assert [b["story_role"] for b in contract["beats"]] == [
        "setup", "escalation", "reaction", "payoff"
    ]


def test_story_engine_rejects_exposition_only_story():
    weak = _strong_story()
    weak["hook"] = "Something happened today."
    weak["beats"] = [
        {"beat": "setup", "asset_action": "finish_work", "voice": "Okay.", "visual": "PIP sits."},
        {"beat": "setup", "asset_action": "finish_work", "voice": "Okay.", "visual": "PIP sits."},
        {"beat": "setup", "asset_action": "finish_work", "voice": "Okay.", "visual": "PIP sits."},
        {"beat": "setup", "asset_action": "finish_work", "voice": "Okay.", "visual": "PIP sits."},
    ]
    result = StoryEngine().evaluate(weak)
    assert not result.passed
    assert result.blockers
    assert any(d.score < QUALITY_FLOOR for d in result.dimensions)


def test_story_engine_outputs_repair_guidance():
    weak = _strong_story()
    weak["angle"] = "generic"
    weak["beats"] = [
        {**row, "asset_action": "finish_work", "voice": "Okay."} for row in weak["beats"]
    ]
    result = StoryEngine().evaluate(weak)
    assert result.repairs
    assert all(isinstance(item, str) and item for item in result.repairs)


def test_deep_analysis_accepts_visible_comedy_escalation():
    result = StoryEngine().evaluate(_strong_story())
    analysis = result.to_dict()["story_analysis"]
    assert result.passed, result.to_dict()
    assert analysis["fake_escalation"] is False
    assert analysis["dimensions"]
    assert all(d["score"] >= QUALITY_FLOOR for d in analysis["dimensions"])
    assert any(t["visible_change"] for t in analysis["transitions"])


def test_deep_analysis_blocks_flat_repetitive_fake_escalation():
    flat = _strong_story()
    flat["beats"] = [
        {"beat": role, "asset_action": "finish_work", "voice": voice,
         "visual": "PIP sits at the desk."}
        for role, voice in [("setup", "One task."), ("escalation", "More task."),
                            ("reaction", "Again, more task."), ("punchline", "Even more task.")]
    ]
    result = StoryEngine().evaluate(flat).to_dict()
    assert not result["passed"]
    assert result["story_analysis"]["fake_escalation"] is True
    assert any("fake_escalation" in b for b in result["story_analysis"]["blockers"])


def test_deep_analysis_flags_unplanted_twist_at_last_beat():
    delayed = _strong_story()
    delayed["beats"][-1] = {
        "beat": "punchline", "asset_action": "deadpan",
        "voice": "Actually, the moon was a receipt.",
        "visual": "A receipt suddenly covers the moon.",
    }
    analysis = StoryEngine().evaluate(delayed).to_dict()["story_analysis"]
    twist = next(d for d in analysis["dimensions"] if d["name"] == "twist_clarity")
    assert twist["score"] < QUALITY_FLOOR
    assert any("final beat without an earlier plant" in e for e in twist["evidence"])


def test_deep_analysis_flags_payoff_without_setup_dependency():
    weak = _strong_story()
    weak["beats"][-1]["voice"] = "Blue Tuesday, the triangle wins."
    weak["beats"][-1]["visual"] = "PIP points at a blue triangle."
    analysis = StoryEngine().evaluate(weak).to_dict()["story_analysis"]
    payoff = next(d for d in analysis["dimensions"] if d["name"] == "payoff_setup_dependency")
    assert payoff["score"] < QUALITY_FLOOR
    assert payoff["repair"]


def test_deep_analysis_rewards_strong_loopable_ending_and_is_backward_compatible():
    story = _strong_story()
    story["beats"][-1]["voice"] = "PIP: One sandwich, coming right back."
    story["beats"][-1]["visual"] = "PIP presses the same cooking button again."
    result = StoryEngine().evaluate(story).to_dict()
    assert "score" in result and "dimensions" in result and "blockers" in result
    ending = next(d for d in result["story_analysis"]["dimensions"]
                  if d["name"] == "ending_strength_loopability")
    assert ending["score"] >= QUALITY_FLOOR
    assert any("echoes the opening" in e for e in ending["evidence"])

def test_control_plane_migrates_legacy_stage_numbers(tmp_path):
    import json
    from app.cartoon_control import CartoonRunStore

    store = CartoonRunStore("legacy-stage-test")
    store.path = tmp_path / "run.json"
    raw_stages = {}
    legacy_names = (
        "1_research", "2_trend_intelligence", "3_idea_generator",
        "4_idea_evaluator", "5_script_engine", "6_storyboard_engine",
        "7_asset_engine", "8_animation_engine", "9_audio_engine",
        "10_caption_engine", "11_video_qa", "12_publish",
        "13_analytics", "14_learning",
    )
    for name in legacy_names:
        raw_stages[name] = {
            "stage": name,
            "status": "passed",
            "started_at": "",
            "finished_at": "",
            "attempts": 1,
            "artifact": {"legacy": True},
            "error": "",
        }
    store.path.write_text(json.dumps({
        "run_id": "legacy-stage-test",
        "created_at": "",
        "current_stage": "9_audio_engine",
        "status": "failed",
        "payload": {},
        "stages": raw_stages,
    }))
    loaded = store.load()
    assert loaded.current_stage == "10_audio_engine"
    assert "5_script_engine" not in loaded.stages
    assert "6_script_engine" in loaded.stages
    assert "16_learning" in loaded.stages


def test_control_plane_migrates_runs_saved_before_the_editor_stage(tmp_path):
    import json
    from app.cartoon_control import CartoonRunStore

    store = CartoonRunStore("pre-editor-run")
    store.path = tmp_path / "run.json"
    names = (
        "1_research", "2_trend_intelligence", "3_idea_generator", "4_idea_evaluator",
        "5_story_engine", "6_script_engine", "7_storyboard_engine", "8_asset_engine",
        "9_animation_engine", "10_audio_engine", "11_caption_engine", "12_video_qa",
        "13_publish", "14_analytics", "15_learning",
    )
    stages = {n: {"stage": n, "status": "passed", "started_at": "", "finished_at": "", "attempts": 1,
                  "artifact": {}, "error": ""} for n in names}
    store.path.write_text(json.dumps({"run_id": "pre-editor-run", "created_at": "", "current_stage": "14_analytics",
                                      "status": "failed", "payload": {}, "stages": stages}))
    loaded = store.load()
    assert loaded.current_stage == "15_analytics"
    assert "13_video_qa" in loaded.stages and "12_video_qa" not in loaded.stages
    assert "16_learning" in loaded.stages
    assert "12_editor_engine" not in loaded.stages


def test_orchestrator_story_stage_persists_locked_contract(tmp_path, monkeypatch):
    import app.cartoon_factory_orchestrator as orch
    from app.cartoon_control import CartoonRunStore

    monkeypatch.setattr(orch, "ARTIFACTS", tmp_path)
    store = CartoonRunStore("story-stage-test")
    store.path = tmp_path / "run.json"
    factory = orch.CartoonFactoryOrchestrator()
    factory.plane.store = store
    run = store.create({
        "winner": {"candidate": _strong_story()},
    })
    result = factory._story_engine(run)
    assert result["status"] == "passed"
    assert result["score"] >= 8.5
    assert result["min_dimension"] >= 8.5
    assert run.payload["story"]["quality"]["all_dimensions_above_floor"] is True

    script_result = factory._script_engine(run)
    assert script_result["story_locked"] is True
    assert run.payload["script"] == run.payload["script_contract"]["text"]
    assert [beat["story_role"] for beat in run.payload["script_contract"]["beats"]] == [
        "setup", "escalation", "reaction", "payoff"
    ]
