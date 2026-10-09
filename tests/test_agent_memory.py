from app.agent_memory import (
    build_next_loop_plan,
    ensure_experiment_identity,
    experiment_id_for,
    record_failure,
    record_provider_observation,
    summarize_failures,
    summarize_providers,
    upsert_experiment,
)


def test_experiment_identity_is_deterministic():
    assert experiment_id_for("run-1") == experiment_id_for("run-1")
    payload = {}
    identity = ensure_experiment_identity(payload, "run-1")
    assert identity["experiment_id"].startswith("exp_")
    assert payload["experiment"]["run_id"] == "run-1"


def test_failure_and_provider_memory_round_trip(tmp_path, monkeypatch):
    import app.agent_memory as memory

    monkeypatch.setattr(memory, "FAILURES_FILE", tmp_path / "failures.json")
    monkeypatch.setattr(memory, "PROVIDERS_FILE", tmp_path / "providers.json")
    record_failure(
        run_id="r1",
        experiment_id="e1",
        stage="animation",
        error_type="RenderError",
        message="broken frame",
        attempt=1,
    )
    record_provider_observation(
        provider="pillow",
        task="animation",
        success=True,
        experiment_id="e1",
        repaired=False,
    )
    failures = summarize_failures()
    providers = summarize_providers()
    assert failures["observations"] == 1
    assert failures["by_error_type"]["RenderError"] == 1
    assert providers["observations"] == 1
    assert providers["by_provider_task_model"]["pillow|animation|"]["success_rate"] == 1.0


def test_next_loop_plan_contains_exploitation_and_exploration(tmp_path, monkeypatch):
    import app.agent_memory as memory

    monkeypatch.setattr(memory, "NEXT_LOOP_FILE", tmp_path / "next_loop.json")
    strategy = {
        "observations": 3,
        "strategy": {
            "exploration_rate": 0.30,
            "preferred_formats": ["dance", "story_comedy"],
            "preferred_hooks": ["pov"],
            "preferred_emotions": ["cute"],
            "preferred_characters": ["MOMO", "PIP"],
        },
        "posteriors": {
            "format_name": {
                "dance": {"observations": 3, "ucb": 0.8},
                "story_comedy": {"observations": 1, "ucb": 0.65},
            }
        },
    }
    plan = build_next_loop_plan(strategy)
    assert plan["exploration_rate"] == 0.30
    assert any(x["type"] == "reinforcement" for x in plan["experiments"])
    assert any(x["type"] == "information_gain" for x in plan["experiments"])


def _action_memory(observations, mean, *, analytics=None):
    stats = {"observations": observations, "raw_mean": mean, "ucb": mean}
    return {
        "observations": observations,
        "analytics_observations": observations if analytics is None else analytics,
        "strategy": {"exploration_rate": .3, "preferred_formats": [], "preferred_hooks": [],
                     "preferred_emotions": [], "preferred_characters": []},
        "posteriors": {
            "format_name": {"winner": stats, "loser": {**stats, "raw_mean": .2}},
            "hook_pattern": {"curiosity": stats},
            "duration_band": {"short_0_8s": stats},
            "emotion": {"amusement": stats},
            "character": {"MOMO": stats},
        },
    }


def test_strategy_actions_cover_strong_winner_with_evidence(tmp_path, monkeypatch):
    import app.agent_memory as memory
    monkeypatch.setattr(memory, "NEXT_LOOP_FILE", tmp_path / "next.json")
    plan = build_next_loop_plan(_action_memory(8, .82))
    action = next(x for x in plan["strategy_actions"] if x["dimension"] == "formats")
    assert action["value"] == "winner"
    assert action["classification"] == "exploit"
    assert action["evidence_count"] == 8 and action["confidence"] > .5
    assert "averaged" in action["rationale"]


def test_strategy_actions_retest_weak_performer(tmp_path, monkeypatch):
    import app.agent_memory as memory
    monkeypatch.setattr(memory, "NEXT_LOOP_FILE", tmp_path / "next.json")
    plan = build_next_loop_plan(_action_memory(6, .75))
    action = next(x for x in plan["strategy_actions"] if x["value"] == "loser")
    assert action["change"] == "deprioritize_and_retest"
    assert action["classification"] == "explore"
    assert action["evidence_count"] == 6


def test_conflicting_dimension_signals_are_preserved(tmp_path, monkeypatch):
    import app.agent_memory as memory
    monkeypatch.setattr(memory, "NEXT_LOOP_FILE", tmp_path / "next.json")
    data = _action_memory(4, .84)
    data["posteriors"]["format_name"] = {"story": {"observations": 3, "raw_mean": .8}}
    data["posteriors"]["hook_pattern"] = {"curiosity": {"observations": 2, "raw_mean": .3}}
    plan = build_next_loop_plan(data)
    assert any(x["dimension"] == "formats" and x["value"] == "story" and x["change"] == "favor" for x in plan["strategy_actions"])
    assert any(x["dimension"] == "hook_styles" and x["value"] == "curiosity" and x["change"] == "deprioritize_and_retest" for x in plan["strategy_actions"])


def test_sparse_evidence_stays_exploratory(tmp_path, monkeypatch):
    import app.agent_memory as memory
    monkeypatch.setattr(memory, "NEXT_LOOP_FILE", tmp_path / "next.json")
    plan = build_next_loop_plan(_action_memory(1, .9))
    action = next(x for x in plan["strategy_actions"] if x["dimension"] == "formats")
    assert action["classification"] == "explore"
    assert action["confidence"] == .25


def test_no_data_does_not_invent_performance(tmp_path, monkeypatch):
    import app.agent_memory as memory
    monkeypatch.setattr(memory, "NEXT_LOOP_FILE", tmp_path / "next.json")
    plan = build_next_loop_plan({"observations": 0, "analytics_observations": 0, "strategy": {}, "posteriors": {}})
    assert plan["strategy_actions"][0]["dimension"] == "experiment_priorities"
    assert plan["strategy_actions"][0]["evidence_count"] == 0
    assert plan["strategy_actions"][0]["confidence"] == 0
    assert not any(x["dimension"] == "formats" for x in plan["strategy_actions"])
