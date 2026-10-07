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
