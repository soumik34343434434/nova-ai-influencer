import app.learning_v2 as learning


def _metric(content_id, fmt, score):
    return {
        "content_id": content_id,
        "platform": "sim",
        "format_name": fmt,
        "hook_pattern": "curiosity",
        "emotion": "amusement",
        "character": "MOMO",
        "views": 1000,
        "likes": 100,
        "comments": 20,
        "shares": 50,
        "saves": 20,
        "followers_gained": 10,
        "retention_pct": 80 if score > 0.7 else 35,
        "duration_seconds": 8,
        "score": score,
    }


def test_orchestrator_applies_learning_policy_to_ideas(monkeypatch, tmp_path):
    import app.cartoon_factory_orchestrator as orch
    from app.cartoon_control import CartoonRun, CartoonRunStore

    monkeypatch.setattr(orch, "ARTIFACTS", tmp_path)
    candidates = [
        {
            "idea_id": "story", "title": "Story", "format_name": "story_comedy",
            "hook_pattern": "curiosity", "emotion": "amusement", "character": "MOMO",
            "beats": [], "trend_score": 0.5,
        },
        {
            "idea_id": "dance", "title": "Dance", "format_name": "dance",
            "hook_pattern": "challenge", "emotion": "cute", "character": "PIP",
            "beats": [], "trend_score": 0.5,
        },
    ]
    monkeypatch.setattr(orch, "generate_ideas", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(
        orch,
        "select_portfolio",
        lambda rows, memory=None, limit=60: [dict(x) for x in rows],
    )

    store = CartoonRunStore("closed-loop-policy-test")
    run = store.create({
        "topic": "test",
        "artifacts": {},
        "versions": {},
        "ranked_signals_v2": [],
        "format_candidates": [],
        "learning_policy": {
            "preferred_formats": ["story_comedy"],
            "preferred_hooks": ["curiosity"],
            "preferred_emotions": ["amusement"],
            "preferred_characters": ["MOMO"],
            "exploration_rate": 0.22,
        },
    })
    orch.CartoonFactoryOrchestrator().plane.store = store
    result = orch.CartoonFactoryOrchestrator()._idea_generator(run)

    ranked = run.payload["candidates"]
    assert result["count"] == 2
    assert ranked[0]["format_name"] == "story_comedy"
    assert ranked[0]["learning_adjustment"] > ranked[1]["learning_adjustment"]


def test_ingest_preserves_metric_history_without_double_counting_strategy_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "METRICS_FILE", tmp_path / "metrics.json")
    monkeypatch.setattr(learning, "METRICS_HISTORY_FILE", tmp_path / "metrics_history.json")
    monkeypatch.setattr(learning, "STRATEGY_FILE", tmp_path / "strategy.json")

    first = _metric("story-1", "story_comedy", 0.70)
    first["captured_at"] = "2026-10-08T10:00:00+00:00"
    second = dict(first)
    second.update({"views": 2400, "likes": 300, "shares": 160, "retention_pct": 88, "score": 0.92, "captured_at": "2026-10-08T16:00:00+00:00"})

    learning.ingest([first])
    learning.ingest([second])

    latest = learning._load_rows(learning.METRICS_FILE)
    history = learning.load_metric_history()
    memory = learning.build_strategy_memory()
    assert len(latest) == 1
    assert latest[0]["views"] == 2400
    assert len(history) == 2
    assert [x["captured_at"] for x in history] == [first["captured_at"], second["captured_at"]]
    assert memory["observations"] == 1


def test_metrics_change_future_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "METRICS_FILE", tmp_path / "metrics.json")
    monkeypatch.setattr(learning, "STRATEGY_FILE", tmp_path / "strategy.json")

    # Loop 1: story_comedy performs strongly; dance performs weakly.
    learning.ingest([
        _metric("story-1", "story_comedy", 0.95),
        _metric("dance-1", "dance", 0.10),
    ])
    memory = learning.build_strategy_memory()
    assert memory["strategy"]["preferred_formats"][0] == "story_comedy"

    # Loop 2 must see that changed policy.
    assert "story_comedy" in memory["strategy"]["preferred_formats"]
    assert "dance" in memory["strategy"]["preferred_formats"] or "dance" in memory["posteriors"]["format_name"]
