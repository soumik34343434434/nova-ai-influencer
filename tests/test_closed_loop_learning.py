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
            # Pin exploitation; exploring runs are covered separately in
            # test_closed_loop_simulation.py.
            "exploration_rate": 0.0,
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


def test_sparse_history_labels_under_tested_and_proxy_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "METRICS_FILE", tmp_path / "metrics.json")
    monkeypatch.setattr(learning, "METRICS_HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(learning, "STRATEGY_FILE", tmp_path / "strategy.json")
    rows = [_metric("a", "dance", .9)]
    memory = learning.build_strategy_memory(rows)
    evidence = memory["evidence"]
    assert evidence["outcome_evidence"] == {"observed": 0, "proxy": 1, "observations": 1}
    assert evidence["learnings"]["under_tested"][0]["confidence"] == "low"
    assert "causality" in evidence["learnings"]["confidence_note"]


def test_balanced_comparison_uses_platform_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "METRICS_FILE", tmp_path / "metrics.json")
    monkeypatch.setattr(learning, "METRICS_HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(learning, "STRATEGY_FILE", tmp_path / "strategy.json")
    rows = []
    for i, (fmt, ret) in enumerate([("dance", 80), ("dance", 82), ("story", 40), ("story", 42)]):
        rows.append({**_metric(str(i), fmt, 0), "platform": "instagram", "retention_pct": ret})
    memory = learning.build_strategy_memory(rows)
    dance = memory["evidence"]["comparisons"]["format_name"]["dance"]
    assert dance["baseline_observations"] == 2
    assert dance["difference"] > 0
    assert dance["comparison"] == "association"


def test_imbalanced_groups_report_sample_size_and_uncertainty(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "METRICS_HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(learning, "STRATEGY_FILE", tmp_path / "strategy.json")
    rows = [{**_metric(f"d{i}", "dance", .9), "platform": "youtube", "retention_pct": 80} for i in range(8)]
    rows += [{**_metric("s", "story", .2), "platform": "youtube", "retention_pct": 20}]
    memory = learning.build_strategy_memory(rows)
    result = memory["evidence"]["comparisons"]["format_name"]["dance"]
    assert (result["observations"], result["baseline_observations"]) == (8, 1)
    winner = next(x for x in memory["evidence"]["learnings"]["winners"] if x["value"] == "dance")
    assert winner["confidence"] == "low"


def test_time_ordered_snapshots_mark_declining_patterns(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "METRICS_HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(learning, "STRATEGY_FILE", tmp_path / "strategy.json")
    history = []
    for i, retention in enumerate((90, 85, 40, 35)):
        history.append({**_metric("same", "dance", 0), "platform": "instagram", "retention_pct": retention,
                        "captured_at": f"2026-10-0{1+i}T00:00:00+00:00"})
    (tmp_path / "history.json").write_text(__import__("json").dumps(history))
    memory = learning.build_strategy_memory([history[-1]])
    decline = memory["evidence"]["learnings"]["declining_patterns"]
    assert decline
    assert decline[0]["change"] < 0
    assert "not established" in decline[0]["interpretation"]
