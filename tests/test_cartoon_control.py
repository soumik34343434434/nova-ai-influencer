from pathlib import Path

from app.cartoon_control import CartoonControlPlane, CartoonRunStore, CartoonStage
from app.learning import MetricRecord, ingest, build_learning_snapshot


def test_control_plane_checkpoints_and_skips_passed_stage(tmp_path):
    store = CartoonRunStore("run-test")
    store.path = tmp_path / "run.json"
    plane = CartoonControlPlane(store=store, max_retries=0)
    calls = []

    def handler(run):
        calls.append(run.current_stage)
        return {"ok": True}

    for stage in CartoonStage:
        plane.register(stage, handler)

    run = store.create({"topic": "demo"})
    completed = plane.run(run)
    assert completed.status == "completed"
    assert len(completed.stages) == len(tuple(CartoonStage))

    resumed = plane.run(store.load())
    assert resumed.status == "completed"
    assert len(calls) == len(tuple(CartoonStage))


def test_control_plane_repair_preserves_previous_artifacts(tmp_path):
    store = CartoonRunStore("repair-test")
    store.path = tmp_path / "run.json"
    plane = CartoonControlPlane(store=store, max_retries=0)

    for stage in CartoonStage:
        plane.register(stage, lambda run: {"stage": run.current_stage})

    run = plane.run(store.create())
    assert run.status == "completed"
    before = store.load().stages[CartoonStage.RESEARCH.value].artifact
    repaired = plane.repair(run.run_id, CartoonStage.ASSET_ENGINE)
    assert repaired.status == "completed"
    assert store.load().stages[CartoonStage.RESEARCH.value].artifact == before


def test_learning_prefers_retention_and_shares(tmp_path, monkeypatch):
    import app.learning as learning

    monkeypatch.setattr(learning, "METRICS_FILE", tmp_path / "metrics.json")
    monkeypatch.setattr(learning, "LEARNING_FILE", tmp_path / "learning.json")
    ingest([
        MetricRecord("a", "instagram", "Winner", views=1000, likes=80, shares=50, retention_pct=82),
        MetricRecord("b", "instagram", "Loser", views=1000, likes=90, shares=3, retention_pct=35),
    ])
    snapshot = build_learning_snapshot()
    assert snapshot["winner"]["content_id"] == "a"
    assert snapshot["loser"]["content_id"] == "b"
