import os

from fastapi.testclient import TestClient

import app.cartoon_control as control
import app.api.n8n_routes as n8n_routes
from app.main import app


def test_n8n_health_does_not_require_auth():
    with TestClient(app) as client:
        response = client.get("/api/n8n/health")
    assert response.status_code == 200
    assert response.json()["integration"] == "n8n"


def test_n8n_run_requires_token(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "RUNS_DIR", tmp_path)
    monkeypatch.setenv("NOVA_API_TOKEN", "test-token")
    with TestClient(app) as client:
        missing = client.post("/api/n8n/runs", json={"topic": "demo"})
        wrong = client.post(
            "/api/n8n/runs",
            json={"topic": "demo"},
            headers={"X-NOVA-API-Key": "wrong"},
        )
        created = client.post(
            "/api/n8n/runs",
            json={"topic": "demo"},
            headers={"X-NOVA-API-Key": "test-token"},
        )
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert created.status_code == 200
    body = created.json()
    assert body["run"]["payload"]["topic"] == "demo"


def test_n8n_can_drive_one_durable_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "RUNS_DIR", tmp_path)
    monkeypatch.setenv("NOVA_API_TOKEN", "test-token")
    created = n8n_routes.factory.create_run("stage-test")

    calls = []

    def fake_run_stage(run, stage):
        calls.append(stage.value)
        store = control.CartoonRunStore(run.run_id)
        store.begin_stage(run, stage)
        store.finish_stage(run, stage, {"fake": True})
        return run

    monkeypatch.setattr(n8n_routes.factory.plane, "run_stage", fake_run_stage)

    with TestClient(app) as client:
        response = client.post(
            f"/api/n8n/runs/{created.run_id}/stages/{control.CartoonStage.RESEARCH.value}",
            headers={"X-NOVA-API-Key": "test-token"},
        )

    assert response.status_code == 200
    assert calls == [control.CartoonStage.RESEARCH.value]
    assert response.json()["stage_status"][control.CartoonStage.RESEARCH.value] == "passed"


def test_n8n_run_status_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "RUNS_DIR", tmp_path)
    monkeypatch.setenv("NOVA_API_TOKEN", "test-token")
    created = n8n_routes.factory.create_run("status-test")

    with TestClient(app) as client:
        response = client.get(
            f"/api/n8n/runs/{created.run_id}",
            headers={"X-NOVA-API-Key": "test-token"},
        )

    assert response.status_code == 200
    assert response.json()["run_id"] == created.run_id
    assert response.json()["status"] == "running"


def test_n8n_api_fails_closed_without_token(monkeypatch):
    monkeypatch.delenv("NOVA_API_TOKEN", raising=False)
    with TestClient(app) as client:
        response = client.post("/api/n8n/runs", json={"topic": "demo"})
    assert response.status_code == 503
