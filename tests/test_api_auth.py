"""Every legacy API route needs the shared token; health stays public; no run starts without auth."""
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import automation_routes, media_routes, routes
from app.api.auth import check_token


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("NOVA_API_TOKEN", "test-token")
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    app.include_router(automation_routes.router, prefix="/api")
    app.include_router(media_routes.router, prefix="/api")
    # Authorised requests reach the database; give them an empty session rather than the real one.
    app.dependency_overrides[routes.db_dep] = lambda: MagicMock()
    return TestClient(app)


PROTECTED = [
    ("post", "/api/content/generate"),
    ("get", "/api/content/queue"),
    ("post", "/api/content/1/approve"),
    ("post", "/api/metrics"),
    ("post", "/api/automation/run"),
    ("post", "/api/media/build-latest"),
]


@pytest.mark.parametrize("method,path", PROTECTED)
def test_legacy_routes_reject_missing_or_wrong_token(client, method, path):
    assert getattr(client, method)(path).status_code == 401
    assert getattr(client, method)(path, headers={"X-NOVA-API-Key": "wrong"}).status_code == 401


def test_health_stays_public(client):
    assert client.get("/api/health").json()["ok"] is True


def test_correct_token_passes_the_guard(client):
    response = client.get("/api/content/queue", headers={"X-NOVA-API-Key": "test-token"})
    assert response.status_code == 200


def test_unconfigured_token_fails_closed(monkeypatch):
    monkeypatch.delenv("NOVA_API_TOKEN", raising=False)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as raised:
        check_token("anything")
    assert raised.value.status_code == 503
