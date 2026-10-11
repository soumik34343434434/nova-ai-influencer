"""Safety rules of the n8n upgrade executor, exercised against a throwaway git repo."""

import subprocess

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import scripts.n8n_upgrade_server as server

TOKEN = "test-token"
GIT_ENV = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}


def _git(repo, *args):
    import os

    env = {**os.environ, **GIT_ENV}
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True, env=env).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / "app" / "module.py").write_text("VALUE = 1\n")
    (root / ".gitignore").write_text(".env\n.venv/\nartifacts/\nnova.db\n")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    (root / ".env").write_text("NOVA_API_TOKEN=secret\n")
    monkeypatch.setattr(server, "REPO", root)
    monkeypatch.setattr(server, "TOKEN", TOKEN)
    return root


def test_protected_paths_win_over_the_allowed_app_prefix():
    assert server._is_protected("app/providers/runway_video.py")
    assert server._is_protected("app/publish_approved.py")
    assert server._is_protected(".github/workflows/ci.yml")
    assert server._is_protected("scripts/n8n_upgrade_server.py")
    assert not server._is_protected("app/agents/idea_generator_v2.py")


def test_generated_output_is_allowed_but_credentials_and_databases_are_not():
    assert server._is_generated_ignored(".venv/lib/python3.13/site-packages/x.py")
    assert server._is_generated_ignored("artifacts/cartoon/x.json")
    assert server._is_generated_ignored("app/__pycache__/x.cpython-313.pyc")
    assert not server._is_generated_ignored(".env")
    assert not server._is_generated_ignored("nova.db")


def test_codex_edit_to_an_allowed_file_passes(repo):
    ignored_before = server._repo_status()[1]
    secrets_before = server._secret_fingerprint()
    (repo / "app" / "module.py").write_text("VALUE = 2\n")

    assert server._ensure_codex_changes_safe(ignored_before, secrets_before) == ["app/module.py"]


def test_codex_edit_to_the_env_file_is_detected_even_though_it_is_ignored(repo):
    ignored_before = server._repo_status()[1]
    secrets_before = server._secret_fingerprint()
    (repo / ".env").write_text("NOVA_API_TOKEN=changed\n")

    with pytest.raises(RuntimeError, match="modified secret files: .env"):
        server._ensure_codex_changes_safe(ignored_before, secrets_before)


def test_codex_creating_a_new_ignored_database_is_a_violation(repo):
    ignored_before = server._repo_status()[1]
    (repo / "nova.db").write_bytes(b"")

    with pytest.raises(RuntimeError, match="ignored, non-generated files: nova.db"):
        server._ensure_codex_changes_safe(ignored_before, server._secret_fingerprint())


def test_codex_creating_generated_output_is_not_a_violation(repo):
    ignored_before = server._repo_status()[1]
    (repo / "artifacts" / "cartoon").mkdir(parents=True)
    (repo / "artifacts" / "cartoon" / "run.json").write_text("{}")

    assert server._ensure_codex_changes_safe(ignored_before, server._secret_fingerprint()) == []


def test_codex_touching_a_protected_publish_gate_is_refused(repo):
    ignored_before = server._repo_status()[1]
    (repo / "app" / "providers").mkdir(parents=True)
    (repo / "app" / "providers" / "publisher.py").write_text("GATE = False\n")

    with pytest.raises(RuntimeError, match="protected paths: app/providers/publisher.py"):
        server._ensure_codex_changes_safe(ignored_before, server._secret_fingerprint())


def test_merge_action_is_disabled(repo):
    client = TestClient(server.app)
    response = client.post(
        "/action",
        json={"action": "merge_pr", "pr_number": 1},
        headers={"X-NOVA-API-Key": TOKEN},
    )
    assert response.status_code == 403
    assert "merging is disabled" in response.json()["detail"]


def test_write_file_refuses_protected_paths(repo):
    client = TestClient(server.app)
    response = client.post(
        "/action",
        json={"action": "write_file", "path": "app/providers/runway_video.py", "content": "x"},
        headers={"X-NOVA-API-Key": TOKEN},
    )
    assert response.status_code == 403
    assert (repo / "app" / "providers" / "runway_video.py").exists() is False


def test_commit_push_refuses_to_commit_on_main(repo):
    (repo / "app" / "module.py").write_text("VALUE = 3\n")
    client = TestClient(server.app)
    response = client.post(
        "/action",
        json={"action": "commit_push", "message": "change on main"},
        headers={"X-NOVA-API-Key": TOKEN},
    )
    assert response.status_code == 409


def test_codex_binary_is_configurable_and_missing_binary_is_reported(monkeypatch):
    monkeypatch.setattr(server, "CODEX_BIN", "definitely-not-installed-codex-xyz")
    with pytest.raises(HTTPException) as caught:
        server._codex_exec("do one upgrade")
    assert caught.value.status_code == 503


def test_codex_binary_default_is_not_a_personal_path():
    assert "/Users/" not in server.CODEX_BIN
    assert "sante" not in server.CODEX_BIN
