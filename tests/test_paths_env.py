"""Repo-relative run paths, legacy absolute-path resolution, and .env loading at entry points only."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import app.cartoon_control as control
import app.env as env_module
import app.paths as paths
from app.paths import resolve_repo_path, to_repo_relative


@pytest.fixture
def repo(monkeypatch, tmp_path):
    """A fake checkout: paths.ROOT and the run store both point inside tmp_path."""
    root = tmp_path / "checkout"
    (root / "artifacts" / "cartoon").mkdir(parents=True)
    monkeypatch.setattr(paths, "ROOT", root)
    monkeypatch.setattr(control, "RUNS_DIR", root / "artifacts" / "cartoon" / "runs")
    return root


def test_paths_inside_the_repo_are_stored_relative(repo):
    inside = repo / "artifacts" / "cartoon" / "episode.mp4"
    assert to_repo_relative(inside) == "artifacts/cartoon/episode.mp4"
    assert to_repo_relative("artifacts/cartoon/episode.mp4") == "artifacts/cartoon/episode.mp4"
    assert to_repo_relative("/elsewhere/episode.mp4") == "/elsewhere/episode.mp4"


def test_relative_paths_resolve_against_the_repo_root(repo):
    assert resolve_repo_path("artifacts/cartoon/episode.mp4") == repo / "artifacts" / "cartoon" / "episode.mp4"


def test_legacy_absolute_path_is_rerooted_when_the_file_exists_here(repo):
    episode = repo / "artifacts" / "cartoon" / "episode.mp4"
    episode.write_bytes(b"video")
    runner_path = "/home/runner/work/nova-ai-influencer/nova-ai-influencer/artifacts/cartoon/episode.mp4"
    assert resolve_repo_path(runner_path) == episode


def test_missing_paths_are_returned_unchanged_so_errors_name_them(repo):
    missing = "/home/someone/else/artifacts/cartoon/gone.mp4"
    assert resolve_repo_path(missing) == Path(missing)


def test_existing_absolute_path_is_returned_as_is(repo, tmp_path):
    outside = tmp_path / "runway_clips" / "shot.mp4"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    assert resolve_repo_path(outside) == outside


def test_run_records_store_repo_relative_paths_and_readers_get_concrete_ones(repo):
    episode = repo / "artifacts" / "cartoon" / "episode.mp4"
    episode.write_bytes(b"video")
    store = control.CartoonRunStore("run-a")
    run = store.create({"video": str(episode), "shot_clips": [str(episode)]})
    on_disk = json.loads(store.path.read_text())
    assert on_disk["payload"]["video"] == "artifacts/cartoon/episode.mp4"
    assert on_disk["payload"]["shot_clips"] == ["artifacts/cartoon/episode.mp4"]

    loaded = control.CartoonRunStore("run-a").load()
    assert loaded.payload["video"] == str(episode)
    assert Path(loaded.payload["video"]).exists()


def test_run_written_on_one_machine_loads_on_another(repo, monkeypatch, tmp_path):
    store = control.CartoonRunStore("run-b")
    store.create({"video": str(repo / "artifacts" / "cartoon" / "episode.mp4")})
    (repo / "artifacts" / "cartoon" / "episode.mp4").write_bytes(b"video")

    other = tmp_path / "other-checkout"
    (other / "artifacts" / "cartoon").mkdir(parents=True)
    (other / "artifacts" / "cartoon" / "episode.mp4").write_bytes(b"video")
    raw = json.loads(store.path.read_text())
    raw["payload"]["video"] = "/home/runner/work/nova-ai-influencer/nova-ai-influencer/artifacts/cartoon/episode.mp4"
    store.path.write_text(json.dumps(raw))
    monkeypatch.setattr(paths, "ROOT", other)
    assert Path(control.CartoonRunStore("run-b").load().payload["video"]) == other / "artifacts" / "cartoon" / "episode.mp4"


def test_load_env_sets_missing_values_and_never_overrides_the_shell(monkeypatch, tmp_path):
    dot_env = tmp_path / ".env"
    dot_env.write_text("NOVA_TEST_FROM_FILE=file-value\nNOVA_TEST_SHELL=file-value\n")
    monkeypatch.delenv("NOVA_TEST_FROM_FILE", raising=False)
    monkeypatch.setenv("NOVA_TEST_SHELL", "shell-value")
    assert env_module.load_env(dot_env) is True
    assert os.environ["NOVA_TEST_FROM_FILE"] == "file-value"
    assert os.environ["NOVA_TEST_SHELL"] == "shell-value"
    monkeypatch.delenv("NOVA_TEST_FROM_FILE", raising=False)


def test_load_env_is_a_no_op_without_a_file(tmp_path):
    assert env_module.load_env(tmp_path / "absent.env") is False


def test_importing_app_modules_never_reads_dotenv():
    """A .env must not leak into imports (tests and libraries); only entry points load it."""
    code = (
        "import dotenv\n"
        "calls = []\n"
        "dotenv.load_dotenv = lambda *a, **k: calls.append(a) or True\n"
        "import app.api.n8n_routes, app.main, app.publish_approved, app.cartoon_factory_orchestrator\n"
        "print('CALLS', len(calls))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=paths.ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "CALLS 0" in result.stdout
