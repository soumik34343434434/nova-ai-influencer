import json
import subprocess

import pytest

import scripts.state_sync as state_sync


def _git(repo, *args):
    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True, env=env)


@pytest.fixture
def state_repo(tmp_path):
    repo = tmp_path / "factory-state"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "factory-state")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "Create factory-state branch")
    return repo


@pytest.fixture
def cartoon_dir(tmp_path):
    cartoon = tmp_path / "artifacts" / "cartoon"
    (cartoon / "runs" / "run1").mkdir(parents=True)
    (cartoon / "strategy_memory.json").write_text('{"wins": 1}')
    (cartoon / "publish_queue.json").write_text("[]")
    (cartoon / "run1_factory.edit_plan.json").write_text("{}")
    (cartoon / "runs" / "run1" / "run.json").write_text(
        json.dumps({"run_id": "run1", "payload": {"video": "/home/runner/work/x/artifacts/cartoon/run1_factory.mp4"}})
    )
    (cartoon / "run1_factory.mp4").write_bytes(b"not-really-an-mp4")
    return cartoon


def _tracked(repo):
    return set(_git(repo, "ls-files").stdout.split())


def test_push_stores_memory_and_run_records_but_not_episode_files(state_repo, cartoon_dir):
    assert state_sync.push(state_repo, "Update factory state", cartoon_dir) is True

    tracked = _tracked(state_repo)
    assert "state/cartoon/strategy_memory.json" in tracked
    assert "state/cartoon/publish_queue.json" in tracked
    assert "state/cartoon/runs/run1/run.json" in tracked
    assert not any("_factory" in path for path in tracked)
    assert not any(path.endswith(".mp4") for path in tracked)


def test_push_commits_only_real_changes(state_repo, cartoon_dir):
    state_sync.push(state_repo, "first", cartoon_dir)
    commits_after_first = int(_git(state_repo, "rev-list", "--count", "HEAD").stdout)

    assert state_sync.push(state_repo, "no change", cartoon_dir) is False
    assert int(_git(state_repo, "rev-list", "--count", "HEAD").stdout) == commits_after_first

    (cartoon_dir / "strategy_memory.json").write_text('{"wins": 2}')
    assert state_sync.push(state_repo, "second", cartoon_dir) is True
    assert int(_git(state_repo, "rev-list", "--count", "HEAD").stdout) == commits_after_first + 1


def test_pull_restores_memory_into_a_fresh_checkout(state_repo, cartoon_dir, tmp_path):
    state_sync.push(state_repo, "seed", cartoon_dir)

    fresh = tmp_path / "fresh" / "artifacts" / "cartoon"
    counts = state_sync.pull(state_repo, fresh)

    assert counts == {"memory": 2, "runs": 1}
    assert json.loads((fresh / "strategy_memory.json").read_text()) == {"wins": 1}
    assert (fresh / "runs" / "run1" / "run.json").is_file()
    assert not (fresh / "run1_factory.edit_plan.json").exists()


def test_pull_without_state_branch_content_starts_empty(state_repo, tmp_path):
    target = tmp_path / "artifacts" / "cartoon"
    assert state_sync.pull(state_repo, target) == {"memory": 0, "runs": 0}
    assert not target.exists()


def test_episode_video_uses_the_file_name_under_artifacts_cartoon(state_repo, cartoon_dir):
    state_sync.push(state_repo, "seed", cartoon_dir)

    path = state_sync.episode_video(state_repo, "run1")

    assert path == "artifacts/cartoon/run1_factory.mp4"


def test_episode_video_fails_clearly_for_unknown_run(state_repo):
    with pytest.raises(SystemExit, match="no run record for missing"):
        state_sync.episode_video(state_repo, "missing")


def test_ensure_branch_creates_the_state_branch_once(tmp_path, monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)

    assert state_sync.ensure_branch(str(remote)) is True
    heads = subprocess.run(
        ["git", "ls-remote", "--heads", str(remote), "factory-state"], text=True, capture_output=True, check=True
    ).stdout
    assert "refs/heads/factory-state" in heads

    assert state_sync.ensure_branch(str(remote)) is False


def test_auth_header_is_only_built_from_a_token(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert state_sync._auth_args() == []

    monkeypatch.setenv("GH_TOKEN", "t0ken")
    args = state_sync._auth_args()
    assert args[0] == "-c" and args[1].startswith("http.https://github.com/.extraheader=AUTHORIZATION: basic ")
    assert "t0ken" not in " ".join(args)
