"""The orchestrator CLI: --help never starts a run, exit codes say what happened, metrics close the loop."""
import json
import os
from types import SimpleNamespace

import pytest

import app.agent_memory as agent_memory
import app.cartoon_control as control
import app.cartoon_factory_orchestrator as factory
import app.env as env_module
import app.paths as paths
from app.cartoon_factory_orchestrator import CartoonFactoryOrchestrator, main


ANALYTICS_VARIABLES = (
    "YOUTUBE_API_KEY", "YOUTUBE_ACCESS_TOKEN", "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET",
    "YOUTUBE_REFRESH_TOKEN", "INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_USER_ID", "TIKTOK_ACCESS_TOKEN",
    "TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET", "TIKTOK_REFRESH_TOKEN",
)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Run store, experiment memory and .env lookup all point into tmp_path."""
    monkeypatch.setattr(control, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(agent_memory, "EXPERIMENTS_FILE", tmp_path / "experiments.json")
    monkeypatch.setattr(env_module, "ROOT", tmp_path)
    for name in ANALYTICS_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _published_run(run_id, content_id="yt-1"):
    store = control.CartoonRunStore(run_id)
    store.create({"published_content_map": {"youtube": {content_id: {"published_at": "2026-10-10T00:00:00+00:00"}}}})
    return store


def _unpublished_run(run_id):
    control.CartoonRunStore(run_id).create({})


def test_help_prints_usage_and_never_starts_a_run(monkeypatch, capsys):
    def forbidden(self, *args, **kwargs):
        raise AssertionError("--help must not start a run")

    monkeypatch.setattr(CartoonFactoryOrchestrator, "run", forbidden)
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
    assert "resume" in capsys.readouterr().out


def test_no_arguments_starts_a_new_run(monkeypatch, capsys):
    calls = []

    def fake_run(self, topic="", run_id=None):
        calls.append(topic)
        return SimpleNamespace(run_id="new-run", status="completed", current_stage="16_learning")

    monkeypatch.setattr(CartoonFactoryOrchestrator, "run", fake_run)
    assert main([]) == 0
    assert calls == [""]
    assert json.loads(capsys.readouterr().out)["run_id"] == "new-run"


def test_a_run_that_did_not_complete_exits_non_zero(monkeypatch):
    monkeypatch.setattr(CartoonFactoryOrchestrator, "run",
                        lambda self, topic="", run_id=None: SimpleNamespace(run_id="x", status="failed", current_stage="5_story_engine"))
    assert main(["run"]) == 1


def test_resume_and_repair_dispatch_to_the_orchestrator(monkeypatch):
    seen = []
    monkeypatch.setattr(CartoonFactoryOrchestrator, "resume",
                        lambda self, run_id: seen.append(("resume", run_id)) or SimpleNamespace(
                            run_id=run_id, status="completed", current_stage="16_learning"))
    monkeypatch.setattr(CartoonFactoryOrchestrator, "repair",
                        lambda self, run_id, stage: seen.append(("repair", run_id, stage.value)) or SimpleNamespace(
                            run_id=run_id, status="completed", current_stage="16_learning"))
    assert main(["resume", "run-9"]) == 0
    assert main(["repair", "run-9", "9_animation_engine"]) == 0
    assert seen == [("resume", "run-9"), ("repair", "run-9", "9_animation_engine")]


def test_repair_rejects_an_unknown_stage_with_usage_error(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["repair", "run-9", "not_a_stage"])
    assert raised.value.code == 2


def test_unknown_run_is_reported_with_exit_code_two(capsys):
    assert main(["resume", "does-not-exist"]) == 2
    assert "Unknown cartoon run" in capsys.readouterr().err


def test_status_prints_stage_status_as_json(capsys):
    _unpublished_run("run-status")
    assert main(["status", "run-status"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run_id"] == "run-status" and out["status"] == "running"


def test_metrics_without_credentials_records_metrics_unavailable(capsys):
    _published_run("published-1")
    assert main(["metrics"]) == 0
    loaded = control.CartoonRunStore("published-1").load()
    assert loaded.payload["platform_metrics_status"]["status"] == "metrics_unavailable"
    assert loaded.payload["metrics_pending"] is True
    report = json.loads(capsys.readouterr().out)
    assert report["configured"] is False
    assert report["runs"] == [{"run_id": "published-1", "status": "metrics_unavailable"}]


def test_metrics_covers_every_published_run_and_skips_unpublished_ones(monkeypatch, capsys):
    _published_run("pub-a")
    _published_run("pub-b", content_id="yt-2")
    _unpublished_run("draft")
    monkeypatch.setenv("YOUTUBE_API_KEY", "public-key")
    refreshed = []

    def fake_refresh(self, run_id):
        refreshed.append(run_id)
        return SimpleNamespace(run_id=run_id, status="completed", payload={"platform_metrics_status": {"status": "metrics_collected"}})

    monkeypatch.setattr(CartoonFactoryOrchestrator, "refresh_metrics_and_learning", fake_refresh)
    assert main(["metrics"]) == 0
    assert refreshed == ["pub-a", "pub-b"]


def test_one_failing_run_does_not_stop_the_others(monkeypatch, capsys):
    _published_run("bad")
    _published_run("good")
    monkeypatch.setenv("YOUTUBE_API_KEY", "public-key")
    refreshed = []

    def fake_refresh(self, run_id):
        if run_id == "bad":
            raise RuntimeError("analytics exploded")
        refreshed.append(run_id)
        return SimpleNamespace(run_id=run_id, status="completed", payload={})

    monkeypatch.setattr(CartoonFactoryOrchestrator, "refresh_metrics_and_learning", fake_refresh)
    assert main(["metrics"]) == 1
    assert refreshed == ["good"]
    report = json.loads(capsys.readouterr().out)
    assert {row["run_id"]: row["status"] for row in report["runs"]} == {"bad": "error", "good": "completed"}


def test_metrics_for_one_run_refuses_an_unpublished_run(capsys):
    _unpublished_run("draft")
    assert main(["metrics", "--run-id", "draft"]) == 2


def test_published_run_ids_skips_unreadable_checkpoints(tmp_path):
    _published_run("ok")
    broken = control.RUNS_DIR / "broken" / "run.json"
    broken.parent.mkdir(parents=True)
    broken.write_text("{not json")
    assert factory.published_run_ids() == ["ok"]


def test_cli_loads_the_repo_env_only_at_the_entry_point(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("NOVA_TEST_CLI_SWITCH=from-dotenv\n")
    # Set then unset through monkeypatch so the variable is removed again at teardown.
    monkeypatch.setenv("NOVA_TEST_CLI_SWITCH", "placeholder")
    monkeypatch.delenv("NOVA_TEST_CLI_SWITCH")
    _unpublished_run("status-env")
    assert main(["status", "status-env"]) == 0
    assert os.environ["NOVA_TEST_CLI_SWITCH"] == "from-dotenv"
