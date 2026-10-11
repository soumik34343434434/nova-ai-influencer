"""Structural checks for the GitHub workflow files.

actionlint validates syntax in CI; these tests pin the rules that matter for
safety and cost: least-privilege permissions, timeouts, shared state
concurrency, no leaked Runway spend, and the approval gate on publishing.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

ALLOWED_ACTIONS = {
    "actions/checkout": "v6",
    "actions/setup-python": "v6",
    "actions/cache": "v5",
}
STATE_WORKFLOWS = {"cartoon-factory.yml", "publish-episode.yml", "metrics-learning.yml"}


def _load(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    # PyYAML reads the bare key ``on`` as a boolean, so accept either spelling.
    data["on"] = data.get("on", data.get(True))
    return data


def _all_workflows() -> list[str]:
    return sorted(p.name for p in WORKFLOWS.glob("*.yml"))


def _steps(job: dict):
    return job.get("steps", [])


@pytest.mark.parametrize("name", _all_workflows())
def test_every_workflow_has_explicit_permissions_and_job_timeouts(name):
    workflow = _load(name)
    assert "permissions" in workflow, f"{name} needs top-level permissions"
    assert workflow["permissions"] != "write-all"
    for job_name, job in workflow["jobs"].items():
        assert "timeout-minutes" in job, f"{name}:{job_name} needs timeout-minutes"
        if "permissions" in job:
            assert job["permissions"] != "write-all"


@pytest.mark.parametrize("name", _all_workflows())
def test_actions_use_the_agreed_major_versions(name):
    for job in _load(name)["jobs"].values():
        for step in _steps(job):
            uses = step.get("uses")
            if not uses:
                continue
            repo, _, ref = uses.partition("@")
            if repo in ALLOWED_ACTIONS:
                assert ref == ALLOWED_ACTIONS[repo], f"{name}: {uses}"
            assert ref.startswith("v"), f"{name}: {uses} is not a version tag"


@pytest.mark.parametrize("name", sorted(STATE_WORKFLOWS))
def test_state_workflows_share_one_serialised_concurrency_group(name):
    concurrency = _load(name)["concurrency"]
    assert concurrency["group"] == "nova-factory-state"
    assert concurrency["cancel-in-progress"] is False


def test_legacy_daily_content_has_no_schedule_but_keeps_manual_dispatch():
    triggers = _load("daily-content.yml")["on"]
    assert "schedule" not in triggers
    assert "workflow_dispatch" in triggers


def test_factory_schedule_is_daily_and_dispatchable():
    triggers = _load("cartoon-factory.yml")["on"]
    assert "workflow_dispatch" in triggers
    assert triggers["schedule"] == [{"cron": "47 3 * * *"}]


def test_ci_runs_on_every_branch_and_on_pull_requests():
    triggers = _load("ci.yml")["on"]
    assert triggers["push"] == {"branches": ["**"]}
    assert "pull_request" in triggers


def test_ci_keeps_the_persona_backtest_out_of_the_blocking_test_job():
    jobs = _load("ci.yml")["jobs"]
    assert jobs["persona-backtest"]["continue-on-error"] is True
    assert "continue-on-error" not in jobs["test"]
    test_commands = " ".join(str(step.get("run", "")) for step in _steps(jobs["test"]))
    assert "app.orchestrator" not in test_commands


def test_factory_job_gets_only_the_research_key_and_no_runway_or_publish_credentials():
    job = _load("cartoon-factory.yml")["jobs"]["episode"]
    text = yaml.safe_dump(job)
    assert "secrets.YOUTUBE_API_KEY" in text
    for forbidden in ("RUNWAY", "NOVA_RUNWAY", "TIKTOK_", "INSTAGRAM_", "_REFRESH_TOKEN"):
        assert forbidden not in text, f"factory job must not reference {forbidden}"


def test_factory_publishes_nothing_and_keeps_episodes_out_of_public_artifacts():
    text = (WORKFLOWS / "cartoon-factory.yml").read_text(encoding="utf-8")
    assert "upload-artifact" not in text
    assert "gh release create" in text and "--draft" in text


def test_publishing_is_manual_and_behind_the_publish_environment():
    workflow = _load("publish-episode.yml")
    assert list(workflow["on"]) == ["workflow_dispatch"]
    assert workflow["jobs"]["publish"]["environment"] == "publish"
    assert "--approve" in " ".join(str(s.get("run", "")) for s in _steps(workflow["jobs"]["publish"]))


def test_metrics_job_runs_the_cli_and_is_scheduled_away_from_the_top_of_the_hour():
    workflow = _load("metrics-learning.yml")
    cron = workflow["on"]["schedule"][0]["cron"]
    assert not cron.startswith("0 ")
    commands = " ".join(str(s.get("run", "")) for s in _steps(workflow["jobs"]["metrics"]))
    assert "python -m app.cartoon_factory_orchestrator metrics" in commands


def test_only_known_optional_switches_are_wired_from_repository_variables():
    text = (WORKFLOWS / "cartoon-factory.yml").read_text(encoding="utf-8")
    wired = {part for part in text.split() if part.startswith("NOVA_") and part.endswith(":")}
    assert wired == {
        "NOVA_VOICE_BACKEND:",
        "NOVA_CARTOON_ANIMATION_BACKEND:",
        "NOVA_MAX_REPAIR_ATTEMPTS:",
    }
