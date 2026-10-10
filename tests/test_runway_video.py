"""Runway execution is opt-in, budget-capped, and produces one clip per shot."""
import json
from pathlib import Path

import pytest

from app.providers import runway_video
from app.providers.runway_video import RunwayExecutionBlocked, RunwayVideoClient, execute_plan


def _plan(tmp_path, durations=(2.4, 3.1)):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return {
        "provider": "runway",
        "shots": [
            {"shot_index": i, "model": "gen-4-turbo", "image": str(image), "duration": d,
             "aspect_ratio": "9:16", "prompt": f"shot {i} prompt"}
            for i, d in enumerate(durations)
        ],
    }


class FakeRunway:
    def __init__(self):
        self.posts = []
        self.polls = {}

    def transport(self, method, url, payload):
        if method == "POST":
            self.posts.append(payload)
            return {"id": f"task-{len(self.posts)}"}
        task_id = url.rsplit("/", 1)[-1]
        self.polls[task_id] = self.polls.get(task_id, 0) + 1
        if self.polls[task_id] < 2:
            return {"status": "RUNNING"}
        return {"status": "SUCCEEDED", "output": [f"https://cdn.example/{task_id}.mp4"]}

    def download(self, url, path):
        path.write_bytes(url.encode())


def _enable(monkeypatch, budget="100"):
    monkeypatch.setenv("NOVA_RUNWAY_EXECUTE", "true")
    monkeypatch.setenv("RUNWAYML_API_SECRET", "test-secret")
    monkeypatch.setenv("NOVA_RUNWAY_MAX_CREDITS", budget)


def test_execution_is_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("NOVA_RUNWAY_EXECUTE", raising=False)
    with pytest.raises(RunwayExecutionBlocked, match="NOVA_RUNWAY_EXECUTE"):
        execute_plan(_plan(tmp_path), tmp_path)


def test_execution_needs_a_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("NOVA_RUNWAY_EXECUTE", "true")
    monkeypatch.delenv("RUNWAYML_API_SECRET", raising=False)
    with pytest.raises(RunwayExecutionBlocked, match="RUNWAYML_API_SECRET"):
        execute_plan(_plan(tmp_path), tmp_path)


def test_budget_counts_billed_seconds_not_shot_seconds(monkeypatch, tmp_path):
    # Two short shots bill 5s each at 5 credits/s = 50 credits.
    _enable(monkeypatch, budget="49")
    with pytest.raises(RunwayExecutionBlocked, match="50 credits"):
        execute_plan(_plan(tmp_path), tmp_path)


def test_plan_executes_and_downloads_one_clip_per_shot(monkeypatch, tmp_path):
    _enable(monkeypatch, budget="50")
    fake = FakeRunway()
    client = RunwayVideoClient("test-secret", transport=fake.transport, downloader=fake.download)
    result = execute_plan(_plan(tmp_path), tmp_path / "clips", client=client, poll_seconds=0)

    assert sorted(result["clips"]) == [0, 1]
    assert all(Path(p).exists() for p in result["clips"].values())
    first = fake.posts[0]
    assert first["model"] == "gen4_turbo"
    assert first["ratio"] == "720:1280"
    assert first["duration"] == 5
    assert first["promptImage"].startswith("data:image/png;base64,")
    assert result["estimated_credits"] == 50


def test_repair_plan_regenerates_only_named_shots(monkeypatch, tmp_path):
    _enable(monkeypatch)
    fake = FakeRunway()
    client = RunwayVideoClient("test-secret", transport=fake.transport, downloader=fake.download)
    plan = _plan(tmp_path, durations=(2.0, 2.0, 2.0))
    repair = {"shots": [dict(plan["shots"][2], prompt="repair")]}
    result = execute_plan(repair, tmp_path / "repair", client=client, poll_seconds=0)
    assert list(result["clips"]) == [2]
    assert len(fake.posts) == 1


def test_unknown_models_need_an_explicit_mapping(monkeypatch, tmp_path):
    _enable(monkeypatch)
    plan = _plan(tmp_path)
    plan["shots"][0]["model"] = "kling-3-standard"
    plan["shots"] = plan["shots"][:1]
    fake = FakeRunway()
    client = RunwayVideoClient("test-secret", transport=fake.transport, downloader=fake.download)
    with pytest.raises(RunwayExecutionBlocked, match="NOVA_RUNWAY_MODEL_MAP"):
        execute_plan(plan, tmp_path, client=client, poll_seconds=0)
    monkeypatch.setenv("NOVA_RUNWAY_MODEL_MAP", json.dumps({"kling-3-standard": "kling3_std"}))
    execute_plan(plan, tmp_path, client=client, poll_seconds=0)
    assert fake.posts[-1]["model"] == "kling3_std"


def test_failed_task_surfaces_the_failure(monkeypatch, tmp_path):
    client = RunwayVideoClient("s", transport=lambda m, u, p: {"id": "t"} if m == "POST" else {"status": "FAILED", "failure": "moderation"})
    with pytest.raises(RuntimeError, match="moderation"):
        client.wait("t", poll_seconds=0)
    assert runway_video.billed_seconds(5.0) == 5 and runway_video.billed_seconds(5.1) == 10
