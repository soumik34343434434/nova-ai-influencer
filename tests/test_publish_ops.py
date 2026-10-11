"""Publishing is idempotent, persists each platform as it succeeds, and fails closed.

These tests run the approval CLI's publish_run offline: HTTP is faked, the run
store and publish queue live in tmp_path, and no real credentials are read.
"""
import json
import urllib.error

import pytest

import app.agent_memory as agent_memory
import app.cartoon_control as control
import app.paths as paths
import app.publish_approved as publish_approved
from app.cartoon_factory_orchestrator import CartoonFactoryOrchestrator
from app.providers.platform_publishers import YouTubeShortsPublisher
from app.providers.publisher import PublishError, PublishJob, PublishResult


@pytest.fixture
def ops(monkeypatch, tmp_path):
    """A run ready for approval, with isolated run store, memory and queue."""
    monkeypatch.setenv("NOVA_ENABLE_REAL_PUBLISH", "true")
    monkeypatch.setattr(control, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(agent_memory, "EXPERIMENTS_FILE", tmp_path / "experiments.json")
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"video-bytes")
    queue = tmp_path / "publish_queue.json"
    queue.write_text(json.dumps([{"run_id": "run-1", "status": "ready_for_review", "video": str(video)}]))
    control.CartoonRunStore("run-1").create({
        "video": str(video),
        "winner": {"candidate": {"title": "PIP vs the button", "caption": "One button."}},
        "final_qa": {"publish_gate": {"decision": "allow"}},
        "content_metadata": {},
    })
    return {"video": video, "queue": queue, "tmp": tmp_path}


class Uploader:
    """Stands in for a live adapter and records every upload attempt."""

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def __call__(self, platform):
        uploader = self

        class Live:
            def publish(self, job, *, dry_run=True):
                uploader.calls.append(job.platform)
                if job.platform in uploader.fail_on:
                    raise PublishError(f"{job.platform} down", retryable=True)
                return PublishResult(job.platform, "published", content_id=f"{job.platform}-1",
                                     published_at="2026-10-10T00:00:00+00:00")

        return Live()


def _job(tmp_path, platform):
    video = tmp_path / "reel.mp4"
    video.write_bytes(b"0123456789")
    return PublishJob.create(video_path=video, platform=platform, title="Title", caption="Caption",
                             experiment_id="exp-1", metadata={"ai_generated": True, "made_for_kids": False},
                             approved=True)


def _publish(ops, platforms, **kwargs):
    return publish_approved.publish_run("run-1", platforms, approved=True, made_for_kids=False,
                                        orchestrator=CartoonFactoryOrchestrator(),
                                        queue_path=ops["queue"], **kwargs)


def test_rerunning_approval_does_not_post_again(monkeypatch, ops):
    uploader = Uploader()
    monkeypatch.setattr(publish_approved, "build_publisher", uploader)
    first = _publish(ops, ["youtube"])
    second = _publish(ops, ["youtube"])
    assert [row["status"] for row in first["results"]] == ["published"]
    assert [row["status"] for row in second["results"]] == ["already_published"]
    assert uploader.calls == ["youtube"]
    assert second["bound"] == {}


def test_force_republish_is_the_only_way_to_upload_again(monkeypatch, ops):
    uploader = Uploader()
    monkeypatch.setattr(publish_approved, "build_publisher", uploader)
    _publish(ops, ["youtube"])
    again = _publish(ops, ["youtube"], force_republish=True)
    assert again["results"][0]["status"] == "published"
    assert uploader.calls == ["youtube", "youtube"]


def test_each_success_is_persisted_before_the_next_platform_fails(monkeypatch, ops):
    monkeypatch.setattr(publish_approved, "build_publisher", Uploader(fail_on={"tiktok"}))
    out = _publish(ops, ["youtube", "tiktok"])
    loaded = control.CartoonRunStore("run-1").load()
    assert "youtube-1" in loaded.payload["published_content_map"]["youtube"]
    assert [row["status"] for row in out["results"]] == ["published", "retryable_error"]


def test_network_failure_on_second_platform_does_not_abort_the_command(monkeypatch, ops):
    class DnsDown:
        def publish(self, job, *, dry_run=True):
            if job.platform == "tiktok":
                raise urllib.error.URLError("Name or service not known")
            return PublishResult(job.platform, "published", content_id="yt-1", published_at="t")

    monkeypatch.setattr(publish_approved, "build_publisher", lambda platform: DnsDown())
    out = _publish(ops, ["youtube", "tiktok"])
    assert [row["status"] for row in out["results"]] == ["published", "retryable_error"]
    assert "youtube" in control.CartoonRunStore("run-1").load().payload["published_content_map"]


def test_urlerror_from_the_http_layer_is_retryable(monkeypatch, tmp_path):
    monkeypatch.setenv("NOVA_ENABLE_REAL_PUBLISH", "true")
    monkeypatch.setenv("YOUTUBE_ACCESS_TOKEN", "yt-token")

    def refused(method, url, headers, body):
        raise urllib.error.URLError("connection refused")

    with pytest.raises(PublishError) as raised:
        YouTubeShortsPublisher(refused).publish(_job(tmp_path, "youtube"), dry_run=False)
    assert raised.value.retryable is True


def test_missing_video_error_names_the_expected_path(monkeypatch, ops):
    ops["video"].unlink()
    monkeypatch.setattr(publish_approved, "build_publisher", Uploader())
    with pytest.raises(FileNotFoundError, match="expected .*episode.mp4"):
        _publish(ops, ["youtube"])


def test_publish_queue_row_reflects_the_outcome(monkeypatch, ops):
    monkeypatch.setattr(publish_approved, "build_publisher", Uploader())
    _publish(ops, ["youtube"])
    row = json.loads(ops["queue"].read_text())[0]
    assert row["status"] == "published"
    assert row["platforms"]["youtube"]["status"] == "published"
    assert len(row["video_sha256"]) == 64


def test_dry_run_never_touches_the_queue_or_the_run(ops):
    before = ops["queue"].read_text()
    out = publish_approved.publish_run("run-1", ["youtube"], approved=False, made_for_kids=None,
                                       orchestrator=CartoonFactoryOrchestrator(), queue_path=ops["queue"])
    assert out["live"] is False and out["results"][0]["status"] == "dry_run"
    assert ops["queue"].read_text() == before
    assert "published_content_map" not in control.CartoonRunStore("run-1").load().payload


def test_unconfirmed_earlier_upload_blocks_until_forced(monkeypatch, ops):
    queue = json.loads(ops["queue"].read_text())
    queue[0]["platforms"] = {"youtube": {"status": "publishing"}}
    ops["queue"].write_text(json.dumps(queue))
    uploader = Uploader()
    monkeypatch.setattr(publish_approved, "build_publisher", uploader)
    out = _publish(ops, ["youtube"])
    assert out["results"][0]["status"] == "blocked"
    assert "previous_publish_unconfirmed" in out["results"][0]["error"]
    assert uploader.calls == []


def test_video_changed_after_review_is_refused(monkeypatch, ops):
    monkeypatch.setattr(publish_approved, "build_publisher", Uploader())
    _publish(ops, ["youtube"])
    queue = json.loads(ops["queue"].read_text())
    queue[0]["video_sha256"] = "0" * 64
    ops["queue"].write_text(json.dumps(queue))
    with pytest.raises(PermissionError, match="video_changed_since_review"):
        _publish(ops, ["tiktok"])


def test_github_runner_absolute_path_resolves_to_the_downloaded_episode(monkeypatch, ops):
    repo = ops["tmp"] / "checkout"
    episode = repo / "artifacts" / "cartoon" / "episode.mp4"
    episode.parent.mkdir(parents=True)
    episode.write_bytes(b"video-bytes")
    monkeypatch.setattr(paths, "ROOT", repo)
    record = control.RUNS_DIR / "run-1" / "run.json"
    raw = json.loads(record.read_text())
    raw["payload"]["video"] = "/home/runner/work/nova-ai-influencer/nova-ai-influencer/artifacts/cartoon/episode.mp4"
    record.write_text(json.dumps(raw))
    monkeypatch.setattr(publish_approved, "build_publisher", Uploader())
    out = _publish(ops, ["youtube"])
    assert out["results"][0]["status"] == "published"
