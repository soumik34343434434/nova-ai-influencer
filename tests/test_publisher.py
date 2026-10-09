import json

import pytest

from app.providers.publisher import (
    DryRunPublisher, PublishError, PublishJob, PublishOutbox,
    PublishValidationError, assert_publish_allowed, classify_publish_error,
    publish_review_item, real_publish_enabled, validate_publish_job,
)


def make_job(video, **kwargs):
    values = dict(video_path=video, platform="instagram", title="Test", caption="Caption",
                  experiment_id="exp-1", metadata={"ai_generated": True})
    values.update(kwargs)
    return PublishJob.create(**values)


def test_dry_run_is_deterministic_and_never_uploads(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    job = make_job(video)
    first = DryRunPublisher("instagram").publish(job)
    second = DryRunPublisher("instagram").publish(job)
    assert first.status == "dry_run"
    assert first.content_id == second.content_id == job.idempotency_key
    assert DryRunPublisher("instagram").publish(job, dry_run=False).status == "not_configured"


@pytest.mark.parametrize("kwargs,expected", [
    ({"title": ""}, "title_required"),
    ({"caption": ""}, "caption_required"),
    ({"experiment_id": ""}, "experiment_id_required"),
    ({"metadata": {}}, "metadata_ai_generated_required"),
])
def test_validation_failures(tmp_path, kwargs, expected):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    values = dict(video_path=video, platform="instagram", title="Title", caption="Caption",
                  experiment_id="exp", metadata={"ai_generated": True})
    values.update(kwargs)
    with pytest.raises(PublishValidationError, match=expected):
        PublishJob.create(**values)


def test_missing_video_and_invalid_schedule_are_rejected(tmp_path):
    with pytest.raises(PublishValidationError, match="video_missing_or_empty"):
        make_job(tmp_path / "missing.mp4")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    with pytest.raises(PublishValidationError, match="scheduled_at_timezone_required"):
        make_job(video, scheduled_at="2026-10-10T09:00:00")


def test_outbox_duplicate_idempotency_and_scheduling_transitions(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    job = make_job(video, scheduled_at="2026-10-10T09:00:00+05:30")
    outbox = PublishOutbox(tmp_path / "queue.json")
    first, inserted = outbox.enqueue(job)
    duplicate, duplicate_inserted = outbox.enqueue(job)
    assert inserted is True and duplicate_inserted is False
    assert first["status"] == "scheduled"
    assert duplicate["idempotency_key"] == first["idempotency_key"]
    assert outbox.transition(job.idempotency_key, "ready_for_review")["status"] == "ready_for_review"
    assert outbox.transition(job.idempotency_key, "publishing")["attempts"] == 1
    assert outbox.transition(job.idempotency_key, "published")["status"] == "published"
    assert len(json.loads((tmp_path / "queue.json").read_text())) == 1


def test_retryable_and_permanent_error_classification():
    assert classify_publish_error(TimeoutError("timeout")) == "retryable_error"
    assert classify_publish_error(PublishError("busy", retryable=True)) == "retryable_error"
    assert classify_publish_error(PublishError("bad request", retryable=False)) == "failed"
    assert classify_publish_error(ValueError("invalid")) == "failed"


def test_live_publish_requires_approval_and_remains_unavailable(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    job = make_job(video)
    monkeypatch.setenv("NOVA_ENABLE_REAL_PUBLISH", "true")
    assert real_publish_enabled()
    with pytest.raises(PermissionError, match="publish_approval_required"):
        assert_publish_allowed(job, dry_run=False)
    with pytest.raises(PermissionError, match="no_real_publisher_configured"):
        assert_publish_allowed(make_job(video, approved=True), dry_run=False)
    assert classify_publish_error(ConnectionError("offline")) == "retryable_error"


def test_review_item_creates_platform_bindings(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    result = publish_review_item(video_path=video, experiment_id="exp-3", title="Test",
                                 caption="Caption", platforms=["instagram", "youtube"])
    assert result["status"] == "dry_run"
    assert {row["platform"] for row in result["results"]} == {"instagram", "youtube"}
    assert all(row["content_id"].startswith("pub_") for row in result["results"])
