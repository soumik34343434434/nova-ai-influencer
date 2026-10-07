from pathlib import Path

from app.providers.publisher import DryRunPublisher, publish_review_item


def test_dry_run_publisher_is_safe_and_deterministic(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    publisher = DryRunPublisher("instagram")
    first = publisher.publish(video, experiment_id="exp-1", dry_run=True)
    second = publisher.publish(video, experiment_id="exp-1", dry_run=True)
    assert first.status == "dry_run"
    assert first.content_id == second.content_id
    assert first.provider == "dry_run"


def test_dry_run_does_not_publish(tmp_path):
    missing = tmp_path / "missing.mp4"
    result = DryRunPublisher("youtube").publish(missing, experiment_id="exp-2")
    assert result.status == "failed"


def test_review_item_creates_platform_bindings(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    result = publish_review_item(
        video_path=video,
        experiment_id="exp-3",
        title="Test",
        caption="Caption",
        platforms=["instagram", "youtube"],
    )
    assert result["status"] == "dry_run"
    assert {row["platform"] for row in result["results"]} == {"instagram", "youtube"}
    assert all(row["content_id"].startswith("dryrun_") for row in result["results"])
