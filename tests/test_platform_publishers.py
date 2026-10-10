"""Live platform adapters stay fail-closed and speak each platform's upload protocol."""
import json

import pytest

from app.providers.platform_publishers import (
    InstagramReelsPublisher,
    TikTokPublisher,
    YouTubeShortsPublisher,
    chunk_plan,
)
from app.providers.publisher import (
    DryRunPublisher,
    PublishJob,
    PublishValidationError,
    assert_publish_allowed,
    build_publisher,
)


def _job(tmp_path, platform, *, approved=True, made_for_kids=False):
    video = tmp_path / "reel.mp4"
    video.write_bytes(b"0123456789" * 100)
    metadata = {"ai_generated": True}
    if made_for_kids is not None:
        metadata["made_for_kids"] = made_for_kids
    return PublishJob.create(video_path=video, platform=platform, title="PIP vs the button",
                             caption="One button. Twelve sandwiches.", experiment_id="exp-1",
                             metadata=metadata, approved=approved)


class FakeHttp:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append({"method": method, "url": url, "headers": headers, "body": body})
        status, resp_headers, payload = self.responses.pop(0)
        return status, resp_headers, json.dumps(payload).encode() if not isinstance(payload, bytes) else payload


@pytest.fixture
def live_env(monkeypatch):
    monkeypatch.setenv("NOVA_ENABLE_REAL_PUBLISH", "true")
    monkeypatch.setenv("YOUTUBE_ACCESS_TOKEN", "yt-token")
    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "ig-token")
    monkeypatch.setenv("INSTAGRAM_USER_ID", "17840000")
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tt-token")


def test_factory_returns_dry_run_unless_enabled_and_configured(monkeypatch, live_env):
    assert isinstance(build_publisher("youtube"), YouTubeShortsPublisher)
    monkeypatch.setenv("NOVA_ENABLE_REAL_PUBLISH", "false")
    assert isinstance(build_publisher("youtube"), DryRunPublisher)
    monkeypatch.setenv("NOVA_ENABLE_REAL_PUBLISH", "true")
    monkeypatch.delenv("TIKTOK_ACCESS_TOKEN")
    assert isinstance(build_publisher("tiktok"), DryRunPublisher)


def test_missing_credentials_name_what_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("NOVA_ENABLE_REAL_PUBLISH", "true")
    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("INSTAGRAM_USER_ID", raising=False)
    with pytest.raises(PermissionError, match="no_real_publisher_configured:instagram:missing=INSTAGRAM_ACCESS_TOKEN"):
        assert_publish_allowed(_job(tmp_path, "instagram"), dry_run=False)


def test_unapproved_job_never_uploads(live_env, tmp_path):
    http = FakeHttp([])
    with pytest.raises(PermissionError, match="publish_approval_required"):
        YouTubeShortsPublisher(http).publish(_job(tmp_path, "youtube", approved=False), dry_run=False)
    assert http.calls == []


def test_audience_classification_must_be_explicit(live_env, tmp_path):
    http = FakeHttp([])
    with pytest.raises(PublishValidationError, match="audience_classification_required"):
        YouTubeShortsPublisher(http).publish(_job(tmp_path, "youtube", made_for_kids=None), dry_run=False)
    assert http.calls == []


def test_youtube_resumable_upload_declares_synthetic_media(live_env, tmp_path):
    http = FakeHttp([(200, {"location": "https://upload.example/session"}, {}), (200, {}, {"id": "yt123"})])
    result = YouTubeShortsPublisher(http).publish(_job(tmp_path, "youtube"), dry_run=False)
    assert result.status == "published" and result.content_id == "yt123"
    meta = json.loads(http.calls[0]["body"])
    assert meta["status"]["containsSyntheticMedia"] is True
    assert meta["status"]["selfDeclaredMadeForKids"] is False
    assert meta["status"]["privacyStatus"] == "private"
    assert "AI-generated" in meta["snippet"]["description"]
    assert http.calls[1]["method"] == "PUT" and http.calls[1]["url"] == "https://upload.example/session"


def test_instagram_reel_uploads_polls_and_publishes(live_env, tmp_path):
    http = FakeHttp([
        (200, {}, {"id": "c1", "uri": "https://rupload.facebook.com/ig-api-upload/v26.0/c1"}),
        (200, {}, {"success": True}),
        (200, {}, {"status_code": "IN_PROGRESS"}),
        (200, {}, {"status_code": "FINISHED"}),
        (200, {}, {"id": "media9"}),
    ])
    result = InstagramReelsPublisher(http, poll_seconds=0).publish(_job(tmp_path, "instagram"), dry_run=False)
    assert result.content_id == "media9"
    assert "media_type=REELS" in http.calls[0]["url"] and "upload_type=resumable" in http.calls[0]["url"]
    from urllib.parse import parse_qs, urlparse
    caption = parse_qs(urlparse(http.calls[0]["url"]).query)["caption"][0]
    assert caption.endswith("AI-generated fictional cartoon.")
    assert http.calls[1]["headers"]["Authorization"] == "OAuth ig-token"
    assert "creation_id=c1" in http.calls[-1]["url"]


def test_tiktok_direct_post_flags_ai_content(live_env, tmp_path):
    http = FakeHttp([
        (200, {}, {"data": {"publish_id": "p1", "upload_url": "https://upload.tiktok.example/p1"}, "error": {"code": "ok"}}),
        (200, {}, {}),
        (200, {}, {"data": {"status": "PROCESSING_UPLOAD"}}),
        (200, {}, {"data": {"status": "PUBLISH_COMPLETE", "publicaly_available_post_id": [777]}}),
    ])
    result = TikTokPublisher(http, poll_seconds=0).publish(_job(tmp_path, "tiktok"), dry_run=False)
    assert result.content_id == "777"
    init = json.loads(http.calls[0]["body"])
    assert init["post_info"]["is_aigc"] is True
    assert init["post_info"]["privacy_level"] == "SELF_ONLY"
    assert init["source_info"]["total_chunk_count"] == 1
    assert http.calls[1]["headers"]["Content-Range"] == "bytes 0-999/1000"


def test_platform_errors_are_classified_for_retry(live_env, tmp_path):
    from app.providers.publisher import PublishError

    http = FakeHttp([(503, {}, {"error": "busy"})])
    with pytest.raises(PublishError) as raised:
        YouTubeShortsPublisher(http).publish(_job(tmp_path, "youtube"), dry_run=False)
    assert raised.value.retryable is True


def test_large_tiktok_files_are_chunked():
    assert chunk_plan(5_000_000) == (5_000_000, 1)
    size, count = chunk_plan(100 * 1024 * 1024)
    assert size == 10 * 1024 * 1024 and count == 10


def test_publish_approved_binds_live_results_to_the_run(monkeypatch, tmp_path, live_env):
    import app.agent_memory as agent_memory
    import app.cartoon_control as control
    import app.publish_approved as publish_approved
    from app.cartoon_factory_orchestrator import CartoonFactoryOrchestrator
    from app.providers.publisher import PublishResult

    monkeypatch.setattr(control, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(agent_memory, "EXPERIMENTS_FILE", tmp_path / "experiments.json")
    video = tmp_path / "reel.mp4"
    video.write_bytes(b"video-bytes")
    store = control.CartoonRunStore("approved-run")
    store.create({
        "video": str(video),
        "winner": {"candidate": {"title": "PIP vs the button", "caption": "One button."}},
        "final_qa": {"publish_gate": {"decision": "allow"}},
    })

    class FakeLive:
        def publish(self, job, *, dry_run=True):
            assert job.approved and not dry_run and job.metadata["made_for_kids"] is False
            return PublishResult(job.platform, "published", content_id=f"{job.platform}-id", published_at="2026-10-10T00:00:00+00:00")

    monkeypatch.setattr(publish_approved, "build_publisher", lambda platform: FakeLive())
    out = publish_approved.publish_run("approved-run", ["youtube", "tiktok"], approved=True, made_for_kids=False,
                                       orchestrator=CartoonFactoryOrchestrator())
    assert out["live"] is True
    loaded = control.CartoonRunStore("approved-run").load()
    assert set(loaded.payload["published_content_map"]) == {"youtube", "tiktok"}
    assert "youtube-id" in loaded.payload["published_content_map"]["youtube"]
