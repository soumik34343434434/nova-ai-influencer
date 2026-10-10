"""Live upload adapters for YouTube Shorts, Instagram Reels and TikTok.

Every adapter is fail-closed. A real upload happens only when all of these hold
(see ``publisher.assert_publish_allowed``):

* the job was explicitly approved by a person (``approved=True``);
* ``NOVA_ENABLE_REAL_PUBLISH=true``;
* the platform's credentials are configured (``required_env`` below);
* the job carries an explicit audience decision (``metadata["made_for_kids"]``),
  because platform audience classification is a human review step.

Each upload declares the content as AI-generated wherever the platform API
supports it (YouTube ``containsSyntheticMedia``, TikTok ``is_aigc``) and always
in the caption. HTTP is injectable so the adapters are testable offline.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# (method, url, headers, body) -> (status, response headers, response body)
Http = Callable[[str, str, dict[str, str], "bytes | None"], tuple[int, dict[str, str], bytes]]

AI_DISCLOSURE = "AI-generated fictional cartoon."


def urllib_http(method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in (exc.headers or {}).items()}, exc.read() or b""


def _json(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body.decode() or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def disclosed_caption(caption: str) -> str:
    text = " ".join(str(caption).split())
    return text if AI_DISCLOSURE.lower() in text.lower() else f"{text}\n\n{AI_DISCLOSURE}".strip()


class LivePublisherBase:
    platform = ""
    provider = ""
    required_env: tuple[str, ...] = ()

    def __init__(self, http: Http | None = None, *, poll_seconds: float = 5.0, timeout_seconds: float = 600.0):
        self.http = http or urllib_http
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    @classmethod
    def configured(cls) -> bool:
        return all(os.getenv(name, "").strip() for name in cls.required_env)

    def publish(self, job: Any, *, dry_run: bool = True) -> Any:
        from app.providers.publisher import (
            PublishError,
            PublishResult,
            PublishValidationError,
            assert_publish_allowed,
            validate_publish_job,
        )

        validate_publish_job(job)
        if dry_run:
            return PublishResult(job.platform, "dry_run", content_id=job.idempotency_key,
                                 provider=f"{self.provider}:dry_run", idempotency_key=job.idempotency_key)
        assert_publish_allowed(job, dry_run=False)
        if not isinstance(job.metadata.get("made_for_kids"), bool):
            raise PublishValidationError("audience_classification_required")
        try:
            content_id = self._upload(job)
        except PublishError:
            raise
        except (TimeoutError, ConnectionError) as exc:
            raise PublishError(f"{self.platform}: {exc}", retryable=True) from exc
        return PublishResult(job.platform, "published", content_id=content_id,
                             published_at=datetime.now(timezone.utc).isoformat(),
                             provider=self.provider, idempotency_key=job.idempotency_key)

    def _check(self, status: int, body: bytes, step: str) -> dict[str, Any]:
        from app.providers.publisher import PublishError

        if status >= 400:
            raise PublishError(f"{self.platform} {step} failed with HTTP {status}: {body[:300]!r}",
                               retryable=status in {408, 429} or status >= 500)
        return _json(body)

    def _upload(self, job: Any) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class YouTubeShortsPublisher(LivePublisherBase):
    """YouTube Data API v3 resumable upload (Shorts are vertical videos under 60s)."""

    platform = "youtube"
    provider = "youtube_data_api_v3"
    required_env = ("YOUTUBE_ACCESS_TOKEN",)
    UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"

    def _upload(self, job: Any) -> str:
        from app.providers.publisher import PublishError

        token = os.environ["YOUTUBE_ACCESS_TOKEN"].strip()
        data = Path(job.video_path).read_bytes()
        metadata = {
            "snippet": {
                "title": job.title[:100],
                "description": disclosed_caption(job.caption),
                "categoryId": os.getenv("YOUTUBE_CATEGORY_ID", "1"),  # Film & Animation
            },
            "status": {
                "privacyStatus": os.getenv("YOUTUBE_PRIVACY_STATUS", "private"),
                "selfDeclaredMadeForKids": bool(job.metadata["made_for_kids"]),
                "containsSyntheticMedia": True,
            },
        }
        status, headers, body = self.http("POST", self.UPLOAD_URL, {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(len(data)),
        }, json.dumps(metadata).encode())
        self._check(status, body, "upload session")
        location = headers.get("location", "")
        if not location:
            raise PublishError("youtube upload session returned no Location", retryable=True)
        status, _, body = self.http("PUT", location, {"Authorization": f"Bearer {token}", "Content-Type": "video/mp4"}, data)
        video_id = str(self._check(status, body, "upload").get("id", ""))
        if not video_id:
            raise PublishError("youtube upload returned no video id", retryable=False)
        return video_id


class InstagramReelsPublisher(LivePublisherBase):
    """Instagram Graph API Reels via resumable upload, container polling and publish."""

    platform = "instagram"
    provider = "instagram_graph_api"
    required_env = ("INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_USER_ID")

    def _upload(self, job: Any) -> str:
        from app.providers.publisher import PublishError

        token = os.environ["INSTAGRAM_ACCESS_TOKEN"].strip()
        user = os.environ["INSTAGRAM_USER_ID"].strip()
        version = os.getenv("INSTAGRAM_GRAPH_VERSION", "v26.0")
        graph = f"https://graph.facebook.com/{version}"
        data = Path(job.video_path).read_bytes()
        params = urllib.parse.urlencode({
            "media_type": "REELS",
            "upload_type": "resumable",
            "caption": disclosed_caption(job.caption),
            "share_to_feed": "true",
            "access_token": token,
        })
        status, _, body = self.http("POST", f"{graph}/{user}/media?{params}", {}, None)
        created = self._check(status, body, "container create")
        container = str(created.get("id", ""))
        upload_uri = str(created.get("uri") or f"https://rupload.facebook.com/ig-api-upload/{version}/{container}")
        if not container:
            raise PublishError("instagram returned no container id", retryable=True)
        status, _, body = self.http("POST", upload_uri, {
            "Authorization": f"OAuth {token}",
            "offset": "0",
            "file_size": str(len(data)),
        }, data)
        self._check(status, body, "video upload")

        started = time.monotonic()
        while True:
            query = urllib.parse.urlencode({"fields": "status_code", "access_token": token})
            status, _, body = self.http("GET", f"{graph}/{container}?{query}", {}, None)
            state = str(self._check(status, body, "container status").get("status_code", "")).upper()
            if state == "FINISHED":
                break
            if state in {"ERROR", "EXPIRED"}:
                raise PublishError(f"instagram container {container} ended in {state}", retryable=False)
            if time.monotonic() - started > self.timeout_seconds:
                raise TimeoutError(f"instagram container {container} still processing")
            time.sleep(self.poll_seconds)

        query = urllib.parse.urlencode({"creation_id": container, "access_token": token})
        status, _, body = self.http("POST", f"{graph}/{user}/media_publish?{query}", {}, None)
        media_id = str(self._check(status, body, "publish").get("id", ""))
        if not media_id:
            raise PublishError("instagram publish returned no media id", retryable=True)
        return media_id


class TikTokPublisher(LivePublisherBase):
    """TikTok Content Posting API direct post with chunked file upload."""

    platform = "tiktok"
    provider = "tiktok_content_posting_api"
    required_env = ("TIKTOK_ACCESS_TOKEN",)
    INIT_URL = "https://open.tiktokapis.com/v2/post/publish/video/init/"
    STATUS_URL = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
    SINGLE_CHUNK_LIMIT = 64 * 1024 * 1024
    CHUNK = 10 * 1024 * 1024

    def _upload(self, job: Any) -> str:
        from app.providers.publisher import PublishError

        token = os.environ["TIKTOK_ACCESS_TOKEN"].strip()
        data = Path(job.video_path).read_bytes()
        size = len(data)
        chunk = size if size <= self.SINGLE_CHUNK_LIMIT else self.CHUNK
        chunks = 1 if chunk == size else max(1, size // chunk)
        auth = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"}
        payload = {
            "post_info": {
                "title": disclosed_caption(job.caption)[:2200],
                # Unaudited TikTok apps may only post privately.
                "privacy_level": os.getenv("TIKTOK_PRIVACY_LEVEL", "SELF_ONLY"),
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
                "is_aigc": True,
            },
            "source_info": {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk, "total_chunk_count": chunks},
        }
        status, _, body = self.http("POST", self.INIT_URL, auth, json.dumps(payload).encode())
        init = self._check(status, body, "init")
        error = init.get("error", {})
        if isinstance(error, dict) and error.get("code") not in (None, "ok"):
            raise PublishError(f"tiktok init error: {error}", retryable=False)
        publish_id = str(init.get("data", {}).get("publish_id", ""))
        upload_url = str(init.get("data", {}).get("upload_url", ""))
        if not publish_id or not upload_url:
            raise PublishError("tiktok init returned no publish_id/upload_url", retryable=True)

        for index in range(chunks):
            start = index * chunk
            end = size if index == chunks - 1 else start + chunk  # last chunk takes the remainder
            status, _, body = self.http("PUT", upload_url, {
                "Content-Type": "video/mp4",
                "Content-Range": f"bytes {start}-{end - 1}/{size}",
            }, data[start:end])
            self._check(status, body, f"chunk {index + 1}/{chunks}")

        started = time.monotonic()
        while True:
            status, _, body = self.http("POST", self.STATUS_URL, auth, json.dumps({"publish_id": publish_id}).encode())
            info = self._check(status, body, "status").get("data", {})
            state = str(info.get("status", "")).upper()
            if state == "PUBLISH_COMPLETE":
                ids = info.get("publicaly_available_post_id") or info.get("publicly_available_post_id") or []
                return str(ids[0]) if ids else publish_id
            if state == "FAILED":
                raise PublishError(f"tiktok publish failed: {info.get('fail_reason', 'unknown')}", retryable=False)
            if time.monotonic() - started > self.timeout_seconds:
                raise TimeoutError(f"tiktok publish {publish_id} still processing")
            time.sleep(self.poll_seconds)


LIVE_PUBLISHERS: dict[str, type[LivePublisherBase]] = {
    "youtube": YouTubeShortsPublisher,
    "instagram": InstagramReelsPublisher,
    "tiktok": TikTokPublisher,
}


def chunk_plan(size: int) -> tuple[int, int]:
    """(chunk_size, total_chunk_count) TikTok will be told for a file of ``size`` bytes."""
    if size <= TikTokPublisher.SINGLE_CHUNK_LIMIT:
        return size, 1
    return TikTokPublisher.CHUNK, max(1, math.floor(size / TikTokPublisher.CHUNK))
