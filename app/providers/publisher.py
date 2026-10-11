"""Provider-neutral publish jobs with an intentionally safe outbox default.

This module performs no network I/O. Live uploads live in
``platform_publishers`` and are reachable only for an approved job when real
publishing is explicitly enabled and the platform's credentials are configured.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
import hashlib
import json
import os


class PublishValidationError(ValueError):
    """Raised when a job lacks required media, identity, or publishing metadata."""


class PublishError(RuntimeError):
    """Provider error carrying an explicit retry classification."""

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class PublishResult:
    platform: str
    status: str
    content_id: str = ""
    published_at: str = ""
    provider: str = ""
    error: str = ""
    retryable: bool = False
    idempotency_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PublishJob:
    """Immutable-in-meaning request to publish one asset to one platform."""

    video_path: str
    platform: str
    title: str
    caption: str
    experiment_id: str
    content_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    scheduled_at: str = ""
    approved: bool = False
    idempotency_key: str = ""
    status: str = "pending"
    attempts: int = 0
    last_error: str = ""

    @classmethod
    def create(cls, *, video_path: str | Path, platform: str, title: str,
               caption: str, experiment_id: str, content_id: str = "",
               metadata: dict[str, Any] | None = None, scheduled_at: str = "",
               approved: bool = False, idempotency_key: str = "") -> "PublishJob":
        platform = platform.strip().lower()
        metadata = dict(metadata or {})
        identity = idempotency_key.strip() or _idempotency_key(
            video_path, platform, experiment_id, content_id, title, caption, metadata
        )
        job = cls(str(video_path), platform, title.strip(), caption.strip(),
                  experiment_id.strip(), str(content_id), metadata,
                  scheduled_at.strip(), approved, identity,
                  status="scheduled" if scheduled_at.strip() else "ready_for_review")
        validate_publish_job(job)
        return job

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _idempotency_key(video_path: str | Path, platform: str, experiment_id: str,
                     content_id: str, title: str, caption: str,
                     metadata: dict[str, Any]) -> str:
    path = Path(video_path)
    try:
        stat = path.stat()
        asset_identity = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        asset_identity = str(path)
    stable = json.dumps([platform, experiment_id, content_id, title.strip(), caption.strip(),
                         asset_identity, metadata], sort_keys=True, default=str)
    return "pub_" + hashlib.sha256(stable.encode()).hexdigest()[:32]


def validate_publish_job(job: PublishJob) -> None:
    problems = []
    if not job.platform.strip(): problems.append("platform_required")
    if not job.experiment_id.strip(): problems.append("experiment_id_required")
    if not job.title.strip(): problems.append("title_required")
    if not job.caption.strip(): problems.append("caption_required")
    path = Path(job.video_path)
    if not path.is_file() or path.stat().st_size <= 0: problems.append("video_missing_or_empty")
    if not job.idempotency_key.strip(): problems.append("idempotency_key_required")
    if not isinstance(job.metadata, dict):
        problems.append("metadata_must_be_object")
    elif "ai_generated" not in job.metadata or not isinstance(job.metadata.get("ai_generated"), bool):
        problems.append("metadata_ai_generated_required")
    if job.scheduled_at:
        try:
            scheduled = datetime.fromisoformat(job.scheduled_at.replace("Z", "+00:00"))
            if scheduled.tzinfo is None: problems.append("scheduled_at_timezone_required")
        except ValueError:
            problems.append("scheduled_at_invalid")
    if problems:
        raise PublishValidationError(",".join(problems))


class Publisher(Protocol):
    platform: str
    def publish(self, job: PublishJob, *, dry_run: bool = True) -> PublishResult: ...


class DryRunPublisher:
    """Validates and records intent only; never uploads, even when dry_run=False."""

    def __init__(self, platform: str):
        self.platform = platform.strip().lower()

    def publish(self, job: PublishJob | str | Path, *, title: str = "", caption: str = "",
                experiment_id: str = "", dry_run: bool = True, metadata: dict[str, Any] | None = None,
                approved: bool = False) -> PublishResult:
        # Compatibility for the original path-based dry-run API.
        if not isinstance(job, PublishJob):
            job = PublishJob.create(video_path=job, platform=self.platform, title=title or "Dry run",
                                    caption=caption or "Dry run", experiment_id=experiment_id or "dry-run",
                                    metadata=metadata or {"ai_generated": True}, approved=approved)
        validate_publish_job(job)
        if not dry_run:
            return PublishResult(job.platform, "not_configured", provider="outbox",
                                 error="Real publishing is disabled; no upload was attempted",
                                 idempotency_key=job.idempotency_key)
        return PublishResult(job.platform, "dry_run", content_id=job.idempotency_key,
                             provider="dry_run", idempotency_key=job.idempotency_key)


class PublishOutbox:
    """JSON outbox with duplicate-safe insertion and scheduling/retry updates."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def enqueue(self, job: PublishJob | dict[str, Any]) -> tuple[dict[str, Any], bool]:
        row = job.to_dict() if isinstance(job, PublishJob) else dict(job)
        if not isinstance(job, PublishJob):
            try:
                validate_publish_job(PublishJob(
                    video_path=str(row.get("video_path", row.get("video", ""))),
                    platform=str(row.get("platform", "")), title=str(row.get("title", "")),
                    caption=str(row.get("caption", "")), experiment_id=str(row.get("experiment_id", "")),
                    content_id=str(row.get("content_id", "")), metadata=row.get("metadata", {}),
                    scheduled_at=str(row.get("scheduled_at", "")), approved=bool(row.get("approved", False)),
                    idempotency_key=str(row.get("idempotency_key", "")),
                ))
            except (TypeError, AttributeError) as exc:
                raise PublishValidationError("invalid_publish_job") from exc
        key = row.get("idempotency_key")
        if not key:
            raise PublishValidationError("idempotency_key_required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            rows = json.loads(self.path.read_text()) if self.path.exists() else []
        except (json.JSONDecodeError, OSError):
            rows = []
        for existing in rows:
            if existing.get("idempotency_key") == key:
                return existing, False
        now = datetime.now(timezone.utc).isoformat()
        row.setdefault("status", "scheduled" if row.get("scheduled_at") else "ready_for_review")
        row.setdefault("created_at", now)
        row.setdefault("attempts", 0)
        row.setdefault("ai_generated", bool(row.get("metadata", {}).get("ai_generated", True)))
        rows.append(row)
        self.path.write_text(json.dumps(rows, indent=2))
        return row, True

    def transition(self, idempotency_key: str, status: str, *, error: str = "") -> dict[str, Any]:
        allowed = {"scheduled", "ready_for_review", "publishing", "published", "retryable_error", "failed"}
        if status not in allowed: raise ValueError(f"unsupported publish status: {status}")
        rows = json.loads(self.path.read_text()) if self.path.exists() else []
        for row in rows:
            if row.get("idempotency_key") == idempotency_key:
                row["status"] = status
                row["last_error"] = error
                if status in {"publishing", "retryable_error", "failed"}: row["attempts"] = int(row.get("attempts", 0)) + (status == "publishing")
                row["updated_at"] = datetime.now(timezone.utc).isoformat()
                self.path.write_text(json.dumps(rows, indent=2))
                return row
        raise KeyError(idempotency_key)


def classify_publish_error(error: Exception) -> str:
    if isinstance(error, PublishError): return "retryable_error" if error.retryable else "failed"
    if isinstance(error, (TimeoutError, ConnectionError)): return "retryable_error"
    return "failed"


def assert_publish_allowed(job: PublishJob, *, dry_run: bool) -> None:
    """Explicit approval and deployment gates for any future live adapter."""
    validate_publish_job(job)
    if dry_run:
        return
    if not job.approved:
        raise PermissionError("publish_approval_required")
    if not real_publish_enabled():
        raise PermissionError("real_publishing_disabled")
    from app.providers.platform_publishers import LIVE_PUBLISHERS

    adapter = LIVE_PUBLISHERS.get(job.platform)
    if adapter is None or not adapter.configured():
        # No adapter, or its credentials are missing: never fall through to an upload.
        missing = adapter.missing_env() if adapter else []
        raise PermissionError("no_real_publisher_configured" + (f":{job.platform}:missing={','.join(missing)}" if missing else f":{job.platform}"))


def build_publisher(platform: str) -> Publisher:
    """Return the live adapter only when real publishing is enabled and configured.

    Otherwise return the non-network dry-run adapter. A live adapter still
    refuses to upload unless the job itself is approved (assert_publish_allowed).
    """
    from app.providers.platform_publishers import LIVE_PUBLISHERS

    adapter = LIVE_PUBLISHERS.get(platform.strip().lower())
    if real_publish_enabled() and adapter is not None and adapter.configured():
        return adapter()
    return DryRunPublisher(platform)


def publish_review_item(*, video_path: str | Path, experiment_id: str, title: str,
                        caption: str, platforms: list[str], content_id: str = "",
                        metadata: dict[str, Any] | None = None, scheduled_at: str = "",
                        approved: bool = False) -> dict[str, Any]:
    results = []
    for platform in platforms:
        job = PublishJob.create(video_path=video_path, platform=platform, title=title,
                                caption=caption, experiment_id=experiment_id, content_id=content_id,
                                metadata=metadata or {"ai_generated": True}, scheduled_at=scheduled_at,
                                approved=approved)
        results.append(build_publisher(platform).publish(job, dry_run=True).to_dict())
    return {"status": "dry_run", "experiment_id": experiment_id, "results": results}


def real_publish_enabled() -> bool:
    """Fail-closed gate: live adapters are used only when this is explicitly true."""
    return os.getenv("NOVA_ENABLE_REAL_PUBLISH", "false").strip().lower() == "true"
