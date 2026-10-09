"""Provider-neutral publishing contract.

The zero-cost default is an outbox. External adapters can be added without
changing the content/media/analytics phases.
"""

from dataclasses import dataclass
from pathlib import Path

from app.providers.publisher import PublishJob, PublishOutbox


@dataclass(frozen=True)
class PublishPayload:
    content_id: int
    video: Path
    caption: str
    ai_generated: bool = True
    title: str = ""
    experiment_id: str = ""
    platform: str = "review"
    metadata: dict | None = None
    scheduled_at: str = ""


class Publisher:
    name = "abstract"

    def publish(self, payload: PublishPayload) -> dict:
        raise NotImplementedError


class OutboxPublisher(Publisher):
    name = "outbox"

    def __init__(self, writer):
        self.writer = writer

    def publish(self, payload: PublishPayload) -> dict:
        metadata = dict(payload.metadata or {})
        metadata["ai_generated"] = payload.ai_generated
        job = PublishJob.create(
            video_path=payload.video, platform=payload.platform,
            title=payload.title or f"Content {payload.content_id}", caption=payload.caption,
            experiment_id=payload.experiment_id or f"content-{payload.content_id}",
            content_id=str(payload.content_id), metadata=metadata,
            scheduled_at=payload.scheduled_at, approved=False,
        )
        item = job.to_dict()
        item.update({"content_id": payload.content_id, "video": str(payload.video),
                     "ai_generated": payload.ai_generated, "publish_mode": "outbox"})
        # Existing callers may pass a writer callback; path-backed outboxes gain
        # idempotent persistence while callback integrations retain their API.
        if isinstance(self.writer, (str, Path)):
            item, _ = PublishOutbox(self.writer).enqueue(item)
        else:
            self.writer(item)
        return item
