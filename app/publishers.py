"""Provider-neutral publishing contract.

The zero-cost default is an outbox. External adapters can be added without
changing the content/media/analytics phases.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublishPayload:
    content_id: int
    video: Path
    caption: str
    ai_generated: bool = True


class Publisher:
    name = "abstract"

    def publish(self, payload: PublishPayload) -> dict:
        raise NotImplementedError


class OutboxPublisher(Publisher):
    name = "outbox"

    def __init__(self, writer):
        self.writer = writer

    def publish(self, payload: PublishPayload) -> dict:
        item = {
            "content_id": payload.content_id,
            "video": str(payload.video),
            "caption": payload.caption,
            "ai_generated": payload.ai_generated,
            "status": "ready_for_review",
        }
        self.writer(item)
        return item
