"""Provider-neutral publishing contract and safe dry-run adapter.

The adapter deliberately does not post anywhere. Real platform adapters can implement
publish() later while preserving the experiment/content binding contract.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Protocol
import json
import uuid


@dataclass
class PublishResult:
    platform: str
    status: str
    content_id: str = ""
    published_at: str = ""
    provider: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Publisher(Protocol):
    platform: str

    def publish(
        self,
        video_path: str | Path,
        *,
        title: str = "",
        caption: str = "",
        experiment_id: str = "",
        dry_run: bool = True,
    ) -> PublishResult:
        ...


class DryRunPublisher:
    """Safe publisher used until an explicit platform adapter is enabled."""

    def __init__(self, platform: str) -> None:
        self.platform = platform

    def publish(
        self,
        video_path: str | Path,
        *,
        title: str = "",
        caption: str = "",
        experiment_id: str = "",
        dry_run: bool = True,
    ) -> PublishResult:
        path = Path(video_path)
        if not path.exists():
            return PublishResult(
                platform=self.platform,
                status="failed",
                provider="dry_run",
                error=f"video_not_found:{path}",
            )
        if not dry_run:
            return PublishResult(
                platform=self.platform,
                status="not_configured",
                provider="dry_run",
                error="No real publisher is enabled",
            )
        seed = f"{self.platform}:{experiment_id}:{path.resolve()}"
        content_id = f"dryrun_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:16]}"
        return PublishResult(
            platform=self.platform,
            status="dry_run",
            content_id=content_id,
            provider="dry_run",
        )


def build_publisher(platform: str) -> Publisher:
    """Return the safest available provider for a platform."""
    return DryRunPublisher(platform.strip().lower())


def publish_review_item(
    *,
    video_path: str | Path,
    experiment_id: str,
    title: str,
    caption: str,
    platforms: list[str],
) -> dict[str, Any]:
    """Create deterministic dry-run content bindings without publishing."""
    results = []
    for platform in platforms:
        result = build_publisher(platform).publish(
            video_path,
            title=title,
            caption=caption,
            experiment_id=experiment_id,
            dry_run=True,
        )
        results.append(result.to_dict())
    return {
        "status": "dry_run",
        "experiment_id": experiment_id,
        "results": results,
    }
