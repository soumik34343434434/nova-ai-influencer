"""Optional platform analytics connectors.

All clients are read-only. Missing credentials return a capability result instead
of fabricating metrics. This keeps the learning loop honest and lets NOVA upgrade
from local/manual metrics to first-party data when accounts are connected.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import os
from typing import Any, Iterable

import httpx


@dataclass
class PlatformMetricSnapshot:
    content_id: str
    platform: str
    title: str = ""
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    retention_pct: float = 0.0
    avg_watch_seconds: float = 0.0
    total_watch_seconds: float = 0.0
    followers_gained: int = 0
    profile_visits: int = 0
    captured_at: str = ""
    published_at: str = ""
    duration_seconds: float = 0.0
    raw: dict[str, Any] | None = None


@dataclass
class ConnectorStatus:
    platform: str
    enabled: bool
    status: str
    detail: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


class InstagramInsightsClient:
    """Instagram Professional account insights via Graph API/Instagram Login."""

    def __init__(self, access_token: str | None = None, user_id: str | None = None, version: str | None = None) -> None:
        self.access_token = access_token or os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
        self.user_id = user_id or os.getenv("INSTAGRAM_USER_ID", "")
        self.version = version or os.getenv("INSTAGRAM_GRAPH_VERSION", "v26.0")

    def status(self) -> ConnectorStatus:
        if not self.access_token or not self.user_id:
            return ConnectorStatus("instagram", False, "missing_credentials", "Set INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_USER_ID")
        return ConnectorStatus("instagram", True, "configured")

    def fetch_recent_reels(self, limit: int = 25, timeout: float = 15.0) -> list[PlatformMetricSnapshot]:
        if not self.status().enabled:
            return []
        base = f"https://graph.facebook.com/{self.version}"
        params = {
            "fields": "id,caption,media_type,media_product_type,timestamp,permalink,like_count,comments_count",
            "limit": limit,
            "access_token": self.access_token,
        }
        try:
            response = httpx.get(f"{base}/{self.user_id}/media", params=params, timeout=timeout)
            response.raise_for_status()
            media = response.json().get("data", [])
        except Exception:
            return []

        snapshots: list[PlatformMetricSnapshot] = []
        metrics = (
            "views,reach,likes,comments,shares,saved,total_interactions,"
            "follows,profile_visits,ig_reels_avg_watch_time,ig_reels_video_view_total_time"
        )
        for item in media:
            if str(item.get("media_product_type", "")).upper() != "REELS":
                continue
            try:
                ins = httpx.get(
                    f"{base}/{item['id']}/insights",
                    params={"metric": metrics, "access_token": self.access_token},
                    timeout=timeout,
                )
                if ins.status_code >= 400:
                    continue
                values: dict[str, float] = {}
                for row in ins.json().get("data", []):
                    name = str(row.get("name", ""))
                    value = row.get("values", [{}])[-1].get("value", 0)
                    try:
                        values[name] = float(value or 0)
                    except (TypeError, ValueError):
                        continue
                duration = 0.0
                snapshots.append(PlatformMetricSnapshot(
                    content_id=str(item["id"]),
                    platform="instagram",
                    title=str(item.get("caption", "")).split("\n", 1)[0][:200],
                    views=int(values.get("views", 0)),
                    likes=int(values.get("likes", item.get("like_count", 0) or 0)),
                    comments=int(values.get("comments", item.get("comments_count", 0) or 0)),
                    shares=int(values.get("shares", 0)),
                    saves=int(values.get("saved", 0)),
                    avg_watch_seconds=values.get("ig_reels_avg_watch_time", 0.0),
                    total_watch_seconds=values.get("ig_reels_video_view_total_time", 0.0),
                    followers_gained=int(values.get("follows", 0)),
                    profile_visits=int(values.get("profile_visits", 0)),
                    captured_at=_now(),
                    published_at=str(item.get("timestamp", "")),
                    duration_seconds=duration,
                    raw={"media": item, "insights": values},
                ))
            except Exception:
                continue
        return snapshots


class YouTubeClient:
    """YouTube Data API + optional Analytics API for the owned channel."""

    def __init__(self, api_key: str | None = None, access_token: str | None = None) -> None:
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY", "")
        self.access_token = access_token or os.getenv("YOUTUBE_ACCESS_TOKEN", "")
        self.channel_id = os.getenv("YOUTUBE_CHANNEL_ID", "")

    def status(self) -> ConnectorStatus:
        if self.api_key or self.access_token:
            return ConnectorStatus("youtube", True, "configured")
        return ConnectorStatus("youtube", False, "missing_credentials", "Set YOUTUBE_API_KEY for public stats or YOUTUBE_ACCESS_TOKEN for owner analytics")

    def fetch_video_stats(self, video_ids: Iterable[str], timeout: float = 12.0) -> list[PlatformMetricSnapshot]:
        ids = [str(x) for x in video_ids if x]
        if not ids or not self.api_key:
            return []
        out = []
        try:
            response = httpx.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={"part": "snippet,statistics,contentDetails", "id": ",".join(ids[:50]), "key": self.api_key},
                timeout=timeout,
            )
            response.raise_for_status()
            for item in response.json().get("items", []):
                stats = item.get("statistics", {})
                content = item.get("contentDetails", {})
                out.append(PlatformMetricSnapshot(
                    content_id=str(item["id"]),
                    platform="youtube",
                    title=str(item.get("snippet", {}).get("title", "")),
                    views=int(stats.get("viewCount", 0) or 0),
                    likes=int(stats.get("likeCount", 0) or 0),
                    comments=int(stats.get("commentCount", 0) or 0),
                    captured_at=_now(),
                    published_at=str(item.get("snippet", {}).get("publishedAt", "")),
                    raw={"statistics": stats, "contentDetails": content},
                ))
        except Exception:
            return []
        return out

    def fetch_owner_analytics(self, video_id: str, days: int = 28, timeout: float = 15.0) -> PlatformMetricSnapshot | None:
        if not self.access_token:
            return None
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=max(1, days))
        params = {
            "ids": "channel==MINE",
            "startDate": str(start),
            "endDate": str(end),
            "metrics": "views,likes,comments,shares,averageViewDuration,averageViewPercentage,subscribersGained,estimatedMinutesWatched",
            "dimensions": "video",
            "filters": f"video=={video_id}",
            "access_token": self.access_token,
        }
        try:
            response = httpx.get("https://youtubeanalytics.googleapis.com/v2/reports", params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            headers = [h["name"] for h in data.get("columnHeaders", [])]
            rows = data.get("rows", [])
            if not rows:
                return None
            row = dict(zip(headers, rows[0]))
            avg_watch = float(row.get("averageViewDuration", 0) or 0)
            watched_minutes = float(row.get("estimatedMinutesWatched", 0) or 0)
            return PlatformMetricSnapshot(
                content_id=video_id,
                platform="youtube",
                views=int(row.get("views", 0) or 0),
                likes=int(row.get("likes", 0) or 0),
                comments=int(row.get("comments", 0) or 0),
                shares=int(row.get("shares", 0) or 0),
                retention_pct=float(row.get("averageViewPercentage", 0) or 0),
                avg_watch_seconds=avg_watch,
                total_watch_seconds=watched_minutes * 60,
                followers_gained=int(row.get("subscribersGained", 0) or 0),
                captured_at=_now(),
                raw={"rows": data},
            )
        except Exception:
            return None


class TikTokClient:
    """TikTok Display API video statistics for the connected creator."""

    def __init__(self, access_token: str | None = None) -> None:
        self.access_token = access_token or os.getenv("TIKTOK_ACCESS_TOKEN", "")

    def status(self) -> ConnectorStatus:
        if not self.access_token:
            return ConnectorStatus("tiktok", False, "missing_credentials", "Set TIKTOK_ACCESS_TOKEN")
        return ConnectorStatus("tiktok", True, "configured")

    def fetch_video_stats(self, video_ids: Iterable[str], timeout: float = 12.0) -> list[PlatformMetricSnapshot]:
        if not self.status().enabled:
            return []
        ids = [str(x) for x in video_ids if x]
        if not ids:
            return []
        fields = "id,title,video_description,create_time,duration,like_count,comment_count,share_count,view_count,is_aigc"
        try:
            response = httpx.post(
                "https://open.tiktokapis.com/v2/video/query/",
                params={"fields": fields},
                headers=_headers(self.access_token),
                json={"filters": {"video_ids": ids[:20]}},
                timeout=timeout,
            )
            response.raise_for_status()
            out = []
            for item in response.json().get("data", {}).get("videos", []):
                out.append(PlatformMetricSnapshot(
                    content_id=str(item["id"]),
                    platform="tiktok",
                    title=str(item.get("title") or item.get("video_description", "")),
                    views=int(item.get("view_count", 0) or 0),
                    likes=int(item.get("like_count", 0) or 0),
                    comments=int(item.get("comment_count", 0) or 0),
                    shares=int(item.get("share_count", 0) or 0),
                    captured_at=_now(),
                    published_at=datetime.fromtimestamp(int(item.get("create_time", 0)), tz=timezone.utc).isoformat() if item.get("create_time") else "",
                    duration_seconds=float(item.get("duration", 0) or 0),
                    raw=item,
                ))
            return out
        except Exception:
            return []


def collect_platform_metrics(content_map: dict[str, dict[str, Any]] | None = None) -> tuple[list[PlatformMetricSnapshot], dict[str, Any]]:
    """Collect what is currently configured; never fabricates unsupported metrics."""
    content_map = content_map or {}
    records: list[PlatformMetricSnapshot] = []
    statuses: list[dict[str, Any]] = []

    ig = InstagramInsightsClient()
    statuses.append(asdict(ig.status()))
    if ig.status().enabled:
        records.extend(ig.fetch_recent_reels())

    yt = YouTubeClient()
    statuses.append(asdict(yt.status()))
    if yt.api_key:
        records.extend(yt.fetch_video_stats(content_map.get("youtube", {}).keys()))
    if yt.access_token:
        for video_id in content_map.get("youtube", {}).keys():
            owner = yt.fetch_owner_analytics(video_id)
            if owner:
                records.append(owner)

    tt = TikTokClient()
    statuses.append(asdict(tt.status()))
    if tt.access_token:
        records.extend(tt.fetch_video_stats(content_map.get("tiktok", {}).keys()))

    # De-duplicate provider snapshots by platform/content id, preferring rows with
    # retention/average-watch information when both public and owner APIs return data.
    best: dict[tuple[str, str], PlatformMetricSnapshot] = {}
    for row in records:
        key = (row.platform, row.content_id)
        old = best.get(key)
        if old is None:
            best[key] = row
        else:
            old_richness = int(old.retention_pct > 0) + int(old.avg_watch_seconds > 0)
            new_richness = int(row.retention_pct > 0) + int(row.avg_watch_seconds > 0)
            if new_richness >= old_richness:
                best[key] = row
    return list(best.values()), {"connectors": statuses, "records": len(best)}


def metric_to_record(snapshot: PlatformMetricSnapshot, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    row = asdict(snapshot)
    metadata = metadata or {}
    for key in ("format_name", "hook_pattern", "emotion", "character", "trend_id", "experiment_id"):
        if key in metadata:
            row[key] = metadata[key]
    return row
