"""Evidence-based trend intelligence v2 for NOVA.

This module separates *observations* from *inferences*. Social APIs are optional:
the engine works with free RSS/public sources and upgrades itself when supported
credentials are present. It never pretends that a proxy signal is a direct
platform trend metric.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from app.agents.trend_opportunity import score_content_opportunity
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote
import xml.etree.ElementTree as ET

import httpx


HOOK_PATTERNS: dict[str, tuple[str, ...]] = {
    "pov": ("pov:", "pov ", "when you ", "when your "),
    "curiosity_question": ("why ", "what if ", "did you know", "can you"),
    "confession": ("i tried", "i thought", "i accidentally", "i finally"),
    "challenge": ("challenge", "try not to", "attempt", "day ", "for 24"),
    "reveal": ("wait for it", "watch till", "you won't believe", "revealed", "reveal"),
    "before_after": ("before", "after", "then vs", "transformation", "glow up"),
    "listicle": ("3 ", "5 ", "top ", "things ", "ways ", "reasons "),
    "reaction": ("reaction", "reacts", "when ", "my face", "their reaction"),
    "story": ("storytime", "story time", "today i", "so this happened"),
}

FORMAT_PATTERNS: dict[str, tuple[str, ...]] = {
    "trend_audio": ("sound", "song", "audio", "dance", "remix", "chorus"),
    "dance": ("dance", "dancing", "choreography", "routine"),
    "reaction": ("reaction", "reacts", "reacting", "duet", "stitch"),
    "pov": ("pov", "when you", "when your"),
    "story_comedy": ("story", "funny", "meme", "office", "work", "awkward"),
    "challenge": ("challenge", "try", "attempt", "24 hours", "day"),
    "transformation": ("before", "after", "upgrade", "transform", "glow up"),
    "tutorial": ("how to", "tutorial", "tips", "guide", "learn"),
    "listicle": ("top", "3 ", "5 ", "7 ", "things", "reasons", "ways"),
    "cute_animal": ("cat", "dog", "puppy", "kitten", "cute animal", "pet"),
    "absurdist": ("weird", "chaos", "absurd", "unexpected", "cursed", "unhinged"),
}

EMOTION_TERMS: dict[str, tuple[str, ...]] = {
    "amusement": ("funny", "meme", "lol", "laugh", "comedy", "hilarious"),
    "surprise": ("surprise", "unexpected", "shocking", "reveal", "wait", "sudden"),
    "curiosity": ("why", "how", "what if", "secret", "mystery", "discover"),
    "cute": ("cute", "adorable", "kitten", "puppy", "baby", "wholesome"),
    "frustration": ("fail", "fails", "annoyed", "frustrating", "broken", "problem"),
    "awe": ("amazing", "incredible", "wow", "insane", "epic"),
    "nostalgia": ("throwback", "nostalgia", "remember", "old school", "childhood"),
    "anxiety": ("stress", "panic", "deadline", "awkward", "scared", "anxiety"),
    "satisfaction": ("clean", "perfect", "satisfying", "fixed", "organized"),
    "cringe": ("cringe", "embarrassing", "awkward", "secondhand"),
}

POSITIVE_EMOTION_WEIGHTS = {
    "amusement": 1.0, "surprise": 1.0, "curiosity": 0.95, "cute": 0.9,
    "awe": 0.9, "satisfaction": 0.85, "nostalgia": 0.8,
    "frustration": 0.8, "anxiety": 0.65, "cringe": 0.65,
}


@dataclass
class TrendSignalV2:
    trend_id: str
    title: str
    source: str
    source_type: str
    url: str = ""
    published_at: str = ""
    region: str = "IN"
    language: str = "en"
    category: str = ""
    description: str = ""
    hashtags: tuple[str, ...] = ()
    music: str = ""
    duration_seconds: float = 0.0
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    source_count: int = 1
    mention_count: int = 1
    hook_patterns: tuple[str, ...] = ()
    formats: tuple[str, ...] = ()
    emotions: tuple[str, ...] = ()
    freshness_score: float = 0.0
    velocity_score: float = 0.0
    cross_platform_score: float = 0.0
    saturation_score: float = 0.0
    novelty_score: float = 0.0
    opportunity_score: float = 0.0
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    capability_notes: tuple[str, ...] = ()
    content_opportunity_score: float = 0.0
    best_character: str = ""
    best_format: str = ""
    character_fit: dict[str, float] | None = None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _similarity(a: str, b: str) -> float:
    at, bt = _tokens(a), _tokens(b)
    if not at or not bt:
        return 0.0
    return len(at & bt) / len(at | bt)


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    raw = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw).astimezone(timezone.utc)
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(value).astimezone(timezone.utc)
        except Exception:
            return None


def freshness_score(published_at: str, now: datetime | None = None) -> float:
    dt = _parse_date(published_at)
    if not dt:
        return 0.35
    now = now or datetime.now(timezone.utc)
    hours = max(0.0, (now - dt).total_seconds() / 3600.0)
    if hours <= 3: return 1.0
    if hours <= 12: return 0.94
    if hours <= 24: return 0.86
    if hours <= 48: return 0.74
    if hours <= 72: return 0.62
    if hours <= 168: return 0.40
    if hours <= 336: return 0.20
    return 0.08


def extract_hook_patterns(text: str) -> tuple[str, ...]:
    text_l = (text or "").lower()
    found = []
    for name, phrases in HOOK_PATTERNS.items():
        if any(p in text_l for p in phrases):
            found.append(name)
    if text_l.strip().endswith("?") or "?" in text_l[:80]:
        found.append("question")
    if "!" in text_l[:100]:
        found.append("exclamation")
    return tuple(dict.fromkeys(found))


def infer_formats(text: str, category: str = "", duration_seconds: float = 0.0) -> tuple[str, ...]:
    blob = f"{text} {category}".lower()
    scores = []
    for fmt, terms in FORMAT_PATTERNS.items():
        hits = sum(term in blob for term in terms)
        if hits:
            scores.append((fmt, hits))
    if duration_seconds and duration_seconds <= 9.0:
        scores.append(("micro_short", 1))
    if duration_seconds and 9.0 < duration_seconds <= 20.0:
        scores.append(("short_story", 1))
    return tuple(x[0] for x in sorted(scores, key=lambda x: (-x[1], x[0]))[:5])


def infer_emotions(text: str) -> tuple[str, ...]:
    blob = (text or "").lower()
    scored = []
    for emotion, terms in EMOTION_TERMS.items():
        hits = sum(t in blob for t in terms)
        if hits:
            scored.append((emotion, hits * POSITIVE_EMOTION_WEIGHTS.get(emotion, 0.7)))
    return tuple(x[0] for x in sorted(scored, key=lambda x: (-x[1], x[0]))[:4])


def _engagement_rate(row: dict[str, Any]) -> float:
    views = max(int(row.get("views", 0) or 0), 1)
    return min(1.0, (
        (int(row.get("likes", 0) or 0) / views) * 0.20
        + (int(row.get("comments", 0) or 0) / views) * 0.60
        + (int(row.get("shares", 0) or 0) / views) * 1.0
    ))


def _dedupe_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for row in rows:
        title = row.get("title", "")
        placed = False
        for group in groups:
            if _similarity(title, group[0].get("title", "")) >= 0.62:
                group.append(row)
                placed = True
                break
        if not placed:
            groups.append([row])
    return groups


def enrich_signals(rows: Iterable[dict[str, Any]], now: datetime | None = None) -> list[TrendSignalV2]:
    raw = [dict(x) for x in rows]
    groups = _dedupe_groups(raw)
    source_types_by_group = []
    for group in groups:
        source_types = {str(x.get("source_type") or x.get("source") or "unknown") for x in group}
        source_types_by_group.append(source_types)

    output: list[TrendSignalV2] = []
    for idx, group in enumerate(groups):
        head = max(group, key=lambda x: (
            int(x.get("views", 0) or 0),
            x.get("published_at", ""),
        ))
        all_text = " ".join(
            _norm(" ".join(str(x.get(k, "")) for k in ("title", "description", "category")))
            for x in group
        )
        hooks = extract_hook_patterns(all_text)
        formats = infer_formats(all_text, str(head.get("category", "")), float(head.get("duration_seconds", 0) or 0))
        emotions = infer_emotions(all_text)
        fresh = max(freshness_score(str(x.get("published_at", "")), now) for x in group)
        mention_count = len(group)
        source_count = len({str(x.get("source_type") or x.get("source") or "") for x in group})
        # Velocity is a proxy unless a source supplies historical snapshots. A
        # source-backed view count/engagement rate gets a small bonus; cross-source
        # acceleration prevents raw popularity from dominating the score.
        engagement = max((_engagement_rate(x) for x in group), default=0.0)
        velocity = min(1.0, 0.55 * fresh + 0.25 * min(1.0, source_count / 4) + 0.20 * engagement)
        cross_platform = min(1.0, source_count / 3.0)
        # Saturation is deliberately a *proxy*: repetition + age + source spread.
        # We never label this as platform-wide saturation without platform data.
        repetition = min(1.0, (mention_count - 1) / 5.0)
        saturation = min(1.0, 0.45 * repetition + 0.25 * cross_platform + 0.30 * (1.0 - fresh))
        novelty = max(0.05, 1.0 - 0.60 * saturation)
        opportunity = (
            0.24 * velocity
            + 0.20 * cross_platform
            + 0.20 * novelty
            + 0.16 * max(0.0, 1.0 - saturation)
            + 0.10 * min(1.0, len(formats) / 2)
            + 0.10 * min(1.0, len(emotions) / 2)
        )
        title = _norm(str(head.get("title", "")))
        stable = hashlib.sha256(re.sub(r"[^a-z0-9]+", "-", title.lower()).encode()).hexdigest()[:12]
        trend_id = f"trend-{stable}"
        evidence = [
            f"sources={source_count}",
            f"mentions={mention_count}",
            f"freshness_proxy={fresh:.3f}",
            f"saturation_proxy={saturation:.3f}",
        ]
        content_fit = score_content_opportunity({
            "title": title,
            "description": str(head.get("description", "")),
            "category": str(head.get("category", "")),
            "hashtags": head.get("hashtags") or [],
            "music": str(head.get("music", "")),
            "formats": formats,
            "hook_patterns": hooks,
            "emotions": emotions,
            "velocity_score": velocity,
            "freshness_score": fresh,
            "cross_platform_score": cross_platform,
            "novelty_score": novelty,
            "saturation_score": saturation,
        })
        opportunity = round(0.65 * opportunity + 0.35 * content_fit["content_opportunity_score"], 4)
        notes = []
        if not any(x.get("source_type") == "youtube" for x in group):
            notes.append("no_direct_youtube_signal")
        if not any(x.get("source_type") == "tiktok_research" for x in group):
            notes.append("no_direct_tiktok_signal")
        if not any(x.get("source_type") == "reddit" for x in group):
            notes.append("no_direct_reddit_signal")
        if not any(x.get("source_type") == "instagram" for x in group):
            notes.append("no_direct_instagram_trend_signal")
        output.append(TrendSignalV2(
            trend_id=trend_id,
            title=title,
            source=str(head.get("source", "")),
            source_type=str(head.get("source_type") or head.get("source") or "unknown"),
            url=str(head.get("url", "")),
            published_at=str(head.get("published_at", "")),
            region=str(head.get("region", "IN")),
            language=str(head.get("language", "en")),
            category=str(head.get("category", "")),
            description=_norm(str(head.get("description", ""))),
            hashtags=tuple(str(x) for x in (head.get("hashtags") or []) if str(x)),
            music=str(head.get("music", "")),
            duration_seconds=float(head.get("duration_seconds", 0) or 0),
            views=max(int(x.get("views", 0) or 0) for x in group),
            likes=max(int(x.get("likes", 0) or 0) for x in group),
            comments=max(int(x.get("comments", 0) or 0) for x in group),
            shares=max(int(x.get("shares", 0) or 0) for x in group),
            source_count=source_count,
            mention_count=mention_count,
            hook_patterns=hooks,
            formats=formats,
            emotions=emotions,
            freshness_score=round(fresh, 4),
            velocity_score=round(velocity, 4),
            cross_platform_score=round(cross_platform, 4),
            saturation_score=round(saturation, 4),
            novelty_score=round(novelty, 4),
            opportunity_score=opportunity,
            confidence=round(min(1.0, 0.45 + 0.12 * min(source_count, 3) + 0.08 * bool(hooks)), 4),
            evidence=tuple(evidence),
            capability_notes=tuple(notes),
            content_opportunity_score=content_fit["content_opportunity_score"],
            best_character=content_fit["best_character"],
            best_format=content_fit["best_format"],
            character_fit=content_fit["character_fit"],
        ))
    return sorted(output, key=lambda x: (x.opportunity_score, x.velocity_score), reverse=True)


def _rss_items(url: str, source_type: str, category: str, timeout: float = 8.0) -> list[dict[str, Any]]:
    response = httpx.get(url, timeout=timeout, headers={"User-Agent": "NOVA-Trend-Scout/2.0"})
    response.raise_for_status()
    root = ET.fromstring(response.text)
    out = []
    for item in root.findall(".//item")[:30]:
        out.append({
            "title": _norm(item.findtext("title", "")),
            "source": source_type,
            "source_type": source_type,
            "category": category,
            "url": _norm(item.findtext("link", "")),
            "published_at": _norm(item.findtext("pubDate", "")),
            "description": _norm(item.findtext("description", "")),
            "region": "IN",
        })
    return [x for x in out if x["title"]]


def discover_reddit_rss(timeout: float = 8.0) -> list[dict[str, Any]]:
    subreddits = (
        "memes", "funny", "TikTokCringe", "Instagramreels", "YouTubeShorts",
        "cats", "aicartoon",
    )
    rows: list[dict[str, Any]] = []
    for sub in subreddits:
        try:
            rows.extend(_rss_items(f"https://www.reddit.com/r/{sub}/.rss", "reddit", f"reddit_{sub.lower()}", timeout))
        except Exception:
            continue
    return rows


def discover_google_trends_rss(region_code: str = "IN", timeout: float = 8.0) -> list[dict[str, Any]]:
    """Google Trends daily/short-term feed as a zero-key trend source."""
    urls = (
        f"https://trends.google.com/trending/rss?geo={region_code}",
        f"https://trends.google.com/trending/rss?geo={region_code}&hours=24",
    )
    rows: list[dict[str, Any]] = []
    for url in urls:
        try:
            rows.extend(_rss_items(url, "google_trends", "google_trends", timeout))
        except Exception:
            continue
    return rows


def discover_youtube_most_popular(api_key: str = "", region_code: str = "IN", timeout: float = 8.0) -> list[dict[str, Any]]:
    api_key = api_key or os.getenv("YOUTUBE_API_KEY", "")
    if not api_key:
        return []
    params = {
        "part": "snippet,statistics,contentDetails",
        "chart": "mostPopular",
        "regionCode": region_code,
        "maxResults": 50,
        "key": api_key,
    }
    try:
        response = httpx.get("https://www.googleapis.com/youtube/v3/videos", params=params, timeout=timeout)
        response.raise_for_status()
        out = []
        for item in response.json().get("items", []):
            stats = item.get("statistics", {})
            details = item.get("contentDetails", {})
            out.append({
                "title": _norm(item.get("snippet", {}).get("title", "")),
                "description": _norm(item.get("snippet", {}).get("description", "")),
                "source": "youtube",
                "source_type": "youtube",
                "category": "youtube_most_popular",
                "url": f"https://www.youtube.com/watch?v={item.get('id', '')}",
                "published_at": item.get("snippet", {}).get("publishedAt", ""),
                "views": int(stats.get("viewCount", 0) or 0),
                "likes": int(stats.get("likeCount", 0) or 0),
                "comments": int(stats.get("commentCount", 0) or 0),
                "duration_seconds": 0.0,
                "region": region_code,
            })
        return out
    except Exception:
        return []


def discover_tiktok_research(access_token: str = "", query: str = "", region: str = "", timeout: float = 8.0) -> list[dict[str, Any]]:
    token = access_token or os.getenv("TIKTOK_RESEARCH_ACCESS_TOKEN", "")
    if not token:
        return []
    region = region or os.getenv("TIKTOK_RESEARCH_REGION", "IN")
    keywords = [x.strip() for x in (query or os.getenv(
        "TIKTOK_RESEARCH_KEYWORDS",
        "dance,meme,funny,cat,cartoon,reaction,pov,cute",
    )).split(",") if x.strip()][:10]
    start_date = os.getenv("TIKTOK_RESEARCH_START_DATE", (datetime.now(timezone.utc).date().replace(day=1)).strftime("%Y%m%d"))
    end_date = os.getenv("TIKTOK_RESEARCH_END_DATE", datetime.now(timezone.utc).strftime("%Y%m%d"))
    fields = "id,video_description,create_time,region_code,share_count,view_count,like_count,comment_count,music_id,hashtag_names,video_duration"
    out = []
    for keyword in keywords:
        payload = {
            "query": {
                "and": [
                    {"operation": "IN", "field_name": "region_code", "field_values": [region]},
                    {"operation": "EQ", "field_name": "keyword", "field_values": [keyword]},
                    {"operation": "EQ", "field_name": "video_length", "field_values": ["SHORT"]},
                ]
            },
            "max_count": 50,
            "cursor": 0,
            "start_date": start_date,
            "end_date": end_date,
            "is_random": False,
        }
        try:
            response = httpx.post(
                "https://open.tiktokapis.com/v2/research/video/query/",
                params={"fields": fields},
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            for item in response.json().get("data", {}).get("videos", []):
                out.append({
                    "title": _norm(item.get("video_description", ""))[:180],
                    "description": _norm(item.get("video_description", "")),
                    "source": "tiktok",
                    "source_type": "tiktok_research",
                    "category": "tiktok_public_research",
                    "url": f"https://www.tiktok.com/@_/video/{item.get('id', '')}",
                    "published_at": datetime.fromtimestamp(int(item.get("create_time", 0) or 0), tz=timezone.utc).isoformat() if item.get("create_time") else "",
                    "views": int(item.get("view_count", 0) or 0),
                    "likes": int(item.get("like_count", 0) or 0),
                    "comments": int(item.get("comment_count", 0) or 0),
                    "shares": int(item.get("share_count", 0) or 0),
                    "hashtags": tuple(item.get("hashtag_names") or ()),
                    "music": str(item.get("music_id", "") or ""),
                    "duration_seconds": float(item.get("video_duration", 0) or 0),
                    "region": region,
                })
        except Exception:
            continue
    return out


def discover_web_social_mentions(timeout: float = 8.0) -> list[dict[str, Any]]:
    queries = (
        ("instagram_mentions", "Instagram Reels trend viral cute cartoon meme"),
        ("tiktok_mentions", "TikTok trend viral sound meme dance"),
        ("youtube_short_mentions", "YouTube Shorts trend viral meme"),
    )
    rows: list[dict[str, Any]] = []
    for category, query in queries:
        try:
            url = "https://news.google.com/rss/search?q=" + quote(query) + "&hl=en-IN&gl=IN&ceid=IN:en"
            for row in _rss_items(url, "google_social_search", category, timeout):
                row["source_type"] = "web_social_proxy"
                rows.append(row)
        except Exception:
            continue
    return rows


def discover_social_trends(timeout: float = 8.0) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    status = {
        "reddit": "attempted_public_rss",
        "youtube": "skipped_no_api_key",
        "tiktok": "skipped_no_research_token",
        "instagram": "web_proxy_only_no_public_trend_api",
    }
    rows.extend(discover_reddit_rss(timeout))
    google_trends = discover_google_trends_rss(timeout=timeout)
    if google_trends:
        status["google_trends"] = "direct_public_rss"
        rows.extend(google_trends)
    else:
        status["google_trends"] = "unavailable"
    yt = discover_youtube_most_popular(timeout=timeout)
    if yt:
        status["youtube"] = "direct_public_api"
        rows.extend(yt)
    tt = discover_tiktok_research(timeout=timeout)
    if tt:
        status["tiktok"] = "direct_research_api"
        rows.extend(tt)
    rows.extend(discover_web_social_mentions(timeout))
    return rows, status


def signal_to_dict(signal: TrendSignalV2) -> dict[str, Any]:
    return asdict(signal)
