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

from app.agents.cartoon_safety import contains_term
from app.agents.trend_opportunity import analyze_trend_transfer, score_content_opportunity
import hashlib
import html
import json
import math
import os
import re
import time
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
    "listicle": ("top ", "things ", "ways ", "reasons "),
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
    "listicle": ("top", "things", "reasons", "ways"),
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
    trend_transfer_analysis: dict[str, Any] | None = None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _similarity(a: str, b: str) -> float:
    at, bt = _tokens(a), _tokens(b)
    if not at or not bt:
        return 0.0
    return len(at & bt) / len(at | bt)


def _set_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    left = {str(x).strip().lower() for x in a if str(x).strip()}
    right = {str(x).strip().lower() for x in b if str(x).strip()}
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _trend_related(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Conservatively connect alternate descriptions of the same trend."""
    title_similarity = _similarity(str(a.get("title", "")), str(b.get("title", "")))
    if title_similarity >= 0.62:
        return True

    music_a = str(a.get("music", "") or "").strip().lower()
    music_b = str(b.get("music", "") or "").strip().lower()
    hashtag_overlap = _set_overlap(a.get("hashtags") or (), b.get("hashtags") or ())
    if music_a and music_a == music_b and title_similarity >= 0.28:
        return True
    if hashtag_overlap >= 0.60 and title_similarity >= 0.32:
        return True
    return False


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


# A number only signals a list when a list noun follows it ("5 ways"), not when it is
# a count, a version or a URL parameter ("Season 5", "oc=5").
NUMBERED_LIST = re.compile(r"(?<![a-z0-9])\d+\s+(?:things|reasons|ways|tips|ideas|facts|signs|mistakes|hacks)(?![a-z0-9])")


def extract_hook_patterns(text: str) -> tuple[str, ...]:
    text_l = (text or "").lower()
    found = []
    for name, phrases in HOOK_PATTERNS.items():
        if any(contains_term(text_l, p.strip()) for p in phrases):
            found.append(name)
    if NUMBERED_LIST.search(text_l) and "listicle" not in found:
        found.append("listicle")
    if text_l.strip().endswith("?") or "?" in text_l[:80]:
        found.append("question")
    if "!" in text_l[:100]:
        found.append("exclamation")
    return tuple(dict.fromkeys(found))


def infer_formats(text: str, category: str = "", duration_seconds: float = 0.0) -> tuple[str, ...]:
    blob = f"{text} {category}".lower()
    scores = []
    for fmt, terms in FORMAT_PATTERNS.items():
        hits = sum(contains_term(blob, term) for term in terms)
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
        hits = sum(contains_term(blob, t) for t in terms)
        if hits:
            scored.append((emotion, hits * POSITIVE_EMOTION_WEIGHTS.get(emotion, 0.7)))
    return tuple(x[0] for x in sorted(scored, key=lambda x: (-x[1], x[0]))[:4])


def recompute_opportunity_score(signal: dict[str, Any]) -> dict[str, Any]:
    """Recalculate opportunity from the signal's current evidence, including history."""
    formats = tuple(signal.get("formats") or ())
    emotions = tuple(signal.get("emotions") or ())
    freshness = float(signal.get("freshness_score", 0.0) or 0.0)
    velocity = float(signal.get("velocity_score", 0.0) or 0.0)
    cross_platform = float(signal.get("cross_platform_score", 0.0) or 0.0)
    saturation = float(signal.get("saturation_score", 0.0) or 0.0)
    novelty = float(signal.get("novelty_score", 0.0) or 0.0)
    opportunity = (
        0.24 * velocity
        + 0.20 * cross_platform
        + 0.20 * novelty
        + 0.16 * max(0.0, 1.0 - saturation)
        + 0.10 * min(1.0, len(formats) / 2)
        + 0.10 * min(1.0, len(emotions) / 2)
    )
    content_fit = score_content_opportunity(signal)
    opportunity = round(0.65 * opportunity + 0.35 * content_fit["content_opportunity_score"], 4)
    return {
        "opportunity_score": opportunity,
        "content_opportunity_score": content_fit["content_opportunity_score"],
        "trend_strength_score": content_fit["trend_strength_score"],
        "production_feasibility_score": content_fit["production_feasibility_score"],
        "hook_strength_score": content_fit["hook_strength_score"],
        "emotion_fit_score": content_fit["emotion_fit_score"],
        "originality_score": content_fit["originality_score"],
        "character_fit": content_fit["character_fit"],
        "best_character": content_fit["best_character"],
        "best_format": content_fit["best_format"],
        "trend_transfer_analysis": analyze_trend_transfer(signal),
    }


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
        placed = False
        for group in groups:
            if any(_trend_related(row, existing) for existing in group):
                group.append(row)
                placed = True
                break
        if not placed:
            groups.append([row])
    return groups


def _upstream_source(row: dict[str, Any]) -> str:
    """The platform a row really comes from. Google News search rows count once, whatever wrapper they arrive in."""
    source = str(row.get("source_type") or row.get("source") or "unknown")
    if source in {"google_news", "web_social_proxy"} or str(row.get("source", "")) in {"google_news", "google_social_search"}:
        return "google_news"
    return source


def enrich_signals(rows: Iterable[dict[str, Any]], now: datetime | None = None) -> list[TrendSignalV2]:
    raw = [dict(x) for x in rows]
    groups = _dedupe_groups(raw)
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
        source_count = len({_upstream_source(x) for x in group})
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
        opportunity = recompute_opportunity_score({
            "formats": formats, "emotions": emotions,
            "freshness_score": fresh, "velocity_score": velocity,
            "cross_platform_score": cross_platform, "saturation_score": saturation,
            "novelty_score": novelty, "title": _norm(str(head.get("title", ""))),
            "description": str(head.get("description", "")),
            "category": str(head.get("category", "")), "hashtags": head.get("hashtags") or [],
            "music": str(head.get("music", "")), "hook_patterns": hooks,
        })["opportunity_score"]
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
        recomputed = recompute_opportunity_score({
            "title": title, "description": str(head.get("description", "")),
            "category": str(head.get("category", "")), "hashtags": head.get("hashtags") or [],
            "music": str(head.get("music", "")), "formats": formats,
            "hook_patterns": hooks, "emotions": emotions,
            "velocity_score": velocity, "freshness_score": fresh,
            "cross_platform_score": cross_platform, "novelty_score": novelty,
            "saturation_score": saturation,
        })
        content_fit.update({k: recomputed[k] for k in (
            "content_opportunity_score", "best_character", "best_format", "character_fit"
        )})
        opportunity = recomputed["opportunity_score"]
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
            trend_transfer_analysis=analyze_trend_transfer({
                "title": title, "description": str(head.get("description", "")),
                "category": str(head.get("category", "")), "hashtags": head.get("hashtags") or [],
                "music": str(head.get("music", "")), "duration_seconds": head.get("duration_seconds", 0),
                "views": max(int(x.get("views", 0) or 0) for x in group),
                "likes": max(int(x.get("likes", 0) or 0) for x in group),
                "comments": max(int(x.get("comments", 0) or 0) for x in group),
                "shares": max(int(x.get("shares", 0) or 0) for x in group),
                "source_count": source_count, "mention_count": mention_count,
                "formats": formats, "emotions": emotions, "hook_patterns": hooks,
                "saturation_score": saturation,
            }),
        ))
    return sorted(output, key=lambda x: (x.opportunity_score, x.velocity_score), reverse=True)


REDDIT_SUBREDDITS = ("aww", "cats", "birding", "robots", "wholesomememes")
REDDIT_USER_AGENT = "linux:nova-ai-influencer:2.1 (public RSS trend reader; contact via repository owner)"
REDDIT_PAUSE_SECONDS = 2.0
PUBLIC_USER_AGENT = "NOVA-Trend-Scout/2.1 (public feed reader; contact via repository owner)"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
GOOGLE_TRENDS_NS = "{https://trends.google.com/trending/rss}"
YOUTUBE_FAMILY_CATEGORIES = ("1", "15", "23", "24")  # Film & Animation, Pets & Animals, Comedy, Entertainment
YOUTUBE_SHORTS_MAX_SECONDS = 180.0
_ISO_DURATION = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$")
_TAG = re.compile(r"<[^>]+>")


def _error_label(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(exc, httpx.HTTPStatusError) and response is not None:
        return f"http_{response.status_code}"
    return f"error:{type(exc).__name__}"


def _plain(text: str) -> str:
    """Visible text of a feed description: entities decoded, markup removed."""
    return _norm(html.unescape(_TAG.sub(" ", html.unescape(text or ""))))


def _without_publisher(title: str, publisher: str) -> str:
    suffix = f" - {publisher}" if publisher else ""
    return title[: -len(suffix)].rstrip() if suffix and title.endswith(suffix) else title


def _views(value: str | None) -> int:
    digits = re.sub(r"\D", "", value or "")
    return int(digits) if digits else 0


def _atom_link(entry: ET.Element) -> str:
    links = entry.findall(f"{ATOM_NS}link")
    for link in links:
        if link.get("rel", "alternate") == "alternate" and link.get("href"):
            return link.get("href", "")
    return next((link.get("href", "") for link in links if link.get("href")), "")


def _iso_seconds(value: str) -> float:
    """Seconds from an ISO-8601 duration such as PT1M30S (YouTube contentDetails.duration)."""
    match = _ISO_DURATION.match(value or "")
    if not match:
        return 0.0
    days, hours, minutes, seconds = (float(part or 0) for part in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _rss_items(url: str, source_type: str, category: str, timeout: float = 8.0,
               user_agent: str = PUBLIC_USER_AGENT) -> list[dict[str, Any]]:
    """Items from an RSS 2.0 feed or an Atom 1.0 feed (Reddit serves Atom)."""
    response = httpx.get(url, timeout=timeout, headers={"User-Agent": user_agent})
    response.raise_for_status()
    root = ET.fromstring(response.text)
    out: list[dict[str, Any]] = []
    if root.tag == f"{ATOM_NS}feed":
        for entry in root.findall(f"{ATOM_NS}entry")[:30]:
            out.append({
                "title": _plain(entry.findtext(f"{ATOM_NS}title", "")),
                "source": source_type,
                "source_type": source_type,
                "category": category,
                "url": _atom_link(entry),
                "published_at": _norm(entry.findtext(f"{ATOM_NS}published", "") or entry.findtext(f"{ATOM_NS}updated", "")),
                "description": _plain(entry.findtext(f"{ATOM_NS}content", "")),
                "region": "IN",
            })
    else:
        for item in root.findall(".//item")[:30]:
            publisher = _norm(item.findtext("source", ""))
            title = _plain(_without_publisher(_norm(item.findtext("title", "")), publisher))
            news = [_norm(node.findtext(f"{GOOGLE_TRENDS_NS}news_item_title", ""))
                    for node in item.findall(f"{GOOGLE_TRENDS_NS}news_item")]
            description = _plain(item.findtext("description", ""))
            if news:
                description = "; ".join(x for x in news if x) or description
            # Google News repeats the headline in its description; that adds no signal.
            if description.startswith(title):
                description = ""
            row: dict[str, Any] = {
                "title": title,
                "source": source_type,
                "source_type": source_type,
                "category": category,
                "url": _norm(item.findtext("link", "")),
                "published_at": _norm(item.findtext("pubDate", "")),
                "description": description,
                "region": "IN",
            }
            traffic = item.findtext(f"{GOOGLE_TRENDS_NS}approx_traffic")
            if traffic is not None:
                row["views"] = _views(traffic)
            out.append(row)
    return [x for x in out if x["title"]]


def discover_reddit_rss(timeout: float = 8.0, status: dict[str, str] | None = None,
                        pause_seconds: float = REDDIT_PAUSE_SECONDS) -> list[dict[str, Any]]:
    """Public subreddit feeds, requested one at a time and reported per subreddit."""
    status = status if status is not None else {}
    rows: list[dict[str, Any]] = []
    for index, sub in enumerate(REDDIT_SUBREDDITS):
        if index and pause_seconds:
            time.sleep(pause_seconds)
        try:
            found = _rss_items(f"https://www.reddit.com/r/{sub}/.rss", "reddit", f"reddit_{sub.lower()}",
                               timeout, user_agent=REDDIT_USER_AGENT)
        except Exception as exc:
            status[sub] = _error_label(exc)
            continue
        status[sub] = f"ok:{len(found)}" if found else "parsed_0"
        rows.extend(found)
    return rows


def discover_google_trends_rss(region_code: str = "IN", timeout: float = 8.0,
                               status: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Google Trends daily feed (zero-key). Traffic and the news context are kept per term."""
    status = status if status is not None else {}
    try:
        rows = _rss_items(f"https://trends.google.com/trending/rss?geo={region_code}", "google_trends", "google_trends", timeout)
    except Exception as exc:
        status["google_trends"] = _error_label(exc)
        return []
    status["google_trends"] = f"ok:{len(rows)}" if rows else "parsed_0"
    return rows


def discover_youtube_most_popular(api_key: str = "", region_code: str = "IN", timeout: float = 8.0,
                                  status: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Most-popular videos in family-friendly categories, kept only when Shorts-length (<= 180 s)."""
    status = status if status is not None else {}
    api_key = api_key or os.getenv("YOUTUBE_API_KEY", "")
    if not api_key:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        for category in YOUTUBE_FAMILY_CATEGORIES:
            params = {
                "part": "snippet,statistics,contentDetails",
                "chart": "mostPopular",
                "regionCode": region_code,
                "videoCategoryId": category,
                "maxResults": 50,
                "key": api_key,
            }
            response = httpx.get("https://www.googleapis.com/youtube/v3/videos", params=params, timeout=timeout)
            response.raise_for_status()
            for item in response.json().get("items", []):
                video_id = str(item.get("id", ""))
                seconds = _iso_seconds(item.get("contentDetails", {}).get("duration", ""))
                if not video_id or video_id in seen or not 0 < seconds <= YOUTUBE_SHORTS_MAX_SECONDS:
                    continue
                seen.add(video_id)
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                out.append({
                    "title": _norm(snippet.get("title", "")),
                    "description": _norm(snippet.get("description", "")),
                    "source": "youtube",
                    "source_type": "youtube",
                    "category": f"youtube_category_{category}",
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "published_at": snippet.get("publishedAt", ""),
                    "views": int(stats.get("viewCount", 0) or 0),
                    "likes": int(stats.get("likeCount", 0) or 0),
                    "comments": int(stats.get("commentCount", 0) or 0),
                    "duration_seconds": seconds,
                    "region": region_code,
                })
    except Exception as exc:
        status["youtube"] = _error_label(exc)
        return out
    status["youtube"] = f"ok:{len(out)}" if out else "parsed_0"
    return out


TIKTOK_TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
TIKTOK_QUERY_URL = "https://open.tiktokapis.com/v2/research/video/query/"


def _tiktok_access_token(timeout: float, status: dict[str, str]) -> str:
    """Client-credential tokens expire after about two hours, so one is minted per run."""
    key = os.getenv("TIKTOK_RESEARCH_CLIENT_KEY", "").strip()
    secret = os.getenv("TIKTOK_RESEARCH_CLIENT_SECRET", "").strip()
    if not key or not secret:
        return os.getenv("TIKTOK_RESEARCH_ACCESS_TOKEN", "").strip()
    try:
        response = httpx.post(
            TIKTOK_TOKEN_URL,
            data={"client_key": key, "client_secret": secret, "grant_type": "client_credentials"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
        response.raise_for_status()
        return str(response.json().get("access_token", "")).strip()
    except Exception as exc:
        status["token"] = _error_label(exc)
        return ""


def discover_tiktok_research(access_token: str = "", query: str = "", region: str = "", timeout: float = 8.0,
                             status: dict[str, str] | None = None) -> list[dict[str, Any]]:
    status = status if status is not None else {}
    token = access_token or _tiktok_access_token(timeout, status)
    if not token:
        return []
    # Research API regions are a list; the owner's default targets markets where TikTok is available.
    regions = [r.strip().upper() for r in (region or os.getenv("TIKTOK_RESEARCH_REGION", "US,GB")).split(",") if r.strip()]
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
                    {"operation": "IN", "field_name": "region_code", "field_values": regions},
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
                TIKTOK_QUERY_URL,
                params={"fields": fields},
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            videos = response.json().get("data", {}).get("videos", [])
        except Exception as exc:
            status[f"keyword:{keyword}"] = _error_label(exc)
            continue
        status[f"keyword:{keyword}"] = f"ok:{len(videos)}" if videos else "parsed_0"
        for item in videos:
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
                "region": item.get("region_code", ""),
            })
    return out


def discover_web_social_mentions(timeout: float = 8.0, status: dict[str, str] | None = None) -> list[dict[str, Any]]:
    status = status if status is not None else {}
    queries = (
        ("instagram_mentions", "Instagram Reels trend viral cute cartoon meme"),
        ("tiktok_mentions", "TikTok trend viral sound meme dance"),
        ("youtube_short_mentions", "YouTube Shorts trend viral meme"),
    )
    rows: list[dict[str, Any]] = []
    for category, query in queries:
        url = "https://news.google.com/rss/search?q=" + quote(f"{query} when:2d") + "&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            found = _rss_items(url, "google_social_search", category, timeout)
        except Exception as exc:
            status[category] = _error_label(exc)
            continue
        status[category] = f"ok:{len(found)}" if found else "parsed_0"
        for row in found:
            row["source_type"] = "web_social_proxy"
            rows.append(row)
    return rows


DEFAULT_INSTAGRAM_TREND_HASHTAGS = "cartoon,animation,funnycats,catsofinstagram,robot,cutebirds,relatable,pov"


def discover_instagram_hashtag_trends(timeout: float = 8.0, http_get: Any = None) -> list[dict[str, Any]]:
    """Top Reels/posts for chosen hashtags via the Instagram Graph API hashtag search.

    Instagram has no public "trending" endpoint. Hashtag search is the official
    route: it needs a Business/Creator account (the same token and user id used
    for publishing) and Meta's Instagram Public Content Access feature, and it
    is limited to 30 unique hashtags per rolling 7 days.
    """
    token = os.getenv("INSTAGRAM_ACCESS_TOKEN", "").strip()
    user = os.getenv("INSTAGRAM_USER_ID", "").strip()
    if not token or not user:
        return []
    version = os.getenv("INSTAGRAM_GRAPH_VERSION", "v26.0")
    graph = f"https://graph.facebook.com/{version}"
    tags = [t.strip().lstrip("#") for t in os.getenv("INSTAGRAM_TREND_HASHTAGS", DEFAULT_INSTAGRAM_TREND_HASHTAGS).split(",") if t.strip()][:10]
    get = http_get or (lambda url, params: httpx.get(url, params=params, timeout=timeout))
    rows: list[dict[str, Any]] = []
    for tag in tags:
        try:
            found = get(f"{graph}/ig_hashtag_search", {"user_id": user, "q": tag, "access_token": token})
            found.raise_for_status()
            hashtag_id = (found.json().get("data") or [{}])[0].get("id")
            if not hashtag_id:
                continue
            media = get(f"{graph}/{hashtag_id}/top_media", {
                "user_id": user,
                "fields": "id,caption,media_type,like_count,comments_count,timestamp,permalink",
                "limit": 25,
                "access_token": token,
            })
            media.raise_for_status()
        except Exception:
            continue
        for item in media.json().get("data", []):
            caption = _norm(item.get("caption", ""))
            rows.append({
                "title": caption[:140] or f"#{tag}",
                "description": caption,
                "source": "instagram",
                "source_type": "instagram",
                "category": f"instagram_hashtag_{tag}",
                "url": item.get("permalink", ""),
                "published_at": item.get("timestamp", ""),
                "likes": int(item.get("like_count", 0) or 0),
                "comments": int(item.get("comments_count", 0) or 0),
                "hashtags": [tag],
                "duration_seconds": 0.0,
            })
    return rows


def trend_source_status() -> dict[str, dict[str, Any]]:
    """Which trend connectors are configured (credentials present). Reachability is not tested here."""
    def env(*names: str) -> list[str]:
        return [n for n in names if not os.getenv(n, "").strip()]

    # A static access token still works for a while; client credentials are preferred
    # because they are exchanged for a fresh token on every run.
    tiktok_needs = [] if os.getenv("TIKTOK_RESEARCH_ACCESS_TOKEN", "").strip() else env(
        "TIKTOK_RESEARCH_CLIENT_KEY", "TIKTOK_RESEARCH_CLIENT_SECRET")
    sources = {
        "google_news_gdelt": {"needs": [], "note": "public RSS/API, no key"},
        "reddit": {"needs": [], "note": "public Atom feeds, no key; may be blocked from some hosts"},
        "google_trends": {"needs": [], "note": "public RSS, no key; may be blocked from some hosts"},
        "youtube": {"needs": env("YOUTUBE_API_KEY"), "note": "YouTube Data API v3 key (mostPopular chart)"},
        "tiktok": {"needs": tiktok_needs,
                   "note": "TikTok Research API (approved non-commercial researchers only)"},
        "instagram": {"needs": env("INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_USER_ID"),
                      "note": "Graph API hashtag search; needs Instagram Public Content Access"},
    }
    for row in sources.values():
        row["configured"] = not row["needs"]
        # Kept for existing readers; it means "configured", not "reachable".
        row["active"] = row["configured"]
    return sources


def _status_summary(prefix: str, detail: dict[str, str]) -> str:
    if not detail:
        return prefix
    return prefix + ":" + ",".join(f"{key}={value}" for key, value in sorted(detail.items()))


def discover_social_trends(timeout: float = 8.0) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Connector rows plus one status string per source, e.g. ``public_rss:aww=ok:12,cats=http_403``."""
    rows: list[dict[str, Any]] = []
    status = {
        "youtube": "skipped_no_api_key",
        "tiktok": "skipped_no_research_credentials",
        "instagram": "skipped_no_graph_credentials",
    }
    reddit_detail: dict[str, str] = {}
    rows.extend(discover_reddit_rss(timeout, reddit_detail))
    status["reddit"] = _status_summary("public_rss", reddit_detail)
    trends_detail: dict[str, str] = {}
    google_trends = discover_google_trends_rss(timeout=timeout, status=trends_detail)
    status["google_trends"] = trends_detail.get("google_trends", "unavailable")
    if google_trends:
        rows.extend(google_trends)
    yt_detail: dict[str, str] = {}
    rows.extend(discover_youtube_most_popular(timeout=timeout, status=yt_detail))
    if yt_detail:
        status["youtube"] = yt_detail["youtube"]
    tt_detail: dict[str, str] = {}
    rows.extend(discover_tiktok_research(timeout=timeout, status=tt_detail))
    if tt_detail:
        status["tiktok"] = _status_summary("research_api", tt_detail)
    ig = discover_instagram_hashtag_trends(timeout=timeout)
    if ig:
        status["instagram"] = "graph_api_hashtag_search"
        rows.extend(ig)
    web_detail: dict[str, str] = {}
    rows.extend(discover_web_social_mentions(timeout, web_detail))
    status["google_social_search"] = _status_summary("public_rss", web_detail)
    return rows, status


if __name__ == "__main__":
    print(json.dumps(trend_source_status(), indent=2))


def signal_to_dict(signal: TrendSignalV2) -> dict[str, Any]:
    return asdict(signal)
