"""Zero-cost trend intelligence for NOVA.

Ranks public feed items by freshness, relevance to NOVA's pillars, comedy
potential, visual potential, and novelty rather than turning every headline
into a video.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
import xml.etree.ElementTree as ET

import httpx

FEEDS = {
    "google_news_ai": "https://news.google.com/rss/search?q=AI%20technology&hl=en-IN&gl=IN&ceid=IN:en",
    "google_news_social": "https://news.google.com/rss/search?q=social%20media%20trends&hl=en-IN&gl=IN&ceid=IN:en",
    "google_news_creator": "https://news.google.com/rss/search?q=creator%20economy&hl=en-IN&gl=IN&ceid=IN:en",
}

@dataclass
class Trend:
    title: str
    source: str
    url: str = ""
    published_at: str = ""
    category: str = ""
    score: float = 0.0
    why_now: str = ""
    freshness_score: float = 0.0
    relevance_score: float = 0.0
    comedy_score: float = 0.0
    visual_score: float = 0.0
    novelty_score: float = 1.0


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _parse_pub(value: str) -> str:
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat()
    except Exception:
        return value


def parse_rss(xml_text: str, source: str, category: str, limit: int = 10) -> list[Trend]:
    root = ET.fromstring(xml_text)
    items = []
    for item in root.findall(".//item")[:limit]:
        title = _clean(item.findtext("title", ""))
        link = _clean(item.findtext("link", ""))
        pub = _parse_pub(item.findtext("pubDate", ""))
        if title:
            items.append(Trend(title=title, source=source, url=link, published_at=pub, category=category))
    return items


def _freshness(published_at: str) -> float:
    if not published_at:
        return 0.35
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        hours = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600)
    except Exception:
        return 0.25
    if hours <= 6:
        return 1.0
    if hours <= 24:
        return 0.85
    if hours <= 72:
        return 0.65
    if hours <= 168:
        return 0.40
    return 0.15


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def score_trend_fit(trend: Trend, seen_titles: list[str] | None = None) -> Trend:
    seen_titles = seen_titles or []
    text = trend.title.lower()

    relevance_terms = (
        "ai", "artificial intelligence", "agent", "chatgpt", "automation",
        "creator", "instagram", "tiktok", "social media", "productivity",
        "workplace", "software", "tech", "robot", "internet",
    )
    comedy_terms = (
        "meeting", "office", "work", "manager", "productivity", "meme",
        "viral", "weird", "awkward", "fails", "chaos", "trend",
    )
    visual_terms = (
        "robot", "ai", "app", "phone", "screen", "creator", "instagram",
        "tiktok", "meme", "office", "fashion", "video", "launch",
    )

    relevance = min(1.0, 0.20 + 0.12 * sum(term in text for term in relevance_terms))
    comedy = min(1.0, 0.20 + 0.15 * sum(term in text for term in comedy_terms))
    visual = min(1.0, 0.20 + 0.15 * sum(term in text for term in visual_terms))
    freshness = _freshness(trend.published_at)

    novelty = 1.0
    tokens = set(re.findall(r"[a-z0-9']+", text))
    for old in seen_titles:
        old_tokens = set(re.findall(r"[a-z0-9']+", old.lower()))
        union = tokens | old_tokens
        if union and len(tokens & old_tokens) / len(union) > 0.65:
            novelty = 0.25
            break

    category_bonus = 0.12 if trend.category in {"ai", "creator", "social"} else 0.0
    score = (
        freshness * 0.24
        + relevance * 0.32
        + comedy * 0.18
        + visual * 0.14
        + novelty * 0.08
        + category_bonus
    )
    trend.freshness_score = round(freshness, 3)
    trend.relevance_score = round(relevance, 3)
    trend.comedy_score = round(comedy, 3)
    trend.visual_score = round(visual, 3)
    trend.novelty_score = round(novelty, 3)
    trend.score = round(min(score, 1.0), 3)
    trend.why_now = (
        "Strong NOVA fit: "
        f"freshness={trend.freshness_score}, relevance={trend.relevance_score}, "
        f"comedy={trend.comedy_score}, visual={trend.visual_score}."
    )
    return trend


def rank_trends(trends: list[Trend]) -> list[Trend]:
    seen: list[str] = []
    ranked = []
    for trend in trends:
        scored = score_trend_fit(trend, seen)
        ranked.append(scored)
        if scored.novelty_score > 0.5:
            seen.append(scored.title)
    return sorted(ranked, key=lambda t: t.score, reverse=True)


def discover_trends(timeout: float = 10.0) -> list[Trend]:
    results = []
    for name, url in FEEDS.items():
        try:
            category = name.replace("google_news_", "")
            response = httpx.get(url, timeout=timeout, headers={"User-Agent": "NOVA-Trend-Scout/0.3"})
            response.raise_for_status()
            results.extend(parse_rss(response.text, name, category))
        except Exception:
            continue

    seen = set()
    unique = []
    for trend in results:
        key = re.sub(r"W+", " ", trend.title.lower()).strip()
        if key not in seen:
            seen.add(key)
            unique.append(trend)
    return rank_trends(unique)


def trend_to_dict(trend: Trend) -> dict:
    return asdict(trend)
