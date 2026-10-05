"""Zero-cost trend discovery and ranking using public RSS feeds."""

from dataclasses import dataclass, asdict
from datetime import timezone
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

def rank_trends(trends: list[Trend]) -> list[Trend]:
    keywords = {
        "ai": 1.2,
        "artificial intelligence": 1.5,
        "chatgpt": 1.4,
        "robot": 1.0,
        "technology": 0.8,
        "instagram": 0.8,
        "tiktok": 0.9,
        "creator": 0.6,
        "viral": 1.2,
        "meme": 1.0,
        "work": 0.5,
        "money": 0.6,
    }
    for trend in trends:
        text = trend.title.lower()
        score = 1.0 + sum(weight for key, weight in keywords.items() if key in text)
        if trend.published_at:
            score += 0.25
        trend.score = round(min(score / 5.5, 1.0), 3)
        trend.why_now = "High-signal topic from a current public feed; use it only as inspiration."
    return sorted(trends, key=lambda t: t.score, reverse=True)

def discover_trends(timeout: float = 10.0) -> list[Trend]:
    results = []
    for name, url in FEEDS.items():
        try:
            category = name.replace("google_news_", "")
            response = httpx.get(url, timeout=timeout, headers={"User-Agent": "NOVA-Trend-Scout/0.2"})
            response.raise_for_status()
            results.extend(parse_rss(response.text, name, category))
        except Exception:
            continue

    seen = set()
    unique = []
    for trend in results:
        key = re.sub(r"\W+", " ", trend.title.lower()).strip()
        if key not in seen:
            seen.add(key)
            unique.append(trend)
    return rank_trends(unique)

def trend_to_dict(trend: Trend) -> dict:
    return asdict(trend)
