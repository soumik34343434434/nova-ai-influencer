"""Multi-source, free web research for the cartoon content factory."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import httpx


GOOGLE_QUERIES = (
    ("ai_gadgets", "AI gadgets funny consumer technology"),
    ("internet_humor", "internet meme weird funny trends"),
    ("school_tech", "school homework technology trends"),
    ("home_tech", "smart home kitchen robot gadgets"),
    ("animation", "animation cartoon creator trends"),
)
GDELT_QUERIES = (
    "AI gadget OR robot OR smart home",
    "funny OR meme OR viral technology",
    "homework OR school technology",
)


@dataclass(frozen=True)
class ResearchSignal:
    title: str
    source: str
    category: str
    url: str = ""
    published_at: str = ""
    score: float = 0.0


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _date(value: str) -> str:
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat()
    except Exception:
        return value or ""


def _rss(url: str, category: str, timeout: float) -> list[ResearchSignal]:
    response = httpx.get(url, timeout=timeout, headers={"User-Agent": "CartoonFactory-Research/1.0"})
    response.raise_for_status()
    root = ET.fromstring(response.text)
    signals: list[ResearchSignal] = []
    for item in root.findall(".//item")[:12]:
        title = _clean(item.findtext("title", ""))
        if title:
            signals.append(ResearchSignal(
                title=title,
                source="google_news",
                category=category,
                url=_clean(item.findtext("link", "")),
                published_at=_date(item.findtext("pubDate", "")),
            ))
    return signals


def _gdelt(query: str, timeout: float) -> list[ResearchSignal]:
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc?query=" + quote(query)
        + "&mode=artlist&maxrecords=15&timespan=24h&sort=datedesc&format=json"
    )
    response = httpx.get(url, timeout=timeout, headers={"User-Agent": "CartoonFactory-Research/1.0"})
    response.raise_for_status()
    data = response.json()
    results: list[ResearchSignal] = []
    for row in data.get("articles", [])[:15]:
        title = _clean(str(row.get("title", "")))
        if title:
            results.append(ResearchSignal(
                title=title,
                source="gdelt",
                category="global_news",
                url=str(row.get("url", "")),
                published_at=str(row.get("seendate", "")),
            ))
    return results


def _topic_tokens(title: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", title.lower()))


def rank(signals: list[ResearchSignal], recent_topics: list[str] | None = None) -> list[ResearchSignal]:
    recent_topics = recent_topics or []
    scored: list[ResearchSignal] = []
    fun = {"funny", "meme", "viral", "weird", "fails", "chaos", "awkward", "robot", "hack"}
    visual = {"robot", "gadget", "phone", "smart", "kitchen", "school", "home", "animation", "app"}
    current_year = datetime.now(timezone.utc).year
    for signal in signals:
        text = signal.title.lower()
        tokens = _topic_tokens(signal.title)
        f = min(1.0, 0.25 + 0.12 * len(tokens & fun))
        v = min(1.0, 0.25 + 0.10 * len(tokens & visual))
        fresh = 0.55
        if str(current_year) in text:
            fresh += 0.1
        novelty = 1.0
        for old in recent_topics:
            old_tokens = _topic_tokens(old)
            union = tokens | old_tokens
            if union and len(tokens & old_tokens) / len(union) > 0.65:
                novelty = 0.3
                break
        score = round(min(1.0, f * 0.4 + v * 0.3 + fresh * 0.2 + novelty * 0.1), 3)
        scored.append(ResearchSignal(**{**asdict(signal), "score": score}))
    # Deduplicate by normalized title while retaining highest score.
    best: dict[str, ResearchSignal] = {}
    for signal in scored:
        key = re.sub(r"[^a-z0-9]+", " ", signal.title.lower()).strip()
        if key and (key not in best or signal.score > best[key].score):
            best[key] = signal
    return sorted(best.values(), key=lambda x: x.score, reverse=True)


def discover(limit: int = 30, timeout: float = 8.0) -> list[ResearchSignal]:
    jobs = []
    with ThreadPoolExecutor(max_workers=min(8, len(GOOGLE_QUERIES) + len(GDELT_QUERIES))) as pool:
        for category, query in GOOGLE_QUERIES:
            url = (
                "https://news.google.com/rss/search?q=" + quote(query)
                + "&hl=en-IN&gl=IN&ceid=IN:en"
            )
            jobs.append(pool.submit(_rss, url, category, timeout))
        for query in GDELT_QUERIES:
            jobs.append(pool.submit(_gdelt, query, timeout))

        signals: list[ResearchSignal] = []
        for job in as_completed(jobs):
            try:
                signals.extend(job.result())
            except Exception:
                continue

    return rank(signals)[: max(1, limit)]
