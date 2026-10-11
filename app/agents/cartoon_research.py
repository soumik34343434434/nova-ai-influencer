"""Multi-source, free web research for the cartoon content factory.

Every source reports its own outcome (``ok:N``, ``empty``, ``error:...``) so a run
can tell a quiet news day from a broken connector. Topics are screened by the
family-safety filter before they leave this module; an unsafe topic is replaced
by a generic evergreen premise rather than being passed on.
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

import httpx

from app.agents.cartoon_safety import sanitize_topic


GOOGLE_QUERIES = (
    ("ai_gadgets", "AI gadgets funny consumer technology"),
    ("internet_humor", "internet meme weird funny trends"),
    ("school_tech", "school homework technology trends"),
    ("home_tech", "smart home kitchen robot gadgets"),
    ("animation", "animation cartoon creator trends"),
)
# GDELT DOC 2.0 needs OR'd terms wrapped in parentheses and phrases quoted. A single
# request also stays far below the public limit of one request every five seconds.
GDELT_QUERY = '(robot OR gadget OR "smart home" OR meme OR homework) sourcelang:english'
GDELT_QUERIES = (GDELT_QUERY,)
USER_AGENT = "NOVA-CartoonFactory-Research/1.1 (public feed reader; contact via repository owner)"

# Generic, family-safe premises. They replace unsafe topics and cover degraded runs,
# so a day with no live signals still gets a varied pool instead of one fixed idea.
EVERGREEN_TOPICS: tuple[dict[str, Any], ...] = (
    {"title": "a robot vacuum meets a curious cat", "category": "evergreen_home",
     "formats": ("story_comedy", "cute_animal"), "emotions": ("amusement", "cute")},
    {"title": "homework that refuses to get finished", "category": "evergreen_school",
     "formats": ("story_comedy", "challenge"), "emotions": ("amusement", "frustration")},
    {"title": "a phone that buzzes at the wrong moment", "category": "evergreen_tech",
     "formats": ("reaction", "pov"), "emotions": ("surprise", "amusement")},
    {"title": "a tiny robot tries to fix the kitchen", "category": "evergreen_tech",
     "formats": ("transformation", "absurdist"), "emotions": ("curiosity", "surprise")},
    {"title": "the best sandwich recipe goes off track", "category": "evergreen_food",
     "formats": ("micro_short", "listicle"), "emotions": ("amusement", "satisfaction")},
    {"title": "a cupcake bakery opens on a Monday", "category": "evergreen_food",
     "formats": ("story_comedy", "reaction"), "emotions": ("cute", "amusement")},
    {"title": "a garden plant that grows too fast", "category": "evergreen_nature",
     "formats": ("absurdist", "transformation"), "emotions": ("awe", "surprise")},
    {"title": "a dance routine for one tiny bird", "category": "evergreen_music",
     "formats": ("dance", "trend_audio"), "emotions": ("cute", "amusement")},
    {"title": "a bedtime story that keeps getting longer", "category": "evergreen_home",
     "formats": ("pov", "story_comedy"), "emotions": ("amusement", "cute")},
    {"title": "a park ball that never stops bouncing", "category": "evergreen_sport",
     "formats": ("challenge", "micro_short"), "emotions": ("surprise", "amusement")},
    {"title": "a sleepy cat and a very loud alarm", "category": "evergreen_home",
     "formats": ("pov", "reaction"), "emotions": ("amusement", "cringe")},
    {"title": "a new gadget with too many buttons", "category": "evergreen_tech",
     "formats": ("reaction", "listicle", "absurdist"), "emotions": ("curiosity", "amusement")},
)
EVERGREEN_TITLES = tuple(str(topic["title"]) for topic in EVERGREEN_TOPICS)


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


def _publisher_free(title: str, publisher: str) -> str:
    """Google News titles end in ' - Publisher'; the publisher is not part of the topic."""
    suffix = f" - {publisher}" if publisher else ""
    if suffix and title.endswith(suffix):
        return title[: -len(suffix)].rstrip()
    return title


def _rss(url: str, category: str, timeout: float) -> list[ResearchSignal]:
    response = httpx.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    root = ET.fromstring(response.text)
    signals: list[ResearchSignal] = []
    for item in root.findall(".//item")[:12]:
        publisher = _clean(item.findtext("source", ""))
        title = _clean(_publisher_free(_clean(item.findtext("title", "")), publisher))
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
        + "&mode=artlist&maxrecords=25&timespan=24h&sort=datedesc&format=json"
    )
    response = httpx.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    # GDELT answers invalid queries and rate limits with HTTP 200 plain text, so
    # the body is parsed explicitly and a non-JSON answer is reported, not hidden.
    try:
        data = json.loads(response.text)
    except ValueError as exc:
        raise ValueError(f"non_json:{_clean(response.text)[:120]}") from exc
    results: list[ResearchSignal] = []
    for row in (data.get("articles") or [])[:25]:
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


def _freshness(published_at: str, now: datetime) -> float:
    if not published_at:
        return 0.4
    try:
        when = parsedate_to_datetime(published_at)
    except (TypeError, ValueError):
        try:
            when = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except ValueError:
            return 0.4
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    hours = (now - when.astimezone(timezone.utc)).total_seconds() / 3600.0
    if hours <= 24:
        return 0.9
    if hours <= 48:
        return 0.7
    if hours <= 72:
        return 0.5
    return 0.3


def rank(signals: list[ResearchSignal], recent_topics: list[str] | None = None) -> list[ResearchSignal]:
    recent_topics = recent_topics or []
    scored: list[ResearchSignal] = []
    fun = {"funny", "meme", "viral", "weird", "fails", "chaos", "awkward", "robot", "hack"}
    visual = {"robot", "gadget", "phone", "smart", "kitchen", "school", "home", "animation", "app"}
    now = datetime.now(timezone.utc)
    current_year = now.year
    recent_tokens = [_topic_tokens(old) for old in recent_topics]
    for signal in signals:
        text = signal.title.lower()
        tokens = _topic_tokens(signal.title)
        f = min(1.0, 0.25 + 0.12 * len(tokens & fun))
        v = min(1.0, 0.25 + 0.10 * len(tokens & visual))
        fresh = _freshness(signal.published_at, now)
        if str(current_year) in text:
            fresh = min(1.0, fresh + 0.1)
        novelty = 1.0
        for old_tokens in recent_tokens:
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


def screen_signals(signals: list[ResearchSignal], day: str = "") -> tuple[list[ResearchSignal], int]:
    """Replace unsafe or real-person topics with evergreen premises. Returns (signals, replaced_count)."""
    screened: list[ResearchSignal] = []
    replaced = 0
    for signal in signals:
        title, status = sanitize_topic(signal.title, EVERGREEN_TITLES, f"{day}|{signal.source}|{signal.url}")
        if status != "ok":
            replaced += 1
            signal = ResearchSignal(**{**asdict(signal), "title": title, "category": "evergreen_replacement"})
        screened.append(signal)
    return screened, replaced


def discover_report(
    limit: int = 30,
    timeout: float = 8.0,
    recent_topics: list[str] | None = None,
) -> tuple[list[ResearchSignal], dict[str, str]]:
    """Collect and rank signals, returning them with the outcome of every source."""
    status: dict[str, str] = {}
    signals: list[ResearchSignal] = []
    with ThreadPoolExecutor(max_workers=min(8, len(GOOGLE_QUERIES) + len(GDELT_QUERIES))) as pool:
        jobs: dict[Any, str] = {}
        for category, query in GOOGLE_QUERIES:
            url = (
                "https://news.google.com/rss/search?q=" + quote(f"{query} when:2d")
                + "&hl=en-IN&gl=IN&ceid=IN:en"
            )
            jobs[pool.submit(_rss, url, category, timeout)] = f"google_news:{category}"
        for query in GDELT_QUERIES:
            jobs[pool.submit(_gdelt, query, timeout)] = "gdelt:global_news"

        for job in as_completed(jobs):
            label = jobs[job]
            try:
                found = job.result()
            except Exception as exc:  # recorded per source; the run keeps going
                status[label] = f"error:{type(exc).__name__}:{str(exc)[:160]}"
                continue
            status[label] = f"ok:{len(found)}" if found else "empty"
            signals.extend(found)

    signals, replaced = screen_signals(signals, date.today().isoformat())
    status["safety"] = f"replaced:{replaced}"
    return rank(signals, recent_topics)[: max(1, limit)], status


def discover(limit: int = 30, timeout: float = 8.0, recent_topics: list[str] | None = None) -> list[ResearchSignal]:
    return discover_report(limit=limit, timeout=timeout, recent_topics=recent_topics)[0]


def evergreen_topics(day: str = "", recent_titles: list[str] | tuple[str, ...] = (), count: int = 8) -> list[dict[str, Any]]:
    """A deterministic, date-rotated set of safe premises that avoids recent topics when possible."""
    day = day or date.today().isoformat()
    recent = {re.sub(r"[^a-z0-9]+", " ", str(title).lower()).strip() for title in recent_titles}
    ordered = sorted(
        EVERGREEN_TOPICS,
        key=lambda topic: hashlib.sha256(f"{day}|{topic['title']}".encode()).hexdigest(),
    )
    fresh = [t for t in ordered if re.sub(r"[^a-z0-9]+", " ", t["title"]).strip() not in recent]
    stale = [t for t in ordered if t not in fresh]
    return [dict(t) for t in (fresh + stale)[: max(1, count)]]


def evergreen_signals(day: str = "", recent_titles: list[str] | tuple[str, ...] = (), count: int = 8) -> list[ResearchSignal]:
    now = datetime.now(timezone.utc).isoformat()
    return [
        ResearchSignal(title=str(topic["title"]), source="evergreen_bank", category=str(topic["category"]),
                       published_at=now)
        for topic in evergreen_topics(day, recent_titles, count)
    ]
