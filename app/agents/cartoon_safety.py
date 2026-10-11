"""Family-safety and real-person filter shared by research, ideation and the gates.

The cartoon brand is for kids and families. Any research topic that is violent,
about death or disasters, crime, politics, adult content, hate, self-harm,
weapons or alcohol, or that appears to name a real person, is replaced by a
generic evergreen premise before it can reach an idea, a title or a caption.

Matching is whole-word (case-insensitive), so harmless words such as "whatever"
or "chateau" never trip a term like "hate".
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

UNSAFE_TERMS: dict[str, tuple[str, ...]] = {
    "violence": (
        "kill", "kills", "killed", "killing", "murder", "murders", "murdered", "stab", "stabbed",
        "stabbing", "shooting", "shootout", "gunman", "gunmen", "gun", "guns", "gunfire", "bomb",
        "bombs", "bombing", "bomb threat", "explode", "explodes", "exploded", "explosion",
        "explosions", "blast", "attack", "attacks", "attacked", "assault", "assaulted", "war",
        "wars", "terror", "terrorist", "terrorists", "terrorism", "hostage", "hostages", "violence",
        "violent", "blood", "bloody", "gore", "gory", "massacre", "riot", "riots", "weapon",
        "weapons", "knife", "knives", "rifle", "missile", "missiles",
    ),
    "death": (
        "dead", "death", "deaths", "died", "dies", "dying", "funeral", "funerals", "corpse",
        "corpses", "fatal", "fatally", "drown", "drowned", "tragedy", "tragic", "mourn", "mourning",
    ),
    "disaster": (
        "flood", "floods", "flooding", "earthquake", "earthquakes", "tsunami", "cyclone", "hurricane",
        "wildfire", "wildfires", "landslide", "landslides", "avalanche", "disaster", "disasters",
        "crash", "crashes", "crashed", "collapse", "collapsed", "stampede",
    ),
    "crime": (
        "arrest", "arrested", "arrests", "drug", "drugs", "cocaine", "heroin", "smuggling", "smuggle",
        "theft", "stolen", "robbery", "robbed", "scam", "scams", "fraud", "kidnap", "kidnapped",
        "kidnapping", "rape", "molest", "trafficking", "extortion", "fugitive", "criminal",
        "criminals", "crime", "crimes",
    ),
    "politics": (
        "election", "elections", "politics", "political", "politician", "politicians", "minister",
        "ministers", "government", "parliament", "parliamentary", "prime minister", "president",
        "senator", "congress", "protest", "protests", "protesters",
    ),
    "adult": (
        "sex", "sexy", "sexual", "sexually", "nude", "nudity", "naked", "porn", "porno", "onlyfans",
        "strip club", "erotic", "lingerie", "nsfw", "xxx",
    ),
    "hate": (
        "hate", "hateful", "racist", "racism", "slur", "slurs", "sexist", "bigot", "bigotry", "nazi",
        "extremist", "extremism",
    ),
    "self_harm": ("suicide", "suicidal", "self harm", "self-harm"),
    "substances": (
        "alcohol", "alcoholic", "beer", "wine", "vodka", "whiskey", "whisky", "drunk", "drunken",
        "cigarette", "cigarettes", "vape", "vaping", "weed", "marijuana", "casino", "gambling",
    ),
    "real_person_or_reuse": (
        "celebrity", "celebrities", "lookalike", "private person", "real child", "minor child",
        "underage", "face swap", "repost", "copyrighted character",
    ),
}

# Capitalised words that do not make a person's name on their own (brands, outlets, tech).
_NAME_ALLOWLIST = frozenset({
    "google", "news", "india", "today", "times", "hindustan", "hindu", "ndtv", "live", "score",
    "breaking", "youtube", "shorts", "reels", "instagram", "tiktok", "apple", "smart", "home",
    "robot", "vacuum", "artificial", "intelligence", "ai", "the", "new", "how", "why", "what",
    "delhi", "mumbai", "kerala", "bengaluru", "chennai", "australia", "vs", "of", "and",
})

_CAPITALISED_RUN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def _term_regex(term: str) -> re.Pattern[str]:
    words = r"\s+".join(re.escape(part) for part in term.split())
    return re.compile(rf"(?<![a-z0-9]){words}(?:s|es)?(?![a-z0-9])", re.IGNORECASE)


@lru_cache(maxsize=None)
def _compiled() -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple(
        (category, _term_regex(term))
        for category, terms in UNSAFE_TERMS.items()
        for term in terms
    )


def contains_term(text: str, term: str) -> bool:
    """Whole-word, case-insensitive containment with an optional plural 's' or 'es'."""
    return bool(_term_regex(term).search(text or ""))


def unsafe_reasons(text: str) -> list[str]:
    """Unsafe categories matched in ``text``; an empty list means no unsafe term was found."""
    value = text or ""
    return sorted({category for category, pattern in _compiled() if pattern.search(value)})


def looks_like_real_person(text: str) -> bool:
    """True when the text has two or more consecutive capitalised name-like words."""
    for match in _CAPITALISED_RUN.finditer(text or ""):
        words = match.group(0).split()
        if sum(1 for word in words if word.lower() not in _NAME_ALLOWLIST) >= 2:
            return True
    return False


def topic_problems(text: str) -> list[str]:
    """Reasons a research topic cannot be used as-is; an empty list means it is usable."""
    problems = unsafe_reasons(text)
    if looks_like_real_person(text):
        problems.append("real_person_name")
    return problems


def evergreen_pick(options: tuple[str, ...], *seed: str) -> str:
    """Deterministic, seed-stable choice from a non-empty tuple."""
    digest = hashlib.sha256("||".join(seed).encode()).hexdigest()
    return options[int(digest, 16) % len(options)]


def sanitize_topic(title: str, options: tuple[str, ...], seed: str) -> tuple[str, str]:
    """Return ``(topic, status)``: the topic itself when safe, else a generic evergreen premise.

    ``status`` is ``"ok"`` or ``"replaced:<reason>[,<reason>]"``.
    """
    problems = topic_problems(title)
    if not problems:
        return title, "ok"
    return evergreen_pick(options, seed, title), "replaced:" + ",".join(problems)


def public_text_problems(*texts: str) -> list[str]:
    """Safety reasons for viewer-facing text (titles, captions, lines). Any hit fails closed."""
    reasons: set[str] = set()
    for text in texts:
        reasons.update(unsafe_reasons(text))
    return sorted(reasons)
