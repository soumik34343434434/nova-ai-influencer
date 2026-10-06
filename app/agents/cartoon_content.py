"""Deterministic engine for wholesome cartoon comedy shorts.

The goal is original, family-friendly visual comedy built from a small cast of
recurring non-real characters. It does not target or depict a real person.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class CartoonEpisode:
    character: str
    title: str
    hook: str
    beats: tuple[dict, ...]
    caption: str
    audience_mode: str = "family_friendly"


CHARACTERS = (
    {
        "name": "PIP",
        "type": "tiny mischievous robot",
        "traits": ("curious", "overconfident", "kind"),
    },
    {
        "name": "MOMO",
        "type": "sleepy orange cat",
        "traits": ("dramatic", "lazy", "clever"),
    },
    {
        "name": "BUBU",
        "type": "round blue bird",
        "traits": ("hyper", "optimistic", "chaotic"),
    },
)

PREMISES = (
    {
        "topic": "smart kitchen",
        "title": "PIP vs the Smart Kitchen",
        "hook": "PIP asked the smart kitchen to make one sandwich.",
        "beats": (
            ("setup", "PIP proudly presses the cooking button.", "curious"),
            ("escalation", "The kitchen makes twelve sandwiches.", "shocked"),
            ("reaction", "PIP looks at the giant sandwich pile.", "overloaded"),
            ("punchline", "PIP: Apparently the kitchen misunderstood 'one'.", "deadpan"),
        ),
    },
    {
        "topic": "homework robot",
        "title": "The Homework Robot",
        "hook": "PIP bought a robot to do homework faster.",
        "beats": (
            ("setup", "PIP hands the robot one tiny math problem.", "confident"),
            ("escalation", "The robot prints a 300-page solution.", "shocked"),
            ("reaction", "PIP hugs the calculator in confusion.", "overloaded"),
            ("punchline", "PIP: I asked for help, not a library.", "deadpan"),
        ),
    },
    {
        "topic": "bedtime",
        "title": "PIP and the Bedtime Alarm",
        "hook": "PIP set one bedtime alarm. The alarm had opinions.",
        "beats": (
            ("setup", "PIP taps the bedtime button.", "sleepy"),
            ("escalation", "A giant alarm clock appears and rings.", "shocked"),
            ("reaction", "PIP hides under a blanket.", "scared"),
            ("punchline", "PIP: Okay. Tomorrow I am sleeping manually.", "deadpan"),
        ),
    },
)


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def pick_episode(topic_hint: str | None = None) -> CartoonEpisode:
    key = (topic_hint or "default").strip().lower()
    matching = [p for p in PREMISES if p["topic"] in key or key in p["topic"]]
    premise = matching[0] if matching else PREMISES[_seed(key) % len(PREMISES)]
    character = CHARACTERS[_seed(key + ":character") % len(CHARACTERS)]
    beats = tuple(
        {
            "beat": name,
            "visual": visual,
            "expression": expression,
            "duration": 2.4 if i < 3 else 3.0,
        }
        for i, (name, visual, expression) in enumerate(premise["beats"])
    )
    return CartoonEpisode(
        character=character["name"],
        title=premise["title"],
        hook=premise["hook"],
        beats=beats,
        caption="Funny original cartoon short. AI-generated fictional characters.",
    )
