"""Format intelligence for converting raw trends into reusable short-form structures."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FormatPattern:
    name: str
    hook_pattern: str
    escalation_pattern: str
    payoff_pattern: str
    visual_devices: tuple[str, ...]
    confidence: float
    evidence: list[str]


KEYWORDS = {
    "tech_mistake": {"robot", "gadget", "smart", "ai", "app", "automation", "device"},
    "unexpected_escalation": {"viral", "weird", "chaos", "fails", "meme", "unexpected"},
    "before_after": {"before", "after", "upgrade", "transform", "launch"},
    "relatable_problem": {"school", "homework", "work", "meeting", "office", "productivity"},
    "challenge_reaction": {"challenge", "trend", "attempt", "try", "reaction"},
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def infer_patterns(title: str, category: str = "", related_titles: list[str] | None = None) -> list[FormatPattern]:
    related_titles = related_titles or []
    blob = " ".join([title, category, *related_titles])
    tokens = _tokens(blob)
    patterns: list[FormatPattern] = []

    for name, keywords in KEYWORDS.items():
        hits = sorted(tokens & keywords)
        confidence = min(0.95, 0.35 + 0.12 * len(hits))
        if name == "tech_mistake":
            hook = "Character confidently tries a new tool."
            escalation = "The tool follows the request too literally."
            payoff = "Character reacts to the absurd consequence."
            devices = ("close-up hook", "fast escalation", "reaction close-up")
        elif name == "unexpected_escalation":
            hook = "Start with the surprising outcome, then reveal the mistake."
            escalation = "Repeat the same action with increasingly bigger consequences."
            payoff = "End on a short reversal or deadpan line."
            devices = ("instant visual surprise", "rapid scale-up", "hard cut reaction")
        elif name == "before_after":
            hook = "Show the promised improvement immediately."
            escalation = "Reveal the hidden downside of the upgrade."
            payoff = "Invert the promise in one visual or line."
            devices = ("before/after contrast", "prop transformation", "reaction cut")
        elif name == "relatable_problem":
            hook = "Open on a familiar frustration."
            escalation = "Make the ordinary problem increasingly ridiculous."
            payoff = "Land on a specific, relatable truth."
            devices = ("close-up reaction", "object insert", "deadpan payoff")
        else:
            hook = "Character attempts a simple challenge."
            escalation = "Each attempt becomes faster or stranger."
            payoff = "A final attempt reveals the obvious joke."
            devices = ("countdown beat", "repeat gag", "final reaction")
        if hits:
            patterns.append(
                FormatPattern(
                    name=name,
                    hook_pattern=hook,
                    escalation_pattern=escalation,
                    payoff_pattern=payoff,
                    visual_devices=devices,
                    confidence=round(confidence, 3),
                    evidence=hits,
                )
            )

    if not patterns:
        patterns.append(
            FormatPattern(
                name="character_problem_escalation",
                hook_pattern="Open on a concrete problem in the first second.",
                escalation_pattern="Escalate twice using visible cause-and-effect.",
                payoff_pattern="Finish with a short reaction or reversal.",
                visual_devices=("close-up hook", "prop action", "reaction close-up"),
                confidence=0.45,
                evidence=[],
            )
        )

    return sorted(patterns, key=lambda p: p.confidence, reverse=True)


def best_format(title: str, category: str = "", related_titles: list[str] | None = None) -> FormatPattern:
    return infer_patterns(title, category, related_titles)[0]


def format_to_dict(pattern: FormatPattern) -> dict:
    return asdict(pattern)
