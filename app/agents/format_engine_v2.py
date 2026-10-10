"""Empirical format intelligence v2.

Separates format *identification* from format *performance*. A trend can say
"this looks like POV" while owned-platform metrics can later say whether POV
actually works for NOVA. No single viral outlier is allowed to dominate.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

from app.learning_v2 import build_strategy_memory


@dataclass(frozen=True)
class FormatPatternV2:
    name: str
    hook_pattern: str
    escalation_pattern: str
    payoff_pattern: str
    visual_devices: tuple[str, ...]
    detected_score: float
    learned_score: float
    confidence: float
    evidence: tuple[str, ...] = ()


BASE_FORMATS: dict[str, dict[str, Any]] = {
    "dance": {
        "hooks": ("movement in first frame", "recognizable beat", "cute commitment"),
        "escalation": "add a visual mistake or unexpected beat change",
        "payoff": "reaction or reversal on the beat",
        "devices": ("wide body shot", "beat-synced cut", "expression switch"),
    },
    "trend_audio": {
        "hooks": ("recognizable sound", "visual action begins immediately"),
        "escalation": "subvert the expected audio action",
        "payoff": "character-specific twist",
        "devices": ("audio-led opening", "beat hit", "reaction close-up"),
    },
    "reaction": {
        "hooks": ("start with the incoming stimulus", "face reaction"),
        "escalation": "reveal why the stimulus matters",
        "payoff": "deadpan or exaggerated reaction",
        "devices": ("insert shot", "face close-up", "micro-pause"),
    },
    "pov": {
        "hooks": ("POV label", "immediately recognizable situation"),
        "escalation": "make the ordinary problem increasingly specific",
        "payoff": "relatable reversal",
        "devices": ("first-person framing", "prop insert", "reaction"),
    },
    "story_comedy": {
        "hooks": ("concrete situation", "unfinished question"),
        "escalation": "two visible consequence steps",
        "payoff": "reversal or deadpan line",
        "devices": ("establishing beat", "prop action", "reaction close-up"),
    },
    "challenge": {
        "hooks": ("simple rule", "countdown", "visible goal"),
        "escalation": "attempts become harder",
        "payoff": "unexpected failure or win",
        "devices": ("counter", "repeated attempts", "impact reaction"),
    },
    "transformation": {
        "hooks": ("before/after promise", "immediate contrast"),
        "escalation": "show hidden cost",
        "payoff": "invert the transformation promise",
        "devices": ("match framing", "prop transformation", "reaction"),
    },
    "cute_animal": {
        "hooks": ("cute first frame", "small action"),
        "escalation": "tiny action creates oversized consequence",
        "payoff": "adorable or deadpan acceptance",
        "devices": ("close character framing", "cute sound", "reaction"),
    },
    "absurdist": {
        "hooks": ("impossible premise immediately"),
        "escalation": "accept the impossible as normal and escalate",
        "payoff": "deadpan acceptance",
        "devices": ("hard premise", "prop escalation", "deadpan close-up"),
    },
    "listicle": {
        "hooks": ("count or promise", "first item immediately"),
        "escalation": "items become stranger",
        "payoff": "best/worst item last",
        "devices": ("counter", "fast cuts", "final reaction"),
    },
    "micro_short": {
        "hooks": ("single clear conflict in first second"),
        "escalation": "one compact reversal",
        "payoff": "instant punchline",
        "devices": ("close hook", "single prop", "fast payoff"),
    },
}


def _learned_score(memory: dict[str, Any], name: str) -> float:
    row = memory.get("posteriors", {}).get("format_name", {}).get(name)
    if not row:
        return 0.5
    return float(row.get("ucb", row.get("posterior_mean", 0.5)))


def infer_format_patterns(signal: dict[str, Any], memory: dict[str, Any] | None = None) -> list[FormatPatternV2]:
    memory = memory or build_strategy_memory()
    detected = list(signal.get("formats") or ())
    if not detected:
        detected = ["micro_short"]

    source_count = int(signal.get("source_count", 1) or 1)
    cross = float(signal.get("cross_platform_score", 0.0) or 0.0)
    results = []
    for name in dict.fromkeys(detected):
        if name not in BASE_FORMATS:
            continue
        spec = BASE_FORMATS[name]
        learned = _learned_score(memory, name)
        prior = min(1.0, 0.35 + 0.12 * min(source_count, 4) + 0.15 * cross)
        detected_score = float(signal.get("opportunity_score", 0.0) or 0.0) * 0.7 + prior * 0.3
        confidence = min(1.0, 0.40 + 0.12 * min(source_count, 3) + 0.10 * bool(signal.get("hook_patterns")))
        results.append(FormatPatternV2(
            name=name,
            hook_pattern="; ".join(spec["hooks"]),
            escalation_pattern=str(spec["escalation"]),
            payoff_pattern=str(spec["payoff"]),
            visual_devices=tuple(spec["devices"]),
            detected_score=round(detected_score, 4),
            learned_score=round(learned, 4),
            confidence=round(confidence, 4),
            evidence=tuple(list(signal.get("evidence") or ()) + list(signal.get("hook_patterns") or ())),
        ))
    return sorted(results, key=lambda x: (0.55 * x.detected_score + 0.45 * x.learned_score, x.confidence), reverse=True)


def rank_format_candidates(signals: Iterable[dict[str, Any]], memory: dict[str, Any] | None = None,
                           limit: int = 20) -> list[dict[str, Any]]:
    memory = memory or build_strategy_memory()
    rows = []
    for signal in signals:
        for pattern in infer_format_patterns(signal, memory):
            rows.append({
                "trend_id": signal.get("trend_id", ""),
                "trend_title": signal.get("title", ""),
                **asdict(pattern),
                "trend_score": float(signal.get("opportunity_score", 0.0) or 0.0),
                "saturation_score": float(signal.get("saturation_score", 0.0) or 0.0),
                "velocity_score": float(signal.get("velocity_score", 0.0) or 0.0),
                "emotion_candidates": list(signal.get("emotions") or ()),
            })
    for row in rows:
        row["score"] = round(
            0.38 * float(row["detected_score"])
            + 0.32 * float(row["learned_score"])
            + 0.18 * float(row["velocity_score"])
            + 0.12 * (1.0 - float(row["saturation_score"])),
            6,
        )
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:max(1, limit)]
