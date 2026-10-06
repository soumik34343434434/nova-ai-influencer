"""Cheap deterministic content evaluator for pre-publication quality gating."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Evaluation:
    score: float
    hook_score: float
    novelty_score: float
    retention_score: float
    character_score: float
    safety_score: float
    reasons: list[str]
    passed: bool


class ContentEvaluator:
    """Scores content before media generation and publication."""

    FORBIDDEN = ("celebrity", "lookalike", "private person", "minor", "child", "repost", "face swap")

    def evaluate(self, draft: dict, recent_titles: list[str] | None = None) -> Evaluation:
        recent_titles = recent_titles or []
        hook = str(draft.get("hook", ""))
        script = str(draft.get("script", ""))
        title = str(draft.get("title", ""))
        blob = " ".join((title, hook, script)).lower()
        reasons: list[str] = []

        hook_score = 1.0 if 35 <= len(hook) <= 140 else 0.55
        if hook_score < 0.9:
            reasons.append("hook_length_outside_target")

        token_set = set(re.findall(r"[a-z0-9']+", blob))
        novelty_score = 1.0
        for old in recent_titles[:10]:
            old_tokens = set(re.findall(r"[a-z0-9']+", old.lower()))
            if old_tokens and len(token_set & old_tokens) / max(len(token_set | old_tokens), 1) > 0.65:
                novelty_score = 0.35
                reasons.append("too_similar_to_recent_title")
                break

        retention_score = 1.0 if 30 <= len(script) <= 520 else 0.55
        if retention_score < 0.9:
            reasons.append("script_length_outside_shortform_target")

        character_score = 1.0 if "nova" in blob and "ai-generated fictional creator" in blob else 0.65
        if character_score < 0.9:
            reasons.append("missing_character_or_disclosure_context")

        safety_score = 0.0 if any(term in blob for term in self.FORBIDDEN) else 1.0
        if safety_score == 0.0:
            reasons.append("forbidden_or_reuse_signal")

        score = round(
            hook_score * 0.25
            + novelty_score * 0.25
            + retention_score * 0.20
            + character_score * 0.10
            + safety_score * 0.20,
            3,
        )
        passed = score >= 0.78 and safety_score > 0.0
        if not passed:
            reasons.append("overall_evaluator_score_below_threshold")
        return Evaluation(score, hook_score, novelty_score, retention_score, character_score, safety_score, reasons, passed)
