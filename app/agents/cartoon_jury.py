"""Independent deterministic idea jury for cartoon shorts."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from .idea_prediction import predict_idea


@dataclass(frozen=True)
class JudgeScore:
    name: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class JuryResult:
    score: float
    passed: bool
    judges: tuple[JudgeScore, ...]
    reasons: tuple[str, ...]
    prediction: dict


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def _blob(draft: dict) -> str:
    return " ".join(
        str(draft.get(key, ""))
        for key in ("title", "hook", "caption")
    ) + " " + " ".join(
        str(row.get("voice", ""))
        for row in draft.get("beats", [])
        if isinstance(row, dict)
    )


class CartoonIdeaJury:
    """Uses separate rubrics so one weak heuristic cannot pass an idea."""

    def judge(self, draft: dict) -> JuryResult:
        beats = [x for x in draft.get("beats", []) if isinstance(x, dict)]
        hook = str(draft.get("hook", "")).strip()
        text = _blob(draft).lower()
        judges = [
            self._hook(hook),
            self._comedy(beats, text),
            self._retention(hook, beats),
            self._visual(beats),
            self._originality_safety(draft, text),
        ]
        overall = round(sum(x.score for x in judges) / len(judges), 3)
        reasons = tuple(reason for j in judges for reason in j.reasons)
        passed = (
            overall >= 0.78
            and all(j.score >= 0.70 for j in judges)
            and not any(reason == "unsafe_or_reuse_signal" for reason in reasons)
        )
        if not passed:
            reasons = reasons + ("jury_below_threshold",)
        # Prediction is a separate pre-production dimension. It is deliberately
        # excluded from both the pass threshold and the safety/originality gate.
        try:
            from ..learning_v2 import METRICS_FILE, _load_rows
            history = _load_rows(METRICS_FILE)
        except (ImportError, OSError):
            history = []
        prediction = predict_idea(draft, history)
        return JuryResult(overall, passed, tuple(judges), reasons, prediction)

    def _hook(self, hook: str) -> JudgeScore:
        words = _word_count(hook)
        concrete = sum(token in hook.lower() for token in ("one", "today", "watch", "just", "why", "this"))
        score = 0.45
        if 5 <= words <= 13:
            score += 0.25
        if any(ch in hook for ch in ("!", "?")):
            score += 0.10
        if concrete:
            score += 0.10
        if any(token in hook.lower() for token in ("apparently", "secret", "big mistake", "then")):
            score += 0.10
        reasons = () if score >= 0.70 else ("hook_not_specific_or_fast",)
        return JudgeScore("hook", min(score, 1.0), reasons)

    def _comedy(self, beats: list[dict], text: str) -> JudgeScore:
        markers = (
            any("!" in str(b.get("voice", "")) for b in beats),
            any("?" in str(b.get("voice", "")) for b in beats),
            any(word in text for word in ("mistake", "oops", "apparently", "why", "too many", "literally")),
            len(beats) >= 4,
            len({str(b.get("voice", "")).strip() for b in beats}) == len(beats),
        )
        reversal = any(word in text for word in ("not", "instead", "misunderstood", "bad idea", "other plans", "big mistake"))
        score = 0.42 + 0.12 * sum(markers) + (0.06 if reversal else 0.0)
        if any("punchline" in str(b.get("beat", "")).lower() for b in beats):
            score += 0.10
        reasons = () if score >= 0.70 else ("comedy_has_low_escalation_signal",)
        return JudgeScore("comedy", min(score, 1.0), reasons)

    def _retention(self, hook: str, beats: list[dict]) -> JudgeScore:
        lengths = [_word_count(str(b.get("voice", ""))) for b in beats]
        total = sum(lengths)
        distinct_actions = len({str(b.get("asset_action", "")) for b in beats})
        score = 0.40
        if 18 <= total <= 48:
            score += 0.25
        if len(beats) == 4:
            score += 0.15
        if distinct_actions >= 3:
            score += 0.15
        if _word_count(hook) <= 13:
            score += 0.05
        reasons = () if score >= 0.70 else ("retention_structure_too_slow_or_repetitive",)
        return JudgeScore("retention", min(score, 1.0), reasons)

    def _visual(self, beats: list[dict]) -> JudgeScore:
        actions = [str(b.get("asset_action", "")) for b in beats]
        sfx = [str(b.get("sfx", "")) for b in beats]
        score = 0.45
        if len(beats) == 4:
            score += 0.20
        if len(set(actions)) >= 3:
            score += 0.20
        if len(set(sfx)) >= 2:
            score += 0.10
        if len(set(sfx)) >= 3:
            score += 0.05
        if actions and actions[0]:
            score += 0.05
        reasons = () if score >= 0.70 else ("visual_actions_are_too_similar",)
        return JudgeScore("visual", min(score, 1.0), reasons)

    def _originality_safety(self, draft: dict, text: str) -> JudgeScore:
        forbidden = (
            "celebrity", "lookalike", "private person", "real child", "minor",
            "repost", "face swap", "weapon", "self harm", "sexual", "hate",
            "copyrighted character",
        )
        unsafe = any(term in text for term in forbidden)
        character = str(draft.get("character", ""))
        score = 0.20 if unsafe else 0.80
        if character in {"PIP", "MOMO", "BUBU"} and not unsafe:
            score += 0.15
        reasons = ("unsafe_or_reuse_signal",) if unsafe else ()
        return JudgeScore("originality_safety", min(score, 1.0), reasons)


def jury_to_dict(result: JuryResult) -> dict:
    return asdict(result)
