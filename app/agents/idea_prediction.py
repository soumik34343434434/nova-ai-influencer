"""Deterministic pre-production estimates for idea outcomes.

These are heuristic estimates, never platform analytics. Historical metrics are
used only as shrinkage priors when matching first-party observations exist.
"""
from __future__ import annotations

from typing import Any, Iterable


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def _history_value(row: dict[str, Any], target: str) -> float | None:
    if target == "retention":
        val = float(row.get("retention_pct_normalized", row.get("retention_pct", 0)) or 0)
        if val <= 0 and float(row.get("avg_watch_seconds", 0) or 0) > 0:
            val = min(100.0, float(row["avg_watch_seconds"]) / max(float(row.get("duration_seconds", 0) or 0.1), .1) * 100)
        return _clamp(val / 100) if val > 0 else None
    views = int(row.get("views", 0) or 0)
    shares = float(row.get("share_rate", 0) or 0) if "share_rate" in row else float(row.get("shares", 0) or 0) / max(views, 1)
    return _clamp(shares / .05) if views > 0 else None  # 5% share rate maps to 1.0


def _features(idea: dict[str, Any]) -> dict[str, float]:
    hook = str(idea.get("hook", ""))
    words = hook.split()
    beats = [b for b in idea.get("beats", []) if isinstance(b, dict)]
    actions = [str(b.get("asset_action", "")) for b in beats if b.get("asset_action")]
    emotion = str(idea.get("emotion", "")).lower()
    duration = float(idea.get("duration_target", idea.get("duration_seconds", 8)) or 8)
    trend = _clamp(float(idea.get("trend_score", idea.get("opportunity_score", 0)) or 0))
    novelty = _clamp(1 - float(idea.get("saturation_score", 0) or 0))
    cost = _clamp(float(idea.get("production_cost", .5) or .5))
    return {
        "hook_strength": _clamp(.35 + .3 * (5 <= len(words) <= 13) + .2 * any(x in hook.lower() for x in ("why", "big mistake", "then", "one", "!", "?")) + .15 * bool(hook.strip())),
        "structure": _clamp(.25 + .18 * min(len(beats), 4) + .12 * any("escalation" in str(b.get("beat", "")) for b in beats) + .15 * any("punchline" in str(b.get("beat", "")) for b in beats)),
        "action_diversity": _clamp(len(set(actions)) / max(3, min(len(beats), 4))),
        "emotion": .72 if emotion in {"amusement", "surprise", "cute", "curiosity", "awe", "satisfaction"} else (.58 if emotion else .5),
        "character": .72 if idea.get("character") in {"MOMO", "PIP", "BUBU"} else (.58 if idea.get("character") else .5),
        "format": .72 if idea.get("format_name") else .5,
        "duration": _clamp(1 - abs(duration - 8) / 16),
        "trend_fit": trend,
        "novelty": novelty,
        "production_cost": 1 - cost,
    }


def predict_idea(idea: dict[str, Any], history: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    """Return bounded retention/shareability estimates and explicit evidence state."""
    rows = list(history)
    features = _features(idea)
    retention = _clamp(.22 * features["hook_strength"] + .20 * features["structure"] + .15 * features["action_diversity"] + .08 * features["emotion"] + .05 * features["character"] + .06 * features["format"] + .08 * features["duration"] + .06 * features["trend_fit"] + .03 * features["novelty"] + .02 * features["production_cost"])
    shareability = _clamp(.12 * features["hook_strength"] + .12 * features["structure"] + .10 * features["action_diversity"] + .12 * features["emotion"] + .12 * features["character"] + .08 * features["format"] + .04 * features["duration"] + .12 * features["trend_fit"] + .14 * features["novelty"] + .03 * features["production_cost"])

    dimensions = {"format_name": idea.get("format_name"), "emotion": idea.get("emotion"), "character": idea.get("character"), "trend_id": idea.get("trend_id")}
    # Distinct content metrics are the sample unit, preventing one row matching
    # several dimensions from gaining extra weight.
    matched = []
    for row in rows:
        if any(value and str(row.get(dim, "")) == str(value) for dim, value in dimensions.items()):
            matched.append(row)
    rv = [_history_value(r, "retention") for r in matched]
    sv = [_history_value(r, "share") for r in matched]
    rv = [v for v in rv if v is not None]
    sv = [v for v in sv if v is not None]
    if rv:
        retention = _clamp((retention * 3 + sum(rv)) / (3 + len(rv)))
    if sv:
        shareability = _clamp((shareability * 3 + sum(sv)) / (3 + len(sv)))
    n = min(len(rv), len(sv)) if rv and sv else max(len(rv), len(sv))
    confidence = _clamp(.15 + .12 * min(n, 5))
    uncertainty = _clamp(.45 - .07 * min(n, 5))
    contributors = sorted(features.items(), key=lambda item: (-item[1], item[0]))
    return {
        "predicted_retention": retention,
        "predicted_shareability": shareability,
        "metric_scale": "0_to_1; shareability is normalized share-rate potential, not a forecast of actual shares",
        "uncertainty": uncertainty,
        "confidence": confidence,
        "evidence_status": "sufficient_evidence" if n >= 3 else "insufficient_evidence",
        "historical_observations": n,
        "estimate_type": "pre_production_prediction",
        "observed_analytics": False,
        "contributing_factors": [{"factor": name, "value": value} for name, value in contributors],
    }
