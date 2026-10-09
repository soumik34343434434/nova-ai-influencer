"""Deterministic, explainable subsystem prioritization for upgrade cycles."""
from __future__ import annotations

from typing import Any, Iterable


# Deliberately favors proof and integration work early: they often expose or
# unblock downstream quality work that cannot be evaluated in isolation.
SAFE_FALLBACK_ORDER = (
    "runtime_render_proof",
    "integrations",
    "performance_animation",
    "audio",
    "editor",
    "final_qa",
    "publishing",
    "analytics",
)

WEIGHTS = {
    "quality_gap": 0.35,
    "failure_frequency": 0.20,
    "user_impact": 0.20,
    "evidence_confidence": 0.15,
    "unblock_leverage": 0.10,
}

DEFAULT_CANDIDATES = {
    "runtime_render_proof": {"user_impact": 9.0, "unblock_leverage": 10.0,
                              "evidence_confidence": 0.0},
    "integrations": {"user_impact": 8.0, "unblock_leverage": 8.0,
                     "evidence_confidence": 0.0},
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(10.0, n))


def _fallback_index(name: str) -> int:
    try:
        return SAFE_FALLBACK_ORDER.index(name)
    except ValueError:
        return len(SAFE_FALLBACK_ORDER)


def _score_candidate(candidate: dict[str, Any], states: dict[str, Any]) -> dict[str, Any]:
    name = str(candidate.get("name", "unknown"))
    areas = candidate.get("areas") or {}
    # Lowest measured area is used so a strong sibling dimension cannot hide a gap.
    measured = [_number(v) for v in areas.values() if v is not None]
    has_measurement = candidate.get("score") is not None or bool(measured)
    score = _number(candidate.get("score"), min(measured) if measured else 0.0)
    target = _number(candidate.get("target", 8.5), 8.5)
    gap = (max(0.0, min(10.0, (target - score) / max(target, 1.0) * 10.0))
           if has_measurement else 0.0)

    evidence_present = bool(candidate.get("evidence")) or bool(measured) or candidate.get("score") is not None
    confidence = _number(candidate.get("evidence_confidence", 6.0 if evidence_present else 0.0))
    # Failures are counts, mapped logarithmically so one noisy subsystem cannot
    # dominate every other signal. An explicit normalized value may be supplied.
    if candidate.get("failure_frequency") is not None:
        failures = _number(candidate["failure_frequency"])
    else:
        count = max(0.0, _number(candidate.get("failures", 0.0)))
        failures = min(10.0, count * 2.0)
    impact = _number(candidate.get("user_impact", 5.0))
    unlock = _number(candidate.get("unblock_leverage", 0.0))
    dependencies = list(candidate.get("dependencies") or [])
    unmet = [dep for dep in dependencies if states.get(dep, {}).get("status") != "complete"]
    blocked = bool(unmet)
    components = {
        "quality_gap": round(gap, 3),
        "failure_frequency": round(failures, 3),
        "user_impact": round(impact, 3),
        "evidence_confidence": round(confidence, 3),
        "unblock_leverage": round(unlock, 3),
    }
    raw = sum(components[key] * weight for key, weight in WEIGHTS.items())
    # No evidence means no positive quality-gap claim. Still eligible via safe
    # fallback, but it cannot outrank measured, high-impact work on invented data.
    if not evidence_present:
        raw *= 0.55
    score_value = round(raw * (0.25 if blocked else 1.0), 3)
    reasons = [f"{key}={value:g} × {WEIGHTS[key]:.2f}" for key, value in components.items()]
    if blocked:
        reasons.append("blocked by incomplete dependencies: " + ", ".join(map(str, unmet)))
    elif unlock:
        reasons.append(f"can unblock downstream work (leverage {unlock:g}/10)")
    if not evidence_present:
        # Keep missing evidence as the terminal explanation: callers can safely
        # surface it as the reason no quality gap was claimed.
        reasons.append("evidence missing; confidence and score discounted")
    return {
        "name": name,
        "priority_score": score_value,
        "components": components,
        "blocked": blocked,
        "unmet_dependencies": unmet,
        "evidence_present": evidence_present,
        "explanation": reasons,
        "_fallback": _fallback_index(name),
    }


def select_subsystem_priority(
    progress: dict[str, Any], candidates: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Choose the next eligible subsystem and return a full score breakdown.

    ``progress`` accepts the QUALITY_PROGRESS shape. Optional candidates can
    supply failures, impact, confidence, dependencies and unblock leverage.
    Missing evidence is kept explicit and falls back to a stable safe order.
    """
    engines = progress.get("engines") if isinstance(progress, dict) else None
    engines = engines if isinstance(engines, dict) else {}
    overrides = {str(c.get("name")): c for c in (candidates or []) if isinstance(c, dict) and c.get("name")}
    names = list(dict.fromkeys([*SAFE_FALLBACK_ORDER, *engines.keys(), *overrides.keys()]))
    rows = []
    for name in names:
        state = engines.get(name, {})
        if not isinstance(state, dict):
            state = {}
        if state.get("status") == "complete":
            continue
        candidate = {**DEFAULT_CANDIDATES.get(name, {}), **state,
                     **overrides.get(name, {}), "name": name}
        rows.append(_score_candidate(candidate, engines))
    if not rows:
        return {"selected": SAFE_FALLBACK_ORDER[0], "priority_score": 0.0,
                "reason": "All recorded subsystems are complete; safe fallback selected.",
                "ranked": [], "fallback_order": list(SAFE_FALLBACK_ORDER)}
    rows.sort(key=lambda row: (-row["priority_score"], row["_fallback"], row["name"]))
    eligible = [row for row in rows if not row["blocked"]]
    chosen = eligible[0] if eligible else min(rows, key=lambda row: (row["_fallback"], row["name"]))
    ranked = [{k: v for k, v in row.items() if k != "_fallback"} for row in rows]
    return {"selected": chosen["name"], "priority_score": chosen["priority_score"],
            "reason": "; ".join(chosen["explanation"]), "ranked": ranked,
            "fallback_order": list(SAFE_FALLBACK_ORDER)}
