"""Provider-neutral, deterministic full-body acting accents.

The output is expressed as small offsets for the existing pose rig. It does not
own rendering or replace ShotPerformanceSpec / the shared kinematic primitives.
"""
from __future__ import annotations

import math


_ACTION_BIAS = {
    "checks_phone": (0.65, -0.25, -0.45),
    "finish_work": (0.0, 0.35, 0.10),
    "sees_calendar": (-0.45, -0.05, 0.35),
    "reaches_for_coffee": (0.55, 0.10, -0.25),
    "glances_at_many_tabs": (-0.25, 0.0, 0.15),
    "leans_toward_camera": (0.05, -0.55, 0.0),
    "deadpan_camera_reaction": (-0.2, 0.3, 0.0),
    # Cartoon cast action packs (config/cartoon_action_packs.json).
    "press_button": (0.45, -0.20, -0.30),
    "shock": (-0.55, 0.45, 0.20),
    "overwhelmed": (0.10, 0.50, 0.15),
    "deadpan": (0.0, 0.10, 0.0),
    "walk_in": (0.50, 0.05, 0.10),
    "wave_hello": (0.30, -0.15, -0.40),
    "yawn_stretch": (-0.10, -0.50, 0.30),
    "panic_run": (0.70, -0.30, 0.25),
    "curl_up": (0.05, 0.60, 0.20),
    "worried_close": (-0.10, 0.20, 0.0),
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _timeline(spec: dict | None, duration: float | None) -> tuple[dict[str, float], float]:
    if not isinstance(spec, dict):
        return {}, 0.0
    try:
        total = max(0.001, float(spec.get("duration", duration or 1.0)))
    except (TypeError, ValueError):
        total = 1.0
    phases: dict[str, float] = {}
    rows = spec.get("phases")
    if not isinstance(rows, (list, tuple)):
        return phases, total
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            start, end = float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= start <= end <= total and end > start:
            phases[str(row.get("name", ""))] = (start + end) / (2.0 * total)
    return phases, total


def acting_state(progress: float, beat: str, performance: dict | None = None) -> dict[str, float]:
    """Return bounded articulation and mass offsets for a normalized shot time."""
    p = _clamp(float(progress), 0.0, 1.0)
    spec = performance.get("shot_spec") if isinstance(performance, dict) else None
    action = spec.get("source_action", spec.get("action")) if isinstance(spec, dict) else None
    if action not in _ACTION_BIAS:
        return {"acting_peak": .45, "acting_impulse": 0.0, "weight_shift": 0.0,
                "torso_arc": 0.0, "joint_articulation": 0.0, "overlap": 0.0,
                "reaction": 0.0, "acceleration": 0.0}
    bias = _ACTION_BIAS[str(action)]
    phases, _ = _timeline(spec, None)
    peak = phases.get("impact", phases.get("action", {"setup": .46, "escalation": .46, "reaction": .40, "punchline": .64}.get(beat, .45)))
    width = _clamp((phases.get("recoil", peak + .15) - peak) * .72, .055, .20)
    q = (p - peak) / width
    impulse = math.exp(-2.2 * q * q)
    # Smooth acceleration into the event; exponentially damped recovery after it.
    prep = _clamp((peak - p) / max(width, .06), 0.0, 1.0) ** 2
    follow = math.exp(-5.0 * max(0.0, p - peak)) * math.sin(math.pi * min(2.0, max(0.0, p - peak) / max(width, .06)))
    if isinstance(spec, dict) and p >= phases.get("settle", 1.01):
        tail = _clamp((1.0 - p) / max(.03, 1.0 - phases.get("settle", 1.0)), 0.0, 1.0)
        follow *= tail
    weight = _clamp((impulse - .20 * prep) * 0.8, -0.25, 0.8)
    return {
        "acting_peak": peak,
        "acting_impulse": impulse,
        "weight_shift": _clamp(bias[0] * weight, -0.35, 0.35),
        "torso_arc": _clamp(bias[1] * impulse + .10 * follow, -0.65, 0.65),
        "joint_articulation": _clamp(bias[2] * impulse + .12 * follow, -0.55, 0.55),
        "overlap": _clamp(follow, -0.45, 0.45),
        "reaction": _clamp(impulse * (1.0 + abs(bias[0])) + .12 * follow, 0.0, 1.25),
        "acceleration": _clamp((p - peak) / max(width, .06), -1.0, 1.0),
    }
