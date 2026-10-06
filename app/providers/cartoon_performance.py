"""Beat-aware character performance for the local cartoon rig."""
from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

from app.providers.cartoon_rig import CharacterRig


@dataclass(frozen=True)
class PerformancePhase:
    name: str
    start: float
    end: float


@dataclass(frozen=True)
class PerformancePlan:
    beat: str
    phases: tuple[PerformancePhase, ...]
    energy: float
    freeze_after: float | None = None


PLANS = {
    "setup": PerformancePlan(
        "setup",
        (PerformancePhase("anticipate", 0.00, 0.18), PerformancePhase("action", 0.18, 0.58),
         PerformancePhase("settle", 0.58, 1.00)),
        0.45,
    ),
    "escalation": PerformancePlan(
        "escalation",
        (PerformancePhase("anticipate", 0.00, 0.16), PerformancePhase("action", 0.16, 0.42),
         PerformancePhase("impact", 0.42, 0.50), PerformancePhase("recoil", 0.50, 0.68),
         PerformancePhase("settle", 0.68, 1.00)),
        1.35,
    ),
    "reaction": PerformancePlan(
        "reaction",
        (PerformancePhase("anticipate", 0.00, 0.12), PerformancePhase("action", 0.12, 0.36),
         PerformancePhase("impact", 0.36, 0.44), PerformancePhase("recoil", 0.44, 0.64),
         PerformancePhase("settle", 0.64, 1.00)),
        1.20,
    ),
    "punchline": PerformancePlan(
        "punchline",
        (PerformancePhase("settle", 0.00, 0.34), PerformancePhase("deadpan", 0.34, 1.00)),
        0.28,
        freeze_after=0.72,
    ),
}


def performance_plan(beat: str) -> PerformancePlan:
    return PLANS.get(beat, PLANS["setup"])


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _phase_progress(p: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0
    return max(0.0, min(1.0, (p - start) / (end - start)))


def arm_motion(progress: float, beat: str, side: str) -> tuple[float, float, float, float]:
    """Return dx, dy, scale, rotation for a small independent arm gesture."""
    p = max(0.0, min(1.0, progress))
    sign = -1.0 if side == "left" else 1.0
    if beat == "setup":
        gesture = math.sin(math.pi * p)
        return sign * 2.0 * gesture, -2.0 * gesture, 1.0 + 0.008 * gesture, sign * 1.2 * gesture
    if beat == "escalation":
        anticipation = max(0.0, 1.0 - p / 0.16)
        action = math.sin(math.pi * _phase_progress(p, 0.16, 0.46))
        recoil = math.sin(math.pi * _phase_progress(p, 0.46, 0.70))
        dx = sign * (-4.0 * anticipation + 15.0 * action - 8.0 * recoil)
        dy = 3.0 * anticipation - 8.0 * action + 4.0 * recoil
        rot = sign * (-2.0 * anticipation + 7.0 * action - 3.0 * recoil)
        scale = 1.0 + 0.030 * action - 0.012 * recoil
        return dx, dy, scale, rot
    if beat == "reaction":
        recoil = math.sin(math.pi * _phase_progress(p, 0.36, 0.66))
        return sign * 7.0 * recoil, -5.0 * recoil, 1.0 + 0.018 * recoil, sign * 4.0 * recoil
    sway = math.sin(math.pi * p * 0.8)
    return sign * 1.0 * sway, 0.6 * sway, 1.0 + 0.004 * sway, sign * 0.7 * sway


def head_motion(progress: float, beat: str) -> tuple[float, float, float, float]:
    plan = performance_plan(beat)
    p = max(0.0, min(1.0, progress))
    if plan.freeze_after is not None and p >= plan.freeze_after:
        p = plan.freeze_after

    energy = plan.energy
    # A continuous anticipation -> snap -> recoil -> settle curve.
    if beat == "escalation":
        anticipation = max(0.0, 1.0 - p / 0.16)
        action = math.sin(math.pi * _phase_progress(p, 0.16, 0.50))
        settle = math.sin(math.pi * _phase_progress(p, 0.50, 1.00))
        dx = -6.0 * anticipation + 13.0 * action - 6.0 * settle
        dy = 2.5 * anticipation - 7.0 * action + 4.0 * settle
        rotation = -2.0 * anticipation + 3.0 * action - 1.8 * settle
        scale = 1.0 + 0.034 * action - 0.010 * settle
    elif beat == "reaction":
        anticipation = max(0.0, 1.0 - p / 0.12)
        snap = math.sin(math.pi * _phase_progress(p, 0.12, 0.44))
        recoil = math.sin(math.pi * _phase_progress(p, 0.44, 0.64))
        dx = 5.0 * anticipation - 16.0 * snap + 8.0 * recoil
        dy = 5.0 * anticipation - 10.0 * snap + 5.0 * recoil
        rotation = 2.0 * anticipation - 4.0 * snap + 2.2 * recoil
        scale = 1.0 + 0.042 * snap - 0.014 * recoil
    elif beat == "punchline":
        sway = math.sin(math.pi * p * 0.8)
        dx = 1.6 * sway
        dy = -0.8 * sway
        rotation = 0.7 * sway
        scale = 1.0 + 0.008 * sway
    else:
        breathe = math.sin(math.tau * p)
        dx = 2.5 * breathe * energy
        dy = -1.4 * breathe * energy
        rotation = 0.6 * breathe * energy
        scale = 1.0 + 0.006 * breathe * energy
    return dx, dy, scale, rotation
