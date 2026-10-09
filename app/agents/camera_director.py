"""Shot-grammar camera direction for short-form cartoon acting."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraDirective:
    focus_weight: float
    zoom_delta: float
    lateral_bias: float
    lead_seconds: float
    hold_strength: float


def plan_camera(beat: str, progress: float, impact: float, settle: float) -> CameraDirective:
    """Return bounded camera intent that supports the actor instead of competing with it."""
    p = max(0.0, min(1.0, float(progress)))
    impact = max(0.0, min(1.0, float(impact)))
    settle = max(-1.0, min(1.0, float(settle)))
    if beat == "setup":
        return CameraDirective(
            focus_weight=0.20 + 0.28 * p,
            zoom_delta=-0.10 * (1.0 - p),
            lateral_bias=0.02 * p,
            lead_seconds=0.05,
            hold_strength=0.20,
        )
    if beat == "escalation":
        return CameraDirective(
            focus_weight=0.34 + 0.58 * impact,
            zoom_delta=-0.13 * impact + 0.025 * max(0.0, -settle),
            lateral_bias=0.055 * min(1.0, impact + 0.25),
            lead_seconds=0.08,
            hold_strength=0.18 + 0.36 * impact,
        )
    if beat == "reaction":
        return CameraDirective(
            focus_weight=0.30 + 0.46 * impact,
            zoom_delta=0.075 * impact - 0.025 * max(0.0, -settle),
            lateral_bias=-0.065 * min(1.0, impact + 0.20),
            lead_seconds=0.06,
            hold_strength=0.24 + 0.28 * impact,
        )
    return CameraDirective(
        focus_weight=0.40 + 0.28 * impact,
        zoom_delta=-0.045 * impact + 0.018 * abs(settle),
        lateral_bias=0.045 * (1.0 - p),
        lead_seconds=0.04,
        hold_strength=0.48 + 0.40 * max(0.0, 1.0 - p),
    )
