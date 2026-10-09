from __future__ import annotations

import math
from dataclasses import dataclass

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


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _event_fraction(beat: str, voice_performance: dict | None) -> float:
    defaults = {"setup": 0.46, "escalation": 0.16, "reaction": 0.40, "punchline": 0.64}
    if voice_performance:
        # Director AI can provide an explicit temporal acting plan. Prefer its
        # impact/action timing over a generic beat default so local animation
        # and hosted video prompts share the same performance choreography.
        spec = voice_performance.get("shot_spec")
        if isinstance(spec, dict):
            phases = spec.get("phases")
            if isinstance(phases, list) and phases:
                preferred = next((p for p in phases if p.get("name") == "impact"), None)
                if preferred is None:
                    preferred = next((p for p in phases if p.get("name") == "action"), None)
                if isinstance(preferred, dict):
                    try:
                        start = float(preferred.get("start"))
                        end = float(preferred.get("end"))
                        if 0.0 <= start <= end <= 1.0 and end > start:
                            return (start + end) / 2.0
                    except (TypeError, ValueError):
                        pass
        measured = voice_performance.get("performance_peak_fraction", voice_performance.get("peak_fraction"))
        if measured is not None:
            try:
                value = float(measured)
                # Voice peak is the strongest available timing cue, but avoid
                # collapsing an entire shot onto a near-zero onset/noise peak.
                if 0.05 <= value <= 0.90:
                    return value
            except (TypeError, ValueError):
                pass
    return defaults.get(beat, 0.45)


def _gaussian(p: float, center: float, width: float) -> float:
    width = max(0.02, width)
    x = (p - center) / width
    return math.exp(-2.8 * x * x)


def _camera_event_push(progress: float, event_fraction: float, beat: str) -> float:
    """Build a short camera push that anticipates and lands on an action event."""
    if beat not in {"escalation", "reaction"}:
        return 0.0
    p = _clamp(progress)
    lead = _gaussian(p, event_fraction - 0.075, 0.055)
    impact = _gaussian(p, event_fraction, 0.075)
    release = _gaussian(p, event_fraction + 0.16, 0.12)
    amplitude = 0.018 if beat == "escalation" else 0.014
    return amplitude * (0.42 * lead + impact - 0.30 * release)


def _anticipation(p: float, center: float, width: float) -> float:
    """Small pre-movement that prepares the body for a later action."""
    if p >= center:
        return 0.0
    q = _clamp((center - p) / max(0.02, width))
    return q * q * (1.0 - q * 0.35)


def _settle_oscillation(p: float, center: float, frequency: float = 7.0) -> float:
    """Damped follow-through after an impulse with a visible but finite tail."""
    if p <= center:
        return 0.0
    t = p - center
    fast = math.exp(-6.4 * t) * math.sin(frequency * t * math.pi)
    slow = 0.38 * math.exp(-2.6 * t) * math.sin(2.0 * frequency * t * math.pi + 0.35)
    return fast + slow


def kinematic_state(progress: float, beat: str, voice_performance: dict | None = None) -> dict[str, float]:
    """Shared body/face/arm/camera motion state.

    The important difference from independent part sines is that every moving
    part receives the same center-of-mass impulse and settle curve. Local arm/head
    accents are layered on top of that shared root movement, which reads much more
    like one character with mass instead of several unrelated sprites.
    """
    p = _clamp(progress)
    if beat == "punchline" and p >= 0.72:
        p = 0.72

    peak = _event_fraction(beat, voice_performance)
    width = {"setup": 0.14, "escalation": 0.105, "reaction": 0.13, "punchline": 0.17}.get(beat, 0.13)
    impact = _gaussian(p, peak, width)
    anticipation = _anticipation(p, peak, width * 0.85)
    settle = _settle_oscillation(p, peak)
    # Hands trail the torso impulse slightly, then overshoot and settle. Keep
    # this local to the arms so the shared center-of-mass motion stays coherent.
    hand_lag = _settle_oscillation(p, peak + 0.045, frequency=5.4)
    energy = performance_plan(beat).energy
    spec = voice_performance.get("shot_spec") if isinstance(voice_performance, dict) else None
    if isinstance(spec, dict):
        peak_phase = next((p for p in spec.get("phases", []) if p.get("name") == "impact"), None)
        if isinstance(peak_phase, dict):
            try:
                energy *= max(0.85, min(1.25, float(peak_phase.get("intensity", 1.0))))
            except (TypeError, ValueError):
                pass
    idle_phase = 0.65 if beat == "setup" else 1.25 if beat == "reaction" else 2.10 if beat == "punchline" else 0.90
    idle_gate = max(0.12, 1.0 - min(1.0, impact * 1.8))
    idle = math.sin(math.tau * (p * 1.18) + idle_phase) * idle_gate
    idle_breath = math.sin(math.tau * (p * 0.59) + idle_phase * 0.7) * idle_gate

    # Character root / center-of-mass trajectory. This is intentionally modest;
    # the camera follows it rather than inventing a second independent motion.
    root = {
        "setup": (-5.0 * anticipation + 7.5 * impact + 2.5 * settle,
                  -1.5 * anticipation - 4.5 * impact + 1.8 * settle,
                  -0.60 * anticipation + 1.45 * impact - 0.55 * settle),
        "escalation": (-9.0 * anticipation + 21.0 * impact + 7.0 * settle,
                       2.8 * anticipation - 10.0 * impact + 4.5 * settle,
                       -1.4 * anticipation + 3.8 * impact - 1.35 * settle),
        "reaction": (6.0 * anticipation - 18.0 * impact - 5.0 * settle,
                     4.0 * anticipation + 10.0 * impact - 4.0 * settle,
                     1.1 * anticipation - 3.3 * impact + 1.2 * settle),
        "punchline": (1.3 * impact + 0.8 * settle,
                      -0.9 * impact + 0.3 * settle,
                      0.32 * impact),
    }.get(beat, (2.0 * impact, -1.0 * impact, 0.3 * impact))
    root_dx, root_dy, root_rot = root
    root_dx += 0.85 * idle
    root_dy += 0.55 * idle_breath
    root_rot += 0.10 * idle

    if beat == "setup":
        left = (-2.0 * anticipation + 0.6 * idle, -1.0 * anticipation, -2.0 * anticipation + 0.4 * idle, 1.0 + 0.012 * impact)
        right = (8.0 * impact + 0.7 * idle, -7.0 * impact + 0.5 * idle_breath, 8.5 * impact + 0.5 * idle, 1.0 + 0.025 * impact)
        head = (1.8 * impact + 0.55 * idle, -1.4 * impact, 0.010 * impact, -1.0 * impact + 0.12 * idle)
    elif beat == "escalation":
        left = (-8.0 * impact, -13.0 * impact, -7.5 * impact, 1.0 + 0.045 * impact)
        right = (10.0 * impact, -15.0 * impact, 8.5 * impact, 1.0 + 0.050 * impact)
        head = (4.0 * impact, -8.0 * impact, 0.028 * impact, 2.3 * impact)
    elif beat == "reaction":
        left = (-9.0 * impact, 7.0 * impact, -7.0 * impact, 1.0 + 0.032 * impact)
        right = (12.0 * impact, 5.0 * impact, 8.0 * impact, 1.0 + 0.030 * impact)
        head = (-4.0 * impact, 6.0 * impact, 0.035 * impact, -3.0 * impact)
    else:
        gesture = math.sin(math.pi * _clamp((p - 0.22) / 0.58)) * (0.55 + 0.45 * energy)
        left = (-2.5 * gesture + 0.35 * idle, 0.8 * gesture, -2.0 * gesture + 0.25 * idle, 1.0 + 0.010 * gesture)
        right = (3.5 * gesture + 0.45 * idle, 0.6 * gesture + 0.3 * idle_breath, 2.2 * gesture + 0.3 * idle, 1.0 + 0.012 * gesture)
        head = (1.0 * impact + 0.4 * idle, -0.8 * impact, 0.010 * impact, 0.4 * impact + 0.10 * idle)

    # A speech envelope adds tiny articulation movement, never replacing the
    # larger emotional body beat. The speech activity is normalized in voice.py.
    speech = 0.0
    speech_activity = (voice_performance or {}).get("activity") if voice_performance else None
    if isinstance(speech_activity, list) and speech_activity:
        idx = min(len(speech_activity) - 1, int(round(_clamp(progress) * (len(speech_activity) - 1))))
        try:
            speech = _clamp((float(speech_activity[idx]) - 0.18) / 0.62)
        except (TypeError, ValueError):
            speech = 0.0

    head_dx, head_dy, head_scale, head_rot = head
    left = (left[0] + 2.2 * hand_lag, left[1] + 1.1 * hand_lag,
            left[2] - 1.8 * hand_lag, left[3])
    right = (right[0] + 2.8 * hand_lag, right[1] + 1.4 * hand_lag,
             right[2] + 2.0 * hand_lag, right[3])
    attention = {
        "setup": (-0.55, 0.00),
        "escalation": (0.52, -0.04),
        "reaction": (0.38, 0.02),
        "punchline": (0.00, 0.00),
    }.get(beat, (0.0, 0.0))
    attention_x, attention_y = attention
    face_target_dx = 8.0 * attention_x * (0.35 + 0.65 * impact)
    face_target_dy = 5.0 * attention_y * (0.30 + 0.70 * impact)
    return {
        "event_fraction": peak,
        "root_dx": root_dx,
        "root_dy": root_dy,
        "root_rot": root_rot,
        "root_scale": 1.0 + 0.013 * impact * energy,
        "left_dx": left[0],
        "left_dy": left[1],
        "left_rot": left[2],
        "left_scale": left[3],
        "right_dx": right[0],
        "right_dy": right[1],
        "right_rot": right[2],
        "right_scale": right[3],
        "head_dx": head_dx,
        "head_dy": head_dy,
        "head_scale": 1.0 + head_scale,
        "head_rot": head_rot,
        "face_target_dx": face_target_dx,
        "face_target_dy": face_target_dy,
        "attention_x": attention_x,
        "attention_y": attention_y,
        # Reserve a little look space in the direction of the character's
        # attention. The camera consumes this as a framing offset, not as
        # additional character motion.
        "camera_focus_x": attention_x * impact,
        "camera_focus_y": attention_y * impact,
        "speech_pulse": speech,
        # Camera tracks the root with a tiny delay, so the camera feels attached
        # to the scene and not like an independent Ken-Burns animation.
        "camera_dx": root_dx * 0.62,
        "camera_dy": root_dy * 0.48,
        "camera_rot": root_rot * 0.24,
        "camera_zoom": (0.030 * impact if beat in {"escalation", "reaction"} else 0.012 * impact)
        + _camera_event_push(p, peak, beat),
    }


def arm_motion(progress: float, beat: str, side: str, voice_performance: dict | None = None) -> tuple[float, float, float, float]:
    """Return small independent arm/hand gesture layered on shared root motion."""
    state = kinematic_state(progress, beat, voice_performance)
    if side == "left":
        return state["left_dx"], state["left_dy"], state["left_scale"], state["left_rot"]
    return state["right_dx"], state["right_dy"], state["right_scale"], state["right_rot"]


def head_motion(progress: float, beat: str, voice_performance: dict | None = None) -> tuple[float, float, float, float]:
    state = kinematic_state(progress, beat, voice_performance)
    return state["head_dx"], state["head_dy"], state["head_scale"], state["head_rot"]
