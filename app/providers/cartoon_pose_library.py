"""Reusable semantic pose library for NOVA's cutout cartoon rigs."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def _shift(p: list[float], dx: float = 0.0, dy: float = 0.0) -> list[float]:
    return [round(p[0] + dx), round(p[1] + dy)]


def _clone(base: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(base)


def _arm(poses: dict, side: str, shoulder_dx: float, elbow_dx: float, elbow_dy: float, wrist_dx: float, wrist_dy: float) -> None:
    poses[f"{side}_shoulder"] = _shift(poses[f"{side}_shoulder"], shoulder_dx, 0)
    poses[f"{side}_elbow"] = _shift(poses[f"{side}_elbow"], elbow_dx, elbow_dy)
    poses[f"{side}_wrist"] = _shift(poses[f"{side}_wrist"], wrist_dx, wrist_dy)


def build_pose_library(base: dict[str, Any], character: str) -> dict[str, dict[str, Any]]:
    """Create named reusable acting poses from a character's canonical rig points."""
    p = {}

    idle = _clone(base)
    p["idle"] = idle

    lean = _clone(base)
    lean["root"] = _shift(base["root"], 0, -28)
    lean["neck"] = _shift(base["neck"], 0, -34)
    lean["head_center"] = _shift(base["head_center"], 8, -46)
    lean["head_rot"] = -5.0
    lean["root_rot"] = -1.5
    lean["gaze"] = [0.18, -0.03]
    _arm(lean, "left", 8, 28, -70, 36, -120)
    _arm(lean, "right", 8, 44, -68, 58, -110)
    p["lean_in"] = lean

    reach = _clone(base)
    reach["root"] = _shift(base["root"], 6, -10)
    reach["neck"] = _shift(base["neck"], 6, -18)
    reach["head_center"] = _shift(base["head_center"], 8, -18)
    reach["head_rot"] = 2.0
    reach["gaze"] = [0.38, -0.02]
    _arm(reach, "left", 12, 34, -48, 54, -100)
    _arm(reach, "right", 15, 86, -32, 110, -78)
    p["reach"] = reach

    present = _clone(base)
    present["root"] = _shift(base["root"], 0, -8)
    present["head_center"] = _shift(base["head_center"], -4, -8)
    present["head_rot"] = -3.0
    present["gaze"] = [-0.18, -0.01]
    _arm(present, "left", -2, -35, -18, -82, -28)
    _arm(present, "right", 2, 40, -20, 86, -35)
    p["present"] = present

    tap = _clone(base)
    tap["root"] = _shift(base["root"], 0, -12)
    tap["head_center"] = _shift(base["head_center"], 4, -18)
    tap["head_rot"] = 4.0
    tap["gaze"] = [0.16, -0.04]
    _arm(tap, "left", -4, -38, -42, -55, -75)
    _arm(tap, "right", 5, 58, -16, 108, -20)
    p["tap"] = tap

    excited = _clone(base)
    excited["root"] = _shift(base["root"], 0, -32)
    excited["neck"] = _shift(base["neck"], 0, -40)
    excited["head_center"] = _shift(base["head_center"], 0, -48)
    excited["head_rot"] = 0.0
    excited["root_rot"] = 0.0
    excited["eye_open"] = 1.18
    excited["gaze"] = [0.0, -0.04]
    excited["mouth"] = "open"
    _arm(excited, "left", 0, -18, -108, -42, -155)
    _arm(excited, "right", 0, 18, -108, 42, -155)
    p["excited"] = excited

    shock = _clone(base)
    shock["root"] = _shift(base["root"], 0, 18)
    shock["neck"] = _shift(base["neck"], 0, 8)
    shock["head_center"] = _shift(base["head_center"], 0, 6)
    shock["head_rot"] = 0.0
    shock["root_rot"] = 0.0
    shock["eye_open"] = 1.24
    shock["gaze"] = [0.0, -0.02]
    shock["mouth"] = "open"
    _arm(shock, "left", -4, -34, -74, -58, -105)
    _arm(shock, "right", 4, 34, -74, 58, -105)
    p["shock"] = shock

    recoil = _clone(base)
    recoil["root"] = _shift(base["root"], -6, 72)
    recoil["neck"] = _shift(base["neck"], -10, 58)
    recoil["head_center"] = _shift(base["head_center"], -18, 60)
    recoil["head_rot"] = -9.0
    recoil["root_rot"] = -4.0
    recoil["eye_open"] = 0.82
    recoil["gaze"] = [-0.18, 0.05]
    recoil["mouth"] = "frown"
    _arm(recoil, "left", -10, -48, 30, -68, 68)
    _arm(recoil, "right", 4, 44, 28, 72, 64)
    p["recoil"] = recoil

    overwhelmed = _clone(base)
    overwhelmed["root"] = _shift(base["root"], -18, 58)
    overwhelmed["neck"] = _shift(base["neck"], -20, 42)
    overwhelmed["head_center"] = _shift(base["head_center"], -24, 38)
    overwhelmed["head_rot"] = -12.0
    overwhelmed["root_rot"] = -5.0
    overwhelmed["eye_open"] = 0.86
    overwhelmed["gaze"] = [0.30, 0.03]
    overwhelmed["mouth"] = "frown"
    _arm(overwhelmed, "left", -16, -42, 36, -55, 92)
    _arm(overwhelmed, "right", -4, 56, 36, 88, 94)
    p["overwhelmed"] = overwhelmed

    shrug = _clone(base)
    shrug["root"] = _shift(base["root"], 0, 10)
    shrug["head_center"] = _shift(base["head_center"], 0, -4)
    shrug["head_rot"] = 2.0
    shrug["gaze"] = [0.0, 0.0]
    _arm(shrug, "left", -4, -48, -60, -78, -38)
    _arm(shrug, "right", 4, 48, -60, 78, -38)
    p["shrug"] = shrug

    deadpan = _clone(base)
    deadpan["root"] = _shift(base["root"], 6, 8)
    deadpan["neck"] = _shift(base["neck"], 5, 2)
    deadpan["head_center"] = _shift(base["head_center"], 6, 8)
    deadpan["head_rot"] = 1.0
    deadpan["root_rot"] = 0.5
    deadpan["eye_open"] = 0.86
    deadpan["gaze"] = [0.02, 0.0]
    deadpan["mouth"] = "flat"
    _arm(deadpan, "left", 0, -4, 10, -8, 12)
    _arm(deadpan, "right", 0, 4, 10, 8, 12)
    p["deadpan"] = deadpan

    sleepy = _clone(base)
    sleepy["root"] = _shift(base["root"], 0, 26)
    sleepy["neck"] = _shift(base["neck"], 0, 30)
    sleepy["head_center"] = _shift(base["head_center"], 14, 40)
    sleepy["head_rot"] = 8.0
    sleepy["eye_open"] = 0.62
    sleepy["gaze"] = [0.0, 0.16]
    sleepy["mouth"] = "flat"
    _arm(sleepy, "left", 2, -8, 24, -14, 34)
    _arm(sleepy, "right", -2, 8, 24, 14, 34)
    p["sleepy"] = sleepy

    # Character-specific acting geometry. The rig stays reusable, but the
    # silhouette of a sleepy cat, robot and bird should not perform the exact
    # same gesture. These are deliberately still conservative enough for the
    # compiled cutout layers to deform cleanly.
    if character.upper() == "MOMO":
        p["idle"]["eye_open"] = 0.92
        p["sleepy"]["eye_open"] = 0.52
        p["present"]["root"] = _shift(base["root"], -4, -10)
        p["present"]["head_center"] = _shift(base["head_center"], -6, -16)
        _arm(p["present"], "left", -4, -58, -34, -112, -54)
        _arm(p["present"], "right", 6, 62, -34, 116, -50)
        p["present"]["root"] = _shift(base["root"], 18, -18)
        p["present"]["head_center"] = _shift(base["head_center"], 22, -28)
        p["present"]["head_rot"] = 4.5
        _arm(p["present"], "left", -4, -24, -48, -52, -82)
        _arm(p["present"], "right", 10, 110, -58, 175, -104)
        p["shock"]["root"] = _shift(base["root"], 0, 18)
        p["shock"]["neck"] = _shift(base["neck"], 0, 8)
        p["shock"]["head_center"] = _shift(base["head_center"], 4, -12)
        p["shock"]["head_rot"] = 3.5
        p["shock"]["root_rot"] = 2.0
        p["shock"]["eye_open"] = 1.12
        p["shock"]["gaze"] = [0.12, -0.08]
        _arm(p["shock"], "left", -10, -58, -118, -96, -198)
        _arm(p["shock"], "right", 10, 58, -118, 96, -198)
        p["overwhelmed"]["root"] = _shift(base["root"], -22, 78)
        p["overwhelmed"]["neck"] = _shift(base["neck"], -24, 52)
        p["overwhelmed"]["head_center"] = _shift(base["head_center"], -34, 54)
        p["overwhelmed"]["head_rot"] = -16.0
        p["overwhelmed"]["root_rot"] = -6.0
        p["overwhelmed"]["eye_open"] = 0.66
        p["overwhelmed"]["gaze"] = [0.28, 0.08]
        _arm(p["overwhelmed"], "left", -26, -80, 54, -124, 110)
        _arm(p["overwhelmed"], "right", 4, 82, 54, 142, 112)
        p["deadpan"]["root"] = _shift(base["root"], 12, 14)
        p["deadpan"]["neck"] = _shift(base["neck"], 8, 8)
        p["deadpan"]["head_center"] = _shift(base["head_center"], 14, 16)
        p["deadpan"]["head_rot"] = 2.0
        p["deadpan"]["gaze"] = [-0.10, 0.0]
        _arm(p["deadpan"], "left", -2, -14, 22, -22, 30)
        _arm(p["deadpan"], "right", 8, 28, -62, 56, -126)
    elif character.upper() == "BUBU":
        p["excited"]["root"] = _shift(p["excited"]["root"], 0, -18)
        p["excited"]["eye_open"] = 1.24
        p["present"]["root"] = _shift(base["root"], 0, -16)
        p["present"]["head_center"] = _shift(base["head_center"], 8, -22)
        p["present"]["head_rot"] = -5.0
        _arm(p["present"], "left", -10, -62, -24, -120, -36)
        _arm(p["present"], "right", 12, 68, -24, 124, -42)
        p["shock"]["root"] = _shift(base["root"], 0, 4)
        p["shock"]["head_center"] = _shift(base["head_center"], 0, -8)
        p["shock"]["eye_open"] = 1.30
        _arm(p["shock"], "left", -6, -52, -102, -86, -150)
        _arm(p["shock"], "right", 6, 52, -102, 86, -150)
        p["overwhelmed"]["root"] = _shift(base["root"], -22, 70)
        p["overwhelmed"]["head_center"] = _shift(base["head_center"], -20, 44)
        p["overwhelmed"]["head_rot"] = -10.0
        _arm(p["overwhelmed"], "left", -22, -72, 52, -104, 132)
        _arm(p["overwhelmed"], "right", -4, 82, 52, 132, 128)
        p["tap"]["root"] = _shift(base["root"], 0, -8)
        _arm(p["tap"], "left", -12, -72, -36, -106, -62)
        _arm(p["tap"], "right", 12, 92, -10, 158, 8)
        p["present"]["gaze"] = [0.20, -0.04]
        p["shock"]["gaze"] = [-0.06, -0.05]
        p["overwhelmed"]["gaze"] = [0.22, 0.06]

    return p


DEFAULT_ACTION_POSES = {
    "press_button": "reach",
    "checks_phone": "reach",
    "finish_work": "deadpan",
    "give_homework": "present",
    "tap_alarm": "tap",
    "open_fridge": "reach",
    "tap_shoes": "tap",
    "shock": "shock",
    "overwhelmed": "overwhelmed",
    "deadpan": "deadpan",
}

# These are intentionally data-shaped rather than baked into Blender. A character
# profile can override any timing or pose displacement while the pure-Python
# helper remains deterministic and testable.
DEFAULT_ACTION_TRANSITION_PROFILES: dict[str, dict[str, float]] = {
    "default": {
        "anticipation_at": 0.22,
        "anticipation_t": -0.18,
        "contact_approach_at": 0.46,
        "contact_approach_t": 0.72,
        "contact_at": 0.60,
        "contact_t": 1.00,
        "overshoot_at": 0.70,
        "overshoot_t": 1.10,
        "recoil_at": 0.82,
        "recoil_t": 0.93,
        "settle_at": 1.00,
        "settle_t": 1.00,
    },
    "press_button": {
        "arc_strength_px": 22.0,
        "anticipation_t": -0.24,
        "contact_approach_t": 0.76,
        "overshoot_t": 1.16,
        "recoil_t": 0.90,
    },
    "tap_alarm": {
        "arc_strength_px": 26.0,
        "anticipation_t": -0.20,
        "contact_approach_t": 0.78,
        "overshoot_t": 1.14,
        "recoil_t": 0.90,
    },
    "shock": {
        "arc_strength_px": 12.0,
        "anticipation_t": -0.14,
        "contact_approach_t": 0.76,
        "overshoot_t": 1.10,
        "recoil_t": 0.91,
    },
    "overwhelmed": {
        "arc_strength_px": 18.0,
        "anticipation_t": -0.14,
        "contact_approach_t": 0.74,
        "overshoot_t": 1.07,
        "recoil_t": 0.93,
    },
    "reach": {"arc_strength_px": 30.0, "anticipation_t": -0.18, "overshoot_t": 1.12, "recoil_t": 0.92},
    "present": {"arc_strength_px": 24.0, "anticipation_t": -0.10, "overshoot_t": 1.06, "recoil_t": 0.95},
    "tap": {"arc_strength_px": 28.0, "anticipation_t": -0.18, "overshoot_t": 1.10, "recoil_t": 0.92},
    "deadpan": {"arc_strength_px": 8.0, "anticipation_t": -0.08, "overshoot_t": 1.02, "recoil_t": 0.97},
    "sleepy": {"arc_strength_px": 10.0, "anticipation_t": -0.06, "overshoot_t": 1.02, "recoil_t": 0.98},
}


def make_action_pose_chain(
    prev: dict[str, Any],
    target: dict[str, Any],
    action: str,
    span: int = 100,
    transition_profiles: dict[str, dict[str, float]] | None = None,
) -> list[tuple[int, dict[str, Any], str]]:
    """Build deterministic anticipation/contact/impact motion for Blender and tests."""
    profiles = transition_profiles or DEFAULT_ACTION_TRANSITION_PROFILES
    profile = dict(profiles.get("default", DEFAULT_ACTION_TRANSITION_PROFILES["default"]))
    profile.update(profiles.get(action, {}))

    def blend(a: dict[str, Any], b: dict[str, Any], t: float) -> dict[str, Any]:
        # Smooth the normal in-range travel, but preserve true extrapolation for
        # anticipation (<0) and overshoot (>1). This gives a soft acceleration
        # into contact without erasing the physical opposition and follow-through.
        u = float(t)
        if 0.0 < u < 1.0:
            u = u * u * (3.0 - 2.0 * u)
        out = deepcopy(a)
        for k in (
            "root", "neck", "head_center",
            "left_shoulder", "left_elbow", "left_wrist",
            "right_shoulder", "right_elbow", "right_wrist",
        ):
            out[k] = [a[k][i] + (b[k][i] - a[k][i]) * u for i in (0, 1)]

        # Cartoon hand-drawn motion rarely travels in a perfect straight line.
        # Add a small configurable arc to elbows/wrists while preserving the
        # real anticipation/overshoot extrapolation outside [0, 1].
        arc_px = float(profile.get("arc_strength_px", 0.0))
        if arc_px and 0.0 <= float(t) <= 1.0:
            import math
            arc = math.sin(math.pi * float(t)) * arc_px
            for side, sign in (("left", 1.0), ("right", -1.0)):
                dx = float(b[f"{side}_wrist"][0] - a[f"{side}_wrist"][0])
                dy = float(b[f"{side}_wrist"][1] - a[f"{side}_wrist"][1])
                length = max(1.0, math.hypot(dx, dy))
                nx, ny = -dy / length, dx / length
                out[f"{side}_elbow"][0] += nx * arc * 0.55 * sign
                out[f"{side}_elbow"][1] += ny * arc * 0.55 * sign
                out[f"{side}_wrist"][0] += nx * arc * sign
                out[f"{side}_wrist"][1] += ny * arc * sign
        for k in ("head_rot", "root_rot", "eye_open"):
            out[k] = a[k] + (b[k] - a[k]) * u
        out["gaze"] = [a["gaze"][i] + (b["gaze"][i] - a["gaze"][i]) * u for i in (0, 1)]
        out["mouth"] = b["mouth"] if u >= 0.65 else a["mouth"]
        return out

    span = max(8, int(span))
    phases = [
        (0.0, 0.00, "approach"),
        (profile["anticipation_at"], profile["anticipation_t"], "anticipation"),
        (profile["contact_approach_at"], profile["contact_approach_t"], "contact_approach"),
        (profile["contact_at"], profile["contact_t"], "contact"),
        (profile["overshoot_at"], profile["overshoot_t"], "overshoot"),
        (profile["recoil_at"], profile["recoil_t"], "recoil"),
        (profile["settle_at"], profile["settle_t"], "settle"),
    ]
    return [(round(span * when), blend(prev, target, t), phase) for when, t, phase in phases]
