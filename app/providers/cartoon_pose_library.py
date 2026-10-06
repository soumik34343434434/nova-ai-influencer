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

    # Character personality slightly biases the expressive library.
    if character.upper() == "MOMO":
        p["idle"]["eye_open"] = 0.92
        p["sleepy"]["eye_open"] = 0.52
    elif character.upper() == "BUBU":
        p["excited"]["root"] = _shift(p["excited"]["root"], 0, -18)
        p["excited"]["eye_open"] = 1.24

    return p


DEFAULT_ACTION_POSES = {
    "press_button": "reach",
    "give_homework": "present",
    "tap_alarm": "tap",
    "open_fridge": "reach",
    "tap_shoes": "tap",
    "shock": "shock",
    "overwhelmed": "overwhelmed",
    "deadpan": "deadpan",
}
