"""Scene-aware performance planner for NOVA."""

from __future__ import annotations


PERFORMANCE = {
    "deadpan_camera_reaction": {
        "expression": "deadpan",
        "gesture": "freeze",
        "eye_brow": 0.0,
        "head": 0.15,
    },
    "checks_phone": {
        "expression": "surprised",
        "gesture": "phone_glance",
        "eye_brow": 0.65,
        "head": 0.35,
    },
    "sees_calendar": {
        "expression": "annoyed",
        "gesture": "point_calendar",
        "eye_brow": 0.25,
        "head": 0.28,
    },
    "reaches_for_coffee": {
        "expression": "tired",
        "gesture": "coffee_reach",
        "eye_brow": 0.15,
        "head": 0.20,
    },
    "glances_at_many_tabs": {
        "expression": "overloaded",
        "gesture": "tab_scan",
        "eye_brow": 0.40,
        "head": 0.42,
    },
    "leans_toward_camera": {
        "expression": "curious",
        "gesture": "lean",
        "eye_brow": 0.55,
        "head": 0.50,
    },
    "talks_with_small_hand_gesture": {
        "expression": "neutral",
        "gesture": "talk",
        "eye_brow": 0.20,
        "head": 0.22,
    },
}


def plan_performance(action: str, dialogue: str, shot_index: int = 0) -> dict:
    key = action if action in PERFORMANCE else "talks_with_small_hand_gesture"
    plan = dict(PERFORMANCE[key])

    lower = dialogue.lower()
    if "?" in dialogue:
        plan["expression"] = "skeptical"
        plan["eye_brow"] = max(plan["eye_brow"], 0.50)
    if "!" in dialogue:
        plan["expression"] = "surprised"
        plan["eye_brow"] = max(plan["eye_brow"], 0.65)
    if any(word in lower for word in ("finished", "result", "conclusion")):
        plan["expression"] = "confident"
    if shot_index == 0:
        plan["attention"] = "camera"
    else:
        plan["attention"] = "scene"

    plan["speaking_style"] = "measured_deadpan"
    return plan
