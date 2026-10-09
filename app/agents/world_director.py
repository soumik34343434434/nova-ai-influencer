"""Depth and environmental-acting plan for NOVA scenes."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class WorldDepthPlan:
    back_layer: str
    mid_layer: str
    foreground_layer: str
    parallax: tuple[float, float, float]
    light_motion: str
    prop_action: str
    continuity_lock: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"parallax": list(self.parallax), "continuity_lock": list(self.continuity_lock)}


SCENE_PROFILES = {
    "office": ("glass-room silhouettes", "desk and shelf dressing", "monitor edge / desk trim", "cool practicals", "micro monitor glow"),
    "home": ("distant city lights", "desk, lamp and wall plane", "mug / laptop foreground", "warm practical lamp", "soft lamp pulse"),
    "cafe": ("window daylight and distant patrons", "tables, plants and counter", "cup / notebook foreground", "natural daylight", "subtle window-light drift"),
    "metro": ("blurred carriage windows", "seats, poles and commuters", "phone / handrail foreground", "overhead carriage light", "tiny passing-light sweep"),
}


def plan_world_depth(scene: str, prop: str, action: str) -> WorldDepthPlan:
    back, mid, foreground, light, light_motion = SCENE_PROFILES.get(scene, SCENE_PROFILES["office"])
    prop_action = {
        "printer": "emit / advance paper only during the action contact window",
        "phone": "wake / glow at attention beat, remain fixed afterward",
        "coffee": "settle with a tiny body-follow cue on contact",
        "papers": "spread or lift on impact, then settle without drifting",
        "alarm": "pulse once at contact, then remain spatially locked",
        "fridge": "door movement only when action explicitly opens it",
        "shoes": "remain grounded; no independent drift",
    }.get(prop, "remain spatially locked to the scene")
    parallax = (0.08, 0.35, 0.75) if scene != "metro" else (0.12, 0.28, 0.62)
    if action in {"shock", "overwhelmed", "deadpan"}:
        light_motion = f"{light_motion}; no camera-driven background drift during reaction"
    return WorldDepthPlan(back, mid, foreground, parallax, light_motion, prop_action, (
        "preserve scene identity across all shots",
        "keep primary prop in the same spatial world as the character",
        "do not introduce recognizable brands or logos",
        "foreground motion must stay subordinate to the acting performance",
    ))
