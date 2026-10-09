"""Recurring visual-world planner for NOVA."""

from __future__ import annotations

import json
from pathlib import Path

from app.agents.world_director import plan_world_depth


ROOT = Path(__file__).resolve().parents[2]
WORLD_PATH = ROOT / "config" / "world.json"


class WorldEngine:
    def __init__(self, path: Path = WORLD_PATH):
        self.data = json.loads(path.read_text())

    def choose_scene(self, text: str, shot_index: int = 0) -> str:
        lower = text.lower()
        scenes = self.data["scenes"]
        # Explicit location/context signals outrank generic objects such as
        # "notification", which can occur in several scenes.
        if any(term in lower for term in ("metro", "commute", "train carriage")):
            return "metro"
        if any(term in lower for term in ("cafe", "coffee shop", "caffeine")):
            return "cafe"
        if any(term in lower for term in ("bedroom", "home desk", "late night", "doomscrolling")):
            return "home"

        direct = {
            "office": ("meeting", "manager", "calendar", "presentation", "deadline", "office", "backlog", "task", "tasks", "work", "phone", "notification", "message"),
            "home": ("tabs", "browser", "bedroom", "home desk"),
            "cafe": ("coffee", "cafe", "caffeine", "weekend"),
            "metro": ("earbuds",),
        }
        hits = [(sum(term in lower for term in terms), name) for name, terms in direct.items()]
        hits.sort(reverse=True)
        if hits[0][0] > 0:
            return hits[0][1]

        ranked = []
        for name, scene in scenes.items():
            count = sum(term in lower for term in scene["best_for"])
            ranked.append((count, name))
        ranked.sort(reverse=True)
        if ranked[0][0] > 0:
            return ranked[0][1]
        names = list(scenes)
        return names[shot_index % len(names)]

    def wardrobe(self, scene: str) -> str:
        return self.data["wardrobes"].get(f"{scene}_default", self.data["wardrobes"]["office_default"])

    def prop(self, scene: str, text: str, shot_index: int = 0) -> str:
        lower = text.lower()
        props = self.data["scenes"][scene]["props"]
        aliases = {
            "laptop": ("laptop", "tab", "tabs", "browser", "screen"),
            "phone": ("phone", "notification", "message", "call"),
            "coffee": ("coffee", "cafe", "caffeine"),
            "calendar": ("calendar", "meeting", "invite", "schedule"),
            "headphones": ("headphone", "earbud", "earbuds"),
            "notebook": ("notebook", "notes", "write"),
            "metro card": ("metro card", "transit card"),
            "earbuds": ("earbud", "earbuds", "headphone"),
        }
        for prop in props:
            if any(term in lower for term in aliases.get(prop, (prop,))):
                return prop
        return props[shot_index % len(props)]

    def plan_shot(self, text: str, shot_index: int = 0) -> dict:
        scene = self.choose_scene(text, shot_index)
        prop = self.prop(scene, text, shot_index)
        action = str({
            "phone": "checks_phone",
            "calendar": "sees_calendar",
            "coffee": "reaches_for_coffee",
        }.get(prop, "small_scene_interaction"))
        return {
            "scene": scene,
            "scene_label": self.data["scenes"][scene]["label"],
            "setting": self.data["scenes"][scene]["setting"],
            "wardrobe": self.wardrobe(scene),
            "prop": prop,
            "mood": self.data["scenes"][scene]["mood"],
            "depth_plan": plan_world_depth(scene, prop, action).to_dict(),
        }
