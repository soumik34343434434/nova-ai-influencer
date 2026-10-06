"""Storyboard planner for NOVA short-form videos.

Turns one script into several visual shots so the renderer can create a real
sequence instead of holding a single static frame for the whole Reel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.agents.world_engine import WorldEngine
from app.agents.performance import plan_performance


@dataclass(frozen=True)
class Shot:
    index: int
    duration: float
    camera: str
    action: str
    dialogue: str
    visual: str
    world: dict
    performance: dict


@dataclass(frozen=True)
class Storyboard:
    hook: str
    total_duration: float
    shots: list[Shot]

    def to_dict(self) -> dict:
        return {
            "hook": self.hook,
            "total_duration": self.total_duration,
            "shots": [asdict(s) for s in self.shots],
        }


def _paragraphs(script: str) -> list[str]:
    blocks = []
    for raw in script.replace("\r", "").split("\n\n"):
        text = " ".join(part.strip() for part in raw.splitlines() if part.strip())
        if text:
            blocks.append(text)
    return blocks


def _camera_for(index: int, text: str) -> str:
    lower = text.lower()
    if index == 0:
        return "hook_close_up"
    if "manager" in lower or "meeting" in lower:
        return "over_shoulder"
    if "looks at camera" in lower or "stares" in lower:
        return "reaction_close_up"
    if "result" in lower or "conclusion" in lower:
        return "medium"
    return "wide_desk"


def _action_for(camera: str, text: str) -> str:
    lower = text.lower()
    if "phone" in lower or "notification" in lower:
        return "checks_phone"
    if "meeting" in lower or "calendar" in lower:
        return "sees_calendar"
    if "looks at camera" in lower or "stares" in lower:
        return "deadpan_camera_reaction"
    if "coffee" in lower:
        return "reaches_for_coffee"
    if "tabs" in lower:
        return "glances_at_many_tabs"
    if camera == "hook_close_up":
        return "leans_toward_camera"
    return "talks_with_small_hand_gesture"


def build_storyboard(draft: dict) -> Storyboard:
    hook = " ".join(str(draft.get("hook", "")).split())
    paragraphs = _paragraphs(str(draft.get("script", "")))

    # A shortform Reel works better as a handful of decisive visual beats.
    selected = paragraphs[:5]
    if not selected:
        selected = [hook]

    world_engine = WorldEngine()
    shots: list[Shot] = []
    for index, text in enumerate(selected):
        camera = _camera_for(index, text)
        action = _action_for(camera, text)
        world = world_engine.plan_shot(text, index)
        performance = plan_performance(action, text, index)
        duration = max(2.6, min(4.8, 2.6 + len(text) / 45.0))
        shots.append(
            Shot(
                index=index,
                duration=round(duration, 2),
                camera=camera,
                action=action,
                dialogue=text,
                visual=(
                    f"NOVA {action.replace('_', ' ')} in a {camera.replace('_', ' ')} shot "
                    f"at {world['scene_label']}, with {world['prop']}."
                ),
                world=world,
                performance=performance,
            )
        )

    # Keep the edit punchy. The final reaction can loop back into the hook.
    total = round(sum(shot.duration for shot in shots), 2)
    if total > 18 and shots:
        factor = 18 / total
        shots = [
            Shot(
                s.index,
                round(max(2.4, s.duration * factor), 2),
                s.camera,
                s.action,
                s.dialogue,
                s.visual,
                s.world,
                s.performance,
            )
            for s in shots
        ]
        total = round(sum(shot.duration for shot in shots), 2)

    return Storyboard(hook=hook, total_duration=total, shots=shots)
