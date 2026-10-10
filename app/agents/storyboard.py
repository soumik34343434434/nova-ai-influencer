"""Storyboard planner for NOVA short-form videos.

Turns one script into several visual shots so the renderer can create a real
sequence instead of holding a single static frame for the whole Reel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.agents.world_engine import WorldEngine
from app.agents.performance import plan_performance
from app.agents.shot_performance import build_shot_performance_spec
from app.agents.director_decision import build_director_decision_packet
from app.agents.reference_preflight import build_reference_manifest, assert_reference_preflight, required_reference_props
from app.agents.character_world_bible import validate_storyboard_continuity
from app.agents.action_packs import action_spec


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
    director_decision: dict


@dataclass(frozen=True)
class Storyboard:
    hook: str
    total_duration: float
    shots: list[Shot]

    def to_dict(self) -> dict:
        continuity_errors = validate_storyboard_continuity([
            {
                "character": shot.world.get("character", ""),
                "character_plan": shot.world.get("character_plan", {}),
                "framing": shot.world.get("framing", ""),
                **({"signature_props": shot.world["signature_props"]} if "signature_props" in shot.world else {}),
            }
            for shot in self.shots
        ])
        return {
            "hook": self.hook,
            "total_duration": self.total_duration,
        "shots": [asdict(s) for s in self.shots],
            "validation": {"passed": not continuity_errors, "errors": continuity_errors},
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
    if index == 3:
        return "reaction_close_up"
    if "manager" in lower or "meeting" in lower:
        return "over_shoulder"
    if "looks at camera" in lower or "looks at the camera" in lower or "stares" in lower:
        return "reaction_close_up"
    if "result" in lower or "conclusion" in lower:
        return "medium"
    return "wide_desk"


def _action_for(camera: str, text: str) -> str:
    lower = text.lower()
    if "looks at camera" in lower or "looks at the camera" in lower or "stares" in lower or camera == "reaction_close_up":
        return "deadpan_camera_reaction"
    if "phone" in lower or "notification" in lower:
        return "checks_phone"
    if "done" in lower or "finished" in lower or "finally" in lower:
        return "finish_work"
    if "meeting" in lower or "calendar" in lower:
        return "sees_calendar"
    if "coffee" in lower:
        return "reaches_for_coffee"
    if "tabs" in lower:
        return "glances_at_many_tabs"
    if camera == "hook_close_up":
        return "leans_toward_camera"
    return "talks_with_small_hand_gesture"


def _beat_text(beat: dict) -> str:
    return " ".join(str(beat.get("dialogue") or beat.get("voice_text") or beat.get("voice") or "").split())


def build_storyboard(draft: dict) -> Storyboard:
    hook = " ".join(str(draft.get("hook", "")).split())
    paragraphs = _paragraphs(str(draft.get("script", "")))
    source_beats = [b for b in (draft.get("beats") or []) if isinstance(b, dict)]
    # When the script supplies beats, plan exactly one shot per rendered beat so
    # the Director's per-shot decisions line up with the frames that are drawn.
    beat_locked = bool(source_beats) and all(_beat_text(b) for b in source_beats)

    # A shortform Reel works better as a handful of decisive visual beats.
    selected = [_beat_text(b) for b in source_beats] if beat_locked else paragraphs[:5]
    if not selected:
        selected = [hook]
    subject = str(draft.get("character") or "NOVA").upper()

    world_engine = WorldEngine()
    shots: list[Shot] = []
    for index, text in enumerate(selected):
        source_beat = source_beats[index] if index < len(source_beats) else {}
        camera = _camera_for(index, text)
        if str(source_beat.get("framing", "")) == "close_up":
            camera = "reaction_close_up"
        # Prefer the beat's authored, renderable action over text heuristics.
        action = str(source_beat.get("asset_action") or source_beat.get("action") or "") or _action_for(camera, text)
        world = world_engine.plan_shot(text, index)
        world["bible_version"] = world_engine.bible.get("version", 1)
        character = str(draft.get("character", "")).upper()
        if character in world_engine.bible.get("characters", {}):
            world["character"] = character
            world["identity_lock"] = world_engine.bible["characters"][character]
            world["style"] = world_engine.bible["style"]
        try:
            authored = float(source_beat.get("duration") or 0.0)
        except (TypeError, ValueError):
            authored = 0.0
        duration = authored if beat_locked and authored > 0 else max(2.6, min(4.8, 2.6 + len(text) / 45.0))
        performance = plan_performance(action, text, index)
        performance["shot_spec"] = build_shot_performance_spec(
            action=action,
            dialogue=text,
            duration=duration,
            camera=camera,
        )
        performance["expected_scene"] = str(world["scene"])
        performance["reference_manifest"] = build_reference_manifest(
            character=str(draft.get("character", "")),
            scene=str(world["scene"]),
            props=sorted(set([world["prop"], *required_reference_props(action)])),
        )
        assert_reference_preflight(
            action,
            str(draft.get("character", "")),
            performance["reference_manifest"],
            expected_scene=str(world["scene"]),
        )
        shots.append(
            Shot(
                index=index,
                duration=round(duration, 2),
                camera=camera,
                action=action,
                dialogue=text,
                visual=(
                    f"{subject} {action.replace('_', ' ')} in a {camera.replace('_', ' ')} shot "
                    f"at {world['scene_label']}, with {world['prop']}."
                ),
                world=world,
                performance=performance,
                director_decision=build_director_decision_packet(
                    index=index, dialogue=text, camera=camera, action=action,
                    world={**world, "sfx": str(source_beat.get("sfx", ""))},
                    performance=performance, duration=duration,
                    story_role=str(source_beat.get("story_role", source_beat.get("beat", ""))),
                    story_intent=str(source_beat.get("story_intent", "")),
                    audio_emphasis=str(source_beat.get("audio_emphasis", "")),
                    prop_action=str(source_beat.get("prop_interaction", "")),
                ).to_dict(),
            )
        )

    # Keep the edit punchy. The final reaction can loop back into the hook.
    total = round(sum(shot.duration for shot in shots), 2)
    if total > 18 and shots and not beat_locked:
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
                {
                    **s.performance,
                    "shot_spec": build_shot_performance_spec(
                        action=s.action,
                        dialogue=s.dialogue,
                        duration=round(max(2.4, s.duration * factor), 2),
                        camera=s.camera,
                    ),
                },
                build_director_decision_packet(
                    index=s.index, dialogue=s.dialogue, camera=s.camera, action=s.action,
                    world=s.world, performance={**s.performance, "shot_spec": build_shot_performance_spec(
                        action=s.action, dialogue=s.dialogue,
                        duration=round(max(2.4, s.duration * factor), 2), camera=s.camera,
                    )}, duration=round(max(2.4, s.duration * factor), 2),
                    story_role=s.director_decision["story"]["role"],
                    story_intent=s.director_decision["story"]["intent"],
                    audio_emphasis=s.director_decision["audio"]["emphasis"],
                    prop_action=s.director_decision["prop_interaction"]["intent"],
                ).to_dict(),
            )
            for s in shots
        ]
        total = round(sum(shot.duration for shot in shots), 2)

    character = str(draft.get("character", "")).upper()
    if character in world_engine.bible.get("characters", {}):
        plans = [_bible_shot_plan(character, shot.action) for shot in shots]
        for shot, plan in zip(shots, plans):
            shot.world["character_plan"] = plan["character_plan"]
            shot.world["framing"] = plan["framing"]
            if "signature_props" in plan:
                shot.world["signature_props"] = plan["signature_props"]
        errors = validate_storyboard_continuity(plans, world_engine.bible)
        if errors:
            raise ValueError("Character/world bible validation failed: " + "; ".join(errors))
    return Storyboard(hook=hook, total_duration=total, shots=shots)


# What the local renderer does to a character image within a shot.
RENDER_TRANSFORMATIONS = ("pose", "gaze", "facial_expression", "rotation", "screen_position", "scale_uniform")


def _bible_shot_plan(character: str, action: str) -> dict:
    """Describe a shot in the bible's terms using the action pack's authored data."""
    spec = action_spec(character, action) or {}
    plan = {
        "character": character,
        "framing": str(spec.get("framing", "medium")),
        "character_plan": {"transformations": list(RENDER_TRANSFORMATIONS)},
    }
    if isinstance(spec.get("visible_props"), list):
        plan["signature_props"] = list(spec["visible_props"])
    return plan
