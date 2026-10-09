"""Provider-neutral orchestration contract for one directed shot."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DirectorDecisionPacket:
    version: str
    shot_id: str
    story: dict[str, Any]
    emotion: dict[str, str]
    action: dict[str, str]
    camera: dict[str, str]
    character_performance: dict[str, Any]
    world: dict[str, Any]
    prop_interaction: dict[str, str]
    audio: dict[str, str]
    timing: dict[str, Any]
    continuity_locks: tuple[str, ...]
    negative_constraints: tuple[str, ...]
    priorities: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["continuity_locks"] = list(self.continuity_locks)
        data["negative_constraints"] = list(self.negative_constraints)
        data["handoff"] = self.handoff()
        return data

    def handoff(self) -> dict[str, Any]:
        """Stable renderer/provider handoff; contains intent, not implementation."""
        return {"contract": "nova.director_decision.v1", "shot_id": self.shot_id,
                "duration": self.timing["duration"], "camera": self.camera,
                "performance": self.character_performance, "world": self.world,
                "prop_interaction": self.prop_interaction, "audio": self.audio,
                "continuity_locks": list(self.continuity_locks),
                "negative_constraints": list(self.negative_constraints),
                "priorities": self.priorities}


def validate_packet(packet: DirectorDecisionPacket) -> None:
    duration = float(packet.timing.get("duration", 0))
    if duration <= 0:
        raise ValueError("director packet duration must be positive")
    if packet.camera.get("framing") != packet.character_performance.get("camera"):
        raise ValueError("camera framing conflicts with character performance camera")
    if not packet.world.get("scene") or packet.priorities.get("camera_actor_world") != "camera > actor > world":
        raise ValueError("invalid camera/actor/world priority contract")
    phases = packet.timing.get("phases", [])
    previous = 0.0
    for phase in phases:
        start, end = float(phase["start"]), float(phase["end"])
        # Shot lengths are presented to two decimals while performance phase
        # boundaries retain millisecond precision.
        if start < previous - 1e-6 or end < start or end > duration + 0.011:
            raise ValueError("director packet timing phases conflict or exceed duration")
        previous = end
    if phases and abs(float(phases[-1]["end"]) - duration) > 0.011:
        raise ValueError("director packet timing phases must cover the shot duration")


def build_director_decision_packet(*, index: int, dialogue: str, camera: str,
        action: str, world: dict[str, Any], performance: dict[str, Any],
        duration: float, story_role: str = "", story_intent: str = "",
        audio_emphasis: str = "", prop_action: str = "") -> DirectorDecisionPacket:
    spec = performance.get("shot_spec", {})
    packet = DirectorDecisionPacket(
        version="1.0", shot_id=f"shot-{index:02d}",
        story={"role": story_role or ("hook" if index == 0 else "beat"),
               "intent": story_intent or dialogue, "dialogue": dialogue},
        emotion={"start": str(spec.get("emotion_start", "neutral")),
                 "peak": str(spec.get("emotion_peak", "reactive")),
                 "end": str(spec.get("emotion_end", "deadpan"))},
        action={"name": action, "intent": str(spec.get("action", action))},
        camera={"framing": camera, "intent": f"{camera.replace('_', ' ')}; preserve readable action"},
        character_performance=spec,
        world=dict(world),
        prop_interaction={"prop": str(world.get("prop", "")), "intent": prop_action or action},
        audio={"emphasis": audio_emphasis or "dialogue-led", "sfx": str(world.get("sfx", ""))},
        timing={"duration": round(float(duration), 2),
                "start": 0.0, "end": round(float(duration), 2),
                "phases": list(spec.get("phases", []))},
        continuity_locks=tuple(spec.get("continuity", ())) +
            (f"preserve world scene '{world.get('scene', '')}' and prop placement",),
        negative_constraints=tuple(spec.get("negative_constraints", ())) +
            ("do not change the approved story intent or dialogue",),
        priorities={"camera_actor_world": "camera > actor > world",
                    "primary": "story readability", "conflict_resolution": "preserve story and continuity; reduce secondary motion"},
    )
    validate_packet(packet)
    return packet
