"""Provider-neutral AI performance routing for NOVA.

The router is intentionally credit-aware: it selects the cheapest configured
provider/model that can satisfy a shot's requirements. Hosted execution is
kept behind a provider boundary so NOVA's brain never depends on one vendor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.agents.shot_performance import performance_prompt
from app.agents.reference_preflight import assert_reference_preflight, required_reference_props


@dataclass(frozen=True)
class PerformanceRequirement:
    """Capabilities required by one shot."""

    duration: float
    acting: bool = False
    facial_expression: bool = False
    hands: bool = False
    camera_motion: bool = False
    lip_sync: bool = False
    continuity_critical: bool = False


@dataclass(frozen=True)
class ModelOption:
    """A candidate model with normalized capability/cost metadata."""

    model: str
    credits_per_second: float
    capabilities: frozenset[str]
    minimum_seconds: float = 0.0

    def supports(self, req: PerformanceRequirement) -> bool:
        required = set()
        if req.acting:
            required.add("acting")
        if req.facial_expression:
            required.add("facial_expression")
        if req.hands:
            required.add("hands")
        if req.camera_motion:
            required.add("camera_motion")
        if req.lip_sync:
            required.add("lip_sync")
        if req.continuity_critical:
            required.add("continuity")
        return required.issubset(self.capabilities) and req.duration >= self.minimum_seconds

    @property
    def estimated_credits(self) -> float:
        return self.credits_per_second * max(self.duration_for_estimate, self.minimum_seconds)

    duration_for_estimate: float = 0.0


# Current Runway options. Keep this table deliberately conservative: a model is
# only selected automatically when we have an explicit capability contract.
# Prices are used for routing, not billing; the provider remains authoritative.
RUNWAY_MODELS = (
    ModelOption("gen-4-turbo", 5.0, frozenset({"acting", "facial_expression", "hands", "camera_motion", "continuity"})),
    ModelOption("kling-3-standard", 9.0, frozenset({"acting", "facial_expression", "hands", "camera_motion", "continuity"})),
    ModelOption("gen-4.5", 12.0, frozenset({"acting", "facial_expression", "hands", "camera_motion", "continuity"})),
    ModelOption("seedance-2-mini", 16.0, frozenset({"acting", "facial_expression", "hands", "camera_motion", "continuity"}), minimum_seconds=5.0),
    ModelOption("kling-o3-pro", 12.0, frozenset({"acting", "facial_expression", "hands", "camera_motion", "continuity"})),
    ModelOption("seedance-2", 36.0, frozenset({"acting", "facial_expression", "hands", "camera_motion", "continuity"}), minimum_seconds=5.0),
)


class PerformanceProvider(Protocol):
    name: str

    def render(
        self,
        image: Path,
        output: Path,
        duration: float,
        shot_index: int,
        performance: dict | None = None,
        model: str | None = None,
    ) -> "PerformanceResult":
        ...


@dataclass(frozen=True)
class PerformanceResult:
    output: Path
    provider: str
    model: str
    estimated_credits: float


def choose_cheapest_model(requirement: PerformanceRequirement, models=RUNWAY_MODELS) -> ModelOption:
    """Return the cheapest capable model; never silently choose an expensive fallback."""
    candidates = []
    for option in models:
        if requirement.duration < option.minimum_seconds:
            continue
        candidates.append(
            ModelOption(
                option.model,
                option.credits_per_second,
                option.capabilities,
                option.minimum_seconds,
                requirement.duration,
            )
        )
    capable = [m for m in candidates if m.supports(requirement)]
    if not capable:
        raise ValueError("No configured performance model satisfies the shot requirements")
    return min(capable, key=lambda m: (m.estimated_credits, m.credits_per_second, m.model))


def choose_cheapest_available_model(
    requirement: PerformanceRequirement,
    available_models: set[str] | list[str] | tuple[str, ...],
    models=RUNWAY_MODELS,
) -> ModelOption:
    """Choose the cheapest capable model from models actually exposed by a worker."""
    available = {str(name) for name in available_models}
    configured = tuple(option for option in models if option.model in available)
    if not configured:
        raise ValueError("No configured performance models are available in the connected workspace")
    return choose_cheapest_model(requirement, configured)


RUNWAY_REPAIR_RULES: dict[str, str] = {
    "insufficient_character_motion": "Increase readable character performance, not camera movement. Use a clear anticipation before the action, a visible action beat, then recoil and settle.",
    "too_many_near_duplicate_frames": "Avoid holding nearly identical poses. Add small connected head, torso and gaze changes while preserving the character's silhouette.",
    "possible_pose_popping": "Reduce abrupt pose changes. Blend between poses with a short anticipation, contact/impact and recoil instead of snapping.",
    "excessive_transition_spike": "Keep the shot internally continuous. Do not create a sudden camera or pose jump.",
    "face_deformation": "Lock facial structure to the reference. Keep eyes, muzzle, ears and facial proportions stable; use only controlled eyelid, brow and mouth changes.",
    "hand_deformation": "Keep hands/paws simple and coherent. Do not animate fingers or invent extra digits; preserve the reference silhouette.",
    "limb_deformation": "Keep limb anatomy stable. Use restrained joint movement and never stretch, duplicate or detach limbs.",
    "identity_drift": "Preserve the exact character identity, proportions, costume and colors from the start frame throughout the shot.",
    "gaze_failure": "Make the gaze target explicit and visible in the composition. Move the eyes first, then let the head and body follow subtly.",
    "no_anticipation": "Add a readable preparatory motion before the main action; do not start the action from a static pose with an instantaneous jump.",
    "no_recoil": "Add a small overshoot/recoil immediately after the main action, then settle.",
    "secondary_motion_missing": "Add restrained secondary motion such as scarf/fur/body follow-through without changing the character design.",
    "camera_drift": "Lock the camera. Remove unintended reframing, zoom, pan or background drift.",
    "background_drift": "Preserve the exact environment and object placement from the reference frame.",
    "timing_failure": "Follow the authored performance timeline exactly and keep each beat readable.",
}


def build_runway_repair_request(
    request: dict[str, Any],
    qa_result: dict[str, Any],
) -> dict[str, Any]:
    """Create a zero-execution targeted Runway repair request from QA diagnostics."""
    reasons = list(dict.fromkeys(str(r) for r in qa_result.get("reasons", []) if r))
    supplied_plan = qa_result.get("plan") if isinstance(qa_result.get("plan"), dict) else {}
    instructions = list(supplied_plan.get("instructions", [])) or [RUNWAY_REPAIR_RULES[r] for r in reasons if r in RUNWAY_REPAIR_RULES]
    target_shots = sorted({int(i) for i in qa_result.get("repair_shots", []) if int(i) >= 0})
    base_prompt = str(request.get("prompt", "")).strip()
    repair_prompt = base_prompt
    if instructions:
        repair_prompt += " Targeted repair: " + " ".join(instructions)
    repair_prompt += (
        " This is a repair render, not a redesign. Preserve the reference character, "
        "scene, framing and continuity. Change only the diagnosed failure."
    )
    return {
        **request,
        "prompt": repair_prompt,
        "repair": {
            "required": True,
            "target_shots": target_shots,
            "reasons": reasons,
            "instructions": instructions,
            "domains": supplied_plan.get("domains", {}),
            "confidence": supplied_plan.get("confidence", 0.0),
            "scope": supplied_plan.get("scope", "targeted-shots" if target_shots else "all-shots"),
            "strategy": "targeted-rerender",
            "max_attempts": 1,
        },
    }


class RunwayPerformanceProvider:
    """Runway bridge boundary.

    NOVA builds a complete, deterministic shot request here. The connected
    Runway worker may execute that request, but model selection and performance
    intent stay inside NOVA so a vendor cannot silently change the director's
    decision or bypass the cheapest-first policy.
    """

    name = "runway"

    def plan(self, requirement: PerformanceRequirement) -> ModelOption:
        return choose_cheapest_model(requirement)

    def build_request(
        self,
        image: Path,
        duration: float,
        shot_index: int,
        performance: dict[str, Any] | None = None,
        aspect_ratio: str = "9:16",
        resolution: str = "720p",
    ) -> dict[str, Any]:
        performance = performance or {}
        requirement = requirement_from_performance(duration, performance)
        selected = choose_cheapest_model(requirement)
        spec = performance.get("shot_spec", performance)
        prompt = performance_prompt(spec) if "phases" in spec else str(performance.get("prompt", "Animate the existing character with controlled, coherent motion."))

        action = str(spec.get("source_action", performance.get("action", "")))
        reference_manifest = performance.get("reference_manifest")
        character = str(
            performance.get(
                "character",
                performance.get("character_name", reference_manifest.get("character", "") if isinstance(reference_manifest, dict) else ""),
            )
        )
        required_props = list(required_reference_props(action))
        reference_preflight = {
            "required_props": required_props,
            "verified": False,
        }
        if required_props:
            reference_preflight = assert_reference_preflight(
                action,
                character,
                reference_manifest,
                expected_scene=str(
                    performance.get(
                        "expected_scene",
                        spec.get("expected_scene", ""),
                    )
                    or ""
                ) or None,
            )
        elif reference_manifest:
            reference_preflight = {
                "passed": True,
                "required_props": [],
                "reference_asset_id": reference_manifest.get("asset_id"),
                "character": character,
            }

        return {
            "provider": self.name,
            "model": selected.model,
            "estimated_credits": selected.estimated_credits,
            "shot_index": int(shot_index),
            "image": str(image),
            "duration": float(duration),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "prompt": prompt,
            "reference_preflight": reference_preflight,
            "requirements": {
                "acting": requirement.acting,
                "facial_expression": requirement.facial_expression,
                "hands": requirement.hands,
                "camera_motion": requirement.camera_motion,
                "lip_sync": requirement.lip_sync,
                "continuity_critical": requirement.continuity_critical,
            },
        }

    def build_storyboard_plan(
        self,
        shots: list[dict[str, Any]],
        images: list[Path],
        aspect_ratio: str = "9:16",
        resolution: str = "720p",
    ) -> dict[str, Any]:
        """Compile storyboard shots into executable Runway requests without generating.

        This is the zero-credit handoff between NOVA's Director and the external
        Runway worker. Every shot gets its own model choice, prompt and estimate.
        """
        if len(shots) != len(images):
            raise ValueError("shots and images must have the same length")
        requests = [
            self.build_request(
                image=images[i],
                duration=float(shot["duration"]),
                shot_index=int(shot["index"]),
                performance=shot.get("performance", {}),
                aspect_ratio=aspect_ratio,
                resolution=resolution,
            )
            for i, shot in enumerate(shots)
        ]
        return {
            "provider": self.name,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "shots": requests,
            "estimated_total_credits": round(sum(r["estimated_credits"] for r in requests), 2),
        }

    def render(self, image: Path, output: Path, duration: float, shot_index: int, performance: dict | None = None, model: str | None = None) -> PerformanceResult:
        requirement = requirement_from_performance(duration, performance or {})
        selected = choose_cheapest_model(requirement)
        if model and model != selected.model:
            raise ValueError(f"Refusing to override cheapest viable model {selected.model} with {model}")
        raise RuntimeError(
            "Runway execution is intentionally external to the local repo. "
            f"NOVA selected {selected.model} at an estimated {selected.estimated_credits:.0f} credits for this shot."
        )


def requirement_from_performance(duration: float, performance: dict) -> PerformanceRequirement:
    """Translate Director performance intent into routing requirements.

    Storyboard shots carry the richer temporal intent under ``shot_spec``. When
    present, that spec is the source of truth; legacy top-level fields remain
    supported for older callers.
    """
    spec = performance.get("shot_spec") or {}
    hands_text = str(spec.get("hands", performance.get("hands", performance.get("gesture", ""))))
    camera_text = str(spec.get("camera", performance.get("camera", "")))
    return PerformanceRequirement(
        duration=float(spec.get("duration", duration)),
        acting=bool(spec.get("action") or performance.get("acting") or performance.get("gesture") or performance.get("expression")),
        facial_expression=bool(spec.get("facial_animation") or performance.get("facial_expression") or performance.get("expression")),
        hands=bool(hands_text),
        camera_motion=bool(camera_text and camera_text.lower() not in {"stable", "static", "locked"}),
        # Lip-sync is explicit; dialogue is handled by the separate audio stage.
        lip_sync=bool(performance.get("lip_sync", False)),
        continuity_critical=bool(spec.get("continuity") or performance.get("continuity_critical")),
    )
