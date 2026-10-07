"""Reference/action preflight for NOVA shot execution.

The preflight is deterministic and provider-neutral. It does not claim that a
model can see an object that is absent from the supplied reference; it requires
the shot's declared action targets to be represented by an explicit reference
manifest before an external animation worker is allowed to execute.
"""
from __future__ import annotations

from typing import Any


ACTION_REQUIRED_PROPS: dict[str, tuple[str, ...]] = {
    "checks_phone": ("phone",),
    "sees_calendar": ("calendar",),
    "reaches_for_coffee": ("coffee",),
    "glances_at_many_tabs": ("laptop",),
}


def required_reference_props(action: str) -> tuple[str, ...]:
    return ACTION_REQUIRED_PROPS.get(str(action), ())


def build_reference_manifest(
    character: str,
    scene: str,
    props: list[str] | tuple[str, ...],
    asset_id: str | None = None,
) -> dict[str, Any]:
    return {
        "character": str(character),
        "scene": str(scene),
        "props": sorted({str(value).strip().lower() for value in props if str(value).strip()}),
        "asset_id": asset_id,
        "verified": True,
    }


def validate_reference_preflight(
    action: str,
    character: str,
    reference_manifest: dict[str, Any] | None,
    expected_scene: str | None = None,
) -> dict[str, Any]:
    required = required_reference_props(action)
    errors: list[str] = []
    warnings: list[str] = []

    if not reference_manifest:
        warnings.append("reference_manifest_missing")
        return {
            "passed": not required,
            "errors": ["reference_manifest_required_for_action_target"] if required else [],
            "warnings": warnings,
            "required_props": list(required),
            "character": character,
        }

    if reference_manifest.get("character") != character:
        errors.append("reference_character_mismatch")
    if expected_scene and reference_manifest.get("scene") != expected_scene:
        errors.append("reference_scene_mismatch")

    props = {
        str(value).strip().lower()
        for value in reference_manifest.get("props", [])
        if str(value).strip()
    }
    missing = [prop for prop in required if prop not in props]
    for prop in missing:
        errors.append(f"reference_missing_required_prop_{prop}")

    if reference_manifest.get("verified") is not True:
        errors.append("reference_manifest_not_verified")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "required_props": list(required),
        "missing_props": missing,
        "character": character,
        "reference_asset_id": reference_manifest.get("asset_id"),
    }


def assert_reference_preflight(
    action: str,
    character: str,
    reference_manifest: dict[str, Any] | None,
    expected_scene: str | None = None,
) -> dict[str, Any]:
    result = validate_reference_preflight(
        action,
        character,
        reference_manifest,
        expected_scene=expected_scene,
    )
    if not result["passed"]:
        raise ValueError(
            "Reference/action preflight failed: " + ", ".join(result["errors"])
        )
    return result
