"""Structured character and world rules with lightweight plan validators."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BIBLE_PATH = ROOT / "config" / "character_world_bible.json"
CHARACTER_FIELDS = ("identity", "proportions", "palette", "signature_props", "allowed_transformations", "forbidden_transformations")


def load_bible(path: Path = BIBLE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_bible(bible: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("style", "characters", "environments", "lighting_presets", "reusable_action_props", "continuity_constraints"):
        if field not in bible:
            errors.append(f"missing_field:{field}")
    for name, entry in bible.get("characters", {}).items():
        for field in CHARACTER_FIELDS:
            if field not in entry:
                errors.append(f"missing_field:characters.{name}.{field}")
    return errors


def validate_character_plan(character: str, plan: dict[str, Any], bible: dict[str, Any] | None = None) -> list[str]:
    """Check requested transformations and optional locked identity values."""
    bible = bible or load_bible()
    entry = bible.get("characters", {}).get(str(character).upper())
    if entry is None:
        return [f"unknown_character:{character}"]
    errors: list[str] = []
    for field in ("identity", "proportions", "palette", "signature_props"):
        if field in plan and plan[field] != entry[field]:
            errors.append(f"identity_violation:{character}:{field}")
    for transformation in plan.get("transformations", []):
        if transformation in entry["forbidden_transformations"]:
            errors.append(f"identity_violation:{character}:{transformation}")
        elif transformation not in entry["allowed_transformations"]:
            errors.append(f"unsupported_transformation:{character}:{transformation}")
    return errors


def validate_storyboard_continuity(shots: list[dict[str, Any]], bible: dict[str, Any] | None = None) -> list[str]:
    """Validate identity locks and signature props across assembled shot plans."""
    bible = bible or load_bible()
    errors: list[str] = []
    seen_character: str | None = None
    for index, shot in enumerate(shots):
        character = str(shot.get("character", "")).upper()
        if not character:
            continue  # Legacy/NOVA plans remain valid without cartoon cast metadata.
        if seen_character is not None and character != seen_character:
            errors.append(f"continuity_violation:shot_{index}:character_changed")
        seen_character = character
        plan = shot.get("character_plan", {})
        errors.extend(f"shot_{index}:{error}" for error in validate_character_plan(character, plan, bible))
        entry = bible.get("characters", {}).get(character)
        if entry and "signature_props" in shot:
            missing = sorted(set(entry["signature_props"]) - set(shot["signature_props"]))
            if missing:
                errors.append(f"continuity_violation:shot_{index}:missing_signature_props:{','.join(missing)}")
    return errors
