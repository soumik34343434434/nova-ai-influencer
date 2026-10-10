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
    """Validate identity locks and signature props across assembled shot plans.

    A close-up may crop a signature prop out of frame; every other framing must
    keep it visible.
    """
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
        if entry and "signature_props" in shot and shot.get("framing") != "close_up":
            missing = sorted(set(entry["signature_props"]) - set(shot["signature_props"]))
            if missing:
                errors.append(f"continuity_violation:shot_{index}:missing_signature_props:{','.join(missing)}")
    return errors


# Hue ranges (degrees) for palette names used by the bible.
PALETTE_HUES: dict[str, tuple[float, float]] = {
    "blue": (185.0, 235.0),
    "sky_blue": (185.0, 235.0),
    "orange": (10.0, 40.0),
    "teal": (160.0, 190.0),
    "yellow": (42.0, 65.0),
}


def palette_shares(image: Any, bbox: tuple[int, int, int, int] | None = None) -> dict[str, float]:
    """Share of saturated pixels in ``bbox`` whose hue falls in each palette colour."""
    import numpy as np

    region = image.crop(bbox) if bbox else image
    hsv = np.asarray(region.convert("RGB").convert("HSV"), dtype=np.float32).reshape(-1, 3)
    hue = hsv[:, 0] * (360.0 / 255.0)
    saturated = (hsv[:, 1] / 255.0 > 0.35) & (hsv[:, 2] / 255.0 > 0.25)
    if not saturated.any():
        return {name: 0.0 for name in PALETTE_HUES}
    hue = hue[saturated]
    return {name: round(float(((hue >= lo) & (hue <= hi)).mean()), 4) for name, (lo, hi) in PALETTE_HUES.items()}


def validate_asset_identity(character: str, image: Any, bbox: tuple[int, int, int, int] | None = None,
                            bible: dict[str, Any] | None = None) -> list[str]:
    """Pixel-level identity check: the character region must match its bible palette.

    Catches a wrong or swapped action image (for example MOMO art in a BUBU shot)
    before any render time is spent on it.
    """
    bible = bible or load_bible()
    entry = bible.get("characters", {}).get(str(character).upper())
    if entry is None:
        return [f"unknown_character:{character}"]
    rules = entry.get("pixel_identity")
    if not rules:
        return []
    shares = palette_shares(image, bbox)
    errors = []
    for colour, minimum in rules.get("min_share", {}).items():
        if shares.get(colour, 0.0) < float(minimum):
            errors.append(f"identity_violation:{character}:palette_{colour}_{shares.get(colour, 0.0):.2f}<{float(minimum):.2f}")
    for colour, maximum in rules.get("max_share", {}).items():
        if shares.get(colour, 0.0) > float(maximum):
            errors.append(f"identity_violation:{character}:palette_{colour}_{shares.get(colour, 0.0):.2f}>{float(maximum):.2f}")
    return errors
