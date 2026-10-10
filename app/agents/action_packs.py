"""Registry of renderable character actions (config/cartoon_action_packs.json).

The registry is the single source of truth for which actions a character can
perform on the local renderer, which story role each action serves, and where
the character sits in each action image. Idea generation, asset lookup and the
rig all read it, so an idea can only request actions that can actually render.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "config" / "cartoon_action_packs.json"
ACTION_DIR = ROOT / "assets" / "cartoon" / "actions"
ROLE_ORDER = ("hook", "escalation", "reaction", "punchline")


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    return _load(str(path))


def character_actions(character: str, registry: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    registry = registry or load_registry()
    return dict(registry.get("characters", {}).get(str(character).upper(), {}).get("actions", {}))


def action_spec(character: str, action: str, registry: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return character_actions(character, registry).get(str(action))


def action_file(character: str, action: str, registry: dict[str, Any] | None = None) -> Path | None:
    """Return the action image path when it is registered and present on disk."""
    spec = action_spec(character, action, registry)
    if not spec or not spec.get("file"):
        return None
    path = ACTION_DIR / str(spec["file"])
    return path if path.exists() else None


def role_actions(character: str, registry: dict[str, Any] | None = None) -> dict[str, str]:
    """Map each story role to the first renderable action that serves it.

    Returns an empty mapping when any role in ROLE_ORDER is uncovered, because
    an episode with a missing beat image cannot be rendered locally.
    """
    chosen: dict[str, str] = {}
    actions = character_actions(character, registry)
    for role in ROLE_ORDER:
        for name, spec in actions.items():
            if role in spec.get("roles", ()) and name not in chosen.values() and action_file(character, name, registry):
                chosen[role] = name
                break
        if role not in chosen:
            return {}
    return chosen


def renderable_characters(characters: tuple[str, ...] | list[str], registry: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Characters with a complete, on-disk action pack for every story role."""
    return tuple(c for c in characters if role_actions(c, registry))


def regions_for_file(filename: str, registry: dict[str, Any] | None = None) -> dict[str, tuple[float, float, float, float]] | None:
    """Rig regions (cover-frame fractions) recorded for a generated action image."""
    registry = registry or load_registry()
    for entry in registry.get("characters", {}).values():
        for spec in entry.get("actions", {}).values():
            if spec.get("file") == filename and isinstance(spec.get("regions"), dict):
                regions = spec["regions"]
                try:
                    return {key: tuple(float(v) for v in regions[key]) for key in ("body", "head", "face")}
                except (KeyError, TypeError, ValueError):
                    return None
    return None


def character_for_file(filename: str, registry: dict[str, Any] | None = None) -> str | None:
    registry = registry or load_registry()
    for character, entry in registry.get("characters", {}).items():
        if any(spec.get("file") == filename for spec in entry.get("actions", {}).values()):
            return character
    return None
