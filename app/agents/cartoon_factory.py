"""Scalable episode planning for the original cartoon universe."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
CHARACTER_DIR = ROOT / "assets" / "cartoon" / "characters"
ACTION_DIR = ROOT / "assets" / "cartoon" / "actions"


@dataclass(frozen=True)
class CharacterProfile:
    name: str
    asset: str
    style: str
    comedy_traits: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeSpec:
    slug: str
    title: str
    hook: str
    character: str
    beats: tuple[dict, ...]
    caption: str


CAST = {
    "PIP": CharacterProfile(
        "PIP", "pip.png",
        "tiny sky-blue mischievous robot with oversized expressive eyes and orange backpack",
        ("curious", "overconfident", "kind"),
    ),
    "MOMO": CharacterProfile(
        "MOMO", "momo.png",
        "sleepy orange cartoon cat with oversized expressive eyes and tiny teal scarf",
        ("dramatic", "lazy", "clever"),
    ),
    "BUBU": CharacterProfile(
        "BUBU", "bubu.png",
        "round blue cartoon bird with oversized expressive eyes and tiny yellow backpack",
        ("hyper", "optimistic", "chaotic"),
    ),
}


PREMISES = (
    {
        "slug": "momo_47_tasks",
        "title": "MOMO vs 47 New Tasks",
        "hook": "MOMO FINALLY FINISHED. THEN THE PHONE BUZZED.",
        "character": "MOMO",
        "beats": (
            ("setup", "finish_work", "Done. Finally.", "none"),
            ("escalation", "checks_phone", "My phone is buzzing... wait, backlog cleared?", "notification"),
            ("reaction", "checks_phone", "I reread the phone. New tasks: forty-seven?!", "notification"),
            ("punchline", "deadpan", "I should have left the phone on silent.", "punchline"),
        ),
    },
    {
        "slug": "smart_kitchen",
        "title": "PIP vs the Smart Kitchen",
        "hook": "PIP asked a smart kitchen for ONE sandwich. It misunderstood.",
        "character": "PIP",
        "beats": (
            ("setup", "press_button", "One sandwich, please.", "button"),
            ("escalation", "shock", "TWELVE?!", "chaos"),
            ("reaction", "overwhelmed", "Why did I press that?", "overwhelmed"),
            ("punchline", "deadpan", "Apparently, the kitchen misunderstood one.", "punchline"),
        ),
    },
    {
        "slug": "homework_robot",
        "title": "MOMO vs the Homework Robot",
        "hook": "MOMO asked a homework robot for ONE answer. Big mistake.",
        "character": "MOMO",
        "beats": (
            ("setup", "give_homework", "Just one answer.", "button"),
            ("escalation", "shock", "WHY IS IT STILL PRINTING?!", "chaos"),
            ("reaction", "overwhelmed", "This is literally a library.", "overwhelmed"),
            ("punchline", "deadpan", "I asked for help—not a textbook.", "punchline"),
        ),
    },
    {
        "slug": "bedtime_alarm",
        "title": "BUBU vs the Bedtime Alarm",
        "hook": "BUBU set ONE bedtime alarm. It had other plans.",
        "character": "BUBU",
        "beats": (
            ("setup", "tap_alarm", "Okay. Bedtime.", "button"),
            ("escalation", "shock", "WHY IS IT SO LOUD?!", "chaos"),
            ("reaction", "overwhelmed", "I was already sleepy.", "overwhelmed"),
            ("punchline", "deadpan", "Tomorrow I am sleeping manually.", "punchline"),
        ),
    },
    {
        "slug": "magic_fridge",
        "title": "PIP and the Magic Fridge",
        "hook": "PIP opened the fridge for ONE snack. That was optimistic.",
        "character": "PIP",
        "beats": (
            ("setup", "open_fridge", "Just one snack.", "button"),
            ("escalation", "shock", "Why are there twenty?", "chaos"),
            ("reaction", "overwhelmed", "I only have two hands.", "overwhelmed"),
            ("punchline", "deadpan", "The fridge has no portion control.", "punchline"),
        ),
    },
    {
        "slug": "automatic_shoes",
        "title": "BUBU and the Automatic Shoes",
        "hook": "BUBU bought shoes that walk by themselves. Bad idea.",
        "character": "BUBU",
        "beats": (
            ("setup", "tap_shoes", "Okay, tiny walk.", "button"),
            ("escalation", "shock", "WAIT. COME BACK!", "chaos"),
            ("reaction", "overwhelmed", "The shoes have escaped.", "overwhelmed"),
            ("punchline", "deadpan", "At least they know the route.", "punchline"),
        ),
    },
)


def _seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def list_episode_specs(limit: int = 5) -> list[EpisodeSpec]:
    ordered = sorted(PREMISES, key=lambda p: _seed(p["slug"]))
    return [
        EpisodeSpec(
            slug=p["slug"],
            title=p["title"],
            hook=p["hook"],
            character=p["character"],
            beats=tuple(
                {
                    "beat": name,
                    "asset_action": asset_action,
                    "voice": voice,
                    "sfx": sfx,
                    "duration": 2.3 if i < 3 else 2.8,
                }
                for i, (name, asset_action, voice, sfx) in enumerate(p["beats"])
            ),
            caption={
                "smart_kitchen": "Asked for ONE sandwich. Got a catering order. 🤖🥪",
                "homework_robot": "Asked for ONE answer. Got a whole textbook. 🤖📚",
                "bedtime_alarm": "BUBU wanted bedtime. The alarm wanted a full production. ⏰",
                "magic_fridge": "One snack became twenty. The fridge won. 🥪",
                "automatic_shoes": "Bought automatic shoes. Forgot to ask where they were going. 👟",
            }.get(p["slug"], f"{p['title']} — funny original cartoon short."),
        )
        for p in ordered[: max(1, min(limit, len(PREMISES)))]
    ]


def asset_path(character: str) -> Path:
    profile = CAST[character]
    path = CHARACTER_DIR / profile.asset
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical asset for {character}: {path}")
    return path


def action_asset(character: str, asset_action: str) -> Path | None:
    names = {
        "PIP": {
            "press_button": "pip_press_button.png",
            "shock": "pip_shock.png",
            "overwhelmed": "pip_overwhelmed.png",
            "deadpan": "pip_deadpan.png",
        }
    }
    filename = names.get(character, {}).get(asset_action)
    if not filename:
        # Blender can animate a clean compiled master directly; do not require
        # four pre-rendered flattened action images for MOMO/BUBU. The legacy
        # Pillow path stays unchanged unless Blender is explicitly selected.
        if os.environ.get("NOVA_CARTOON_ANIMATION_BACKEND", "pillow").strip().lower() == "blender" and character in {"MOMO", "BUBU"}:
            return asset_path(character)
        return None
    path = ACTION_DIR / filename
    return path if path.exists() else None


def renderable_actions(character: str, actions: Iterable[str]) -> tuple[str, ...]:
    """Return the subset of ``actions`` the active backend can render for ``character``."""
    return tuple(action for action in actions if action_asset(character, action) is not None)


def coverage(spec: EpisodeSpec) -> float:
    available = sum(1 for beat in spec.beats if action_asset(spec.character, beat["asset_action"]))
    return available / max(1, len(spec.beats))


def quality_gate(spec: EpisodeSpec) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    duration = sum(b["duration"] for b in spec.beats)
    if not 8.0 <= duration <= 15.0:
        reasons.append("duration_outside_short_form_range")
    if len(spec.beats) < 4:
        reasons.append("not_enough_comedy_beats")
    if len(spec.hook.split()) > 12:
        reasons.append("hook_too_long")
    if coverage(spec) < 1.0:
        reasons.append("missing_action_asset_coverage")
    if spec.character not in CAST:
        reasons.append("unknown_character")
    return not reasons, reasons
