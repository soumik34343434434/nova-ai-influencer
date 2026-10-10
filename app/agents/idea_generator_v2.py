"""High-volume, diversity-aware idea generation for NOVA.

The generator creates many cheap structured concepts first, then lets the jury
decide what deserves expensive script/animation work. It intentionally does not
copy trend wording; trends become constraints (format, hook, emotion, pacing).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import re
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class IdeaCandidateV2:
    idea_id: str
    title: str
    hook: str
    character: str
    format_name: str
    trend_id: str
    emotion: str
    angle: str
    duration_target: float
    beats: tuple[dict[str, Any], ...]
    production_cost: float
    source_evidence: tuple[str, ...] = ()


FORMAT_RECIPES: dict[str, dict[str, Any]] = {
    "dance": {
        "duration": 7.0,
        "angles": ("adorable_commitment", "wrong_move", "sudden_mood_change", "beat_drop_surprise", "overconfident_start"),
    },
    "trend_audio": {
        "duration": 8.0,
        "angles": ("audio_mismatch", "cute_reveal", "deadpan_switch", "beat_misread", "unexpected_pause"),
    },
    "reaction": {
        "duration": 8.0,
        "angles": ("shock", "overconfidence", "deadpan", "tiny_detail_big_reaction", "wrong_notification"),
    },
    "pov": {
        "duration": 9.0,
        "angles": ("relatable_work", "social_anxiety", "small_problem_big_reaction", "monday_mood", "quiet_minute"),
    },
    "story_comedy": {
        "duration": 10.0,
        "angles": ("escalation", "misunderstanding", "reverse_expectation", "false_win", "loop_back"),
    },
    "challenge": {
        "duration": 10.0,
        "angles": ("attempts", "countdown", "cheat_then_fail", "almost_win", "rules_change"),
    },
    "transformation": {
        "duration": 8.0,
        "angles": ("upgrade_backfire", "before_after", "expectation_reality", "wrong_upgrade", "side_effect"),
    },
    "cute_animal": {
        "duration": 7.0,
        "angles": ("tiny_hero", "adorable_failure", "comfort", "protective", "tiny_solution_big_effect"),
    },
    "absurdist": {
        "duration": 8.0,
        "angles": ("logic_break", "escalation", "deadpan_acceptance", "reality_glitch", "nonsense_rule"),
    },
    "listicle": {
        "duration": 9.0,
        "angles": ("three_failures", "three_reactions", "countdown", "top_bad_choices", "one_more"),
    },
    "micro_short": {
        "duration": 7.0,
        "angles": ("single_gag", "instant_reversal", "reaction_loop", "setup_payoff_echo", "micro_twist"),
    },
}


# Acting actions understood by the story engine and the Blender rig. Each
# format stages its four beats with its own action plan; when a character
# cannot render a planned action, the role preferences pick a substitute.
IDEA_ACTIONS = ("finish_work", "checks_phone", "press_button", "shock", "overwhelmed", "deadpan")

# Every idea is four beats: a hook that states the goal, an escalation that
# reverses it, a reaction, and a character-owned punchline that calls back to
# the opening so the clip loops. The hook line is also the idea's hook, which
# the script engine speaks on the first beat. Placeholders: {c} character,
# {prop} the prop suggested by the trend.
BEAT_ROLES = ("hook", "escalation", "reaction", "punchline")
DEFAULT_BEATS = (
    ("{c} had one job. Then the {prop} started talking!", "notification"),
    ("Turns out the {prop} had other plans!", "impact"),
    ("Wait, what?", "reaction_hit"),
    ("{c}: of course. Same job, again.", "punchline"),
)
FORMAT_BEATS: dict[str, tuple[tuple[str, str], ...]] = {
    "dance": (
        ("{c} heard one song. Then the dance took over!", "beat"),
        ("Instead, {c} invents a brand new move!", "impact"),
        ("Wait, is that allowed?", "reaction_hit"),
        ("{c}: still counts as dancing.", "punchline"),
    ),
    "trend_audio": (
        ("{c} tried one trending sound. Then it backfired!", "beat"),
        ("Turns out the sound was an alarm!", "impact"),
        ("Why is it so loud?", "reaction_hit"),
        ("{c}: still dancing to it, apparently.", "punchline"),
    ),
    "reaction": (
        ("{c} got one normal notification. Big mistake!", "notification"),
        ("Instead of one message, there are forty-seven!", "impact"),
        ("Wait. Forty-seven?", "reaction_hit"),
        ("{c}: of course. Back to silent mode.", "punchline"),
    ),
    "pov": (
        ("POV: {c} finally gets one quiet minute!", "room"),
        ("Then the {prop} lights up instead!", "notification"),
        ("Why now?", "reaction_hit"),
        ("{c}: one quiet minute. Apparently too much to ask.", "punchline"),
    ),
    "challenge": (
        ("{c} said yes before hearing the rules. Big mistake!", "countdown"),
        ("Turns out the challenge has three more rounds!", "impact"),
        ("Three more?!", "reaction_hit"),
        ("{c}: still not reading the rules.", "punchline"),
    ),
    "transformation": (
        ("{c} upgraded one tiny thing. Big mistake!", "spark"),
        ("Instead of faster, everything is upside down!", "impact"),
        ("Oops. Wait, what?", "reaction_hit"),
        ("{c}: put it back the way it was.", "punchline"),
    ),
    "cute_animal": (
        ("{c} is trying to help. Nobody asked!", "cute"),
        ("Instead, the tiny plan becomes a giant problem!", "impact"),
        ("Oops.", "reaction_hit"),
        ("{c}: still counts as helping.", "punchline"),
    ),
    "listicle": (
        ("Three things {c} should not do. Number one?", "countdown"),
        ("Turns out number two is pressing every button!", "impact"),
        ("Why so many buttons?", "reaction_hit"),
        ("{c}: number one is still deleting the app.", "punchline"),
    ),
}

DEFAULT_ACTION_PLAN = ("finish_work", "checks_phone", "shock", "deadpan")
FORMAT_ACTION_PLANS: dict[str, tuple[str, str, str, str]] = {
    "dance": ("press_button", "shock", "overwhelmed", "deadpan"),
    "trend_audio": ("checks_phone", "press_button", "shock", "deadpan"),
    "reaction": ("checks_phone", "shock", "overwhelmed", "deadpan"),
    "pov": ("finish_work", "checks_phone", "overwhelmed", "deadpan"),
    "challenge": ("press_button", "checks_phone", "shock", "deadpan"),
    "transformation": ("press_button", "overwhelmed", "shock", "deadpan"),
    "cute_animal": ("finish_work", "overwhelmed", "shock", "deadpan"),
    "listicle": ("finish_work", "press_button", "shock", "deadpan"),
}

ROLE_ACTION_PREFERENCES: dict[str, tuple[str, ...]] = {
    "hook": ("finish_work", "press_button", "checks_phone", "shock"),
    "escalation": ("shock", "overwhelmed", "checks_phone", "press_button"),
    "reaction": ("overwhelmed", "shock", "deadpan"),
    "punchline": ("deadpan", "overwhelmed", "shock"),
}


def _stage_actions(beats: tuple[dict[str, Any], ...], plan: tuple[str, ...],
                   available: Iterable[str] | None) -> tuple[dict[str, Any], ...] | None:
    """Assign renderable actions to beats, or return None if the character cannot stage them."""
    if available is None:
        return tuple(dict(beat, asset_action=action) for beat, action in zip(beats, plan))
    renderable = set(available)
    used: list[str] = []
    staged = []
    for index, (beat, planned) in enumerate(zip(beats, plan)):
        options = [a for a in (planned, *ROLE_ACTION_PREFERENCES.get(beat["beat"], ()), *IDEA_ACTIONS) if a in renderable]
        # Keep actions the plan stages later free for those beats.
        reserved = set(plan[index + 1:])
        choice = next((a for a in options if a not in used and a not in reserved), None)
        if choice is None:
            choice = next((a for a in options if a not in used), None)
        if choice is None:
            choice = next((a for a in options if not used or a != used[-1]), None)
        if choice is None:
            return None
        used.append(choice)
        staged.append(dict(beat, asset_action=choice))
    # The jury and story gates need visibly changing action across the beats.
    return tuple(staged) if len(set(used)) >= 3 else None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:70]


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("||".join(parts).encode()).hexdigest(), 16)


def _emotion_phrase(emotion: str) -> str:
    return {
        "amusement": "funny",
        "surprise": "unexpected",
        "curiosity": "mysterious",
        "cute": "adorable",
        "frustration": "annoying",
        "awe": "impressive",
        "nostalgia": "nostalgic",
        "anxiety": "stressful",
        "satisfaction": "satisfying",
        "cringe": "awkward",
    }.get(emotion, "unexpected")


def _prop_from_signal(text: str) -> str:
    words = set(re.findall(r"[a-z0-9']+", text.lower()))
    for key, prop in (
        ("phone", "phone"), ("meeting", "laptop"), ("song", "speaker"),
        ("dance", "phone"), ("ai", "laptop"), ("cat", "toy"),
        ("food", "snack"), ("homework", "papers"), ("office", "laptop"),
        ("music", "speaker"), ("robot", "gadget"), ("meme", "phone"),
    ):
        if key in words:
            return prop
    return "mystery object"


def _anchor_from_signal(text: str) -> str:
    stop = {
        "pov", "the", "and", "with", "for", "viral", "trend", "trending",
        "reaction", "funny", "cute", "song", "video", "this", "that",
        "from", "new", "today", "shorts", "reels", "tiktok", "instagram",
    }
    words = [w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if w not in stop and len(w) > 2]
    return " ".join(words[:4]) or "a strange little trend"


def _angle_phrase(angle: str) -> str:
    return {
        "adorable_commitment": "commits too hard",
        "wrong_move": "does it completely wrong",
        "sudden_mood_change": "loses the mood instantly",
        "audio_mismatch": "the sound does not match reality",
        "cute_reveal": "the cute part has a catch",
        "deadpan_switch": "goes deadpan at the worst moment",
        "shock": "gets surprised by a tiny detail",
        "overconfidence": "is way too confident",
        "deadpan": "pretends nothing happened",
        "relatable_work": "tries to protect one quiet minute",
        "social_anxiety": "avoids one awkward moment",
        "small_problem_big_reaction": "turns a tiny problem into chaos",
        "escalation": "lets the small problem grow",
        "misunderstanding": "misreads the situation",
        "reverse_expectation": "gets the opposite result",
        "attempts": "keeps trying after the warning",
        "countdown": "runs out of patience",
        "cheat_then_fail": "takes a shortcut that backfires",
        "upgrade_backfire": "upgrades the wrong thing",
        "before_after": "gets a very different after",
        "expectation_reality": "gets the exact opposite",
        "tiny_hero": "tries to save the day",
        "adorable_failure": "fails in the cutest way",
        "comfort": "turns chaos into comfort",
        "logic_break": "accepts impossible logic",
        "deadpan_acceptance": "acts like this is normal",
        "three_failures": "collects three bad decisions",
        "three_reactions": "has three increasingly worse reactions",
        "single_gag": "has exactly one job",
        "instant_reversal": "wins for one second then loses",
        "reaction_loop": "reacts to the same problem twice",
    }.get(angle, angle.replace("_", " "))


def _make_beats(character: str, recipe: str, signal: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    prop = _prop_from_signal(str(signal.get("title", "")))
    return tuple(
        {"beat": role, "voice": voice.format(c=character, prop=prop), "sfx": sfx}
        for role, (voice, sfx) in zip(BEAT_ROLES, FORMAT_BEATS.get(recipe, DEFAULT_BEATS))
    )


def _is_near_duplicate(item: dict[str, Any], tokens: set[str],
                       chosen: list[dict[str, Any]], seen_tokens: list[set[str]]) -> bool:
    for old, old_tokens in zip(chosen, seen_tokens):
        union = tokens | old_tokens
        if union and len(tokens & old_tokens) / len(union) >= 0.72:
            # Keep diversity even when wording is similar.
            if item["format_name"] == old["format_name"] and item["character"] == old["character"]:
                return True
    return False


def generate_ideas(trends: Iterable[dict[str, Any]], formats: Iterable[dict[str, Any]] | None = None,
                   count: int = 60, characters: tuple[str, ...] = ("MOMO", "PIP", "BUBU"),
                   renderable_actions: Mapping[str, Iterable[str]] | None = None) -> list[dict[str, Any]]:
    """Generate a diverse idea pool that spans every supplied trend.

    ``renderable_actions`` maps each character to the actions the active
    renderer can produce. When given, ideas use only those actions and
    characters that cannot stage a format are skipped; when omitted, every
    character is assumed to support all of ``IDEA_ACTIONS``.
    """
    trend_rows = list(trends)
    format_rows = list(formats or [])
    if not trend_rows:
        trend_rows = [{"trend_id": "evergreen", "title": "evergreen cartoon problem", "emotions": ("amusement",), "formats": ("micro_short",)}]

    groups: list[list[dict[str, Any]]] = []
    for trend in trend_rows[:20]:
        trend_formats = list(trend.get("formats") or ())
        if format_rows:
            trend_formats.extend(str(x.get("name")) for x in format_rows[:8] if x.get("name"))
        trend_formats.extend(["micro_short", "reaction", "story_comedy", "absurdist", "cute_animal"])
        unique_formats = list(dict.fromkeys(x for x in trend_formats if x in FORMAT_RECIPES))
        if not unique_formats:
            unique_formats = ["micro_short"]

        emotions = list(trend.get("emotions") or ()) or ["amusement", "surprise", "cute"]
        anchor = _anchor_from_signal(str(trend.get("title", "")))
        group: list[dict[str, Any]] = []
        for fmt in unique_formats[:8]:
            recipe = FORMAT_RECIPES[fmt]
            plan = FORMAT_ACTION_PLANS.get(fmt, DEFAULT_ACTION_PLAN)
            for character in characters:
                available = None if renderable_actions is None else tuple(renderable_actions.get(character, ()))
                for angle in recipe["angles"]:
                    emotion = emotions[_seed(str(trend.get("trend_id")), fmt, character, angle) % len(emotions)]
                    beats = _stage_actions(_make_beats(character, fmt, trend), plan, available)
                    if beats is None:
                        continue
                    title = f"{character} + {anchor}: {fmt.replace('_', ' ')} — {_angle_phrase(angle)}"
                    hook = beats[0]["voice"]
                    idea_id = f"idea-{_slug(title)}-{abs(_seed(title, str(trend.get('trend_id')))) % 100000:05d}"
                    cost = 0.25 + 0.08 * len(beats) + 0.06 * (1 if fmt in {"dance", "trend_audio"} else 0)
                    group.append({
                        "idea_id": idea_id,
                        "title": title,
                        "hook": hook,
                        "character": character,
                        "format_name": fmt,
                        "trend_id": str(trend.get("trend_id", "")),
                        "emotion": emotion,
                        "angle": angle,
                        "duration_target": float(recipe["duration"]),
                        "beats": list(beats),
                        "production_cost": round(min(cost, 1.0), 3),
                        "source_evidence": list(trend.get("evidence", ())),
                        "trend_score": float(trend.get("opportunity_score", trend.get("score", 0.0)) or 0.0),
                        "saturation_score": float(trend.get("saturation_score", 0.0) or 0.0),
                        "velocity_score": float(trend.get("velocity_score", 0.0) or 0.0),
                    })
        if group:
            group.sort(key=lambda x: (x["production_cost"], _seed(x["idea_id"])))
            groups.append(group)

    # Take ideas from each trend in turn, strongest trend first, so one trend's
    # format x character x angle grid cannot fill the whole pool. Near-duplicates
    # are suppressed: we prefer different formats, characters and emotional angles
    # before keeping another semantically similar idea.
    groups.sort(key=lambda g: -g[0]["trend_score"])
    queues = [iter(group) for group in groups]
    chosen: list[dict[str, Any]] = []
    seen_tokens: list[set[str]] = []
    while queues and len(chosen) < count:
        active = []
        for queue in queues:
            for item in queue:
                toks = set(re.findall(r"[a-z0-9]+", (item["title"] + " " + item["hook"]).lower()))
                if not _is_near_duplicate(item, toks, chosen, seen_tokens):
                    chosen.append(item)
                    seen_tokens.append(toks)
                    active.append(queue)
                    break
            if len(chosen) >= count:
                break
        queues = active
    return chosen


def idea_to_dict(item: IdeaCandidateV2 | dict[str, Any]) -> dict[str, Any]:
    return asdict(item) if isinstance(item, IdeaCandidateV2) else dict(item)
