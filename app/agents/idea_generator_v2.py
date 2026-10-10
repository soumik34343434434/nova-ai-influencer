"""High-volume, diversity-aware idea generation for NOVA.

The generator creates many cheap structured concepts first, then lets the jury
decide what deserves expensive script/animation work. It intentionally does not
copy trend wording; trends become constraints (format, hook, emotion, pacing).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import re
from typing import Any, Iterable

from app.agents.action_packs import action_spec, renderable_characters, role_actions


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


CHARACTER_PERSONAS = {
    "MOMO": ("sleepy", "sarcastic", "cute"),
    "PIP": ("mischievous", "curious", "chaotic"),
    "BUBU": ("innocent", "eager", "overconfident"),
}

# Every idea is four beats: hook -> escalation -> reaction (the twist) ->
# punchline. The beat images come from the character's action pack
# (config/cartoon_action_packs.json), so an idea can only ask for actions that
# can actually be rendered. Lines use {C} (character), {P} (prop from the
# trend) and {N} (a concrete count that makes the escalation visibly worse).
#
# Each hook carries a hook-style tag so the learning loop can measure which
# opening styles work, independent of the trend that inspired the idea.
FORMAT_RECIPES: dict[str, dict[str, Any]] = {
    "dance": {
        "duration": 11.0,
        "angles": ("adorable_commitment", "wrong_move", "sudden_mood_change", "beat_drop_surprise", "overconfident_start"),
        "hooks": (("{C} learned one dance move. Then the beat dropped!", "twist_tease"),
                  ("Nobody told {C} this was a dance battle?!", "curiosity")),
        "goals": ("{C} just wants to try one tiny dance.", "{C} plans one simple move for the {P}."),
        "escalations": ("Then the song speeds up. {N} moves at once!", "Then {N} dancers join in. Zero warning!"),
        "twists": ("Wait, why? Turns out it was the slow version.",),
        "punches": ("{C}: of course. Still dancing, apparently.",),
    },
    "trend_audio": {
        "duration": 11.0,
        "angles": ("audio_mismatch", "cute_reveal", "deadpan_switch", "beat_misread", "unexpected_pause"),
        "hooks": (("{C} heard one sound and committed. Then it changed!", "twist_tease"),
                  ("The sound says cute. Nobody warned {C}?!", "curiosity")),
        "goals": ("{C} wants to match one trending sound.", "{C} tries to lip-sync one tiny line."),
        "escalations": ("Then the audio glitches. {N} remixes at once!",),
        "twists": ("Wait, why? Turns out the {P} was playing it backwards.",),
        "punches": ("{C}: apparently that still counts as one take.",),
    },
    "reaction": {
        "duration": 11.0,
        "angles": ("shock", "overconfidence", "deadpan", "tiny_detail_big_reaction", "wrong_notification"),
        "hooks": (("{C} opened one notification. Big mistake!", "warning"),
                  ("{C} saw it. Then {C} regretted it!", "twist_tease")),
        "goals": ("{C} just wants one normal look at the {P}.", "{C} thought one look at the {P} was harmless."),
        "escalations": ("Then it buzzes again. {N} alerts at once!", "Then the {P} lights up. {N} new messages!"),
        "twists": ("Instead of news, it's literally a reminder to relax?",),
        "punches": ("{C}: I should have ignored that one.", "{C}: of course. Still not relaxing."),
    },
    "pov": {
        "duration": 11.0,
        "angles": ("relatable_work", "social_anxiety", "small_problem_big_reaction", "monday_mood", "quiet_minute"),
        "hooks": (("POV: {C} finally gets one quiet minute!", "pov"),
                  ("POV: {C} had one job today?!", "pov")),
        "goals": ("{C} finally sits down for one quiet minute.", "{C} wants one normal minute with the {P}."),
        "escalations": ("Then the {P} lights up. {N} tiny emergencies!",),
        "twists": ("Wait, why? Turns out the quiet minute was the emergency.",),
        "punches": ("{C}: apparently the universe still disagrees.", "{C}: of course. Back to work, again."),
    },
    "story_comedy": {
        "duration": 12.0,
        "angles": ("escalation", "misunderstanding", "reverse_expectation", "false_win", "loop_back"),
        "hooks": (("{C} thought today was easy. Big mistake!", "warning"),
                  ("It started as one tiny problem. Then {C} helped!", "twist_tease")),
        "goals": ("{C} has one simple plan for the {P}.", "{C} tries to fix one tiny thing."),
        "escalations": ("Then the problem multiplies. {N} of them!", "Then it gets worse. {N} problems at once!"),
        "twists": ("Instead of fixing it, the {P} misunderstood everything?", "Actually, why? The {P} had other plans."),
        "punches": ("{C}: of course. Still one job, apparently.", "{C}: put it back. All of it."),
    },
    "challenge": {
        "duration": 11.0,
        "angles": ("attempts", "countdown", "cheat_then_fail", "almost_win", "rules_change"),
        "hooks": (("{C} said yes before hearing the rules. Big mistake!", "warning"),
                  ("One simple challenge. Can {C} survive it?", "challenge")),
        "goals": ("{C} accepts one easy challenge with the {P}.", "{C} tries one attempt. Easy, right?"),
        "escalations": ("Then the rules change. {N} rounds at once!",),
        "twists": ("Wait, why? Turns out the prize was another challenge.",),
        "punches": ("{C}: apparently that's still not in the tutorial.", "{C}: apparently I still lost round one."),
    },
    "transformation": {
        "duration": 11.0,
        "angles": ("upgrade_backfire", "before_after", "expectation_reality", "wrong_upgrade", "side_effect"),
        "hooks": (("{C} upgraded one thing. Big mistake!", "warning"),
                  ("Before: perfect. Then {C} touched the {P}!", "before_after")),
        "goals": ("{C} wants to improve one tiny part of the {P}.", "{C} plans one small upgrade."),
        "escalations": ("Then the upgrade spreads. {N} new features!",),
        "twists": ("Instead of better, the {P} misunderstood the upgrade?",),
        "punches": ("{C}: put it back. Put it all back.",),
    },
    "cute_animal": {
        "duration": 11.0,
        "angles": ("tiny_hero", "adorable_failure", "comfort", "protective", "tiny_solution_big_effect"),
        "hooks": (("{C} is trying to help. Then it happened!", "twist_tease"),
                  ("Nobody asked {C} to help. {C} helped anyway!", "curiosity")),
        "goals": ("{C} wants to help with one tiny thing.", "{C} has one adorable plan to help."),
        "escalations": ("Then the tiny plan grows. {N} problems now!",),
        "twists": ("Wait, why? Turns out the help was literally the problem.",),
        "punches": ("{C}: still counts as helping. Apparently.",),
    },
    "absurdist": {
        "duration": 11.0,
        "angles": ("logic_break", "escalation", "deadpan_acceptance", "reality_glitch", "nonsense_rule"),
        "hooks": (("{C} accepted this way too quickly. Then physics quit!", "twist_tease"),
                  ("This should not be possible. Nobody told {C}?", "curiosity")),
        "goals": ("{C} just wants one normal minute with the {P}.", "{C} tries one ordinary task."),
        "escalations": ("Then gravity takes a break. {N} things float up!",),
        "twists": ("Instead of panicking, the {P} starts giving advice?",),
        "punches": ("{C}: of course. Same time tomorrow, again.",),
    },
    "listicle": {
        "duration": 12.0,
        "angles": ("three_failures", "three_reactions", "countdown", "top_bad_choices", "one_more"),
        "hooks": (("Three things {C} should never do. Number one: this!", "list"),
                  ("{C}'s top three bad decisions today. Big mistake!", "list")),
        "goals": ("{C} tries number three: one tiny button.", "{C} picks one harmless idea from the list."),
        "escalations": ("Then number two happens. {N} times worse!",),
        "twists": ("Wait, why? Turns out number one was the instructions.",),
        "punches": ("{C}: apparently the list still wins. Again.",),
    },
    "micro_short": {
        "duration": 10.0,
        "angles": ("single_gag", "instant_reversal", "reaction_loop", "setup_payoff_echo", "micro_twist"),
        "hooks": (("{C} had one job. Then the {P} moved!", "warning"),
                  ("This took seven seconds to go wrong. Watch {C}!", "countdown")),
        "goals": ("{C} has one job with the {P}.", "{C} finally finishes one tiny task."),
        "escalations": ("Then it breaks. {N} pieces everywhere!",),
        "twists": ("Instead of fixing it, {C} misunderstood the instructions?",),
        "punches": ("{C}: of course. Still one job.",),
    },
}

# Escalation counts are concrete numbers the story engine recognises as a
# bigger consequence than the setup's "one".
ESCALATION_COUNTS = ("twelve", "forty-seven")

# Distinct sound cues per beat role (all synthesised by cartoon_audio.make_sfx).
ROLE_SFX = {
    "hook": ("notification", "pop", "whoosh", "tap"),
    "escalation": ("impact", "chaos"),
    "reaction": ("overwhelmed", "reaction_hit"),
    "punchline": ("punchline",),
}

FORMAT_HOOK_SFX = {"dance": "beat", "trend_audio": "beat", "challenge": "countdown", "listicle": "countdown"}


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


def _pick(options: tuple[Any, ...], *key: str) -> Any:
    return options[_seed(*key) % len(options)]


def _fill(template: str, character: str, prop: str, count: str) -> str:
    text = template.format(C=character, P=prop, N=count)
    # A count can open a sentence ("... grows. {N} problems"); capitalise it.
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def _make_beats(character: str, recipe: str, angle: str, emotion: str, signal: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Build the four renderable beats for one idea, or () if the character has no complete pack."""
    roles = role_actions(character)
    if not roles:
        return ()
    prop = _prop_from_signal(str(signal.get("title", "")))
    kit = FORMAT_RECIPES[recipe]
    key = (character, recipe, angle, str(signal.get("trend_id", "")))
    count = _pick(ESCALATION_COUNTS, *key, "count")
    lines = {
        "hook": _fill(_pick(kit["goals"], *key, "goal"), character, prop, count),
        "escalation": _fill(_pick(kit["escalations"], *key, "esc"), character, prop, count),
        "reaction": _fill(_pick(kit["twists"], *key, "twist"), character, prop, count),
        "punchline": _fill(_pick(kit["punches"], *key, "punch"), character, prop, count),
    }
    hook_sfx = FORMAT_HOOK_SFX.get(recipe) or _pick(ROLE_SFX["hook"], *key, "sfx")
    beats = []
    for role in ("hook", "escalation", "reaction", "punchline"):
        action = roles[role]
        spec = action_spec(character, action) or {}
        words = len(re.findall(r"[A-Za-z0-9']+", lines[role]))
        beats.append({
            "beat": role,
            "asset_action": action,
            "voice": lines[role],
            "visual": str(spec.get("visual", "")),
            "emotion": str(spec.get("emotion", emotion)),
            "framing": str(spec.get("framing", "medium")),
            "sfx": hook_sfx if role == "hook" else _pick(ROLE_SFX[role], *key, role),
            "duration": round(max(2.0, words / 2.8 + 0.6), 2),
        })
    return tuple(beats)


def generate_ideas(trends: Iterable[dict[str, Any]], formats: Iterable[dict[str, Any]] | None = None,
                   count: int = 60, characters: tuple[str, ...] = ("MOMO", "PIP", "BUBU")) -> list[dict[str, Any]]:
    trend_rows = list(trends)
    format_rows = list(formats or [])
    if not trend_rows:
        trend_rows = [{"trend_id": "evergreen", "title": "evergreen cartoon problem", "emotions": ("amusement",), "formats": ("micro_short",)}]
    # Asset capability gate: only cast characters whose action pack covers
    # every beat role, so no generated idea is unrenderable.
    cast = renderable_characters(characters)

    candidates: list[dict[str, Any]] = []
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
        prop = _prop_from_signal(str(trend.get("title", "")))
        for fmt in unique_formats[:8]:
            recipe = FORMAT_RECIPES[fmt]
            for character in cast:
                for angle in recipe["angles"]:
                    emotion = emotions[_seed(str(trend.get("trend_id")), fmt, character, angle) % len(emotions)]
                    angle_text = _angle_phrase(angle)
                    title = f"{character} + {anchor}: {fmt.replace('_', ' ')} — {angle_text}"
                    hook_template, hook_pattern = _pick(recipe["hooks"], title, character)
                    hook = _fill(hook_template, character, prop, "")
                    beats = _make_beats(character, fmt, angle, emotion, trend)
                    if not beats:
                        continue
                    idea_id = f"idea-{_slug(title)}-{abs(_seed(title, str(trend.get('trend_id')))) % 100000:05d}"
                    cost = 0.25 + 0.08 * len(beats) + 0.06 * (1 if fmt in {"dance", "trend_audio"} else 0)
                    candidates.append({
                        "idea_id": idea_id,
                        "title": title,
                        "hook": hook,
                        "hook_pattern": hook_pattern,
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
    # Deterministic near-duplicate suppression. We prefer different formats,
    # characters and emotional angles before keeping another semantically similar idea.
    candidates.sort(key=lambda x: (
        -x.get("trend_score", 0.0),
        x["production_cost"],
        _seed(x["idea_id"]),
    ))
    chosen: list[dict[str, Any]] = []
    seen_tokens: list[set[str]] = []
    for item in candidates:
        toks = set(re.findall(r"[a-z0-9]+", (item["title"] + " " + item["hook"]).lower()))
        duplicate = False
        for old, old_tokens in zip(chosen, seen_tokens):
            union = toks | old_tokens
            if union and len(toks & old_tokens) / len(union) >= 0.72:
                # Keep diversity even when wording is similar.
                if item["format_name"] == old["format_name"] and item["character"] == old["character"]:
                    duplicate = True
                    break
        if duplicate:
            continue
        chosen.append(item)
        seen_tokens.append(toks)
        if len(chosen) >= count:
            break
    return chosen


def idea_to_dict(item: IdeaCandidateV2 | dict[str, Any]) -> dict[str, Any]:
    return asdict(item) if isinstance(item, IdeaCandidateV2) else dict(item)
