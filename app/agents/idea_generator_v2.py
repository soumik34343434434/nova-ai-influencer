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

FORMAT_RECIPES: dict[str, dict[str, Any]] = {
    "dance": {
        "duration": 7.0,
        "angles": ("adorable_commitment", "wrong_move", "sudden_mood_change", "beat_drop_surprise", "overconfident_start"),
        "hooks": ("MOMO heard the song once. That was enough.", "Nobody told MOMO this was a dance challenge."),
    },
    "trend_audio": {
        "duration": 8.0,
        "angles": ("audio_mismatch", "cute_reveal", "deadpan_switch", "beat_misread", "unexpected_pause"),
        "hooks": ("MOMO understood the trend completely wrong.", "The sound says cute. MOMO has other plans."),
    },
    "reaction": {
        "duration": 8.0,
        "angles": ("shock", "overconfidence", "deadpan", "tiny_detail_big_reaction", "wrong_notification"),
        "hooks": ("MOMO saw it. MOMO instantly regretted it.", "The notification looked harmless. It wasn't."),
    },
    "pov": {
        "duration": 9.0,
        "angles": ("relatable_work", "social_anxiety", "small_problem_big_reaction", "monday_mood", "quiet_minute"),
        "hooks": ("POV: MOMO finally gets one quiet minute.", "POV: everything was under control for 3 seconds."),
    },
    "story_comedy": {
        "duration": 10.0,
        "angles": ("escalation", "misunderstanding", "reverse_expectation", "false_win", "loop_back"),
        "hooks": ("MOMO thought today was easy. Cute.", "It started as a tiny problem."),
    },
    "challenge": {
        "duration": 10.0,
        "angles": ("attempts", "countdown", "cheat_then_fail", "almost_win", "rules_change"),
        "hooks": ("MOMO said yes before hearing the rules.", "One simple challenge. One very tired cat."),
    },
    "transformation": {
        "duration": 8.0,
        "angles": ("upgrade_backfire", "before_after", "expectation_reality", "wrong_upgrade", "side_effect"),
        "hooks": ("MOMO upgraded one thing. Big mistake.", "Before: perfect. After: MOMO."),
    },
    "cute_animal": {
        "duration": 7.0,
        "angles": ("tiny_hero", "adorable_failure", "comfort", "protective", "tiny_solution_big_effect"),
        "hooks": ("MOMO is trying to help. It's adorable.", "Nobody asked MOMO to do this."),
    },
    "absurdist": {
        "duration": 8.0,
        "angles": ("logic_break", "escalation", "deadpan_acceptance", "reality_glitch", "nonsense_rule"),
        "hooks": ("MOMO accepted this way too quickly.", "This should not be possible. MOMO doesn't care."),
    },
    "listicle": {
        "duration": 9.0,
        "angles": ("three_failures", "three_reactions", "countdown", "top_bad_choices", "one_more"),
        "hooks": ("Three things MOMO absolutely should not have done.", "MOMO's top 3 bad decisions today."),
    },
    "micro_short": {
        "duration": 7.0,
        "angles": ("single_gag", "instant_reversal", "reaction_loop", "setup_payoff_echo", "micro_twist"),
        "hooks": ("This took 7 seconds to go wrong.", "MOMO had one job."),
    },
}


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


def _make_beats(character: str, recipe: str, angle: str, emotion: str, signal: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    prop = _prop_from_signal(str(signal.get("title", "")))
    persona = CHARACTER_PERSONAS[character][0]
    if recipe in {"dance", "trend_audio"}:
        return (
            {"beat": "hook", "asset_action": "finish_work", "voice": f"{character} hears the sound and commits instantly.", "sfx": "beat"},
            {"beat": "attempt", "asset_action": "checks_phone", "voice": f"The {persona} version of the trend begins.", "sfx": "motion"},
            {"beat": "escalation", "asset_action": "shock", "voice": "Then the trend changes.", "sfx": "impact"},
            {"beat": "punchline", "asset_action": "deadpan", "voice": f"{character}: apparently I missed that part.", "sfx": "punchline"},
        )
    if recipe == "reaction":
        return (
            {"beat": "hook", "asset_action": "finish_work", "voice": f"{character} gets one normal notification.", "sfx": "notification"},
            {"beat": "setup", "asset_action": "checks_phone", "voice": f"{character} checks the {prop}.", "sfx": "phone"},
            {"beat": "escalation", "asset_action": "shock", "voice": f"The {emotion} part appears.", "sfx": "impact"},
            {"beat": "punchline", "asset_action": "deadpan", "voice": f"{character}: I should have ignored that.", "sfx": "punchline"},
        )
    if recipe == "pov":
        return (
            {"beat": "hook", "asset_action": "finish_work", "voice": f"POV: {character} finally gets a quiet minute.", "sfx": "room"},
            {"beat": "setup", "asset_action": "checks_phone", "voice": f"Then the {prop} lights up.", "sfx": "notification"},
            {"beat": "escalation", "asset_action": "shock", "voice": "It should have been one small task.", "sfx": "impact"},
            {"beat": "punchline", "asset_action": "deadpan", "voice": f"{character}: apparently the universe disagrees.", "sfx": "punchline"},
        )
    if recipe == "challenge":
        return (
            {"beat": "hook", "asset_action": "finish_work", "voice": f"{character} says yes before hearing the rules.", "sfx": "countdown"},
            {"beat": "attempt", "asset_action": "checks_phone", "voice": "Attempt one is suspiciously easy.", "sfx": "tap"},
            {"beat": "escalation", "asset_action": "shock", "voice": "Attempt two changes everything.", "sfx": "impact"},
            {"beat": "punchline", "asset_action": "deadpan", "voice": f"{character}: that was not in the tutorial.", "sfx": "punchline"},
        )
    if recipe == "transformation":
        return (
            {"beat": "hook", "asset_action": "finish_work", "voice": f"Before: {character} has a perfect plan.", "sfx": "spark"},
            {"beat": "setup", "asset_action": "checks_phone", "voice": "After: one tiny upgrade.", "sfx": "whoosh"},
            {"beat": "escalation", "asset_action": "shock", "voice": "The upgrade has opinions.", "sfx": "impact"},
            {"beat": "punchline", "asset_action": "deadpan", "voice": f"{character}: put it back.", "sfx": "punchline"},
        )
    if recipe == "cute_animal":
        return (
            {"beat": "hook", "asset_action": "finish_work", "voice": f"{character} is trying very hard to help.", "sfx": "cute"},
            {"beat": "setup", "asset_action": "checks_phone", "voice": "The tiny plan is adorable.", "sfx": "tap"},
            {"beat": "escalation", "asset_action": "shock", "voice": "The tiny plan becomes a giant problem.", "sfx": "impact"},
            {"beat": "punchline", "asset_action": "deadpan", "voice": f"{character}: still counts as helping.", "sfx": "punchline"},
        )
    if recipe == "listicle":
        return (
            {"beat": "hook", "asset_action": "finish_work", "voice": f"Three things {character} should never do.", "sfx": "countdown"},
            {"beat": "setup", "asset_action": "checks_phone", "voice": "Number three starts harmless.", "sfx": "tap"},
            {"beat": "escalation", "asset_action": "shock", "voice": "Number two is somehow worse.", "sfx": "impact"},
            {"beat": "punchline", "asset_action": "deadpan", "voice": f"{character}: number one is deleting the app.", "sfx": "punchline"},
        )
    return (
        {"beat": "hook", "asset_action": "finish_work", "voice": f"{character}: finally. One normal minute.", "sfx": "room"},
        {"beat": "setup", "asset_action": "checks_phone", "voice": f"The {prop} buzzes.", "sfx": "notification"},
        {"beat": "escalation", "asset_action": "shock", "voice": f"The {emotion} part appears.", "sfx": "impact"},
        {"beat": "punchline", "asset_action": "deadpan", "voice": f"{character}: of course it did.", "sfx": "punchline"},
    )


def generate_ideas(trends: Iterable[dict[str, Any]], formats: Iterable[dict[str, Any]] | None = None,
                   count: int = 60, characters: tuple[str, ...] = ("MOMO", "PIP", "BUBU")) -> list[dict[str, Any]]:
    trend_rows = list(trends)
    format_rows = list(formats or [])
    if not trend_rows:
        trend_rows = [{"trend_id": "evergreen", "title": "evergreen cartoon problem", "emotions": ("amusement",), "formats": ("micro_short",)}]

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
        for fmt in unique_formats[:8]:
            recipe = FORMAT_RECIPES[fmt]
            for character in characters:
                for angle in recipe["angles"]:
                    emotion = emotions[_seed(str(trend.get("trend_id")), fmt, character, angle) % len(emotions)]
                    angle_text = _angle_phrase(angle)
                    title = f"{character} + {anchor}: {fmt.replace('_', ' ')} — {angle_text}"
                    base_hook = recipe["hooks"][_seed(title, character) % len(recipe["hooks"])]
                    hook = f"{base_hook} ({angle_text})."
                    beats = _make_beats(character, fmt, angle, emotion, trend)
                    idea_id = f"idea-{_slug(title)}-{abs(_seed(title, str(trend.get('trend_id')))) % 100000:05d}"
                    cost = 0.25 + 0.08 * len(beats) + 0.06 * (1 if fmt in {"dance", "trend_audio"} else 0)
                    candidates.append({
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
