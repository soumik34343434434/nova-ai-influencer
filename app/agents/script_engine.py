"""Production-ready Script Engine for NOVA short-form cartoons.

The Story Engine owns narrative decisions. This module turns the approved Story
Contract into a timed performance script without changing the story structure.
It adds dialogue normalization, delivery intent, captions, SFX cues and a
deterministic quality gate so downstream Director/Audio/Editor stages consume a
single script contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

QUALITY_FLOOR = 8.5
WORDS_PER_SECOND = 2.8
MIN_BEAT_SECONDS = 1.15
MAX_BEAT_SECONDS = 3.6
MAX_CAPTION_WORDS = 8
MAX_CAPTION_CHARS = 42


@dataclass(frozen=True)
class ScriptDimension:
    name: str
    score: float
    evidence: tuple[str, ...]
    repair: str = ""


@dataclass(frozen=True)
class ScriptQuality:
    score: float
    passed: bool
    dimensions: tuple[ScriptDimension, ...]
    blockers: tuple[str, ...]
    repairs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScriptContract:
    text: str
    hook: str
    title: str
    character: str
    target_duration: float
    beats: tuple[dict[str, Any], ...]
    quality: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ROLE_WEIGHTS = {
    "hook": 0.16,
    "setup": 0.22,
    "escalation": 0.26,
    "reaction": 0.18,
    "payoff": 0.18,
}

ROLE_DELIVERY = {
    "hook": "immediate, conversational, confident",
    "setup": "natural, lightly curious",
    "escalation": "faster, surprised, clearly stressed",
    "reaction": "brief, processing the consequence",
    "payoff": "slower, dry, deadpan",
}

ROLE_SFX = {
    "hook": "room",
    "setup": "notification",
    "escalation": "impact",
    "reaction": "reaction_hit",
    "payoff": "punchline",
}

ROLE_CAPTION_STYLE = {
    "hook": "hook",
    "setup": "support",
    "escalation": "emphasis",
    "reaction": "reaction",
    "payoff": "punchline",
}

REQUIRED_ROLES = ("hook", "setup", "escalation", "reaction", "payoff")

ACTION_CUE = {
    "finish_work": "perform the finishing action first, then speak",
    "checks_phone": "eyes lead to the phone, head follows, then deliver the recognition line",
    "shock": "hit the emotional peak before the key word",
    "overwhelmed": "hold the overwhelmed reaction long enough to read",
    "deadpan": "finish the line, then hold the camera reaction",
    "press_button": "complete the button press before the consequence lands",
    "sees_calendar": "look toward the calendar before the realization",
    "reaches_for_coffee": "reach first, then let the relief line land",
    "glances_at_many_tabs": "scan the scene before the overload line",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text or "")


def _word_count(text: str) -> int:
    return len(_words(text))


def _clean_text(text: str) -> str:
    text = " ".join(str(text or "").replace("\r", " ").split())
    text = re.sub(r"^([A-Z][A-Z0-9_ -]{1,15}):\s*", "", text)
    return text.strip()


def _sentence(text: str, role: str) -> str:
    clean = _clean_text(text)
    if not clean:
        return clean
    if role == "escalation":
        return clean.rstrip("!?") + "!"
    if role == "reaction":
        return clean.rstrip(".!?") + ("?" if "?" in clean else ".")
    if role == "payoff":
        return clean.rstrip("!?") + "."
    return clean.rstrip(".!?") + "."


def _caption(text: str, role: str) -> str:
    words = _words(text)
    if not words:
        return ""
    if len(words) <= MAX_CAPTION_WORDS and len(text) <= MAX_CAPTION_CHARS:
        return text
    # Prefer a compact leading phrase; never add words that are absent from the dialogue.
    selected: list[str] = []
    size = 0
    for word in words:
        projected = size + (1 if selected else 0) + len(word)
        if len(selected) >= MAX_CAPTION_WORDS or projected > MAX_CAPTION_CHARS:
            break
        selected.append(word)
        size = projected
    result = " ".join(selected)
    if role == "payoff" and result and result[-1] not in ".!?":
        result += "."
    return result


def _duration_for_beat(role: str, target: float, text: str) -> float:
    base = max(MIN_BEAT_SECONDS, target * ROLE_WEIGHTS.get(role, 1.0 / 4.0))
    speech = _word_count(text) / WORDS_PER_SECOND + 0.10
    return round(min(MAX_BEAT_SECONDS, max(base, speech + 0.20)), 3)


def _normalize_roles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        role = str(item.get("story_role") or item.get("beat") or "").lower().strip()
        if role == "attempt":
            role = "setup"
        if role == "punchline":
            role = "payoff"
        if role not in ROLE_WEIGHTS:
            role = REQUIRED_ROLES[min(index, len(REQUIRED_ROLES) - 1)]
        item["story_role"] = role
        normalized.append(item)
    return normalized


def _repair_dialogue(row: dict[str, Any], role: str, character: str) -> str:
    original = _clean_text(str(row.get("voice") or row.get("dialogue") or ""))
    if original:
        return _sentence(original, role)
    action = str(row.get("asset_action") or "").replace("_", " ")
    if role == "hook":
        return f"{character}: one normal minute." if character else "One normal minute."
    if role == "setup":
        return f"Then the {action} starts."
    if role == "escalation":
        return "Then it gets worse!"
    if role == "reaction":
        return f"{character}: wait, what?" if character else "Wait, what?"
    return f"{character}: of course." if character else "Of course."


def _timing(target: float, rows: list[dict[str, Any]], lines: list[str]) -> list[dict[str, float]]:
    weights = [ROLE_WEIGHTS.get(str(row["story_role"]), 0.2) for row in rows]
    raw = [max(MIN_BEAT_SECONDS, target * weight) for weight in weights]
    # Give longer dialogue enough room without changing the authored target.
    raw = [max(value, _word_count(line) / WORDS_PER_SECOND + 0.30) for value, line in zip(raw, lines)]
    total = sum(raw)
    if total > target:
        scale = target / total
        raw = [max(MIN_BEAT_SECONDS, value * scale) for value in raw]
    # Rebalance any minimum-duration expansion back into the longest beats.
    total = sum(raw)
    if total > target:
        excess = total - target
        order = sorted(range(len(raw)), key=lambda i: raw[i], reverse=True)
        for i in order:
            room = max(0.0, raw[i] - MIN_BEAT_SECONDS)
            take = min(room, excess)
            raw[i] -= take
            excess -= take
            if excess <= 1e-9:
                break
    total = sum(raw) or target
    scale = target / total
    raw = [value * scale for value in raw]
    starts: list[dict[str, float]] = []
    cursor = 0.0
    for value in raw:
        end = cursor + value
        starts.append({"start": round(cursor, 3), "end": round(end, 3), "duration": round(value, 3)})
        cursor = end
    if starts:
        starts[-1]["end"] = round(target, 3)
        starts[-1]["duration"] = round(target - starts[-1]["start"], 3)
    return starts


def _quality_dimensions(
    contract: dict[str, Any],
    rows: list[dict[str, Any]],
) -> ScriptQuality:
    lines = [str(row.get("dialogue", "")) for row in rows]
    captions = [str(row.get("caption", "")) for row in rows]
    roles = [str(row.get("story_role", "")) for row in rows]
    actions = [str(row.get("action", "")) for row in rows]
    target = float(contract.get("target_duration", 8.0))
    total_duration = sum(float(row.get("duration", 0.0)) for row in rows)
    character = str(contract.get("character", "")).strip()
    dimensions: list[ScriptDimension] = []

    first = lines[0] if lines else ""
    hook_score = 5.0
    hook_evidence: list[str] = []
    if 4 <= _word_count(first) <= 12:
        hook_score += 2.0
        hook_evidence.append("opening line is concise")
    if any(mark in first for mark in ("?", "!", ":", "—")):
        hook_score += 1.0
        hook_evidence.append("opening has tension punctuation")
    if character and character.lower() in first.lower():
        hook_score += 1.0
        hook_evidence.append("character owns the opening")
    if first and first == str(contract.get("hook", "")).strip():
        hook_score += 1.0
        hook_evidence.append("script opening matches story hook exactly")
    dimensions.append(ScriptDimension("hook_delivery", min(hook_score, 10.0), tuple(hook_evidence),
                                       "Make the first spoken line concrete and immediate."))

    fidelity = 5.0
    fidelity_evidence: list[str] = []
    if len(rows) >= 4:
        fidelity += 1.5
        fidelity_evidence.append("all core story beats are represented")
    if roles and roles[0] in {"hook", "setup"} and "escalation" in roles and "reaction" in roles and "payoff" in roles:
        fidelity += 1.5
        fidelity_evidence.append("roles preserve the approved story contract")
    if all(row.get("action") for row in rows):
        fidelity += 1.0
        fidelity_evidence.append("every line has a visual action cue")
    if all(row.get("story_intent") for row in rows):
        fidelity += 1.0
        fidelity_evidence.append("story intent carried into script beats")
    dimensions.append(ScriptDimension("story_fidelity", min(fidelity, 10.0), tuple(fidelity_evidence),
                                       "Keep dialogue and beat roles locked to the approved Story Contract."))

    speech = 5.0
    speech_evidence: list[str] = []
    long_lines = [n for n in (_word_count(line) for line in lines) if n > 18]
    if not long_lines:
        speech += 1.5
        speech_evidence.append("no line exceeds 18 words")
    if sum(_word_count(line) for line in lines) <= max(18, int(target * WORDS_PER_SECOND)):
        speech += 1.5
        speech_evidence.append("total speech fits target duration")
    if all(line and not re.search(r"[|<>\[\]]", line) for line in lines):
        speech += 1.0
        speech_evidence.append("dialogue is TTS-safe")
    if len(lines) >= 4 and len(set(lines)) >= 4:
        speech += 1.0
        speech_evidence.append("no repeated dialogue lines")
    dimensions.append(ScriptDimension("dialogue_economy", min(speech, 10.0), tuple(speech_evidence),
                                       "Shorten long lines and remove repeated phrasing."))

    delivery = 5.0
    delivery_evidence: list[str] = []
    deliveries = [str(row.get("delivery", "")) for row in rows]
    if all(deliveries):
        delivery += 2.0
        delivery_evidence.append("every line has explicit delivery direction")
    if len(set(deliveries)) >= 3:
        delivery += 1.0
        delivery_evidence.append("delivery changes across the emotional arc")
    if roles and deliveries[-1].startswith("slower"):
        delivery += 1.0
        delivery_evidence.append("payoff is deliberately slowed for comic timing")
    if roles and any("faster" in value for value in deliveries):
        delivery += 1.0
        delivery_evidence.append("escalation has energy change")
    dimensions.append(ScriptDimension("character_voice", min(delivery, 10.0), tuple(delivery_evidence),
                                       "Give the character a consistent voice with distinct escalation and payoff delivery."))

    timing_score = 5.0
    timing_evidence: list[str] = []
    error = abs(total_duration - target)
    if error <= 0.05:
        timing_score += 2.5
        timing_evidence.append("beat durations exactly match target")
    elif error <= 0.15:
        timing_score += 1.5
        timing_evidence.append("beat durations stay within 150 ms")
    if all(float(row.get("duration", 0.0)) >= MIN_BEAT_SECONDS for row in rows):
        timing_score += 1.0
        timing_evidence.append("every beat is readable")
    if rows and float(rows[-1].get("duration", 0.0)) >= 1.25:
        timing_score += 0.5
        timing_evidence.append("payoff has room to land")
    if all(float(row.get("pause_before", 0.0)) <= 0.12 and float(row.get("pause_after", 0.0)) <= 0.20 for row in rows):
        timing_score += 1.0
        timing_evidence.append("pauses avoid synthetic over-segmentation")
    dimensions.append(ScriptDimension("timing", min(timing_score, 10.0), tuple(timing_evidence),
                                       "Rebalance beat durations and keep pauses short and intentional."))

    visual_score = 5.0
    visual_evidence: list[str] = []
    if all(actions):
        visual_score += 2.0
        visual_evidence.append("all dialogue beats map to visible action")
    if len(set(actions)) >= 3:
        visual_score += 1.5
        visual_evidence.append("visual actions change with the story")
    if any(row.get("performance") for row in rows):
        visual_score += 0.5
        visual_evidence.append("performance direction is attached to each beat")
    if all(row.get("visual") for row in rows):
        visual_score += 1.0
        visual_evidence.append("visual intent is explicit")
    dimensions.append(ScriptDimension("visual_sync", min(visual_score, 10.0), tuple(visual_evidence),
                                       "Attach one clear visible action to every spoken line."))

    reaction_score = 5.0
    reaction_evidence: list[str] = []
    reaction_rows = [row for row in rows if row.get("story_role") == "reaction"]
    payoff_rows = [row for row in rows if row.get("story_role") == "payoff"]
    if reaction_rows:
        reaction_score += 2.0
        reaction_evidence.append("explicit reaction line exists")
    if payoff_rows:
        reaction_score += 1.5
        reaction_evidence.append("explicit payoff line exists")
    if reaction_rows and payoff_rows and reaction_rows[0]["index"] < payoff_rows[0]["index"]:
        reaction_score += 1.0
        reaction_evidence.append("reaction precedes punchline")
    if payoff_rows and _word_count(payoff_rows[-1]["dialogue"]) <= 14:
        reaction_score += 0.5
        reaction_evidence.append("punchline is compact")
    dimensions.append(ScriptDimension("reaction_payoff", min(reaction_score, 10.0), tuple(reaction_evidence),
                                       "Keep a readable reaction beat immediately before a concise payoff."))

    caption_score = 5.0
    caption_evidence: list[str] = []
    if all(_word_count(caption) <= MAX_CAPTION_WORDS for caption in captions):
        caption_score += 2.0
        caption_evidence.append("captions stay under 8 words")
    if all(len(caption) <= MAX_CAPTION_CHARS for caption in captions):
        caption_score += 1.5
        caption_evidence.append("captions fit a compact mobile line")
    if all(caption for caption in captions):
        caption_score += 1.0
        caption_evidence.append("every line has a caption")
    if captions and str(rows[-1].get("caption_style")) == "punchline":
        caption_score += 0.5
        caption_evidence.append("payoff caption has punchline styling")
    dimensions.append(ScriptDimension("caption_readiness", min(caption_score, 10.0), tuple(caption_evidence),
                                       "Keep captions concise and mobile-readable."))

    sound_score = 5.0
    sound_evidence: list[str] = []
    if all(str(row.get("sfx", "")) for row in rows):
        sound_score += 2.0
        sound_evidence.append("every beat has an SFX cue")
    if len(set(str(row.get("sfx", "")) for row in rows)) >= 3:
        sound_score += 1.0
        sound_evidence.append("sound design changes with the story")
    if rows and rows[0].get("sfx") != rows[-1].get("sfx"):
        sound_score += 1.0
        sound_evidence.append("opening and payoff have distinct sonic roles")
    if any("notification" in str(row.get("sfx", "")) for row in rows):
        sound_score += 1.0
        sound_evidence.append("attention cue exists for a notification beat")
    dimensions.append(ScriptDimension("sound_cues", min(sound_score, 10.0), tuple(sound_evidence),
                                       "Add distinct, story-matched sound cues without overloading the mix."))

    performance_score = 5.0
    performance_evidence: list[str] = []
    if all(row.get("performance") for row in rows):
        performance_score += 2.0
        performance_evidence.append("every beat exposes animation intent")
    if all("timing" in row.get("performance", {}) for row in rows):
        performance_score += 1.0
        performance_evidence.append("performance carries timing context")
    if all("gaze" in row.get("performance", {}) for row in rows):
        performance_score += 1.0
        performance_evidence.append("gaze direction is explicit")
    if all("emotion" in row.get("performance", {}) for row in rows):
        performance_score += 1.0
        performance_evidence.append("emotion is explicit for animation")
    dimensions.append(ScriptDimension("performance_direction", min(performance_score, 10.0), tuple(performance_evidence),
                                       "Pass explicit emotion, gaze and timing intent to Director/Animation."))

    avg = round(sum(d.score for d in dimensions) / len(dimensions), 2)
    blockers = tuple(
        f"{d.name}:{d.score:.2f}<{QUALITY_FLOOR:.1f}"
        for d in dimensions
        if d.score < QUALITY_FLOOR
    )
    repairs = tuple(d.repair for d in dimensions if d.repair and d.score < QUALITY_FLOOR)
    passed = avg >= QUALITY_FLOOR and not blockers and len(rows) >= 4
    if len(rows) < 4:
        blockers = blockers + ("script_requires_at_least_4_beats",)
        repairs = repairs + ("Preserve at least hook, setup, escalation, reaction and payoff.",)
        passed = False
    return ScriptQuality(avg, passed, tuple(dimensions), blockers, repairs)


class ScriptEngine:
    """Convert a validated Story Contract into an 8.5+ production script."""

    def evaluate(self, story: dict[str, Any]) -> ScriptQuality:
        contract = self._build_contract(story, for_evaluation=True)
        return _quality_dimensions(contract, contract["beats"])

    def build(self, story: dict[str, Any]) -> ScriptContract:
        contract = self._build_contract(story, for_evaluation=False)
        quality = _quality_dimensions(contract, contract["beats"])
        if not quality.passed:
            raise ValueError("Script quality gate failed: " + "; ".join(quality.blockers))
        quality_dict = quality.to_dict()
        quality_dict["quality_floor"] = QUALITY_FLOOR
        quality_dict["dimension_count"] = len(quality.dimensions)
        quality_dict["min_dimension_score"] = min(d.score for d in quality.dimensions)
        quality_dict["all_dimensions_above_floor"] = all(d.score >= QUALITY_FLOOR for d in quality.dimensions)
        return ScriptContract(
            text=contract["text"],
            hook=contract["hook"],
            title=contract["title"],
            character=contract["character"],
            target_duration=contract["target_duration"],
            beats=tuple(contract["beats"]),
            quality=quality_dict,
        )

    def _build_contract(self, story: dict[str, Any], for_evaluation: bool = False) -> dict[str, Any]:
        if not isinstance(story, dict):
            raise TypeError("story must be a dictionary")
        rows = _normalize_roles([row for row in story.get("beats", []) if isinstance(row, dict)])
        if len(rows) < 4:
            raise ValueError("Story Contract must contain at least four beats")
        character = str(story.get("character", "")).strip()
        target = float(story.get("duration_target", 8.0) or 8.0)
        target = max(5.0, min(30.0, target))

        # Story Engine can emit four beats while the narrative has five logical
        # roles. The Script Engine only inserts an explicit hook beat when the
        # story lacks one; it never invents a new narrative conflict.
        # Keep the Story Engine's approved roles. A four-beat story may start
        # with setup while the separate approved hook is spoken on that beat.

        lines: list[str] = []
        normalized: list[dict[str, Any]] = []
        story_hook = _sentence(str(story.get("hook") or ""), "hook")
        for index, row in enumerate(rows[:5]):
            role = str(row["story_role"])
            line = _repair_dialogue(row, role, character)
            if index == 0 and story_hook and story_hook.lower() not in line.lower():
                # The approved hook is the first spoken idea, while the beat's
                # original setup line remains attached so no narrative intent is lost.
                line = f"{story_hook} {line}".strip()
            lines.append(line)

            action = str(row.get("asset_action") or row.get("action") or "")
            visual = str(row.get("visual") or row.get("visual_intent") or "")
            intent = str(row.get("story_intent") or "")
            duration_placeholder = target * ROLE_WEIGHTS.get(role, 0.2)
            performance = {
                "emotion": {
                    "hook": "curious",
                    "setup": "neutral",
                    "escalation": "surprised",
                    "reaction": "processing",
                    "payoff": "deadpan",
                }.get(role, "neutral"),
                "gaze": {
                    "hook": "camera",
                    "setup": "scene",
                    "escalation": "scene",
                    "reaction": "camera",
                    "payoff": "camera",
                }.get(role, "scene"),
                "timing": ROLE_DELIVERY.get(role, "natural"),
                "action_cue": ACTION_CUE.get(action, f"perform the {action.replace('_', ' ')} action coherently"),
            }
            normalized.append({
                "index": index,
                "story_role": role,
                "story_intent": intent,
                "action": action,
                "dialogue": line,
                "voice_text": line,
                "delivery": ROLE_DELIVERY.get(role, "natural, conversational"),
                "caption": _caption(line, role),
                "caption_style": ROLE_CAPTION_STYLE.get(role, "support"),
                "sfx": str(row.get("sfx") or ROLE_SFX.get(role, "room")),
                "visual": visual,
                "performance": performance,
                "source_voice": _clean_text(str(row.get("voice") or "")),
                "source_asset_action": action,
                "pause_before": 0.04 if role in {"setup", "reaction"} else 0.02,
                "pause_after": {
                    "hook": 0.05,
                    "setup": 0.05,
                    "escalation": 0.08,
                    "reaction": 0.09,
                    "payoff": 0.16,
                }.get(role, 0.05),
            })

        timings = _timing(target, normalized, lines)
        for row, timing in zip(normalized, timings):
            row["timing_start"] = timing["start"]
            row["timing_end"] = timing["end"]
            row["duration"] = timing["duration"]
            row["word_count"] = _word_count(row["dialogue"])

        text = "\n\n".join(row["voice_text"] for row in normalized)
        return {
            "text": text,
            "hook": str(story.get("hook") or normalized[0]["dialogue"]).strip(),
            "title": str(story.get("premise") or story.get("title") or "").strip(),
            "character": character,
            "target_duration": target,
            "beats": normalized,
        }
