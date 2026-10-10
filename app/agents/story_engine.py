"""Quality-gated story engine for NOVA short-form cartoon narratives.

The engine converts a winning idea into an explicit story contract before
scriptwriting. It scores independent narrative dimensions, enforces an 8.5/10
floor per dimension, and emits repair guidance instead of silently accepting a
weak premise.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Any


QUALITY_FLOOR = 8.5
MAX_SCORE = 10.0

BEAT_ROLE_MAP = {"hook":"hook", "setup":"setup", "attempt":"setup", "escalation":"escalation", "reaction":"reaction", "punchline":"payoff", "payoff":"payoff"}

ACTION_MARKERS = {
    "finish_work": {"setup", "false_win", "hook"},
    "checks_phone": {"setup", "discovery", "escalation"},
    "shock": {"escalation", "reaction"},
    "deadpan": {"payoff", "reaction"},
    "overwhelmed": {"reaction"},
    "press_button": {"setup"},
    "sees_calendar": {"discovery", "escalation"},
    "reaches_for_coffee": {"setup", "reaction"},
    "walk_in": {"setup", "hook"},
    "wave_hello": {"setup", "hook"},
    "yawn_stretch": {"escalation", "reaction"},
    "panic_run": {"escalation"},
    "curl_up": {"reaction"},
    "worried_close": {"reaction"},
}

REACTION_ACTIONS = frozenset({"shock", "overwhelmed", "deadpan", "curl_up", "worried_close"})

CONFLICT_MARKERS = (
    "problem", "mistake", "wrong", "but", "then", "instead", "fails", "backfires",
    "buzz", "changes", "unexpected", "too many", "not", "shouldn't", "never", "twelve", "forty-seven", "300", "giant",
)

SURPRISE_MARKERS = (
    "apparently", "unexpected", "big mistake", "wrong", "instead", "misunderstood",
    "of course", "then", "somehow", "should not", "one more",
)

PAYOFF_MARKERS = (
    "apparently", "of course", "i should", "put it back", "that's", "still counts",
    "not in", "ignored", "delete", "tomorrow", "great", "misunderstood",
)


@dataclass(frozen=True)
class StoryDimension:
    name: str
    score: float
    evidence: tuple[str, ...]
    repair: str = ""


@dataclass(frozen=True)
class StoryContract:
    premise: str
    logline: str
    hook: str
    character: str
    desire: str
    conflict: str
    escalation: str
    twist: str
    payoff: str
    ending: str
    beats: tuple[dict[str, Any], ...]
    duration_target: float
    quality: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StoryQuality:
    score: float
    passed: bool
    dimensions: tuple[StoryDimension, ...]
    blockers: tuple[str, ...]
    repairs: tuple[str, ...]
    # Additive field: older consumers can continue reading the original fields.
    story_analysis: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text or "")


def _word_count(text: str) -> int:
    return len(_words(text))


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in markers)


def _beat_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in candidate.get("beats", []) if isinstance(row, dict)]
    return rows[:5]


def _score_hook(candidate: dict[str, Any]) -> StoryDimension:
    hook = str(candidate.get("hook", "")).strip()
    wc = _word_count(hook)
    evidence: list[str] = []
    score = 5.0
    if 5 <= wc <= 13:
        score += 2.0
        evidence.append("5-13 word hook")
    else:
        evidence.append("hook length outside short-form target")
    if "!" in hook or "?" in hook:
        score += 1.0
        evidence.append("question/exclamation tension")
    if _has_any(hook, ("big mistake", "then", "one job", "finally", "nobody", "pov")):
        score += 1.0
        evidence.append("curiosity/tension phrase")
    if any(token.lower() in hook.lower() for token in (candidate.get("character", ""), "one", "tiny", "normal")):
        score += 1.0
        evidence.append("specific subject or concrete detail")
    repair = "" if score >= QUALITY_FLOOR else "Make the first sentence specific, concrete, and curiosity-driven."
    return StoryDimension("hook", round(min(score, 10.0), 2), tuple(evidence), repair)


def _score_setup(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> StoryDimension:
    first = rows[0] if rows else {}
    text = str(first.get("voice", "")) + " " + str(first.get("visual", ""))
    score = 5.0
    evidence: list[str] = []
    if first.get("beat") in {"hook", "setup"}:
        score += 1.5
        evidence.append("opening beat establishes context")
    if first.get("asset_action"):
        score += 1.0
        evidence.append("opening has a visible action")
    if _word_count(text) >= 4:
        score += 1.0
        evidence.append("opening carries concrete information")
    if str(candidate.get("character", "")).lower() in text.lower():
        score += 0.5
        evidence.append("character appears in setup")
    if len(rows) >= 3:
        score += 1.0
        evidence.append("room for setup-to-payoff progression")
    repair = "" if score >= QUALITY_FLOOR else "State who is doing what and what they want within the first beat."
    return StoryDimension("setup", round(min(score, 10.0), 2), tuple(evidence), repair)


def _score_goal(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> StoryDimension:
    text = " ".join(str(row.get("voice", "")) for row in rows[:2])
    candidate_blob = f"{candidate.get('title','')} {candidate.get('angle','')} {text}".lower()
    concrete = any(word in candidate_blob for word in (
        "one", "normal", "quiet", "help", "try", "want", "finish", "plan", "job",
        "answer", "challenge", "save", "understand",
    ))
    score = 6.0
    evidence: list[str] = []
    if concrete:
        score += 2.0
        evidence.append("concrete character objective")
    if any(word in candidate_blob for word in ("finally", "trying", "said yes", "asked", "thought")):
        score += 1.0
        evidence.append("goal implied through intent verb")
    if len(text.split()) >= 8:
        score += 0.5
        evidence.append("early context is explicit")
    if candidate.get("character"):
        score += 0.5
        evidence.append("goal belongs to a named recurring character")
    repair = "" if score >= QUALITY_FLOOR else "Add a one-sentence character desire that the conflict can interrupt."
    return StoryDimension("character_goal", round(min(score, 10.0), 2), tuple(evidence), repair)


def _score_conflict(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> StoryDimension:
    text = " ".join(str(row.get("voice", "")) for row in rows)
    score = 5.0
    evidence: list[str] = []
    if _has_any(text, CONFLICT_MARKERS):
        score += 2.5
        evidence.append("explicit problem/reversal language")
    if re.search(r"\b(?:two|three|four|five|six|seven|eight|nine|ten|twelve|300|forty-seven)\b", text.lower()):
        score += 1.0
        evidence.append("concrete consequence/count")
    if len(rows) >= 2 and rows[1].get("asset_action") != rows[0].get("asset_action"):
        score += 1.0
        evidence.append("visible action changes after setup")
    if str(candidate.get("angle", "")).lower() in {"misunderstanding", "reverse_expectation", "upgrade_backfire", "rules_change", "small_problem_big_reaction"}:
        score += 1.0
        evidence.append("conflict-bearing story angle")
    if any(str(row.get("beat", "")).lower() in {"escalation", "attempt"} for row in rows):
        score += 0.5
        evidence.append("explicit escalation beat exists")
    repair = "" if score >= QUALITY_FLOOR else "Turn the premise into a concrete problem that visibly interrupts the character's goal."
    return StoryDimension("conflict", round(min(score, 10.0), 2), tuple(evidence), repair)


def _score_escalation(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> StoryDimension:
    actions = [str(row.get("asset_action", "")) for row in rows]
    voices = [str(row.get("voice", "")) for row in rows]
    score = 5.0
    evidence: list[str] = []
    if len(rows) >= 4:
        score += 1.0
        evidence.append("four-beat progression")
    if len(set(actions)) >= 3:
        score += 1.5
        evidence.append("three or more distinct visual actions")
    if any("more" in v.lower() or "again" in v.lower() or "then" in v.lower() for v in voices):
        score += 1.0
        evidence.append("linguistic escalation marker")
    if any(str(row.get("beat", "")).lower() in {"escalation", "attempt"} for row in rows):
        score += 1.0
        evidence.append("named escalation beat")
    if str(candidate.get("angle", "")).lower() in {"escalation", "three_failures", "three_reactions", "single_gag", "instant_reversal"}:
        score += 0.5
        evidence.append("escalation-aware angle")
    repair = "" if score >= QUALITY_FLOOR else "Make each beat materially worse, stranger, or more revealing than the previous beat."
    return StoryDimension("escalation", round(min(score, 10.0), 2), tuple(evidence), repair)


def _score_visual(rows: list[dict[str, Any]]) -> StoryDimension:
    actions = [str(row.get("asset_action", "")) for row in rows]
    text = " ".join(str(row.get("visual", "")) for row in rows)
    score = 5.0
    evidence: list[str] = []
    if len(set(actions)) >= 3:
        score += 2.5
        evidence.append("distinct visual actions")
    elif len(set(actions)) >= 2:
        score += 1.0
        evidence.append("some visual variation")
    if any(action in ACTION_MARKERS for action in actions):
        score += 1.0
        evidence.append("production-known acting actions")
    if len(text) >= 40:
        score += 1.0
        evidence.append("visual intent is explicit")
    if any(word in text.lower() for word in ("phone", "button", "coffee", "laptop", "alarm", "pile")):
        score += 1.0
        evidence.append("concrete prop or environment")
    repair = "" if score >= QUALITY_FLOOR else "Replace exposition with visible actions, reactions, and one concrete prop interaction."
    return StoryDimension("visual_storytelling", round(min(score, 10.0), 2), tuple(evidence), repair)


def _score_surprise(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> StoryDimension:
    text = " ".join(str(row.get("voice", "")) for row in rows) + " " + str(candidate.get("hook", ""))
    score = 5.0
    evidence: list[str] = []
    if _has_any(text, SURPRISE_MARKERS):
        score += 2.0
        evidence.append("explicit reversal/surprise marker")
    if len(rows) >= 3 and rows[1].get("asset_action") != rows[-1].get("asset_action"):
        score += 1.0
        evidence.append("outcome differs from initial state")
    if any(str(row.get("beat", "")).lower() in {"escalation", "punchline"} for row in rows):
        score += 1.0
        evidence.append("twist-capable beat structure")
    if str(candidate.get("angle", "")).lower() in {"reverse_expectation", "misunderstanding", "wrong_move", "deadpan_switch", "micro_twist"}:
        score += 1.0
        evidence.append("reversal-driven angle")
    repair = "" if score >= QUALITY_FLOOR else "Introduce one specific reversal that the audience can understand instantly without narration."
    return StoryDimension("surprise_twist", round(min(score, 10.0), 2), tuple(evidence), repair)


def _score_reaction(rows: list[dict[str, Any]]) -> StoryDimension:
    score = 5.0
    evidence: list[str] = []
    reaction_rows = [row for row in rows if str(row.get("beat", "")).lower() in {"reaction", "punchline"} or str(row.get("asset_action", "")) in REACTION_ACTIONS]
    if reaction_rows:
        score += 2.0
        evidence.append("explicit character reaction")
    if any(row.get("expression") or row.get("emotion") for row in reaction_rows):
        score += 1.0
        evidence.append("reaction has emotional state")
    if len(reaction_rows) >= 2:
        score += 1.5
        evidence.append("reaction evolves toward payoff")
    if any("face" in str(row).lower() or "looks" in str(row).lower() for row in reaction_rows):
        score += 0.5
        evidence.append("facial reaction is visually expressible")
    repair = "" if score >= QUALITY_FLOOR else "Give the character a readable reaction beat after the conflict, before the punchline."
    return StoryDimension("character_reaction", round(min(score, 10.0), 2), tuple(evidence), repair)


def _score_payoff(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> StoryDimension:
    last = rows[-1] if rows else {}
    text = " ".join(str(row.get("voice", "")) for row in rows) + " " + str(candidate.get("title", "")) + " " + str(candidate.get("hook", ""))
    score = 5.0
    evidence: list[str] = []
    if str(last.get("beat", "")).lower() in {"punchline", "payoff"}:
        score += 1.5
        evidence.append("explicit payoff beat")
    if _has_any(text, PAYOFF_MARKERS):
        score += 1.5
        evidence.append("punchline/reversal language")
    if _word_count(str(last.get("voice", ""))) <= 14:
        score += 1.0
        evidence.append("short punchline")
    if candidate.get("character") and str(candidate["character"]).lower() in str(last.get("voice", "")).lower():
        score += 1.0
        evidence.append("character owns the punchline")
    if str(candidate.get("angle", "")).lower() in {"deadpan", "deadpan_switch", "single_gag", "loop_back"}:
        score += 1.0
        evidence.append("payoff-aware angle")
    repair = "" if score >= QUALITY_FLOOR else "End with a concise character-owned reversal or punchline."
    return StoryDimension("payoff", round(min(score, 10.0), 2), tuple(evidence), repair)


def _score_ending(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> StoryDimension:
    last = rows[-1] if rows else {}
    text = str(last.get("voice", ""))
    score = 5.0
    evidence: list[str] = []
    if last:
        score += 1.0
        evidence.append("explicit final beat")
    if str(last.get("beat", "")).lower() in {"punchline", "payoff"}:
        score += 1.5
        evidence.append("ending resolves on payoff beat")
    if text:
        score += 0.5
        evidence.append("ending has spoken resolution")
    if any(token in text.lower() for token in ("should have", "apparently", "of course", "still", "put it back", "ignored", "misunderstood")):
        score += 1.0
        evidence.append("ending has aftertaste/reversal")
    if str(candidate.get("angle", "")).lower() in {"loop_back", "reaction_loop", "instant_reversal", "deadpan"}:
        score += 1.0
        evidence.append("loop/replay-friendly ending")
    repair = "" if score >= QUALITY_FLOOR else "Make the final frame resolve the joke cleanly and leave a compact aftertaste."
    return StoryDimension("ending_loopability", round(min(score, 10.0), 2), tuple(evidence), repair)


_INTENSITY = {"one": 1, "single": 1, "small": 1, "tiny": 1, "two": 2, "three": 3,
              "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
              "nine": 9, "ten": 10, "twelve": 12, "dozen": 12, "hundred": 100,
              "thousand": 1000, "giant": 3, "huge": 3, "pile": 2, "flood": 4,
              "explodes": 4, "multiplies": 4, "again": 1, "more": 1}
_ACTION_VERBS = re.compile(r"\b(?:press(?:es|ed)?|opens?|closes?|drops?|throws?|grabs?|runs?|falls?|appears?|piles?|spills?|breaks?|buzz(?:es|ed)?|changes?|turns?|reveals?|hands?|eats?|stares?|looks?|backs?)\b", re.I)


def _visible_signature(row: dict[str, Any]) -> tuple[str, frozenset[str]]:
    """A modest surface representation of what the audience can see happen."""
    action = re.sub(r"[^a-z0-9]+", "_", str(row.get("asset_action", "")).lower()).strip("_")
    visible = (str(row.get("visual", "")) + " " + str(row.get("asset_action", ""))).lower()
    verbs = frozenset(m.group(0).lower() for m in _ACTION_VERBS.finditer(visible))
    return action, verbs


def _intensity(text: str) -> int:
    low = (text or "").lower()
    nums = [int(x) for x in re.findall(r"\b\d+\b", low)]
    nums.extend(v for word, v in _INTENSITY.items() if re.search(rf"\b{word}\b", low))
    return max(nums, default=0)


def _story_analysis(rows: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, production-facing checks beyond the legacy quality rubric."""
    voices = [str(r.get("voice", "")) for r in rows]
    visuals = [str(r.get("visual", "")) for r in rows]
    actions = [str(r.get("asset_action", "")).lower() for r in rows]
    alltext = " ".join(voices + visuals).lower()
    # Textual escalation counts only when a visible action/state also advances.
    transitions = []
    for i in range(1, len(rows)):
        prev_sig, sig = _visible_signature(rows[i-1]), _visible_signature(rows[i])
        before = _intensity(voices[i-1] + " " + visuals[i-1])
        after = _intensity(voices[i] + " " + visuals[i])
        visible_change = sig != prev_sig and bool(sig[0] or sig[1])
        materially_worse = after > before or (visible_change and actions[i] != actions[i-1])
        transitions.append({"from": i-1, "to": i, "visible_change": visible_change,
                            "materially_worse": materially_worse})
    escalation_count = sum(t["materially_worse"] for t in transitions)
    fake_escalation = bool(re.search(r"\b(?:more|again|even worse|bigger|another)\b", alltext)) and escalation_count == 0

    reaction_idxs = [i for i,r in enumerate(rows) if str(r.get("beat", "")).lower() == "reaction"
                     or str(r.get("asset_action", "")).lower() in REACTION_ACTIONS]
    reaction_after = any(i > 0 and str(rows[i-1].get("beat", "")).lower() in {"escalation", "attempt"}
                         for i in reaction_idxs)
    twist_idx = next((i for i,r in enumerate(rows) if str(r.get("beat", "")).lower() in {"twist", "reversal"}
                      or (i > 0 and any(w in str(r.get("voice", "")).lower() for w in ("actually", "instead", "misunderstood", "turns out")))), None)
    twist_clear = twist_idx is not None and (bool(rows[twist_idx].get("visual")) or bool(rows[twist_idx].get("asset_action")))
    stop_words = {"the", "and", "was", "were", "with", "that", "this", "from", "then", "for", "you", "pips"}
    twist_words = set(re.findall(r"[a-z]{3,}", (voices[twist_idx] + " " + visuals[twist_idx]).lower())) - stop_words if twist_idx is not None else set()
    planted_words = set(re.findall(r"[a-z]{3,}", " ".join(voices[:twist_idx]).lower())) - stop_words if twist_idx is not None else set()
    twist_planted = bool(twist_words & planted_words) if twist_idx is not None else False
    setup_tokens = set(re.findall(r"[a-z]{4,}", " ".join(voices[:max(1, len(rows)-1)]).lower()))
    payoff_tokens = set(re.findall(r"[a-z]{4,}", voices[-1].lower())) if voices else set()
    dependency = bool(setup_tokens & payoff_tokens) or bool(re.search(r"\b(?:that|it|again|still|back|one)\b", voices[-1].lower() if voices else ""))
    last = rows[-1] if rows else {}
    ending_text = voices[-1].lower() if voices else ""
    loopable = bool(re.search(r"\b(?:again|back|still|one|same|again)\b", ending_text)) or actions and actions[-1] == actions[0]

    def dim(name: str, score: float, evidence: list[str], repair: str) -> dict[str, Any]:
        return asdict(StoryDimension(name, round(max(0, min(10, score)), 2), tuple(evidence),
                                    "" if score >= QUALITY_FLOOR else repair))
    dims = [
        dim("comedy_escalation", 5 + min(escalation_count, 3)*1.5 + (1 if len(rows)>=4 else 0),
            ([f"{escalation_count} beat transitions visibly/materially advance"] if escalation_count else []) + (["fake escalation: wording changes without visible stake/action change"] if fake_escalation else []),
            "Make each successive beat change the visible action or increase a concrete consequence; remove verbal-only 'more/again' escalation."),
        dim("beat_worsening_reversal", 5 + (2 if escalation_count>=2 else 0) + (2 if twist_idx is not None else 0),
            (["multiple adjacent beats worsen or change the situation"] if escalation_count>=2 else []) + (["a reversal is signaled in a beat"] if twist_idx is not None else []),
            "Give the character at least two distinct worsening turns, then reverse the audience's expectation in an identifiable beat."),
        dim("reaction_timing", 5 + (2 if reaction_idxs else 0) + (2 if reaction_after else 0),
            (["reaction is explicitly staged"] if reaction_idxs else []) + (["reaction follows the escalation before payoff"] if reaction_after else []),
            "Place a readable reaction immediately after the consequence and before the punchline."),
        dim("twist_clarity", 5 + (2 if twist_idx is not None else 0) + (2 if twist_clear else 0) - (2 if twist_idx == len(rows)-1 and not twist_planted else 0),
            (["twist/reversal has an identifiable beat"] if twist_idx is not None else []) + (["visual or action makes the reversal legible"] if twist_clear else []) + (["twist arrives in the final beat without an earlier plant"] if twist_idx == len(rows)-1 and not twist_planted else []),
            "State the reversal in a dedicated beat and make its cause visible so the audience can follow it on first watch."),
        dim("payoff_setup_dependency", 5 + (2.5 if dependency else 0) + (1 if str(last.get("beat", "")).lower() in {"payoff", "punchline"} else 0),
            (["ending reuses or resolves an element planted earlier"] if dependency else []) + (["last beat is marked as payoff"] if str(last.get("beat", "")).lower() in {"payoff", "punchline"} else []),
            "Plant a specific prop, phrase, or rule earlier and make the final joke depend on that setup."),
        dim("ending_strength_loopability", 5 + (2 if str(last.get("beat", "")).lower() in {"payoff", "punchline"} else 0) + (1 if ending_text else 0) + (2 if loopable else 0),
            (["ending lands on a payoff beat"] if str(last.get("beat", "")).lower() in {"payoff", "punchline"} else []) + (["final phrase/action echoes the opening for replay or loop"] if loopable else []),
            "End on a decisive punchline or image that echoes the opening and gives the clip a clean replay point."),
    ]
    blockers = [f"story_analysis:{d['name']}:{d['score']:.2f}<{QUALITY_FLOOR:.1f}" for d in dims if d["score"] < QUALITY_FLOOR]
    if fake_escalation:
        blockers.append("story_analysis:fake_escalation:no_visible_stake_or_action_change")
    return {"dimensions": dims, "transitions": transitions, "fake_escalation": fake_escalation,
            "blockers": blockers, "repairs": [d["repair"] for d in dims if d["repair"]]}


class StoryEngine:
    """Build and gate a production-facing story contract."""

    def evaluate(self, candidate: dict[str, Any]) -> StoryQuality:
        rows = _beat_rows(candidate)
        analysis = _story_analysis(rows, candidate)
        dimensions = (
            _score_hook(candidate),
            _score_setup(candidate, rows),
            _score_goal(candidate, rows),
            _score_conflict(rows, candidate),
            _score_escalation(rows, candidate),
            _score_visual(rows),
            _score_surprise(rows, candidate),
            _score_reaction(rows),
            _score_payoff(rows, candidate),
            _score_ending(rows, candidate),
        )
        blockers = tuple(
            f"{dim.name}:{dim.score:.2f}<={QUALITY_FLOOR:.1f}"
            for dim in dimensions
            if dim.score < QUALITY_FLOOR
        )
        repairs = tuple(dim.repair for dim in dimensions if dim.repair)
        overall = round(sum(dim.score for dim in dimensions) / len(dimensions), 2)
        blockers = blockers + tuple(analysis["blockers"])
        repairs = repairs + tuple(analysis["repairs"])
        passed = overall >= QUALITY_FLOOR and not blockers and len(rows) >= 4
        if len(rows) < 4:
            blockers = blockers + ("story_requires_at_least_4_beats",)
            repairs = repairs + ("Create hook, setup, escalation, reaction, and payoff beats.",)
            passed = False
        return StoryQuality(overall, passed, dimensions, blockers, repairs, analysis)

    def build(self, candidate: dict[str, Any]) -> StoryContract:
        quality = self.evaluate(candidate)
        if not quality.passed:
            raise ValueError("Story quality gate failed: " + "; ".join(quality.blockers))

        rows = _beat_rows(candidate)
        hook = str(candidate.get("hook", "")).strip()
        character = str(candidate.get("character", "")).strip()
        first = str(rows[0].get("voice", "")).strip()
        middle = " ".join(str(row.get("voice", "")).strip() for row in rows[1:-1])
        last = str(rows[-1].get("voice", "")).strip()
        title = str(candidate.get("title", "")).strip()
        angle = str(candidate.get("angle", "")).strip()

        desire = first or f"{character} wants a normal, manageable outcome."
        conflict = middle or "A small problem interrupts the character's plan."
        escalation = str(rows[2].get("voice", "")).strip() if len(rows) >= 3 else conflict
        twist = last or "The outcome is worse than expected."
        payoff = last
        ending = last

        normalized_beats = []
        for index, row in enumerate(rows):
            normalized = dict(row)
            normalized["story_role"] = BEAT_ROLE_MAP.get(str(normalized.get("beat", "")).lower(), ("hook", "setup", "escalation", "reaction", "payoff")[min(index, 4)])
            normalized["duration"] = float(normalized.get("duration", candidate.get("duration_target", 8.0) / max(1, len(rows))))
            normalized["story_intent"] = {
                "hook": "create an unanswered question or immediate tension",
                "setup": "establish the character's goal and visible context",
                "escalation": "make the problem larger, stranger, or harder",
                "reaction": "show the character processing the consequence",
                "payoff": "land the reversal and end decisively",
            }.get(normalized["story_role"], "support continuity")
            normalized_beats.append(normalized)

        quality_dict = quality.to_dict()
        quality_dict["quality_floor"] = QUALITY_FLOOR
        quality_dict["dimension_count"] = len(quality.dimensions)
        quality_dict["min_dimension_score"] = min(dim.score for dim in quality.dimensions)
        quality_dict["all_dimensions_above_floor"] = all(dim.score >= QUALITY_FLOOR for dim in quality.dimensions)
        quality_dict["story_analysis"] = quality.story_analysis

        return StoryContract(
            premise=title,
            logline=f"{character} wants {angle.replace('_', ' ')} to stay simple, but a small problem escalates into an unexpected payoff.",
            hook=hook,
            character=character,
            desire=desire,
            conflict=conflict,
            escalation=escalation,
            twist=twist,
            payoff=payoff,
            ending=ending,
            beats=tuple(normalized_beats),
            duration_target=float(candidate.get("duration_target", 8.0)),
            quality=quality_dict,
        )
