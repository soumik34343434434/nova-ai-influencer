"""Provider-neutral, metadata-only edit decisions for short-form video."""

from __future__ import annotations

from typing import Any


def build_edit_plan(beats: list[dict[str, Any]], durations: list[float] | None = None) -> dict[str, Any]:
    """Map finished beats to a deterministic 9:16 edit plan; never touches media."""
    rows = [row for row in beats if isinstance(row, dict)]
    if not rows:
        return {
            "version": 1,
            "provider": "neutral",
            "aspect_ratio": "9:16",
            "duration_seconds": 0.0,
            "shots": [],
            "cut_points_seconds": [],
            "hook": {"start_seconds": 0.0, "end_seconds": 0.0, "shot_index": None},
            "payoff": {"start_seconds": 0.0, "end_seconds": 0.0, "shot_index": None},
            "reaction_holds": [],
            "loop": {"enabled": False, "from_seconds": 0.0, "to_shot_index": None},
        }

    source_durations = durations if durations is not None else [float(row.get("duration", 2.0)) for row in rows]
    clock = 0.0
    shots = []
    cuts = []
    reactions = []
    payoff = None
    # The hook is an editorial position: the opening shot must carry it even
    # when upstream metadata labels that beat as setup or escalation.
    hook_index = 0
    for index, row in enumerate(rows):
        raw = source_durations[index] if index < len(source_durations) else row.get("duration", 2.0)
        duration = max(0.25, float(raw or 0.25))
        start, end = round(clock, 3), round(clock + duration, 3)
        role = str(row.get("story_role", row.get("beat", ""))).lower()
        shot = {
            "shot_index": index,
            "story_role": role,
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": round(duration, 3),
            "pacing": "hold" if role == "reaction" else ("land" if role in {"payoff", "punchline"} else "cut"),
            "transition_in": "hard_cut" if index else "open",
            "caption": {
                "text": str(row.get("voice", row.get("dialogue", ""))),
                "start_seconds": start,
                "end_seconds": end,
                "placement": "lower_third_safe",
                "anchor": {"x": 0.5, "y": 0.72},
                "safe_bounds": {"x_min": 0.08, "x_max": 0.92, "y_min": 0.58, "y_max": 0.82},
            },
        }
        shots.append(shot)
        if index:
            cuts.append(start)
        if role == "reaction":
            reactions.append({"shot_index": index, "start_seconds": start, "end_seconds": end, "hold_seconds": round(min(0.8, duration), 3)})
        if role in {"payoff", "punchline"}:
            payoff = {"start_seconds": start, "end_seconds": end, "shot_index": index}
        clock += duration

    first = shots[hook_index]
    plan = {
        "version": 1,
        "provider": "neutral",
        "aspect_ratio": "9:16",
        "duration_seconds": round(clock, 3),
        "pacing_constraints": {"min_shot_seconds": 0.25, "max_shot_seconds": None, "ordering": "source_beat_order"},
        "shots": shots,
        "cut_points_seconds": cuts,
        "hook": {"start_seconds": 0.0, "end_seconds": round(min(1.0, first["end_seconds"]), 3), "shot_index": hook_index},
        "payoff": payoff or {"start_seconds": round(clock, 3), "end_seconds": round(clock, 3), "shot_index": None},
        "reaction_holds": reactions,
        "transitions": {"default": "hard_cut", "effects": []},
        "loop": {"enabled": True, "from_seconds": round(clock, 3), "to_shot_index": 0, "match_opening": True},
    }
    return plan
