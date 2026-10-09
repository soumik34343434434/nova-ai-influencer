"""Durable agent memory for NOVA's self-improving loop.

This module keeps experiment lineage, failure memory, provider observations and
the next-loop decision plan separate from transient run checkpoints.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "cartoon"
EXPERIMENTS_FILE = ARTIFACTS / "experiments.json"
FAILURES_FILE = ARTIFACTS / "failure_memory.json"
PROVIDERS_FILE = ARTIFACTS / "provider_memory.json"
NEXT_LOOP_FILE = ARTIFACTS / "next_loop_plan.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def experiment_id_for(run_id: str) -> str:
    digest = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()[:16]
    return f"exp_{digest}"


def _load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value
    except (OSError, json.JSONDecodeError):
        return default


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def ensure_experiment_identity(payload: dict[str, Any], run_id: str) -> dict[str, Any]:
    identity = dict(payload.get("experiment", {}))
    identity.setdefault("experiment_id", experiment_id_for(run_id))
    identity.setdefault("run_id", run_id)
    identity.setdefault("created_at", utc_now())
    payload["experiment"] = identity
    return identity


def upsert_experiment(record: dict[str, Any]) -> dict[str, Any]:
    data = _load(EXPERIMENTS_FILE, {"version": 1, "records": {}})
    if not isinstance(data, dict):
        data = {"version": 1, "records": {}}
    records = data.setdefault("records", {})
    experiment_id = str(record.get("experiment_id", ""))
    if not experiment_id:
        raise ValueError("experiment_id is required")
    existing = dict(records.get(experiment_id, {}))
    existing.update(record)
    existing["updated_at"] = utc_now()
    records[experiment_id] = existing
    data["updated_at"] = existing["updated_at"]
    _write(EXPERIMENTS_FILE, data)
    return existing


def record_failure(
    *,
    run_id: str,
    experiment_id: str,
    stage: str,
    error_type: str,
    message: str,
    attempt: int,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = _load(FAILURES_FILE, {"version": 1, "records": []})
    if not isinstance(data, dict):
        data = {"version": 1, "records": []}
    row = {
        "failure_id": hashlib.sha256(
            f"{run_id}:{stage}:{attempt}:{error_type}:{message}".encode("utf-8")
        ).hexdigest()[:20],
        "captured_at": utc_now(),
        "run_id": run_id,
        "experiment_id": experiment_id,
        "stage": stage,
        "error_type": error_type,
        "message": message[:2000],
        "attempt": int(attempt),
        "context": context or {},
    }
    rows = [x for x in data.get("records", []) if x.get("failure_id") != row["failure_id"]]
    rows.append(row)
    data["records"] = rows[-2000:]
    data["updated_at"] = row["captured_at"]
    _write(FAILURES_FILE, data)
    return row


def record_provider_observation(
    *,
    provider: str,
    task: str,
    success: bool,
    model: str = "",
    experiment_id: str = "",
    cost: float = 0.0,
    repaired: bool = False,
    quality_score: float | None = None,
    failure_types: Iterable[str] = (),
) -> dict[str, Any]:
    data = _load(PROVIDERS_FILE, {"version": 1, "records": []})
    if not isinstance(data, dict):
        data = {"version": 1, "records": []}
    row = {
        "captured_at": utc_now(),
        "provider": provider,
        "task": task,
        "model": model,
        "experiment_id": experiment_id,
        "success": bool(success),
        "cost": float(cost or 0.0),
        "repaired": bool(repaired),
        "quality_score": None if quality_score is None else float(quality_score),
        "failure_types": sorted({str(x) for x in failure_types if x}),
    }
    data["records"] = list(data.get("records", []))[-2000:] + [row]
    data["updated_at"] = row["captured_at"]
    _write(PROVIDERS_FILE, data)
    return row


def summarize_failures() -> dict[str, Any]:
    data = _load(FAILURES_FILE, {"records": []})
    rows = list(data.get("records", [])) if isinstance(data, dict) else []
    counts: dict[str, int] = {}
    stages: dict[str, int] = {}
    for row in rows:
        key = str(row.get("error_type", "UnknownError"))
        counts[key] = counts.get(key, 0) + 1
        stage = str(row.get("stage", "unknown"))
        stages[stage] = stages.get(stage, 0) + 1
    return {
        "observations": len(rows),
        "by_error_type": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "by_stage": dict(sorted(stages.items(), key=lambda item: (-item[1], item[0]))),
    }


def summarize_providers() -> dict[str, Any]:
    data = _load(PROVIDERS_FILE, {"records": []})
    rows = list(data.get("records", [])) if isinstance(data, dict) else []
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        key = f"{row.get('provider','')}|{row.get('task','')}|{row.get('model','')}"
        item = grouped.setdefault(key, {"observations": 0, "successes": 0, "repairs": 0, "cost": 0.0})
        item["observations"] += 1
        item["successes"] += int(bool(row.get("success")))
        item["repairs"] += int(bool(row.get("repaired")))
        item["cost"] += float(row.get("cost", 0.0) or 0.0)
    for item in grouped.values():
        item["success_rate"] = round(item["successes"] / max(item["observations"], 1), 4)
        item["repair_rate"] = round(item["repairs"] / max(item["observations"], 1), 4)
        item["avg_cost"] = round(item["cost"] / max(item["observations"], 1), 4)
    return {"observations": len(rows), "by_provider_task_model": grouped}


def build_next_loop_plan(
    strategy_memory: dict[str, Any],
    *,
    failure_summary: dict[str, Any] | None = None,
    provider_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strategy = strategy_memory.get("strategy", {})
    posteriors = strategy_memory.get("posteriors", {})
    exploration = float(strategy.get("exploration_rate", 0.30) or 0.30)

    # Keep the primary exploitation policy focused on the current leader.
    # Other formats remain eligible through the explicit exploration experiments.
    preferred_formats = list(strategy.get("preferred_formats", []))[:1]
    preferred_hooks = list(strategy.get("preferred_hooks", []))[:3]
    preferred_emotions = list(strategy.get("preferred_emotions", []))[:3]
    preferred_characters = list(strategy.get("preferred_characters", []))[:3]

    experiments = []
    for fmt in preferred_formats[:2]:
        for character in preferred_characters[:2]:
            if fmt and character:
                experiments.append({
                    "type": "reinforcement",
                    "format_name": fmt,
                    "character": character,
                    "reason": "high-UCB learned direction",
                })

    known_format_pool = [
        "dance", "trend_audio", "story_comedy", "reaction", "pov",
        "cute_animal", "challenge", "transformation", "absurdist", "listicle",
    ]
    unknown_formats = [
        name for name, stats in posteriors.get("format_name", {}).items()
        if int(stats.get("observations", 0)) < 2
    ]
    for fmt in [*unknown_formats, *known_format_pool]:
        if fmt and fmt not in {x.get("format_name") for x in experiments}:
            experiments.append({
                "type": "information_gain",
                "format_name": fmt,
                "reason": "under-tested or unexplored format",
            })
        if len([x for x in experiments if x.get("type") == "information_gain"]) >= 3:
            break

    # Explicitly keep multiple experiment dimensions alive so NOVA can learn
    # whether the result came from format, hook, character or duration.
    experiments.extend([
        {"type": "factor_test", "dimension": "hook_pattern", "values": preferred_hooks[:2], "reason": "separate hook effect from format effect"},
        {"type": "factor_test", "dimension": "duration", "values": [6, 8, 10], "reason": "test retention/pacing trade-off"},
    ])

    # Turn the policy suggestions into durable, deduplicated controlled plans.
    # These are assignments for future/local runs; planning never publishes.
    from app.experiment_system import plan_experiment
    for suggestion in experiments:
        dimension = suggestion.get("dimension")
        values = suggestion.get("values") or []
        if not dimension:
            dimension = "format" if suggestion.get("format_name") else "character" if suggestion.get("character") else "hook"
            value = suggestion.get("format_name", suggestion.get("character", ""))
            values = [value, f"alternative:{value}"] if value else []
        if len(values) < 2:
            continue
        plan = plan_experiment(
            dimension=dimension, variants=values,
            hypothesis=f"Changing {dimension} changes the success metric",
            success_metric="normalized_score", mode="exploration" if suggestion.get("type") != "reinforcement" else "exploitation",
            context=str(suggestion.get("format_name", "")),
        )
        suggestion["experiment_id"] = plan["experiment_id"]
        suggestion["variants"] = plan["variants"]
        suggestion["hypothesis"] = plan["hypothesis"]
        suggestion["success_metric"] = plan["success_metric"]

    failure_summary = failure_summary or summarize_failures()
    provider_summary = provider_summary or summarize_providers()

    dominant_failures = list(failure_summary.get("by_error_type", {}).items())[:3]
    provider_rows = provider_summary.get("by_provider_task_model", {})
    preferred_provider_routes = [
        {
            "route": key,
            "success_rate": float(value.get("success_rate", 0.0)),
            "repair_rate": float(value.get("repair_rate", 0.0)),
            "avg_cost": float(value.get("avg_cost", 0.0)),
        }
        for key, value in sorted(
            provider_rows.items(),
            key=lambda item: (
                -float(item[1].get("success_rate", 0.0)),
                float(item[1].get("repair_rate", 0.0)),
                float(item[1].get("avg_cost", 0.0)),
            ),
        )[:5]
    ]

    plan = {
        "version": 1,
        "generated_at": utc_now(),
        "objective": "maximize expected audience outcome and information gain while reducing avoidable production failure and cost",
        "exploration_rate": exploration,
        "reinforce": {
            "formats": preferred_formats,
            "hooks": preferred_hooks,
            "emotions": preferred_emotions,
            "characters": preferred_characters,
        },
        "experiments": experiments[:8],
        "failure_prevention": [
            {"error_type": name, "occurrences": count, "action": "prevent_before_generation"}
            for name, count in dominant_failures
        ],
        "provider_routing": preferred_provider_routes,
        "guardrails": [
            "real platform metrics outrank model predictions",
            "do not treat missing analytics as zero performance",
            "preserve exploration while sample sizes are small",
            "avoid expensive renders without a measurable hypothesis",
        ],
    }
    _write(NEXT_LOOP_FILE, plan)
    # Make the plan directly consumable by the next generation pass. This is
    # deliberately a small policy surface rather than prose: future stages can
    # change behavior from these fields without reinterpreting free-form text.
    plan["policy"] = {
        "preferred_formats": preferred_formats,
        "preferred_hooks": preferred_hooks,
        "preferred_emotions": preferred_emotions,
        "preferred_characters": preferred_characters,
        "exploration_rate": exploration,
        "avoid_error_types": [name for name, _ in dominant_failures],
        "provider_routes": preferred_provider_routes,
    }
    return plan


def update_experiment_learning(
    record: dict[str, Any],
    *,
    strategy_memory: dict[str, Any],
    next_loop_plan: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(record)
    merged["learning"] = {
        "updated_at": utc_now(),
        "observations": strategy_memory.get("observations", 0),
        "preferred_formats": strategy_memory.get("strategy", {}).get("preferred_formats", []),
        "preferred_hooks": strategy_memory.get("strategy", {}).get("preferred_hooks", []),
        "next_loop_experiments": next_loop_plan.get("experiments", []),
        "guardrails": next_loop_plan.get("guardrails", []),
    }
    return upsert_experiment(merged)
