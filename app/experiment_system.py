"""Deterministic, local experiment planning and outcome registry.

Planning never publishes content. A planned variant can be attached to a run,
then follows that run through the review outbox and analytics metadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_FILE = ROOT / "artifacts" / "cartoon" / "experiment_registry.json"
DIMENSIONS = {"hook", "opening", "character", "format", "duration", "ending"}


def _read(path: Path = REGISTRY_FILE) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("experiments"), dict):
            return value
    except (OSError, json.JSONDecodeError):
        pass
    return {"version": 1, "experiments": {}}


def _write(data: dict[str, Any], path: Path = REGISTRY_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _digest(value: str, size: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:size]


def plan_experiment(*, dimension: str, variants: Iterable[Any], hypothesis: str,
                    success_metric: str, min_observations: int = 3,
                    max_observations: int = 30, stop_threshold: float = 0.0,
                    mode: str = "exploration", context: str = "",
                    path: Path = REGISTRY_FILE) -> dict[str, Any]:
    dimension = str(dimension).lower()
    if dimension not in DIMENSIONS:
        raise ValueError(f"unsupported dimension: {dimension}")
    values = list(dict.fromkeys(variants))
    if len(values) < 2:
        raise ValueError("an experiment requires at least two distinct variants")
    if not hypothesis.strip() or not success_metric.strip():
        raise ValueError("one hypothesis and success metric are required")
    if min_observations < 1 or max_observations < min_observations:
        raise ValueError("invalid observation limits")
    if mode not in {"exploration", "exploitation"}:
        raise ValueError("mode must be exploration or exploitation")
    signature = json.dumps([dimension, values, hypothesis.strip(), success_metric.strip(), context], sort_keys=True, ensure_ascii=False)
    experiment_id = "plan_" + _digest(signature)
    data = _read(path)
    old = data["experiments"].get(experiment_id)
    if old:
        return old
    record = {
        "experiment_id": experiment_id, "dimension": dimension,
        "hypothesis": hypothesis.strip(), "success_metric": success_metric.strip(),
        "mode": mode, "context": context,
        "variants": [{"variant_id": f"var_{_digest(experiment_id + ':' + json.dumps(v, sort_keys=True, ensure_ascii=False))}", "value": v} for v in values],
        "min_observations": int(min_observations), "max_observations": int(max_observations),
        "stop_threshold": float(stop_threshold), "observations": 0, "outcomes": [],
        "status": "planned",
    }
    data["experiments"][experiment_id] = record
    _write(data, path)
    return record


def assign_variant(experiment: dict[str, Any], identity: str) -> dict[str, Any]:
    """Stable assignment; the same identity always receives the same variant."""
    variants = experiment.get("variants", [])
    if not variants:
        raise ValueError("experiment has no variants")
    selected = variants[int(_digest(f"{experiment['experiment_id']}:{identity}", 32), 16) % len(variants)]
    return {"experiment_id": experiment["experiment_id"], "variant_id": selected["variant_id"],
            "dimension": experiment["dimension"], "variant_value": selected["value"]}


def record_observation(experiment_id: str, *, variant_id: str, metric_value: float,
                       content_id: str = "", run_id: str = "", analytics_id: str = "",
                       path: Path = REGISTRY_FILE) -> dict[str, Any]:
    data = _read(path)
    if experiment_id not in data["experiments"]:
        raise KeyError(experiment_id)
    record = data["experiments"][experiment_id]
    if record["status"] in {"stopped", "completed"}:
        return record
    if variant_id not in {v["variant_id"] for v in record["variants"]}:
        raise ValueError("variant_id does not belong to experiment")
    if content_id and analytics_id and any(
        outcome.get("content_id") == content_id and outcome.get("analytics_id") == analytics_id
        for outcome in record["outcomes"]
    ):
        return record
    record["outcomes"].append({"variant_id": variant_id, "metric_value": float(metric_value),
                               "content_id": content_id, "run_id": run_id, "analytics_id": analytics_id})
    record["observations"] = len(record["outcomes"])
    # A stop threshold is a minimum absolute gap between the best two variant means.
    groups = {v["variant_id"]: [] for v in record["variants"]}
    for outcome in record["outcomes"]:
        groups[outcome["variant_id"]].append(outcome["metric_value"])
    means = sorted((sum(xs) / len(xs) for xs in groups.values() if xs), reverse=True)
    if record["observations"] >= record["max_observations"]:
        record["status"] = "completed"
    elif record["observations"] >= record["min_observations"] and len(means) > 1 and means[0] - means[1] >= record["stop_threshold"] > 0:
        record["status"] = "stopped"
    else:
        record["status"] = "running"
    data["experiments"][experiment_id] = record
    _write(data, path)
    return record
