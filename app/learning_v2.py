"""Experiment-aware learning memory for NOVA.

Uses normalized content scores plus lightweight Bayesian/UCB-style posteriors.
The goal is to learn durable tendencies without overfitting one viral outlier.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = ROOT / "artifacts" / "cartoon" / "metrics.json"
METRICS_HISTORY_FILE = ROOT / "artifacts" / "cartoon" / "metrics_history.json"
STRATEGY_FILE = ROOT / "artifacts" / "cartoon" / "strategy_memory.json"


@dataclass
class ExperimentMetric:
    content_id: str
    platform: str
    title: str = ""
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    retention_pct: float = 0.0
    avg_watch_seconds: float = 0.0
    duration_seconds: float = 0.0
    followers_gained: int = 0
    captured_at: str = ""
    published_at: str = ""
    experiment_id: str = ""
    format_name: str = ""
    hook_pattern: str = ""
    emotion: str = ""
    character: str = ""
    trend_id: str = ""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_metric(row: dict[str, Any]) -> dict[str, Any]:
    views = max(int(row.get("views", 0) or 0), 1)
    duration = max(float(row.get("duration_seconds", 0) or 0), 0.1)
    avg_watch = float(row.get("avg_watch_seconds", 0) or 0)
    retention = float(row.get("retention_pct", 0) or 0)

    # When native retention is unavailable, derive a conservative watch-ratio
    # from average watch time. Derived values are explicitly marked.
    derived_retention = False
    if retention <= 0 and avg_watch > 0:
        retention = min(100.0, avg_watch / duration * 100.0)
        derived_retention = True

    share_rate = int(row.get("shares", 0) or 0) / views
    save_rate = int(row.get("saves", 0) or 0) / views
    comment_rate = int(row.get("comments", 0) or 0) / views
    like_rate = int(row.get("likes", 0) or 0) / views
    follow_rate = int(row.get("followers_gained", 0) or 0) / views

    # Deliberately reward deep actions more than passive likes.
    score = (
        min(retention / 100.0, 1.0) * 0.42
        + min(share_rate * 100.0, 1.0) * 0.24
        + min(save_rate * 100.0, 1.0) * 0.12
        + min(comment_rate * 100.0, 1.0) * 0.07
        + min(like_rate * 20.0, 1.0) * 0.08
        + min(follow_rate * 200.0, 1.0) * 0.07
    )
    normalized = dict(row)
    normalized.update({
        "retention_pct_normalized": round(retention, 3),
        "score": round(score, 6),
        "share_rate": round(share_rate, 8),
        "save_rate": round(save_rate, 8),
        "comment_rate": round(comment_rate, 8),
        "like_rate": round(like_rate, 8),
        "follow_rate": round(follow_rate, 8),
        "derived_retention": derived_retention,
        "has_observed_analytics": any(
            key in row and row.get(key) is not None
            for key in ("views", "likes", "comments", "shares", "saves", "retention_pct", "avg_watch_seconds")
        ),
    })
    return normalized


def _snapshot_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("content_id", "")),
        str(row.get("platform", "")),
        str(row.get("captured_at", "") or ""),
    )


def load_metric_history() -> list[dict[str, Any]]:
    return _load_rows(METRICS_HISTORY_FILE)


def ingest(rows: Iterable[dict[str, Any] | ExperimentMetric]) -> list[dict[str, Any]]:
    existing = _load_rows(METRICS_FILE)
    history = load_metric_history()
    latest_by_key = {(str(x.get("content_id")), str(x.get("platform"))): x for x in existing}
    history_by_key = {_snapshot_key(x): x for x in history}
    for item in rows:
        row = normalize_metric(asdict(item) if isinstance(item, ExperimentMetric) else dict(item))
        content_key = (str(row.get("content_id", "")), str(row.get("platform", "")))
        # Keep a durable snapshot history, but deduplicate an exact capture.
        history_by_key[_snapshot_key(row)] = row
        current = latest_by_key.get(content_key)
        current_at = str(current.get("captured_at", "")) if current else ""
        new_at = str(row.get("captured_at", ""))
        if current is None or new_at >= current_at:
            latest_by_key[content_key] = row
    latest = list(latest_by_key.values())
    history = list(history_by_key.values())
    history.sort(key=lambda x: (str(x.get("captured_at", "")), str(x.get("content_id", "")), str(x.get("platform", ""))))
    _write_json(METRICS_FILE, latest)
    _write_json(METRICS_HISTORY_FILE, history)
    return latest


def _beta_posterior(values: list[float], prior_alpha: float = 2.0, prior_beta: float = 2.0) -> tuple[float, float, float]:
    """Return posterior alpha, beta and mean for bounded [0,1] outcomes."""
    if not values:
        mean = prior_alpha / (prior_alpha + prior_beta)
        return prior_alpha, prior_beta, mean
    successes = sum(max(0.0, min(1.0, v)) for v in values)
    failures = len(values) - successes
    alpha = prior_alpha + successes
    beta = prior_beta + failures
    return alpha, beta, alpha / (alpha + beta)


def _ucb(mean: float, observations: int, total_observations: int) -> float:
    if observations <= 0:
        return 1.0
    bonus = math.sqrt(2.0 * math.log(max(total_observations, 2)) / observations)
    return min(1.0, mean + 0.18 * bonus)


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key, "") or "").strip()
        if value:
            groups[value].append(float(row.get("score", 0.0) or 0.0))
    total = max(len(rows), 1)
    result = {}
    for value, vals in groups.items():
        alpha, beta, mean = _beta_posterior(vals)
        result[value] = {
            "observations": len(vals),
            "posterior_mean": round(mean, 4),
            "ucb": round(_ucb(mean, len(vals), total), 4),
            "raw_mean": round(sum(vals) / len(vals), 4),
        }
    return result


def _evidence_summary(rows: list[dict[str, Any]], history: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe outcome evidence separately from the legacy selection posterior."""
    observed_fields = ("views", "likes", "comments", "shares", "saves", "retention_pct",
                      "avg_watch_seconds", "followers_gained")
    evidence_rows = []
    for row in rows:
        supplied = [name for name in observed_fields if row.get(name) not in (None, "")]
        proxy = (not supplied or str(row.get("platform", "")).lower() in {"sim", "simulation", "synthetic"}
                 or str(row.get("metric_source", "")).lower() in {"proxy", "synthetic", "simulated"})
        evidence_rows.append({**row, "_evidence_kind": "proxy" if proxy else "observed"})

    def comparison(key: str, value: str) -> dict[str, Any]:
        selected_all = [r for r in evidence_rows if str(r.get(key, "") or "").strip() == value]
        # Compare within platform when possible; this avoids treating platform mix as a creative effect.
        platforms = Counter(str(r.get("platform", "") or "") for r in selected_all)
        platform = platforms.most_common(1)[0][0] if platforms else ""
        pool = [r for r in evidence_rows if not platform or str(r.get("platform", "") or "") == platform]
        selected = [r for r in pool if str(r.get(key, "") or "").strip() == value]
        baseline = [r for r in pool if str(r.get(key, "") or "").strip() != value]
        vals = [float(r.get("score", 0) or 0) for r in selected]
        base_vals = [float(r.get("score", 0) or 0) for r in baseline]
        mean = sum(vals) / len(vals) if vals else None
        base_mean = sum(base_vals) / len(base_vals) if base_vals else None
        # Descriptive standard error over bounded normalized scores, not a causal interval.
        se = (math.sqrt(sum((x - mean) ** 2 for x in vals) / max(len(vals) - 1, 1) / len(vals))
              if len(vals) > 1 and mean is not None else None)
        return {"observations": len(vals), "observed_outcomes": sum(r["_evidence_kind"] == "observed" for r in selected),
                "proxy_outcomes": sum(r["_evidence_kind"] == "proxy" for r in selected),
                "mean": round(mean, 4) if mean is not None else None,
                "standard_error": round(se, 4) if se is not None else None,
                "baseline_observations": len(base_vals),
                "baseline_mean": round(base_mean, 4) if base_mean is not None else None,
                "baseline_standard_error": round(math.sqrt(sum((x - base_mean) ** 2 for x in base_vals) / max(len(base_vals) - 1, 1) / len(base_vals)), 4) if len(base_vals) > 1 and base_mean is not None else None,
                "difference": round(mean - base_mean, 4) if mean is not None and base_mean is not None else None,
                "comparison": "association"}

    comparisons = {}
    for key in ("format_name", "hook_pattern", "emotion", "character"):
        comparisons[key] = {value: comparison(key, value) for value in sorted({str(r.get(key, "") or "").strip() for r in evidence_rows} - {""})}

    ordered = sorted(history, key=lambda r: (str(r.get("captured_at", "")), str(r.get("content_id", ""))))
    timelines: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        combination = "|".join(str(row.get(k, "") or "").strip() for k in ("format_name", "hook_pattern", "character"))
        if combination.strip("|"):
            timelines[combination].append(normalize_metric(row))
    declining = []
    for combo, snapshots in timelines.items():
        if len(snapshots) < 3:
            continue
        split = max(1, len(snapshots) // 2)
        early = [float(r.get("score", 0) or 0) for r in snapshots[:split]]
        recent = [float(r.get("score", 0) or 0) for r in snapshots[split:]]
        delta = sum(recent) / len(recent) - sum(early) / len(early)
        if delta <= -0.08:
            declining.append({"combination": combo, "snapshots": len(snapshots), "early_mean": round(sum(early)/len(early), 4),
                              "recent_mean": round(sum(recent)/len(recent), 4), "change": round(delta, 4), "interpretation": "temporal association; fatigue is not established"})
    winners, losers, under_tested = [], [], []
    for dimension, values in comparisons.items():
        for value, stats in values.items():
            item = {"dimension": dimension, "value": value, **stats,
                    "confidence": "low" if stats["observations"] < 3 or stats["baseline_observations"] < 3 else "moderate"}
            if stats["observations"] < 3:
                under_tested.append(item)
            if stats["difference"] is not None and stats["observations"] >= 2:
                (winners if stats["difference"] >= 0.05 else losers if stats["difference"] <= -0.05 else []).append(item)
    return {"outcome_evidence": {"observed": sum(r["_evidence_kind"] == "observed" for r in evidence_rows),
                                   "proxy": sum(r["_evidence_kind"] == "proxy" for r in evidence_rows),
                                   "observations": len(evidence_rows)},
            "comparisons": comparisons,
            "learnings": {"winners": winners, "losers": losers, "under_tested": under_tested,
                          "declining_patterns": declining,
                          "confidence_note": "Associations are descriptive; they do not establish causality."}}


def build_strategy_memory(rows: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    data = list(rows) if rows is not None else _load_rows(METRICS_FILE)
    normalized = [normalize_metric(x) for x in data]
    if not normalized:
        memory = {
            "version": 2,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "observations": 0,
            "analytics_observations": 0,
            "strategy": {
                "exploration_rate": 0.30,
                "preferred_formats": [],
                "preferred_hooks": [],
                "preferred_emotions": [],
                "preferred_characters": [],
            },
            "posteriors": {},
            "evidence": _evidence_summary([], load_metric_history()),
            "lessons": ["collect first-party platform metrics before claiming a creative winner"],
        }
        _write_json(STRATEGY_FILE, memory)
        return memory

    ranked = sorted(normalized, key=lambda r: float(r.get("score", 0.0)), reverse=True)
    posteriors = {
        key: _group(normalized, key)
        for key in ("format_name", "hook_pattern", "emotion", "character", "trend_id")
    }
    # Duration is learned as a stable, interpretable band for next-cycle planning.
    duration_rows = []
    for row in normalized:
        try:
            duration = float(row.get("duration_seconds", 0) or 0)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            band = "short_0_8s" if duration <= 8 else "medium_9_15s" if duration <= 15 else "long_16s_plus"
            duration_rows.append({**row, "duration_band": band})
    posteriors["duration_band"] = _group(duration_rows, "duration_band")
    observed_rows = [row for row in normalized if row.get("has_observed_analytics")]
    observed_posteriors = {
        key: _group(observed_rows, key)
        for key in ("format_name", "hook_pattern", "emotion", "character")
    }
    observed_duration_rows = [row for row in duration_rows if row.get("has_observed_analytics")]
    observed_posteriors["duration_band"] = _group(observed_duration_rows, "duration_band")
    interactions = {}
    for left, right in (("format_name", "hook_pattern"), ("format_name", "character"), ("hook_pattern", "emotion")):
        groups = defaultdict(list)
        for row in normalized:
            a = str(row.get(left, "") or "").strip()
            b = str(row.get(right, "") or "").strip()
            if a and b:
                groups[f"{a}|{b}"].append(float(row.get("score", 0.0) or 0.0))
        table = {}
        for value, vals in groups.items():
            alpha, beta, mean = _beta_posterior(vals)
            table[value] = {
                "observations": len(vals),
                "posterior_mean": round(mean, 4),
                "ucb": round(_ucb(mean, len(vals), len(normalized)), 4),
                "raw_mean": round(sum(vals) / len(vals), 4),
            }
        interactions[f"{left}__{right}"] = table
    posteriors["interactions"] = interactions

    def top_keys(key: str, n: int = 4) -> list[str]:
        table = posteriors[key]
        return [
            item[0]
            for item in sorted(
                table.items(),
                key=lambda kv: (kv[1]["ucb"], kv[1]["observations"]),
                reverse=True,
            )[:n]
        ]

    preferred_formats = top_keys("format_name")
    preferred_hooks = top_keys("hook_pattern")
    preferred_emotions = top_keys("emotion")
    preferred_characters = top_keys("character")

    exploration = 0.30
    if len(normalized) >= 20:
        exploration = 0.22
    if len(normalized) >= 50:
        exploration = 0.16

    lessons: list[str] = []
    if float(ranked[0].get("retention_pct_normalized", 0)) >= 75:
        lessons.append("preserve hooks and structures that sustain three-quarter-plus completion")
    if float(ranked[0].get("share_rate", 0)) > float(ranked[0].get("like_rate", 0)):
        lessons.append("prioritize shareable visual premises over like-only bait")
    if float(ranked[-1].get("retention_pct_normalized", 100)) < 45:
        lessons.append("shorten or front-load weak openings")
    lessons.append("use posterior uncertainty to keep exploring under-tested formats")

    memory = {
        "version": 2,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "observations": len(normalized),
        "strategy": {
            "exploration_rate": exploration,
            "preferred_formats": preferred_formats,
            "preferred_hooks": preferred_hooks,
            "preferred_emotions": preferred_emotions,
            "preferred_characters": preferred_characters,
        },
        "posteriors": posteriors,
        "evidence": _evidence_summary(normalized, load_metric_history()),
        "observed_posteriors": observed_posteriors,
        "analytics_observations": sum(bool(r.get("has_observed_analytics")) for r in normalized),
        "leaders": [
            {
                "content_id": r.get("content_id"),
                "platform": r.get("platform"),
                "title": r.get("title"),
                "score": r.get("score"),
                "retention_pct": r.get("retention_pct_normalized"),
            }
            for r in ranked[:5]
        ],
        "lessons": lessons,
    }
    _write_json(STRATEGY_FILE, memory)
    return memory


def score_candidate(candidate: dict[str, Any], memory: dict[str, Any] | None = None) -> float:
    memory = memory or (json.loads(STRATEGY_FILE.read_text()) if STRATEGY_FILE.exists() else build_strategy_memory([]))
    strategy = memory.get("strategy", {})
    post = memory.get("posteriors", {})
    exploration = float(strategy.get("exploration_rate", 0.30))

    features = (
        ("format_name", str(candidate.get("format_name", ""))),
        ("hook_pattern", str(candidate.get("hook_pattern", ""))),
        ("emotion", str(candidate.get("emotion", ""))),
        ("character", str(candidate.get("character", ""))),
    )
    known = []
    for key, value in features:
        if value and value in post.get(key, {}):
            known.append(float(post[key][value].get("ucb", 0.5)))
        elif value:
            known.append(0.55 + exploration * 0.25)

    exploitation = sum(known) / len(known) if known else 0.5
    interaction_values = []
    interaction_map = post.get("interactions", {})
    for left, right in (("format_name", "hook_pattern"), ("format_name", "character"), ("hook_pattern", "emotion")):
        a = str(candidate.get(left, "") or "").strip()
        b = str(candidate.get(right, "") or "").strip()
        if a and b:
            row = interaction_map.get(f"{left}__{right}", {}).get(f"{a}|{b}")
            if row:
                interaction_values.append(float(row.get("ucb", row.get("posterior_mean", 0.5))))
    interaction_score = sum(interaction_values) / len(interaction_values) if interaction_values else exploitation
    exploitation = 0.82 * exploitation + 0.18 * interaction_score
    novelty = 1.0 - float(candidate.get("saturation_score", 0.0) or 0.0)
    trend = float(candidate.get("trend_score", 0.0) or 0.0)
    cost = float(candidate.get("production_cost", 0.5) or 0.5)
    return round(
        0.38 * exploitation
        + 0.28 * trend
        + 0.18 * novelty
        + 0.10 * min(1.0, exploration + 0.25)
        + 0.06 * (1.0 - cost),
        6,
    )


def select_portfolio(candidates: Iterable[dict[str, Any]], memory: dict[str, Any] | None = None,
                     limit: int = 12) -> list[dict[str, Any]]:
    rows = [dict(x) for x in candidates]
    memory = memory or build_strategy_memory()
    for row in rows:
        row["learning_score"] = score_candidate(row, memory)
    rows.sort(key=lambda x: (-float(x["learning_score"]), str(x.get("idea_id", ""))))

    # Diversity-aware portfolio: don't let one format/character consume all slots.
    chosen = []
    counts: Counter[str] = Counter()
    for row in rows:
        key = f"{row.get('format_name','')}|{row.get('character','')}"
        penalty = counts[key] / max(1, limit)
        row["portfolio_score"] = round(float(row["learning_score"]) - 0.12 * penalty, 6)
        chosen.append(row)
        counts[key] += 1
    chosen.sort(key=lambda x: (-float(x["portfolio_score"]), str(x.get("idea_id", ""))))
    return chosen[:max(1, limit)]
