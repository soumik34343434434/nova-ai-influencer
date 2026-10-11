"""Persistent trend snapshot history for velocity and saturation inference."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from app.agents.trend_intelligence_v2 import recompute_opportunity_score


ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = ROOT / "artifacts" / "cartoon" / "trend_history.json"


def _load() -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        value = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def snapshot(signals: Iterable[dict[str, Any]], max_rows: int = 5000) -> list[dict[str, Any]]:
    history = _load()
    now = datetime.now(timezone.utc).isoformat()
    rows = list(history)
    for signal in signals:
        rows.append({
            "captured_at": now,
            "trend_id": str(signal.get("trend_id", "")),
            "title": str(signal.get("title", "")),
            "platform": str(signal.get("source_type", "")),
            "source_count": int(signal.get("source_count", 0) or 0),
            "mention_count": int(signal.get("mention_count", 0) or 0),
            "views": int(signal.get("views", 0) or 0),
            "shares": int(signal.get("shares", 0) or 0),
            "opportunity_score": float(signal.get("opportunity_score", 0) or 0),
        })
    rows = rows[-max_rows:]
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def recent_rows(limit: int = 200) -> list[dict[str, Any]]:
    """The latest snapshot rows, newest last, for de-duplicating against recent episodes."""
    return _load()[-max(0, limit):]


def attach_historical_signals(signals: Iterable[dict[str, Any]], history: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    history = history if history is not None else _load()
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in history:
        by_id.setdefault(str(row.get("trend_id", "")), []).append(row)

    enriched = []
    for signal in signals:
        item = dict(signal)
        # Attachment runs before this run's snapshot is written, so the newest
        # stored row is the previous run's observation (not two runs ago).
        past = by_id.get(str(item.get("trend_id", "")), [])
        if past:
            previous = past[-1]
            current_views = int(item.get("views", 0) or 0)
            previous_views = int(previous.get("views", 0) or 0)
            if previous_views > 0:
                growth = (current_views - previous_views) / previous_views
                item["historical_growth_rate"] = round(max(-1.0, min(5.0, growth)), 4)
                item["velocity_score"] = round(
                    min(1.0, max(float(item.get("velocity_score", 0.0)), 0.50 + 0.25 * max(0.0, min(2.0, growth)))),
                    4,
                )
            repeats = len(past)
            item["historical_snapshot_count"] = repeats
            # Repeats across earlier runs add a penalty on top of the in-run proxy.
            # A max() with the in-run proxy would let the penalty cancel out.
            item["saturation_score"] = round(
                min(1.0, float(item.get("saturation_score", 0.0)) + 0.35 * min(1.0, repeats / 8.0)),
                4,
            )
            item.update(recompute_opportunity_score(item))
        else:
            item["historical_snapshot_count"] = len(past)
            item["historical_growth_rate"] = 0.0
        enriched.append(item)
    return enriched
