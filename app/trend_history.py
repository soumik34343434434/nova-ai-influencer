"""Persistent trend snapshot history for velocity and saturation inference."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


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


def attach_historical_signals(signals: Iterable[dict[str, Any]], history: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    history = history if history is not None else _load()
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in history:
        by_id.setdefault(str(row.get("trend_id", "")), []).append(row)

    enriched = []
    for signal in signals:
        item = dict(signal)
        past = by_id.get(str(item.get("trend_id", "")), [])
        if len(past) >= 2:
            previous = past[-2]
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
            # More repeated appearances across snapshots increase saturation,
            # but acceleration can temporarily offset the penalty.
            item["saturation_score"] = round(
                min(
                    1.0,
                    max(
                        float(item.get("saturation_score", 0.0)),
                        min(1.0, repeats / 8.0) * 0.65
                        + 0.35 * max(0.0, 1.0 - float(item.get("velocity_score", 0.0))),
                    ),
                ),
                4,
            )
        else:
            item["historical_snapshot_count"] = len(past)
            item["historical_growth_rate"] = 0.0
        enriched.append(item)
    return enriched
