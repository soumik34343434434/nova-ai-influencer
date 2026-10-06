"""Metrics normalization and learning policy for the cartoon factory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = ROOT / "artifacts" / "cartoon" / "metrics.json"
LEARNING_FILE = ROOT / "artifacts" / "cartoon" / "learning.json"


@dataclass(frozen=True)
class MetricRecord:
    content_id: str
    platform: str
    title: str
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    retention_pct: float = 0.0
    avg_watch_seconds: float = 0.0
    captured_at: str = ""


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def ingest(records: Iterable[MetricRecord]) -> list[dict]:
    existing = _load(METRICS_FILE)
    by_key = {(str(x.get("content_id")), str(x.get("platform"))): x for x in existing}
    for record in records:
        by_key[(record.content_id, record.platform)] = asdict(record)
    merged = list(by_key.values())
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    return merged


def _score(row: dict) -> float:
    views = max(int(row.get("views", 0) or 0), 1)
    retention = min(float(row.get("retention_pct", 0.0) or 0.0) / 100.0, 1.0)
    share_rate = min(float(row.get("shares", 0) or 0) / views * 50.0, 1.0)
    save_rate = min(float(row.get("saves", 0) or 0) / views * 50.0, 1.0)
    comment_rate = min(float(row.get("comments", 0) or 0) / views * 100.0, 1.0)
    like_rate = min(float(row.get("likes", 0) or 0) / views * 25.0, 1.0)
    return retention * 0.45 + share_rate * 0.25 + save_rate * 0.12 + comment_rate * 0.08 + like_rate * 0.10


def build_learning_snapshot() -> dict:
    rows = _load(METRICS_FILE)
    if not rows:
        snapshot = {
            "metrics": 0,
            "winner": None,
            "loser": None,
            "strategy": "collect platform metrics before changing creative strategy",
            "rules": [],
        }
    else:
        ranked = sorted(rows, key=_score, reverse=True)
        winner = ranked[0]
        loser = ranked[-1]
        rules = []
        if float(winner.get("retention_pct", 0) or 0) >= 70:
            rules.append("preserve fast hooks and short setup")
        if float(winner.get("shares", 0) or 0) > float(winner.get("likes", 0) or 0) * 0.15:
            rules.append("favor shareable visual gags")
        if float(loser.get("retention_pct", 0) or 0) < 45:
            rules.append("shorten setup on low-retention formats")
        snapshot = {
            "metrics": len(rows),
            "winner": {"content_id": winner.get("content_id"), "title": winner.get("title"), "score": round(_score(winner), 3)},
            "loser": {"content_id": loser.get("content_id"), "title": loser.get("title"), "score": round(_score(loser), 3)},
            "strategy": " ".join(rules) if rules else "run controlled creative variations",
            "rules": rules,
        }
    LEARNING_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEARNING_FILE.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
    return snapshot


if __name__ == "__main__":
    print(json.dumps(build_learning_snapshot(), indent=2, ensure_ascii=False))
