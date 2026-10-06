"""Research -> idea -> script -> storyboard -> media -> QA -> outbox -> learning."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.agents.cartoon_director import CartoonDirector
from app.agents.cartoon_factory import action_asset
from app.agents.cartoon_quality import CartoonQuality, quality_to_dict
from app.agents.cartoon_research import ResearchSignal, discover
from app.db.init import init_db
from app.db.models import ContentItem, Metric
from app.db.session import SessionLocal
from app.providers.cartoon_media import render_photo_mix


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "cartoon"
QUEUE = ARTIFACTS / "publish_queue.json"


def _recent_titles(limit: int = 30) -> list[str]:
    init_db()
    db = SessionLocal()
    try:
        return [x.title for x in db.query(ContentItem).order_by(ContentItem.created_at.desc()).limit(limit).all()]
    finally:
        db.close()


def _supported(candidate: dict) -> bool:
    return all(action_asset(candidate["character"], beat["asset_action"]) for beat in candidate["beats"])


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


class CartoonAutopilot:
    def __init__(self, director: CartoonDirector | None = None, quality: CartoonQuality | None = None):
        self.director = director or CartoonDirector()
        self.quality = quality or CartoonQuality()

    def research(self, limit: int = 20) -> list[ResearchSignal]:
        signals = discover(limit=limit)
        _save_json(ARTIFACTS / "research.json", [asdict(x) for x in signals])
        return signals

    def choose(self, signals: list[ResearchSignal], candidates_per_signal: int = 3) -> dict | None:
        recent = _recent_titles()
        ranked = []
        for signal in signals[: max(1, len(signals))]:
            for candidate in self.director.candidates(signal.title, candidates_per_signal):
                candidate["research"] = asdict(signal)
                candidate["family_friendly"] = True
                candidate["original_characters"] = True
                candidate["ai_generated"] = True
                candidate["asset_ready"] = _supported(candidate)
                evaluation = self.quality.evaluate(candidate, recent)
                if evaluation.passed and candidate["asset_ready"]:
                    final = round(evaluation.score * 0.75 + signal.score * 0.25, 3)
                    ranked.append((final, candidate, evaluation))
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[0], reverse=True)
        score, candidate, evaluation = ranked[0]
        return {"score": score, "candidate": candidate, "evaluation": quality_to_dict(evaluation), "alternates": [r[0] for r in ranked[1:5]]}

    def render(self, winner: dict, output_name: str | None = None) -> dict:
        candidate = winner["candidate"]
        slug = output_name or candidate["premise_slug"]
        frames = [action_asset(candidate["character"], b["asset_action"]) for b in candidate["beats"]]
        if any(p is None for p in frames):
            raise RuntimeError("Winner lost asset coverage before render")
        durations = [2.2, 2.2, 2.2, 2.8]
        output = ARTIFACTS / f"{slug}_autopilot.mp4"
        render_photo_mix(
            [p for p in frames if p is not None],
            [b["voice"] for b in candidate["beats"]],
            durations,
            output,
            voice_lines=[b["voice"] for b in candidate["beats"]],
            sfx_kinds=[b["sfx"] for b in candidate["beats"]],
        )
        media_qa = self.quality.render_check(output)
        motion_qa = self.quality.motion_check(output)
        if not media_qa["passed"]:
            raise RuntimeError("Render QA failed: " + ", ".join(media_qa["reasons"]))
        if not motion_qa["passed"]:
            raise RuntimeError("Motion QA failed: " + ", ".join(motion_qa["reasons"]))
        metadata = {"video": str(output), "winner": winner, "media_qa": media_qa, "motion_qa": motion_qa}
        _save_json(ARTIFACTS / f"{slug}_autopilot.json", metadata)
        return metadata

    def queue(self, media: dict) -> dict:
        candidate = media["winner"]["candidate"]
        item = {
            "video": media["video"],
            "title": candidate["title"],
            "caption": candidate["caption"],
            "ai_generated": True,
            "status": "ready_for_review",
            "audience_classification_review_required": True,
        }
        current = []
        if QUEUE.exists():
            try:
                current = json.loads(QUEUE.read_text())
            except json.JSONDecodeError:
                current = []
        current.append(item)
        _save_json(QUEUE, current)
        return item

    def run(self, research_limit: int = 12) -> dict:
        signals = self.research(research_limit)
        if not signals:
            raise RuntimeError("No research signals available")
        winner = self.choose(signals)
        if winner is None:
            _save_json(ARTIFACTS / "autopilot_blocked.json", {"reason": "no_quality_and_asset_ready_candidate", "signals": [asdict(x) for x in signals]})
            return {"status": "blocked", "signals": len(signals)}
        media = self.render(winner)
        queued = self.queue(media)
        return {"status": "publish_ready", "research": len(signals), "winner": winner, "media": media["media_qa"], "queued": queued}


def learning_snapshot() -> dict:
    init_db()
    db = SessionLocal()
    try:
        rows = db.query(Metric).all()
        if not rows:
            return {"metrics": 0, "winner": None, "strategy": "collect real platform metrics"}
        def score(row: Metric) -> float:
            retention = min(row.retention_pct / 100.0, 1.0)
            shares = min(row.shares / max(row.views, 1) * 40.0, 1.0)
            comments = min(row.comments / max(row.views, 1) * 100.0, 1.0)
            likes = min(row.likes / max(row.views, 1) * 30.0, 1.0)
            return retention * 0.45 + shares * 0.30 + comments * 0.15 + likes * 0.10
        winner = max(rows, key=score)
        content = db.get(ContentItem, winner.content_id)
        return {"metrics": len(rows), "winner": winner.content_id, "winner_score": round(score(winner), 3), "strategy": f"make new variants of {content.title}" if content else "expand winning format"}
    finally:
        db.close()


if __name__ == "__main__":
    result = CartoonAutopilot().run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
