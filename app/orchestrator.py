"""End-to-end NOVA pipeline with zero-cost fallbacks and compatibility checks."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.agents.trend_agent import Trend, discover_trends, rank_trends
from app.db.init import init_db
from app.db.models import ContentItem, Metric
from app.db.session import SessionLocal
from app.providers.local_media import build_video_from_draft
from app.agents.content_evaluator import ContentEvaluator
from app.state_machine import DurableRun, Phase


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CHARACTER_DIR = ROOT / "assets" / "character"
PUBLISH_QUEUE = ARTIFACTS / "publish_queue.json"


@dataclass
class SafetyResult:
    allowed: bool
    reasons: list[str]


@dataclass
class CycleResult:
    trends: int
    drafts: int
    media: int
    queued: int
    blocked: int
    winner_id: int | None
    mode: str


class SafetyGate:
    """Cheap guardrail between content generation and publication."""

    FORBIDDEN = ("celebrity", "lookalike", "private person", "minor", "child", "repost", "face swap")

    def check(self, draft: dict) -> SafetyResult:
        reasons: list[str] = []
        text = json.dumps(draft).lower()
        if draft.get("originality_score", 0) < 0.80:
            reasons.append("originality_score_below_threshold")
        if draft.get("risk_score", 1) > 0.20:
            reasons.append("risk_score_above_threshold")
        if "ai" not in text:
            reasons.append("missing_ai_context")
        if any(term in text for term in self.FORBIDDEN):
            reasons.append("forbidden_or_reuse_signal")
        return SafetyResult(not reasons, reasons)


class CharacterStore:
    """Uses the real generated character when present, otherwise a free local fallback."""

    def __init__(self, path: Path | None = None):
        configured = os.getenv("NOVA_CHARACTER_IMAGE", "").strip()
        self.path = path or (Path(configured) if configured else (CHARACTER_DIR / "nova.png"))
        self.fallback_path = CHARACTER_DIR / "nova_fallback.png"

    def ensure(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            return self.path

        img = Image.new("RGB", (1080, 1920), "#15171c")
        draw = ImageDraw.Draw(img)
        draw.ellipse((285, 440, 795, 1000), fill="#d7a982", outline="#f0d1bd", width=8)
        draw.pieslice((255, 350, 825, 920), 190, 350, fill="#1b1c20")
        draw.rounded_rectangle((355, 640, 725, 760), radius=40, outline="#c0c4cf", width=12)
        draw.ellipse((420, 675, 465, 720), fill="#16181d")
        draw.ellipse((615, 675, 660, 720), fill="#16181d")
        draw.arc((450, 775, 630, 850), 200, 340, fill="#16181d", width=8)
        draw.rounded_rectangle((325, 970, 755, 1500), radius=55, fill="#252b3a", outline="#8c94ab", width=6)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 58)
        except Exception:
            font = ImageFont.load_default()
        draw.text((70, 90), "NOVA SEN", fill="#ffffff", font=font)
        draw.text((70, 160), "AI-generated fictional creator", fill="#b8bdc9", font=font)
        img.save(self.path)
        return self.path


class TrendEngine:
    def live(self, limit: int = 5) -> list[Trend]:
        return discover_trends()[:limit]

    def synthetic(self) -> list[Trend]:
        return rank_trends([
            Trend("AI agents are becoming everyone's new coworker", "backtest", category="ai", score=0.96),
            Trend("Creators are using AI to speed up short-form video", "backtest", category="creator", score=0.91),
            Trend("The internet is obsessed with productivity hacks", "backtest", category="social", score=0.82),
        ])


class ContentEngine:
    def create(self, db, trend: Trend) -> ContentItem:
        from app.services.content_service import create_draft
        context = (
            f"Trend title: {trend.title}\n"
            f"Category: {trend.category}\n"
            "Create a completely original NOVA scene. Do not copy wording, footage, motion, "
            "creator identity, or visual composition."
        )
        return create_draft(db, context)


class LearningEngine:
    def score(self, metric: Metric) -> float:
        return (
            min(metric.retention_pct / 100.0, 1.0) * 0.45
            + min(metric.shares / max(metric.views, 1) * 40.0, 1.0) * 0.30
            + min(metric.comments / max(metric.views, 1) * 100.0, 1.0) * 0.15
            + min(metric.likes / max(metric.views, 1) * 30.0, 1.0) * 0.10
        )

    def choose_winner(self, db) -> int | None:
        rows = db.query(Metric).all()
        return max(rows, key=self.score).content_id if rows else None

    def next_strategy(self, db) -> dict:
        winner = self.choose_winner(db)
        if winner is None:
            return {"winner_id": None, "strategy": "collect first-week data"}
        row = db.get(ContentItem, winner)
        return {
            "winner_id": winner,
            "strategy": f"create 3 meaningfully different variants of: {row.title}",
            "guardrail": "expand the winning premise; never duplicate the exact script",
        }


class Publisher:
    """Safe default publisher: write to an outbox, never silently publish."""

    def __init__(self):
        self.queue: list[dict] = []

    def enqueue(self, content_id: int, video: Path, caption: str) -> None:
        self.queue.append({
            "content_id": content_id,
            "video": str(video),
            "caption": caption,
            "ai_generated": True,
            "publish_mode": os.getenv("PUBLISH_MODE", "outbox"),
            "status": "ready_for_review",
        })

    def flush(self) -> Path:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        current = []
        if PUBLISH_QUEUE.exists():
            try:
                current = json.loads(PUBLISH_QUEUE.read_text())
            except json.JSONDecodeError:
                current = []
        current.extend(self.queue)
        PUBLISH_QUEUE.write_text(json.dumps(current, indent=2))
        return PUBLISH_QUEUE


class Compatibility:
    """Fail-fast contracts between every phase."""

    @staticmethod
    def check_draft(draft: ContentItem) -> None:
        assert draft.title and draft.hook and draft.script
        assert draft.originality_score >= 0.80
        assert draft.risk_score <= 0.20
        assert draft.status in {"pending_approval", "approved"}

    @staticmethod
    def check_video(path: Path) -> None:
        assert path.exists() and path.stat().st_size > 0
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        assert stream["width"] == 1080
        assert stream["height"] == 1920

    @staticmethod
    def check_queue(path: Path) -> None:
        items = json.loads(path.read_text())
        assert items
        for item in items:
            assert item["ai_generated"] is True
            assert item["status"] in {"ready_for_review", "published"}


def _process_drafts(db, trends: list[Trend], publisher: Publisher, char: Path) -> tuple[int, int, int]:
    gate = SafetyGate()
    evaluator = ContentEvaluator()
    engine = ContentEngine()
    drafts = media = blocked = 0
    recent_titles: list[str] = []

    for trend in trends:
        draft = engine.create(db, trend)
        drafts += 1
        Compatibility.check_draft(draft)
        payload = json.loads(draft.content_json)
        payload["caption"] = draft.platform_caption + "\n\nAI-generated fictional creator."

        evaluation = evaluator.evaluate(
            {
                "title": draft.title,
                "hook": draft.hook,
                "script": draft.script,
                "caption": payload["caption"],
            },
            recent_titles,
        )
        if not evaluation.passed:
            blocked += 1
            continue

        result = gate.check(payload)
        if not result.allowed:
            blocked += 1
            continue
        recent_titles.append(draft.title)

        latest = {
            "id": draft.id,
            "title": draft.title,
            "hook": draft.hook,
            "script": draft.script,
            "caption": draft.platform_caption,
        }
        video = build_video_from_draft(char, latest, ARTIFACTS / "videos")
        Compatibility.check_video(video)
        publisher.enqueue(draft.id, video, payload["caption"])
        media += 1

    return drafts, media, blocked


def run_backtest() -> CycleResult:
    """Synthetic end-to-end regression test for all phases."""
    init_db()
    db = SessionLocal()
    publisher = Publisher()
    run = DurableRun.start(db, {"mode": "backtest"})
    try:
        trends = TrendEngine().synthetic()
        run.transition(Phase.TREND_VALIDATED)
        char = CharacterStore().ensure()
        run.transition(Phase.IDEA_CREATED)
        drafts, media, blocked = _process_drafts(db, trends, publisher, char)
        run.transition(Phase.SCRIPT_READY)
        run.transition(Phase.MEDIA_READY)
        run.transition(Phase.MEDIA_QA)
        queue = publisher.flush()
        Compatibility.check_queue(queue)
        run.transition(Phase.PUBLISH_READY)
        run.transition(Phase.PUBLISHED)

        # Simulated platform results close the learning loop without contacting a platform.
        for index, row in enumerate(db.query(ContentItem).order_by(ContentItem.id.desc()).limit(3).all()):
            db.add(Metric(
                content_id=row.id,
                platform="backtest",
                views=12000 + index * 2500,
                likes=840 + index * 170,
                comments=96 + index * 24,
                shares=180 + index * 30,
                avg_watch_seconds=14.2 + index,
                retention_pct=71 + index * 3,
                revenue=0.0,
            ))
        db.commit()
        winner = LearningEngine().choose_winner(db)
        strategy = LearningEngine().next_strategy(db)
        assert strategy["winner_id"] == winner
        run.transition(Phase.METRICS_CAPTURED)
        run.update_payload(winner_id=winner, strategy=strategy)
        run.transition(Phase.LEARNING_UPDATED)
        return CycleResult(len(trends), drafts, media, len(publisher.queue), blocked, winner, "backtest")
    finally:
        db.close()


def run_live_cycle(limit: int = 3) -> CycleResult:
    init_db()
    db = SessionLocal()
    publisher = Publisher()
    run = DurableRun.start(db, {"mode": "live", "limit": limit})
    try:
        trends = TrendEngine().live(limit)
        run.transition(Phase.TREND_VALIDATED)
        char = CharacterStore().ensure()
        run.transition(Phase.IDEA_CREATED)
        drafts, media, blocked = _process_drafts(db, trends, publisher, char)
        run.transition(Phase.SCRIPT_READY)
        run.transition(Phase.MEDIA_READY)
        run.transition(Phase.MEDIA_QA)
        queue = publisher.flush()
        Compatibility.check_queue(queue)
        run.transition(Phase.PUBLISH_READY)
        winner = LearningEngine().choose_winner(db)
        run.update_payload(winner_id=winner, queued=len(publisher.queue), blocked=blocked)
        run.mark_waiting("external publisher is pending; resume at publish_ready")
        return CycleResult(len(trends), drafts, media, len(publisher.queue), blocked, winner, "live")
    finally:
        db.close()


if __name__ == "__main__":
    mode = os.getenv("NOVA_MODE", "backtest")
    print(json.dumps(asdict(run_live_cycle() if mode == "live" else run_backtest()), indent=2))
