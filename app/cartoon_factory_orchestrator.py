"""Concrete wiring for the 14-stage cartoon production loop.

The orchestrator intentionally reuses the tested research, director, storyboard,
asset, renderer and QA components. The publish stage creates a provider-neutral
outbox entry; a platform adapter can consume it without changing upstream work.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.agents.cartoon_director import CartoonDirector
from app.agents.cartoon_factory import action_asset, coverage
from app.agents.cartoon_format_intelligence import best_format, format_to_dict
from app.agents.cartoon_jury import CartoonIdeaJury, jury_to_dict
from app.agents.cartoon_animatic import render_animatic
from app.agents.cartoon_animatic_judge import AnimaticJudge
from app.agents.cartoon_quality import CartoonQuality
from app.agents.cartoon_research import discover, rank, ResearchSignal
from app.agents.storyboard import build_storyboard
from app.cartoon_control import CartoonControlPlane, CartoonRun, CartoonRunStore, CartoonStage
from app.providers.cartoon_media import render_photo_mix


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "cartoon"
OUTBOX = ARTIFACTS / "publish_queue.json"


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


class CartoonFactoryOrchestrator:
    """End-to-end stage wiring with durable checkpoints and repair."""

    def __init__(self, director: CartoonDirector | None = None, quality: CartoonQuality | None = None,
                 jury: CartoonIdeaJury | None = None, animatic_judge: AnimaticJudge | None = None):
        self.director = director or CartoonDirector()
        self.quality = quality or CartoonQuality()
        self.jury = jury or CartoonIdeaJury()
        self.animatic_judge = animatic_judge or AnimaticJudge()
        self.plane = CartoonControlPlane(max_retries=1)
        self._register()

    def _register(self) -> None:
        self.plane.register(CartoonStage.RESEARCH, self._research)
        self.plane.register(CartoonStage.TREND_INTELLIGENCE, self._trend_intelligence)
        self.plane.register(CartoonStage.IDEA_GENERATOR, self._idea_generator)
        self.plane.register(CartoonStage.IDEA_EVALUATOR, self._idea_evaluator)
        self.plane.register(CartoonStage.SCRIPT_ENGINE, self._script_engine)
        self.plane.register(CartoonStage.STORYBOARD_ENGINE, self._storyboard_engine)
        self.plane.register(CartoonStage.ASSET_ENGINE, self._asset_engine)
        self.plane.register(CartoonStage.ANIMATION_ENGINE, self._animation_engine)
        self.plane.register(CartoonStage.AUDIO_ENGINE, self._audio_engine)
        self.plane.register(CartoonStage.CAPTION_ENGINE, self._caption_engine)
        self.plane.register(CartoonStage.VIDEO_QA, self._video_qa)
        self.plane.register(CartoonStage.PUBLISH, self._publish)
        self.plane.register(CartoonStage.ANALYTICS, self._analytics)
        self.plane.register(CartoonStage.LEARNING, self._learning)

    def create_run(self, topic: str = "") -> CartoonRun:
        store = CartoonRunStore()
        return store.create({"topic": topic, "artifacts": {}, "versions": {"orchestrator": "1.0"}})

    def run(self, topic: str = "", run_id: str | None = None) -> CartoonRun:
        store = CartoonRunStore(run_id) if run_id else CartoonRunStore()
        run = store.load() if run_id else store.create({"topic": topic, "artifacts": {}, "versions": {"orchestrator": "1.0"}})
        self.plane.store = store
        return self.plane.run(run)

    def resume(self, run_id: str) -> CartoonRun:
        store = CartoonRunStore(run_id)
        self.plane.store = store
        return self.plane.run(store.load())

    def repair(self, run_id: str, stage: CartoonStage) -> CartoonRun:
        store = CartoonRunStore(run_id)
        self.plane.store = store
        return self.plane.repair(run_id, stage)

    def _signals(self, run: CartoonRun) -> list[ResearchSignal]:
        return [ResearchSignal(**row) for row in run.payload.get("research_signals", [])]

    def _research(self, run: CartoonRun) -> dict[str, Any]:
        signals = discover(limit=24)
        self.plane.store.set_payload(run, research_signals=[asdict(x) for x in signals])
        return {"count": len(signals), "source": "google_news+gdelt"}

    def _trend_intelligence(self, run: CartoonRun) -> dict[str, Any]:
        signals = rank(self._signals(run), run.payload.get("recent_topics", []))
        chosen = signals[:8]
        enriched = []
        for signal in chosen:
            pattern = best_format(signal.title, signal.category, [x.title for x in chosen[:5]])
            enriched.append({**asdict(signal), "format": format_to_dict(pattern)})
        self.plane.store.set_payload(run, ranked_signals=enriched)
        return {
            "ranked_count": len(enriched),
            "top_score": chosen[0].score if chosen else 0.0,
            "format_intelligence": True,
        }

    def _idea_generator(self, run: CartoonRun) -> dict[str, Any]:
        ranked_rows = list(run.payload.get("ranked_signals", []))
        ranked = [ResearchSignal(**{k: row[k] for k in ("title", "source", "category", "url", "published_at", "score")}) for row in ranked_rows]
        topic = run.payload.get("topic") or (ranked[0].title if ranked else "evergreen cartoon")
        learning = {}
        learning_file = ARTIFACTS / "learning.json"
        if learning_file.exists():
            try:
                learning = json.loads(learning_file.read_text())
            except json.JSONDecodeError:
                learning = {}
        strategy = str(learning.get("strategy", ""))
        candidates: list[dict[str, Any]] = []
        format_by_title = {row["title"]: row.get("format", {}) for row in ranked_rows}
        for signal in ranked[:4]:
            for candidate in self.director.candidates(
                signal.title,
                count=2,
                strategy=json.dumps(format_by_title.get(signal.title, {}), ensure_ascii=False) + " " + strategy,
            ):
                candidate["research"] = asdict(signal)
                candidate["format"] = format_by_title.get(signal.title, {})
                candidate["asset_ready"] = all(
                    action_asset(candidate["character"], beat["asset_action"]) for beat in candidate["beats"]
                )
                candidates.append(candidate)
        self.plane.store.set_payload(run, candidates=candidates, idea_topic=topic)
        return {"count": len(candidates), "topic": topic}

    def _idea_evaluator(self, run: CartoonRun) -> dict[str, Any]:
        candidates = run.payload.get("candidates", [])
        recent = run.payload.get("recent_titles", [])
        preselected = []
        for candidate in candidates:
            if not candidate.get("asset_ready"):
                continue
            deterministic = self.quality.evaluate(candidate, recent)
            jury = self.jury.judge(candidate)
            if deterministic.passed and jury.passed:
                score = round(
                    deterministic.score * 0.45
                    + jury.score * 0.35
                    + float(candidate["research"].get("score", 0.0)) * 0.20,
                    3,
                )
                preselected.append({
                    "score": score,
                    "candidate": candidate,
                    "evaluation": asdict(deterministic),
                    "jury": jury_to_dict(jury),
                })
        preselected.sort(key=lambda item: item["score"], reverse=True)
        attempts = []
        winner = None
        for item in preselected[:8]:
            candidate = item["candidate"]
            images = [action_asset(candidate["character"], beat["asset_action"]) for beat in candidate["beats"]]
            if any(path is None for path in images):
                continue
            animatic = ARTIFACTS / "animatics" / f"{candidate['premise_slug']}_{len(attempts)}.mp4"
            render_animatic([p for p in images if p is not None], [beat["voice"] for beat in candidate["beats"]], animatic)
            animatic_result = self.animatic_judge.inspect(animatic, expected_shots=4)
            attempts.append({"candidate": candidate["title"], "animatic": str(animatic), "result": animatic_result})
            if animatic_result["passed"]:
                item["animatic"] = str(animatic)
                item["animatic_evaluation"] = animatic_result
                winner = item
                break
        if winner is None:
            raise RuntimeError("No candidate passed the idea jury and cheap animatic gate")
        self.plane.store.set_payload(
            run,
            winner=winner,
            alternates=preselected[1:4],
            animatic_attempts=attempts,
        )
        return {
            "winner_score": winner["score"],
            "alternates": len(preselected[1:4]),
            "jury": True,
            "cheap_animatic": True,
            "animatic_attempts": len(attempts),
        }

    def _script_engine(self, run: CartoonRun) -> dict[str, Any]:
        candidate = run.payload["winner"]["candidate"]
        script_lines = [beat["voice"] for beat in candidate["beats"]]
        script = "\n\n".join(script_lines)
        self.plane.store.set_payload(run, script=script, hook=candidate["hook"], title=candidate["title"])
        return {"beats": len(script_lines), "target_seconds": "8-15"}

    def _storyboard_engine(self, run: CartoonRun) -> dict[str, Any]:
        candidate = run.payload["winner"]["candidate"]
        draft = {
            "title": candidate["title"],
            "hook": candidate["hook"],
            "character": candidate["character"],
            "script": run.payload["script"],
        }
        storyboard = build_storyboard(draft)
        data = storyboard.to_dict()
        self.plane.store.set_payload(run, storyboard=data)
        return {"shots": len(storyboard.shots), "duration": storyboard.total_duration}

    def _asset_engine(self, run: CartoonRun) -> dict[str, Any]:
        candidate = run.payload["winner"]["candidate"]
        assets = []
        for beat in candidate["beats"]:
            path = action_asset(candidate["character"], beat["asset_action"])
            if not path:
                raise FileNotFoundError(
                    f"Missing action asset: {candidate['character']}:{beat['asset_action']}"
                )
            assets.append(str(path))
        self.plane.store.set_payload(run, asset_paths=assets)
        return {"coverage": coverage(type("Spec", (), {"character": candidate["character"], "beats": candidate["beats"]})())}

    def _animation_engine(self, run: CartoonRun) -> dict[str, Any]:
        candidate = run.payload["winner"]["candidate"]
        images = [Path(p) for p in run.payload["asset_paths"]]
        captions = [beat["voice"] for beat in candidate["beats"]]
        sfx = [beat["sfx"] for beat in candidate["beats"]]
        durations = [2.0, 2.0, 2.0, 2.4]
        slug = candidate["premise_slug"] + "_factory"
        output = ARTIFACTS / f"{slug}.mp4"
        render_photo_mix(
            images,
            captions,
            durations,
            output,
            voice_lines=captions,
            sfx_kinds=sfx,
        )
        self.plane.store.set_payload(run, video=str(output), render_durations=durations)
        return {"video": str(output), "renderer": "Pillow+FFmpeg"}

    def _audio_engine(self, run: CartoonRun) -> dict[str, Any]:
        video = Path(run.payload["video"])
        audio_candidates = list(video.parent.glob(video.stem + "_frames/episode_audio.wav"))
        has_audio_source = bool(audio_candidates and audio_candidates[0].exists())
        if not has_audio_source:
            raise RuntimeError("Audio stage expected renderer-produced episode_audio.wav")
        return {"voice_sfx_mixed": True, "audio_source": str(audio_candidates[0])}

    def _caption_engine(self, run: CartoonRun) -> dict[str, Any]:
        candidate = run.payload["winner"]["candidate"]
        disclosure = "AI-generated • fictional cartoon"
        return {"captions": [beat["voice"] for beat in candidate["beats"]], "disclosure": disclosure}

    def _video_qa(self, run: CartoonRun) -> dict[str, Any]:
        video = Path(run.payload["video"])
        result = self.quality.render_check(video)
        if not result["passed"]:
            raise RuntimeError("Video QA failed: " + ", ".join(result["reasons"]))
        return result

    def _publish(self, run: CartoonRun) -> dict[str, Any]:
        candidate = run.payload["winner"]["candidate"]
        item = {
            "run_id": run.run_id,
            "video": run.payload["video"],
            "title": candidate["title"],
            "caption": candidate.get("caption", ""),
            "ai_generated": True,
            "status": "ready_for_review",
            "audience_classification_review_required": True,
        }
        queue = []
        if OUTBOX.exists():
            try:
                queue = json.loads(OUTBOX.read_text())
            except json.JSONDecodeError:
                queue = []
        queue.append(item)
        _save(OUTBOX, queue)
        return {"provider": "outbox", "status": "ready_for_review"}

    def _analytics(self, run: CartoonRun) -> dict[str, Any]:
        # Real platform metrics arrive asynchronously after publication. This
        # stage records the dependency instead of pretending metrics exist.
        return {"status": "awaiting_platform_metrics", "metric_sources": ["instagram_reels", "youtube_shorts"]}

    def _learning(self, run: CartoonRun) -> dict[str, Any]:
        from app.learning import build_learning_snapshot
        snapshot = build_learning_snapshot()
        self.plane.store.set_payload(run, learning=snapshot)
        return snapshot


if __name__ == "__main__":
    run = CartoonFactoryOrchestrator().run()
    print(json.dumps({"run_id": run.run_id, "status": run.status, "current_stage": run.current_stage}, indent=2))
