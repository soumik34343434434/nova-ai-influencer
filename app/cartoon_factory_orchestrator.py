"""Concrete wiring for the 14-stage cartoon production loop.

The orchestrator intentionally reuses the tested research, director, storyboard,
asset, renderer and QA components. The publish stage creates a provider-neutral
outbox entry; a platform adapter can consume it without changing upstream work.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.agents.cartoon_director import CartoonDirector
from app.agents.cartoon_factory import action_asset, coverage
from app.agents.cartoon_format_intelligence import best_format, format_to_dict
from app.agents.trend_intelligence_v2 import discover_social_trends, enrich_signals, signal_to_dict
from app.agents.format_engine_v2 import rank_format_candidates
from app.agents.idea_generator_v2 import generate_ideas
from app.learning_v2 import build_strategy_memory, ingest as ingest_learning_metrics, select_portfolio
from app.agent_memory import (
    build_next_loop_plan,
    ensure_experiment_identity,
    explores,
    load_next_loop_plan,
    record_provider_observation,
    summarize_failures,
    summarize_providers,
    update_experiment_learning,
    upsert_experiment,
)
from app.providers.platform_analytics import collect_platform_metrics, metric_to_record
from app.trend_history import snapshot as snapshot_trends, attach_historical_signals, recent_rows as recent_trend_rows
from app.agents.cartoon_jury import CartoonIdeaJury, jury_to_dict
from app.agents.cartoon_animatic import render_animatic
from app.agents.cartoon_animatic_judge import AnimaticJudge
from app.agents.cartoon_motion_qa import CartoonMotionQA
from app.agents.cartoon_quality import CartoonQuality
from app.agents.final_qa import FinalQAAggregator
from app.agents.media_qa import MediaQA
from app.agents.cartoon_research import EVERGREEN_TITLES, ResearchSignal, discover_report, evergreen_signals
from app.agents.cartoon_safety import public_text_problems, sanitize_topic
from app.agents.storyboard import build_storyboard
from app.agents.story_engine import StoryEngine
from app.agents.script_engine import ScriptEngine
from app.cartoon_control import CartoonControlPlane, CartoonRun, CartoonRunStore, CartoonStage
from app.providers.cartoon_animation import render_cartoon_motion, rerender_cartoon_shots
from app.providers.blender_animation import BlenderAnimationProvider
from app.providers.ai_performance import (
    RunwayPerformanceProvider,
    build_runway_repair_request,
)
from app.providers import runway_video
from app.providers.reel_editor import assemble_reel, verify_reel


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "cartoon"
OUTBOX = ARTIFACTS / "publish_queue.json"
# Enough recent history to keep a month of daily episodes out of today's pool.
RECENT_MEMORY_LIMIT = 200
# Trend rows that must remain after removing recently used trends before the filter applies.
MIN_FRESH_TREND_ROWS = 5


# Visual QA -> Repair AI loop budget (override with NOVA_MAX_REPAIR_ATTEMPTS).
DEFAULT_MAX_REPAIR_ATTEMPTS = 2
# Failures a rerender cannot fix: the toolchain itself is missing.
UNREPAIRABLE_VIDEO_REASONS = {"ffprobe_missing"}

# The renderer and voice profiles know setup/escalation/reaction/punchline.
RENDER_BEATS = {"hook": "setup", "attempt": "setup", "payoff": "punchline"}


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def _guard_public_text(candidate: dict[str, Any]) -> None:
    """Fail closed: unsafe viewer-facing text never reaches the publish queue or a platform title."""
    problems = public_text_problems(str(candidate.get("title", "")), str(candidate.get("caption", "")))
    if problems:
        raise RuntimeError("Publish blocked: viewer-facing text failed the family-safety filter: " + ", ".join(problems))


def _recent_episode_memory(limit: int = RECENT_MEMORY_LIMIT) -> tuple[list[str], list[str], list[str]]:
    """Titles and trend ids of queued episodes, plus recent research topics from trend history."""
    titles: list[str] = []
    trend_ids: list[str] = []
    queue: Any = []
    if OUTBOX.exists():
        try:
            queue = json.loads(OUTBOX.read_text())
        except (OSError, json.JSONDecodeError):
            queue = []
    for item in (queue[-limit:] if isinstance(queue, list) else []):
        if not isinstance(item, dict):
            continue
        if item.get("title"):
            titles.append(str(item["title"]))
        trend = str((item.get("experiment_metadata") or {}).get("trend_id", ""))
        if trend:
            trend_ids.append(trend)
    topics = [str(row.get("title", "")) for row in recent_trend_rows(limit) if row.get("title")]
    return titles, trend_ids, topics


def _screen_social_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Replace unsafe or real-person titles from social connectors with evergreen premises."""
    replaced = 0
    for row in rows:
        title, status = sanitize_topic(str(row.get("title", "")), EVERGREEN_TITLES,
                                       f"social|{row.get('source_type', '')}|{row.get('url', '')}")
        if status != "ok":
            replaced += 1
            row["title"] = title
            row["safety_status"] = status
    return rows, replaced


class CartoonFactoryOrchestrator:
    """End-to-end stage wiring with durable checkpoints and repair."""

    def __init__(self, director: CartoonDirector | None = None, quality: CartoonQuality | None = None,
                 jury: CartoonIdeaJury | None = None, animatic_judge: AnimaticJudge | None = None):
        self.director = director or CartoonDirector()
        self.quality = quality or CartoonQuality()
        self.jury = jury or CartoonIdeaJury()
        self.animatic_judge = animatic_judge or AnimaticJudge()
        self.story_engine = StoryEngine()
        self.script_engine = ScriptEngine()
        self.final_qa = FinalQAAggregator(self.quality, self.story_engine)
        self.media_qa = MediaQA()
        self.plane = CartoonControlPlane(max_retries=1)
        self._register()

    def _register(self) -> None:
        self.plane.register(CartoonStage.RESEARCH, self._research)
        self.plane.register(CartoonStage.TREND_INTELLIGENCE, self._trend_intelligence)
        self.plane.register(CartoonStage.IDEA_GENERATOR, self._idea_generator)
        self.plane.register(CartoonStage.IDEA_EVALUATOR, self._idea_evaluator)
        self.plane.register(CartoonStage.STORY_ENGINE, self._story_engine)
        self.plane.register(CartoonStage.SCRIPT_ENGINE, self._script_engine)
        self.plane.register(CartoonStage.STORYBOARD_ENGINE, self._storyboard_engine)
        self.plane.register(CartoonStage.ASSET_ENGINE, self._asset_engine)
        self.plane.register(CartoonStage.ANIMATION_ENGINE, self._animation_engine)
        self.plane.register(CartoonStage.AUDIO_ENGINE, self._audio_engine)
        self.plane.register(CartoonStage.CAPTION_ENGINE, self._caption_engine)
        self.plane.register(CartoonStage.EDITOR_ENGINE, self._editor_engine)
        self.plane.register(CartoonStage.VIDEO_QA, self._video_qa)
        self.plane.register(CartoonStage.PUBLISH, self._publish)
        self.plane.register(CartoonStage.ANALYTICS, self._analytics)
        self.plane.register(CartoonStage.LEARNING, self._learning)

    @staticmethod
    def _new_run_payload(topic: str) -> dict[str, Any]:
        # Carry the last learning pass's policy into this run so the trend and
        # idea stages act on what previous experiments measured.
        return {
            "topic": topic,
            "artifacts": {},
            "versions": {"orchestrator": "1.0"},
            "next_loop_plan": load_next_loop_plan(),
        }

    def create_run(self, topic: str = "") -> CartoonRun:
        store = CartoonRunStore()
        return store.create(self._new_run_payload(topic))

    def run(self, topic: str = "", run_id: str | None = None) -> CartoonRun:
        store = CartoonRunStore(run_id) if run_id else CartoonRunStore()
        run = store.load() if run_id else store.create(self._new_run_payload(topic))
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

    def _sync_experiment(self, run: CartoonRun, **updates: Any) -> dict[str, Any]:
        identity = ensure_experiment_identity(run.payload, run.run_id)
        identity.update({k: v for k, v in updates.items() if v is not None})
        run.payload["experiment"] = identity
        upsert_experiment(identity)
        self.plane.store.set_payload(run, experiment=identity)
        return identity

    def bind_published_content(self, run_id: str, published_content_map: dict[str, dict[str, Any]]) -> CartoonRun:
        """Attach real platform content IDs to an experiment after external publishing."""
        store = CartoonRunStore(run_id)
        run = store.load()
        self.plane.store = store
        # Merge, so platforms published at different times all stay bound.
        merged = {platform: dict(mapping) for platform, mapping in run.payload.get("published_content_map", {}).items()}
        for platform, mapping in published_content_map.items():
            merged.setdefault(platform, {}).update(mapping)
        store.set_payload(
            run,
            published_content_map=merged,
            publish_status="published",
            metrics_pending=True,
        )
        self._sync_experiment(
            run,
            publish_status="published",
            published_content_map=merged,
        )
        return run

    def refresh_metrics_and_learning(self, run_id: str) -> CartoonRun:
        """Re-run only analytics + learning after platform metrics become available."""
        store = CartoonRunStore(run_id)
        run = store.load()
        self.plane.store = store
        store.repair_from(run, CartoonStage.ANALYTICS)
        return self.plane.run(run, from_stage=CartoonStage.ANALYTICS)

    def _signals(self, run: CartoonRun) -> list[ResearchSignal]:
        return [ResearchSignal(**row) for row in run.payload.get("research_signals", [])]

    def _research(self, run: CartoonRun) -> dict[str, Any]:
        recent_titles, recent_trend_ids, recent_topics = _recent_episode_memory()
        signals, source_status = discover_report(limit=30, recent_topics=recent_topics)
        base_rows = [asdict(x) for x in signals]
        social_rows, connector_status = discover_social_trends()
        social_rows, social_replaced = _screen_social_rows(social_rows)
        if base_rows:
            mode = "live"
        elif social_rows:
            mode = "social_only"
        else:
            mode = "degraded_evergreen"
        if mode == "degraded_evergreen":
            # No live signal at all: a varied, date-rotated safe pool instead of one fixed idea.
            base_rows = [asdict(x) for x in evergreen_signals(recent_titles=recent_topics + recent_titles, count=8)]
        if mode != "live":
            print(f"[research] degraded mode={mode}: sources={source_status} social={connector_status}", file=sys.stderr)
        combined = base_rows + social_rows
        trend_v2 = [signal_to_dict(x) for x in enrich_signals(combined)]
        trend_v2 = attach_historical_signals(trend_v2)
        snapshot_trends(trend_v2)
        research_status = {
            "mode": mode,
            "sources": source_status,
            "social_connectors": connector_status,
            "social_topics_replaced": social_replaced,
        }
        self.plane.store.set_payload(
            run,
            research_signals=base_rows,
            trend_signals_v2=trend_v2,
            social_connector_status=connector_status,
            research_status=research_status,
            recent_titles=recent_titles,
            recent_trend_ids=recent_trend_ids,
        )
        return {
            "count": len(signals),
            "social_records": len(social_rows),
            "evergreen_records": len(base_rows) if mode == "degraded_evergreen" else 0,
            "trend_signals_v2": len(trend_v2),
            "mode": mode,
            "degraded": mode != "live",
            "sources_ok": sorted(label for label, value in source_status.items() if value.startswith("ok")),
            "sources": source_status,
            "social_connectors": connector_status,
        }

    def _trend_intelligence(self, run: CartoonRun) -> dict[str, Any]:
        trend_rows = list(run.payload.get("trend_signals_v2", []))
        trend_rows.sort(key=lambda x: (float(x.get("opportunity_score", 0.0)), float(x.get("velocity_score", 0.0))), reverse=True)
        # Do not re-make an episode from a trend that already produced one, unless that leaves too little.
        recent_ids = set(run.payload.get("recent_trend_ids", []))
        fresh_rows = [row for row in trend_rows if row.get("trend_id") not in recent_ids]
        if len(fresh_rows) >= MIN_FRESH_TREND_ROWS:
            trend_rows = fresh_rows
        memory = build_strategy_memory()
        policy = dict(run.payload.get("next_loop_plan", {}).get("policy", {}))
        formats = rank_format_candidates(trend_rows[:20], memory=memory, limit=40)
        preferred = set(policy.get("preferred_formats", []))
        avoid_errors = set(policy.get("avoid_error_types", []))
        if preferred:
            for item in formats:
                # Format candidates name their format under "name".
                item["learned_priority"] = 1.0 if (item.get("format_name") or item.get("name")) in preferred else 0.0
                item["policy_adjusted_score"] = float(item.get("score", 0.0)) + 0.12 * item["learned_priority"]
            formats.sort(key=lambda x: float(x.get("policy_adjusted_score", x.get("score", 0.0))), reverse=True)
        self.plane.store.set_payload(
            run,
            ranked_signals_v2=trend_rows[:20],
            format_candidates=formats,
            learning_policy=policy,
            avoid_error_types=sorted(avoid_errors),
        )
        return {
            "ranked_count": len(trend_rows[:20]),
            "top_score": trend_rows[0].get("opportunity_score", 0.0) if trend_rows else 0.0,
            "format_candidates": len(formats),
            "format_intelligence": True,
            "hook_detection": True,
            "emotion_detection": True,
            "saturation_proxy": True,
        }

    def _idea_generator(self, run: CartoonRun) -> dict[str, Any]:
        trend_rows = list(run.payload.get("ranked_signals_v2", []))
        format_rows = list(run.payload.get("format_candidates", []))
        raw_ideas = generate_ideas(trend_rows[:20], format_rows[:20], count=100)
        memory = build_strategy_memory()
        policy = dict(run.payload.get("learning_policy", {}))
        candidates = select_portfolio(raw_ideas, memory=memory, limit=min(60, len(raw_ideas)))
        preferred_formats = set(policy.get("preferred_formats", []))
        preferred_hooks = set(policy.get("preferred_hooks", []))
        preferred_emotions = set(policy.get("preferred_emotions", []))
        preferred_characters = set(policy.get("preferred_characters", []))
        raw_rate = policy.get("exploration_rate")
        exploration_rate = 0.30 if raw_rate is None else float(raw_rate)
        # Epsilon-greedy over runs: an exploring run deliberately favours
        # formats the policy is not already reinforcing, so under-tested
        # options keep receiving evidence instead of being starved.
        explore = bool(preferred_formats) and explores(run.run_id, exploration_rate)
        selection_mode = "explore" if explore else "exploit"
        for candidate in candidates:
            if explore:
                candidate["learning_adjustment"] = 0.08 * float(candidate.get("format_name") not in preferred_formats)
            else:
                candidate["learning_adjustment"] = (
                    0.10 * float(candidate.get("format_name") in preferred_formats)
                    + 0.06 * float(candidate.get("hook_pattern") in preferred_hooks)
                    + 0.04 * float(candidate.get("emotion") in preferred_emotions)
                    + 0.04 * float(candidate.get("character") in preferred_characters)
                )
            candidate["exploration_rate"] = exploration_rate
            candidate["selection_mode"] = selection_mode
        candidates.sort(key=lambda x: float(x.get("learning_adjustment", 0.0)), reverse=True)
        # Adapt the high-volume idea pool to the production contract used by the
        # existing jury/animatic stages. Generation remains cheap; only a small
        # portfolio is rendered later.
        normalized = []
        for idea in candidates:
            candidate = dict(idea)
            candidate["premise_slug"] = candidate.get("idea_id", "idea")
            candidate["caption"] = f"{candidate.get('hook', '')} An original cartoon short."
            candidate["hook_pattern"] = str(candidate.get("hook_pattern") or "")
            if not candidate["hook_pattern"]:
                candidate["hook_pattern"] = str((candidate.get("hook_patterns") or [""])[0])
            if not candidate["hook_pattern"] and run.payload.get("ranked_signals_v2"):
                candidate["hook_pattern"] = str((run.payload["ranked_signals_v2"][0].get("hook_patterns") or [""])[0])
            candidate["research"] = {
                "title": candidate.get("title", ""),
                "source": "trend_engine_v2",
                "category": candidate.get("format_name", ""),
                "url": "",
                "published_at": "",
                "score": float(candidate.get("trend_score", 0.0) or 0.0),
            }
            candidate["format"] = {"name": candidate.get("format_name", "")}
            candidate["asset_ready"] = all(
                action_asset(candidate["character"], beat["asset_action"]) is not None
                for beat in candidate["beats"]
            ) or __import__("os").environ.get("NOVA_CARTOON_ANIMATION_BACKEND", "pillow").strip().lower() == "blender"
            normalized.append(candidate)
        topic = run.payload.get("topic") or (trend_rows[0].get("title") if trend_rows else "evergreen cartoon")
        self.plane.store.set_payload(run, candidates=normalized, idea_topic=topic, selection_mode=selection_mode)
        return {
            "selection_mode": selection_mode,
            "count": len(normalized),
            "generated_pool": len(raw_ideas),
            "topic": topic,
            "strategy_memory": memory.get("observations", 0),
        }

    def _idea_evaluator(self, run: CartoonRun) -> dict[str, Any]:
        candidates = run.payload.get("candidates", [])
        recent = run.payload.get("recent_titles", [])
        preselected = []
        for candidate in candidates:
            if not candidate.get("asset_ready"):
                continue
            deterministic = self.quality.evaluate(candidate, recent)
            jury = self.jury.judge(candidate)
            story_quality = self.story_engine.evaluate(candidate)
            if deterministic.passed and jury.passed and story_quality.passed:
                score = round(
                    deterministic.score * 0.35
                    + jury.score * 0.30
                    + story_quality.score / 10.0 * 0.15
                    + float(candidate["research"].get("score", 0.0)) * 0.20,
                    3,
                )
                # Keep expected audience outcome visible as a distinct ranking
                # signal; hard gates above remain authoritative.
                prediction = jury.prediction
                score = round(score * 0.95 + 0.05 * (
                    0.55 * float(prediction["predicted_retention"])
                    + 0.45 * float(prediction["predicted_shareability"])
                ), 3)
                # Learned policy nudges the ranking among gate-passing ideas
                # (at most +0.12); it can never rescue an idea that failed a gate.
                score = round(score + 0.5 * float(candidate.get("learning_adjustment", 0.0) or 0.0), 3)
                preselected.append({
                    "score": score,
                    "candidate": candidate,
                    "evaluation": asdict(deterministic),
                    "jury": jury_to_dict(jury),
                    "story_quality": story_quality.to_dict(),
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
            raise RuntimeError("No candidate passed the idea jury, story quality gate, and cheap animatic gate")
        self.plane.store.set_payload(
            run,
            winner=winner,
            alternates=preselected[1:4],
            animatic_attempts=attempts,
        )
        chosen = winner["candidate"]
        self._sync_experiment(
            run,
            idea_id=chosen.get("idea_id", chosen.get("premise_slug", "")),
            trend_id=chosen.get("trend_id", ""),
            format_name=chosen.get("format_name", ""),
            hook_pattern=chosen.get("hook_pattern", ""),
            hook_text=chosen.get("hook", ""),
            emotion=chosen.get("emotion", ""),
            character=chosen.get("character", ""),
            angle=chosen.get("angle", ""),
            duration=chosen.get("duration_target", 0.0),
            production_method=__import__("os").environ.get("NOVA_CARTOON_ANIMATION_BACKEND", "pillow"),
            selection_mode=chosen.get("selection_mode", "exploit"),
        )
        # Bind this local run to a deterministic controlled plan. No posting is
        # involved; the variant identity travels through the review outbox.
        from app.experiment_system import plan_experiment
        dimension = "format"
        chosen_value = chosen.get("format_name", "story_comedy")
        controls = list(dict.fromkeys([chosen_value, "alternative:" + str(chosen_value)]))
        controlled = plan_experiment(
            dimension=dimension, variants=controls,
            hypothesis="Format changes normalized audience outcome",
            success_metric="normalized_score", mode="exploration",
            context=str(chosen.get("trend_id", "")),
        )
        # The selected candidate is the materialized variant for this run.
        # Stable allocator remains available to allocate future candidate pools.
        assignment = next(v for v in controlled["variants"] if v["value"] == chosen_value)
        assignment = {"experiment_id": controlled["experiment_id"], "variant_id": assignment["variant_id"],
                      "dimension": dimension, "variant_value": chosen_value}
        self._sync_experiment(run, controlled_experiment_id=controlled["experiment_id"],
                              variant_id=assignment["variant_id"], variant_dimension=dimension,
                              variant_value=assignment["variant_value"],
                              hypothesis=controlled["hypothesis"], success_metric=controlled["success_metric"])
        return {
            "winner_score": winner["score"],
            "alternates": len(preselected[1:4]),
            "jury": True,
            "cheap_animatic": True,
            "animatic_attempts": len(attempts),
        }

    def _story_engine(self, run: CartoonRun) -> dict[str, Any]:
        candidate = run.payload["winner"]["candidate"]
        story = self.story_engine.build(candidate).to_dict()
        self.plane.store.set_payload(run, story=story, story_quality=story["quality"])
        return {
            "status": "passed",
            "score": story["quality"]["score"],
            "min_dimension": story["quality"]["min_dimension_score"],
            "dimensions": story["quality"]["dimension_count"],
            "quality_floor": story["quality"]["quality_floor"],
        }

    def _script_engine(self, run: CartoonRun) -> dict[str, Any]:
        candidate = run.payload["winner"]["candidate"]
        story = run.payload.get("story") or candidate
        contract = self.script_engine.build(story).to_dict()
        self.plane.store.set_payload(
            run,
            script_contract=contract,
            script=contract["text"],
            hook=contract["hook"],
            title=contract["title"],
        )
        return {
            "status": "passed",
            "beats": len(contract["beats"]),
            "target_seconds": f"{contract['target_duration']:.1f}",
            "quality_score": contract["quality"]["score"],
            "min_dimension": contract["quality"]["min_dimension_score"],
            "quality_floor": contract["quality"]["quality_floor"],
            "dimensions": contract["quality"]["dimension_count"],
            "story_locked": True,
        }

    def _storyboard_engine(self, run: CartoonRun) -> dict[str, Any]:
        candidate = run.payload["winner"]["candidate"]
        draft = {
            "title": candidate["title"],
            "hook": candidate["hook"],
            "character": candidate["character"],
            "script": run.payload["script"],
            "beats": self._script_beats(run),
        }
        storyboard = build_storyboard(draft)
        data = storyboard.to_dict()
        self.plane.store.set_payload(run, storyboard=data)
        return {"shots": len(storyboard.shots), "duration": storyboard.total_duration}

    def _script_beats(self, run: CartoonRun) -> list[dict[str, Any]]:
        """Script beats merged with the candidate's renderable actions, one per rendered beat.

        The Script Engine owns the spoken line (the approved hook replaces the
        first beat) and timing; the idea owns the action image for each beat.
        Falls back to the raw idea beats for runs without a script contract.
        """
        candidate = run.payload["winner"]["candidate"]
        idea_beats = [dict(b) for b in candidate.get("beats", [])]
        script_beats = list(run.payload.get("script_contract", {}).get("beats", []))
        if len(script_beats) != len(idea_beats):
            return [dict(b, dialogue=b.get("voice", "")) for b in idea_beats]
        merged = []
        for idea_beat, script_beat in zip(idea_beats, script_beats):
            row = dict(idea_beat)
            row.update({k: v for k, v in script_beat.items() if k not in {"action"}})
            row["asset_action"] = idea_beat.get("asset_action", script_beat.get("action", ""))
            row["voice"] = script_beat.get("voice_text") or script_beat.get("dialogue") or idea_beat.get("voice", "")
            row["dialogue"] = row["voice"]
            row["beat"] = idea_beat.get("beat", script_beat.get("story_role", ""))
            merged.append(row)
        return merged

    @staticmethod
    def _check_asset_identity(character: str, assets: list[str]) -> dict[str, Any]:
        """Check each action image's character region against the bible palette."""
        from PIL import Image

        from app.agents.character_world_bible import validate_asset_identity
        from app.providers.cartoon_rig import _cover, get_rig

        errors: list[str] = []
        checked = []
        for asset in dict.fromkeys(assets):
            try:
                bbox = get_rig(asset).bbox
            except ValueError:
                bbox = None  # Unknown layout (e.g. a Blender master sheet): check the whole frame.
            with Image.open(asset) as image:
                problems = validate_asset_identity(character, _cover(image), bbox)
            checked.append({"asset": asset, "errors": problems})
            errors.extend(f"{Path(asset).name}:{problem}" for problem in problems)
        return {"checked": checked, "errors": errors}

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
        identity = self._check_asset_identity(candidate["character"], assets)
        if identity["errors"]:
            raise RuntimeError("Character bible identity check failed: " + "; ".join(identity["errors"]))
        self.plane.store.set_payload(run, asset_paths=assets, asset_identity=identity)
        return {
            "coverage": coverage(type("Spec", (), {"character": candidate["character"], "beats": candidate["beats"]})()),
            "identity_checked": len(identity["checked"]),
        }

    def _animation_engine(self, run: CartoonRun) -> dict[str, Any]:
        from app.agents.edit_decision import build_edit_plan

        candidate = run.payload["winner"]["candidate"]
        images = [Path(p) for p in run.payload["asset_paths"]]
        script_beats = self._script_beats(run)
        captions = [beat["voice"] for beat in script_beats]
        sfx = [beat["sfx"] for beat in script_beats]
        beats = [RENDER_BEATS.get(beat["beat"], beat["beat"]) for beat in script_beats]
        durations = [float(beat.get("duration", 2.0)) for beat in script_beats]
        slug = candidate["premise_slug"] + "_factory"
        storyboard_shots = list(run.payload.get("storyboard", {}).get("shots", []))
        shot_specs = (
            [shot.get("performance", {}).get("shot_spec") for shot in storyboard_shots]
            if len(storyboard_shots) == len(images) else None
        )

        # Build the zero-credit Runway handoff for every storyboard shot.
        # Local rendering remains the default worker until a Runway execution
        # adapter is explicitly enabled; the plan is still persisted now so QA
        # and future provider workers consume the same Director intent.
        if storyboard_shots and len(storyboard_shots) == len(images):
            runway_plan = RunwayPerformanceProvider().build_storyboard_plan(
                storyboard_shots,
                images,
                aspect_ratio="9:16",
                resolution="720p",
            )
            runway_plan_path = ARTIFACTS / f"{slug}.runway_plan.json"
            _save(runway_plan_path, runway_plan)
            self.plane.store.set_payload(run, runway_plan=str(runway_plan_path))
        output = ARTIFACTS / f"{slug}.mp4"
        backend = __import__("os").environ.get("NOVA_CARTOON_ANIMATION_BACKEND", "pillow").strip().lower()
        if backend == "blender" and candidate.get("character") in {"PIP", "MOMO", "BUBU"}:
            provider = BlenderAnimationProvider()
            if not provider.available():
                raise RuntimeError("NOVA_CARTOON_ANIMATION_BACKEND=blender but Blender is unavailable")
            blender_output = ARTIFACTS / f"{slug}_blender.mp4"
            bg = Path(__import__("os").environ["NOVA_BLENDER_BACKGROUND"]) if __import__("os").environ.get("NOVA_BLENDER_BACKGROUND") else None
            info = provider.render_episode(
                ROOT, blender_output, captions, beats, durations, sfx,
                resolution_percent=int(__import__("os").environ.get("NOVA_BLENDER_RESOLUTION", "100")),
                action_assets=images,
                background_path=bg,
                character=candidate.get("character", "PIP"),
                actions=[b.get("asset_action", "") for b in candidate.get("beats", [])],
                hook_text=candidate.get("hook", ""),
            )
            output = Path(info["video"])
            self.plane.store.set_payload(run, animation_backend="blender", audio_embedded=True)
        else:
            render_cartoon_motion(
                images,
                captions,
                durations,
                beats,
                output,
                voice_lines=captions,
                sfx_kinds=sfx,
                fps=24,
                shot_specs=shot_specs,
                character=candidate.get("character"),
                scene=candidate.get("scene"),
            )
        motion_metadata = output.with_suffix(".motion.json")
        effective_durations = durations
        try:
            meta = json.loads(motion_metadata.read_text(encoding="utf-8"))
            effective_durations = list(meta.get("audio_sync", {}).get("effective_shot_durations", durations))
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        self.plane.store.set_payload(
            run,
            video=str(output),
            render_durations=effective_durations,
            animation_metadata=str(motion_metadata),
        )
        edit_plan = build_edit_plan(script_beats, effective_durations)
        edit_plan_path = ARTIFACTS / f"{slug}.edit_plan.json"
        _save(edit_plan_path, edit_plan)
        self.plane.store.set_payload(run, edit_plan=edit_plan, edit_plan_path=str(edit_plan_path))
        hosted = None
        if backend == "runway":
            # Hosted shots replace the local pictures; the local render still
            # supplies the voice/SFX track and timing the Editor cuts to.
            plan_path = run.payload.get("runway_plan")
            if not plan_path:
                raise RuntimeError("Runway backend needs one storyboard shot per beat to build a plan")
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            clips_dir = os.environ.get("NOVA_RUNWAY_CLIPS_DIR", "").strip()
            if clips_dir:
                # Clips made in the Runway web app / connector for this plan.
                hosted = runway_video.import_clips(plan, Path(clips_dir))
            else:
                hosted = runway_video.execute_plan(plan, ARTIFACTS / f"{slug}_runway", run_id=run.run_id, kind="generate")
            clips = [hosted["clips"][index] for index in sorted(hosted["clips"])]
            self.plane.store.set_payload(run, shot_clips=clips, runway_execution={
                k: v for k, v in hosted.items() if k != "clips"
            }, animation_backend="runway")
        else:
            # A local render replaces any hosted clips from an earlier attempt;
            # keeping them would assemble the old Runway shots and block the free repair.
            self.plane.store.set_payload(run, shot_clips=None, runway_execution=None, animation_backend=backend)
        return {
            "video": str(output),
            "renderer": "blender-cutout-performance-rig-v5" if backend == "blender" and candidate.get("character") in {"PIP", "MOMO", "BUBU"} else "cartoon-motion-3.6",
            "fps": 24,
            "script_driven": bool(run.payload.get("script_contract")),
            "director_shot_specs": sum(1 for spec in (shot_specs or []) if spec),
            "hosted_shots": len(hosted["clips"]) if hosted else 0,
        }

    def _rerender_for_motion_qa(self, run: CartoonRun, motion_result: dict[str, Any]) -> dict[str, Any]:
        """Surgically rerender failed shots plus their dependent outgoing bridges."""
        # A paid repair that was interrupted keeps its attempt number, so the retry
        # resumes the same Runway manifest instead of paying for the shots again.
        pending = run.payload.get("motion_repair_pending") or {}
        attempt = int(pending.get("attempt") or int(run.payload.get("motion_repair_attempts", 0)) + 1)
        if attempt > self._max_repair_attempts():
            raise RuntimeError(f"Motion QA repair budget exhausted after {attempt - 1} attempt(s)")

        candidate = run.payload["winner"]["candidate"]
        images = [Path(p) for p in run.payload["asset_paths"]]
        script_beats = self._script_beats(run)
        captions = [beat["voice"] for beat in script_beats]
        beats = [RENDER_BEATS.get(beat["beat"], beat["beat"]) for beat in script_beats]
        durations = [float(beat.get("duration", 2.0)) for beat in script_beats]
        reasons = list(motion_result.get("reasons", []))
        target_shots = list(motion_result.get("repair_shots", []))
        if not target_shots:
            target_shots = list(range(len(images)))

        plan = CartoonMotionQA.plan_repair(reasons, target_shots)

        # Build a zero-credit external-worker repair manifest alongside the
        # local rerender. A future Runway worker can consume this manifest
        # without regenerating the entire storyboard or spending credits on
        # unaffected shots.
        runway_plan_path = run.payload.get("runway_plan")
        runway_repair_path = None
        if runway_plan_path:
            try:
                runway_plan = json.loads(Path(runway_plan_path).read_text(encoding="utf-8"))
                runway_shots = list(runway_plan.get("shots", []))
                repair_shots = []
                for shot_index in target_shots:
                    if 0 <= shot_index < len(runway_shots):
                        repair_shots.append(
                            build_runway_repair_request(
                                runway_shots[shot_index],
                                {
                                    "reasons": reasons,
                                    "repair_shots": [shot_index],
                                    "plan": plan,
                                },
                            )
                        )
                if repair_shots:
                    runway_repair = {
                        "provider": runway_plan.get("provider", "runway"),
                        "source_plan": str(runway_plan_path),
                        "strategy": "targeted-rerender",
                        "trigger_reasons": reasons,
                        "shots": repair_shots,
                        "estimated_total_credits": round(
                            sum(float(item.get("estimated_credits", 0.0)) for item in repair_shots),
                            2,
                        ),
                    }
                    runway_repair_path = ARTIFACTS / f"{candidate['premise_slug']}_factory.runway_repair_plan.json"
                    _save(runway_repair_path, runway_repair)
                    self.plane.store.set_payload(
                        run,
                        runway_repair_plan=str(runway_repair_path),
                    )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                runway_repair_path = None

        if run.payload.get("shot_clips"):
            # A hosted reel is never silently re-rendered locally, and a paid
            # regeneration only happens when the owner has opted in.
            if os.environ.get("NOVA_RUNWAY_PAID_REPAIRS", "false").strip().lower() != "true":
                raise RuntimeError(
                    "Runway reel failed video QA (" + ", ".join(reasons) + "); paid repairs are off "
                    "(NOVA_RUNWAY_PAID_REPAIRS=false), so no credits were spent"
                )
            if not runway_repair_path:
                raise RuntimeError("Runway reel failed video QA but no repair plan could be built")
            return self._repair_hosted_shots(run, attempt, reasons, target_shots, plan, Path(runway_repair_path))

        tuning = {
            **plan["profile"],
            "repair_attempt": attempt,
            "repair_reasons": reasons,
        }

        source = Path(run.payload["video"])
        slug = candidate["premise_slug"] + "_factory"
        output = ARTIFACTS / f"{slug}.mp4"
        rerender_cartoon_shots(
            source,
            images,
            captions,
            durations,
            beats,
            output,
            target_shots,
            render_tuning=tuning,
            fps=24,
        )
        history = list(run.payload.get("motion_repair_history", []))
        history.append({
            "attempt": attempt,
            "reasons": reasons,
            "target_shots": target_shots,
            "plan": plan,
            "video": str(output),
        })
        self.plane.store.set_payload(
            run,
            video=str(output),
            animation_metadata=str(output.with_suffix(".motion.json")),
            motion_repair_attempts=attempt,
            motion_repair_target_shots=target_shots,
            motion_repair_history=history,
        )
        return {
            "attempt": attempt,
            "plan": plan,
            "target_shots": target_shots,
            "video": str(output),
        }

    def _repair_hosted_shots(self, run: CartoonRun, attempt: int, reasons: list[str], target_shots: list[int],
                             plan: dict[str, Any], repair_plan_path: Path) -> dict[str, Any]:
        """Targeted hosted repair: regenerate only failing shots, then re-cut the reel.

        The paid clips are stored in ``shot_clips`` before the re-cut, so a failed
        assembly is retried without paying again. A reel that fails the editor's
        checks is never adopted as the episode video.
        """
        candidate = run.payload["winner"]["candidate"]
        repair_plan = json.loads(repair_plan_path.read_text(encoding="utf-8"))
        runway_plan = json.loads(Path(run.payload["runway_plan"]).read_text(encoding="utf-8"))
        pending = {"attempt": attempt, "reasons": reasons, "target_shots": target_shots,
                   "repair_plan": str(repair_plan_path)}
        self.plane.store.set_payload(run, motion_repair_pending=pending)
        result = runway_video.execute_plan(repair_plan, ARTIFACTS / f"{candidate['premise_slug']}_runway_repair_{attempt}",
                                           run_id=run.run_id, kind="repair")
        # shot_clips is ordered by shot number; map each replacement to its slot.
        order = sorted(int(shot.get("shot_index", position)) for position, shot in enumerate(runway_plan.get("shots", [])))
        clips = list(run.payload["shot_clips"])
        for shot, clip in result["clips"].items():
            clips[order.index(int(shot))] = clip
        self.plane.store.set_payload(run, shot_clips=clips, motion_repair_pending=pending)
        audio = run.payload.get("audio_metadata", {}).get("audio_source")
        output = ARTIFACTS / f"{candidate['premise_slug']}_edited_r{attempt}.mp4"
        report = assemble_reel([Path(c) for c in clips], run.payload["edit_plan"], output,
                               audio=Path(audio) if audio else None)
        if not report["passed"]:
            raise RuntimeError("Hosted repair: reel does not match the edit plan: " + ", ".join(report["reasons"]))
        history = list(run.payload.get("motion_repair_history", []))
        history.append({"attempt": attempt, "reasons": reasons, "target_shots": target_shots, "plan": plan,
                        "video": str(output), "strategy": "hosted-targeted-regeneration"})
        self.plane.store.set_payload(run, video=str(output), editor=report,
                                     motion_repair_attempts=attempt, motion_repair_target_shots=target_shots,
                                     motion_repair_history=history, motion_repair_pending=None)
        return {"attempt": attempt, "plan": plan, "target_shots": target_shots, "video": str(output),
                "strategy": "hosted-targeted-regeneration"}

    def _audio_engine(self, run: CartoonRun) -> dict[str, Any]:
        video = Path(run.payload["video"])
        if run.payload.get("audio_embedded") or run.payload.get("animation_backend") == "blender":
            result = {"voice_sfx_mixed": True, "audio_embedded": True, "audio_source": str(video)}
            # Some legacy/unit callers pass lightweight run-like objects; keep
            # evidence persistence for real CartoonRun instances only.
            if hasattr(run, "run_id") and hasattr(run, "stages"):
                self.plane.store.set_payload(run, audio_metadata=result)
            return result
        audio_candidates = list(video.parent.glob(video.stem + "_frames/episode_audio.wav"))
        has_audio_source = bool(audio_candidates and audio_candidates[0].exists())
        if not has_audio_source:
            raise RuntimeError("Audio stage expected renderer-produced episode_audio.wav")
        result = {"voice_sfx_mixed": True, "audio_source": str(audio_candidates[0])}
        if hasattr(run, "run_id") and hasattr(run, "stages"):
            self.plane.store.set_payload(run, audio_metadata=result)
        return result

    def _caption_engine(self, run: CartoonRun) -> dict[str, Any]:
        disclosure = "AI-generated • fictional cartoon"
        result = {"captions": [beat["voice"] for beat in self._script_beats(run)], "disclosure": disclosure}
        self.plane.store.set_payload(run, caption_metadata=result)
        return result

    def _editor_engine(self, run: CartoonRun) -> dict[str, Any]:
        """Editor AI: assemble hosted shot clips, or conform the local reel, to the edit plan."""
        candidate = run.payload["winner"]["candidate"]
        plan = run.payload.get("edit_plan") or {}
        if not plan.get("shots"):
            raise RuntimeError("Editor needs an edit plan from the animation stage")
        clips = list(run.payload.get("shot_clips") or [])
        if clips:
            audio = run.payload.get("audio_metadata", {}).get("audio_source")
            output = ARTIFACTS / f"{candidate['premise_slug']}_edited.mp4"
            report = assemble_reel([Path(c) for c in clips], plan, output, audio=Path(audio) if audio else None)
            mode = "assembled"
            if report["passed"]:
                self.plane.store.set_payload(run, video=str(output))
        else:
            metadata_path = Path(str(run.payload.get("animation_metadata", "")))
            shot_frames, fps = None, 24
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                shot_frames = [int(row["frames"]) for row in metadata.get("shots", [])] or None
                fps = int(metadata.get("fps", 24))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
            report = verify_reel(Path(run.payload["video"]), plan, shot_frames=shot_frames, fps=fps)
            mode = "conformed"
        report["mode"] = mode
        self.plane.store.set_payload(run, editor=report)
        if not report["passed"]:
            raise RuntimeError("Editor: reel does not match the edit plan: " + ", ".join(report["reasons"]))
        return {"mode": mode, "duration": report["duration_seconds"], "cuts": len(report["measured_cut_points_seconds"])}

    def _final_qa(self, run: CartoonRun, media: dict[str, Any] | None, motion: dict[str, Any] | None, *, enforce_publish: bool = False) -> dict[str, Any]:
        self.plane.store.set_payload(run, final_media_qa=media, final_motion_qa=motion)
        result = self.final_qa.evaluate(run.payload)
        self.plane.store.set_payload(run, final_qa=result)
        if enforce_publish and not result["passed"]:
            raise RuntimeError("Final QA blocked publish: " + ", ".join(result["blocking_failures"]))
        return result

    @staticmethod
    def _max_repair_attempts() -> int:
        try:
            return max(0, int(os.environ.get("NOVA_MAX_REPAIR_ATTEMPTS", str(DEFAULT_MAX_REPAIR_ATTEMPTS))))
        except ValueError:
            return DEFAULT_MAX_REPAIR_ATTEMPTS

    def _rerender_full_for_video_qa(self, run: CartoonRun, media: dict[str, Any]) -> dict[str, Any]:
        """Rebuild the reel after a technical failure (bad file, codec, missing audio).

        A hosted reel is re-cut from the clips already downloaded, so a broken
        assembly never spends Runway credits again; a local reel is re-rendered.
        """
        attempt = int(run.payload.get("video_repair_attempts", 0)) + 1
        if run.payload.get("shot_clips"):
            self._editor_engine(run)
            strategy = "reassemble_hosted_clips"
        else:
            self._animation_engine(run)
            strategy = "full_rerender"
        history = list(run.payload.get("video_repair_history", []))
        history.append({"attempt": attempt, "strategy": strategy, "reasons": list(media.get("reasons", [])),
                        "video": run.payload["video"]})
        self.plane.store.set_payload(run, video_repair_attempts=attempt, video_repair_history=history)
        return {"attempt": attempt, "strategy": strategy, "reasons": list(media.get("reasons", [])),
                "video": run.payload["video"]}

    def _video_qa(self, run: CartoonRun) -> dict[str, Any]:
        """Visual QA -> Repair AI -> Visual QA, repeated up to a bounded budget.

        Motion failures get a targeted rerender of only the failing shots;
        technical failures (file, codec, audio, dimensions) get a full rerender.
        The script, assets, voice and approved idea never change during repair.
        """
        max_attempts = self._max_repair_attempts()
        video = Path(run.payload["video"])
        repairs: list[dict[str, Any]] = []
        initial_motion: dict[str, Any] | None = None
        while True:
            media = self.quality.render_check(video)
            motion = self.quality.motion_check(video) if media["passed"] else None
            if media["passed"] and motion is not None and motion["passed"]:
                break
            technical = not media["passed"]
            reasons = list(media["reasons"] if technical else motion["reasons"])
            if technical and set(reasons) & UNREPAIRABLE_VIDEO_REASONS:
                raise RuntimeError("Video QA failed: " + ", ".join(reasons))
            if len(repairs) >= max_attempts:
                label = "Video QA" if technical else "Motion QA"
                raise RuntimeError(f"{label} failed after {len(repairs)} repair attempt(s): " + ", ".join(reasons))
            if technical:
                repair = self._rerender_full_for_video_qa(run, media)
            else:
                initial_motion = initial_motion or motion
                repair = self._rerender_for_motion_qa(run, motion)
            repairs.append(repair)
            video = Path(repair["video"])

        # Measured mp4 checks (loudness, silence, edges, freezes, cuts, captions)
        # run once the rerender loop settles; their failures block at publish.
        media = MediaQA.combine(media, self.media_qa.evaluate(video))
        final_qa = self._final_qa(run, media, motion)
        record_provider_observation(
            provider=str(run.payload.get("animation_backend", "pillow")),
            task="animation",
            success=True,
            experiment_id=str(run.payload.get("experiment", {}).get("experiment_id", "")),
            repaired=bool(repairs),
        )
        result: dict[str, Any] = {
            "passed": True,
            "media": media,
            "motion": motion,
            "final_qa": final_qa,
            "repaired": bool(repairs),
            "repair_attempts": len(repairs),
            "max_repair_attempts": max_attempts,
        }
        if repairs:
            result["repair"] = repairs[-1]
            result["repairs"] = repairs
        if initial_motion is not None:
            result["initial_motion"] = initial_motion
        return result

    def _publish(self, run: CartoonRun) -> dict[str, Any]:
        # Re-evaluate at publish readiness, after story, script, audio, captions,
        # and video QA have all had a chance to record their evidence.
        media = run.payload.get("final_media_qa")
        motion = run.payload.get("final_motion_qa")
        self._final_qa(run, media, motion, enforce_publish=True)
        gate = run.payload.get("final_qa")
        if not isinstance(gate, dict) or gate.get("publish_gate", {}).get("decision") != "allow":
            raise RuntimeError("Publish blocked: final QA publish-gate evidence is missing or failed")
        candidate = run.payload["winner"]["candidate"]
        _guard_public_text(candidate)
        experiment = ensure_experiment_identity(run.payload, run.run_id)
        item = {
            "run_id": run.run_id,
            "experiment_id": experiment.get("experiment_id", ""),
            "video": run.payload["video"],
            "title": candidate["title"],
            "caption": candidate.get("caption", ""),
            "ai_generated": True,
            "status": "ready_for_review",
            "final_qa": gate["publish_gate"],
            "audience_classification_review_required": True,
            "experiment_metadata": {
                "controlled_experiment_id": experiment.get("controlled_experiment_id", ""),
                "variant_id": experiment.get("variant_id", ""),
                "variant_dimension": experiment.get("variant_dimension", ""),
                "variant_value": experiment.get("variant_value", ""),
                "idea_id": candidate.get("idea_id", candidate.get("premise_slug", "")),
                "trend_id": candidate.get("trend_id", ""),
                "format_name": candidate.get("format_name", ""),
                "hook_pattern": candidate.get("hook_pattern", ""),
                "emotion": candidate.get("emotion", ""),
                "character": candidate.get("character", ""),
                "angle": candidate.get("angle", ""),
                "duration_target": candidate.get("duration_target", 0.0),
                "production_cost": candidate.get("production_cost", 0.0),
                "selection_mode": candidate.get("selection_mode", "exploit"),
                "edit_plan": run.payload.get("edit_plan", {}),
            },
        }
        queue = []
        if OUTBOX.exists():
            try:
                queue = json.loads(OUTBOX.read_text())
            except json.JSONDecodeError:
                queue = []
        queue.append(item)
        _save(OUTBOX, queue)
        content_metadata = dict(run.payload.get("content_metadata", {}))
        content_metadata[run.run_id] = dict(item["experiment_metadata"], experiment_id=experiment.get("experiment_id", ""))
        content_metadata[experiment.get("experiment_id", run.run_id)] = content_metadata[run.run_id]
        self.plane.store.set_payload(run, content_metadata=content_metadata)
        self._sync_experiment(
            run,
            publish_status="ready_for_review",
            video=run.payload.get("video", ""),
            experiment_metadata=item["experiment_metadata"],
        )
        return {"provider": "outbox", "status": "ready_for_review", "experiment_metadata": item["experiment_metadata"]}

    def _analytics(self, run: CartoonRun) -> dict[str, Any]:
        # Pull first-party metrics only for content IDs explicitly bound to this
        # experiment. Never ingest unrelated recent platform posts.
        content_map = run.payload.get("published_content_map", {})
        known = {
            (str(platform), str(content_id))
            for platform, mapping in content_map.items()
            if isinstance(mapping, dict)
            for content_id in mapping.keys()
        }
        if not known:
            snapshots, status = [], {"connectors": [], "records": 0, "status": "awaiting_published_content_map"}
        else:
            snapshots, status = collect_platform_metrics(content_map)
            snapshots = [row for row in snapshots if (row.platform, row.content_id) in known]

        if snapshots:
            rows = []
            from app.experiment_system import record_observation
            for snapshot in snapshots:
                metadata = dict(run.payload.get("content_metadata", {}).get(snapshot.content_id, {}))
                metadata.update(content_map.get(snapshot.platform, {}).get(snapshot.content_id, {}))
                metadata["experiment_id"] = run.payload.get("experiment", {}).get("experiment_id", metadata.get("experiment_id", ""))
                controlled = run.payload.get("experiment", {})
                metadata.setdefault("controlled_experiment_id", controlled.get("controlled_experiment_id", ""))
                metadata.setdefault("variant_id", controlled.get("variant_id", ""))
                metadata.setdefault("variant_dimension", controlled.get("variant_dimension", ""))
                metric_row = metric_to_record(snapshot, metadata)
                rows.append(metric_row)
                if metadata.get("controlled_experiment_id") and metadata.get("variant_id"):
                    record_observation(
                        str(metadata["controlled_experiment_id"]),
                        variant_id=str(metadata["variant_id"]),
                        metric_value=float(metric_row.get("score", 0.0)),
                        content_id=str(snapshot.content_id), run_id=run.run_id,
                        analytics_id=str(snapshot.captured_at),
                    )
            merged = ingest_learning_metrics(rows)
            self._sync_experiment(run, platform_metrics=len(rows), last_metrics_at=max((r.get("captured_at", "") for r in rows), default=""))
        else:
            merged = []
        self.plane.store.set_payload(
            run,
            platform_metrics_status=status,
            platform_metric_records=len(merged),
            metrics_pending=not bool(snapshots),
        )
        return {
            "status": "metrics_collected" if snapshots else "awaiting_published_content_map_or_metrics",
            "records": len(snapshots),
            "connectors": status.get("connectors", []),
        }

    def _learning(self, run: CartoonRun) -> dict[str, Any]:
        memory = build_strategy_memory()
        failure_summary = summarize_failures()
        provider_summary = summarize_providers()
        next_loop_plan = build_next_loop_plan(
            memory,
            failure_summary=failure_summary,
            provider_summary=provider_summary,
        )
        # Persist the same auditable action layer alongside posterior memory;
        # existing strategy fields remain intact for older consumers.
        from app import learning_v2
        learning_v2._write_json(learning_v2.STRATEGY_FILE, memory)
        # Keep the legacy snapshot too so older dashboards and tests remain valid.
        from app.learning import build_learning_snapshot
        legacy = build_learning_snapshot()
        experiment = update_experiment_learning(
            {
                **run.payload.get("experiment", {}),
                "learning_state": {
                    "strategy_observations": memory.get("observations", 0),
                    "failure_observations": failure_summary.get("observations", 0),
                    "provider_observations": provider_summary.get("observations", 0),
                    "next_loop": next_loop_plan.get("experiments", []),
                },
            },
            strategy_memory=memory,
            next_loop_plan=next_loop_plan,
        )
        result = {
            "v2": memory,
            "legacy": legacy,
            "failure_memory": failure_summary,
            "provider_memory": provider_summary,
            "next_loop_plan": next_loop_plan,
            "experiment": experiment,
        }
        self.plane.store.set_payload(
            run,
            learning=result,
            strategy_memory=memory,
            next_loop_plan=next_loop_plan,
        )
        return result


if __name__ == "__main__":
    run = CartoonFactoryOrchestrator().run()
    print(json.dumps({"run_id": run.run_id, "status": run.status, "current_stage": run.current_stage}, indent=2))
    # Non-zero when a stage failed, so schedulers (GitHub Actions, cron) see it.
    raise SystemExit(0 if run.status == "completed" else 1)
