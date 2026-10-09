"""Zero-credit end-to-end proof of NOVA's publish -> analytics -> learning -> ideas loop."""

from pathlib import Path

import app.agent_memory as agent_memory
import app.cartoon_control as control
import app.cartoon_factory_orchestrator as orch
import app.learning_v2 as learning
from app.providers.platform_analytics import PlatformMetricSnapshot
from app.providers.publisher import build_publisher


def _make_run(orchestrator, run_id, video_path, fmt):
    store = control.CartoonRunStore(run_id)
    run = store.create(
        {
            "topic": "closed loop simulation",
            "video": str(video_path),
            "winner": {
                "candidate": {
                    "idea_id": f"{fmt}-idea",
                    "title": f"{fmt} experiment",
                    "caption": "simulation",
                    "format_name": fmt,
                    "hook_pattern": "curiosity",
                    "emotion": "amusement",
                    "character": "MOMO",
                    "angle": "synthetic benchmark",
                    "duration_target": 8,
                    "production_cost": 0,
                    "beats": [{"voice": "Test beat"}],
                }
            },
            "final_media_qa": {"passed": True, "reasons": []},
            "final_motion_qa": {"passed": True, "score": 0.95, "reasons": [], "audio_sync_score": 0.98},
            "story_quality": {"passed": True, "score": 9.0, "blockers": []},
            "script_contract": {"quality": {"dimensions": [{"name": "caption_readiness", "score": 9.0}]}},
            "runtime_qa": {"passed": True, "score": 1.0, "reasons": []},
        }
    )
    orchestrator.plane.store = store
    orchestrator._publish(run)
    return store


def test_full_closed_loop_publish_metrics_learning_changes_next_ideas(monkeypatch, tmp_path):
    # Isolate every persistent memory store: this test cannot affect real NOVA history.
    monkeypatch.setattr(control, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(orch, "OUTBOX", tmp_path / "publish_queue.json")
    monkeypatch.setattr(orch, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(learning, "METRICS_FILE", tmp_path / "metrics.json")
    monkeypatch.setattr(learning, "STRATEGY_FILE", tmp_path / "strategy.json")
    monkeypatch.setattr(agent_memory, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(agent_memory, "EXPERIMENTS_FILE", tmp_path / "experiments.json")
    monkeypatch.setattr(agent_memory, "FAILURES_FILE", tmp_path / "failures.json")
    monkeypatch.setattr(agent_memory, "PROVIDERS_FILE", tmp_path / "providers.json")
    monkeypatch.setattr(agent_memory, "NEXT_LOOP_FILE", tmp_path / "next_loop.json")

    video = tmp_path / "simulation.mp4"
    # Final QA intentionally rejects placeholder artifacts; this simulation
    # supplies the minimum complete evidence needed to exercise the publish loop.
    video.write_bytes(b"x" * 100_001)

    orchestrator = orch.CartoonFactoryOrchestrator()

    # EXPERIMENT #1: two published variants with deliberately different outcomes.
    story_store = _make_run(orchestrator, "sim-story-001", video, "story_comedy")
    dance_store = _make_run(orchestrator, "sim-dance-001", video, "dance")

    story = story_store.load()
    dance = dance_store.load()

    story_pub = build_publisher("instagram").publish(
        video, experiment_id=story.payload["experiment"]["experiment_id"], dry_run=True
    )
    dance_pub = build_publisher("instagram").publish(
        video, experiment_id=dance.payload["experiment"]["experiment_id"], dry_run=True
    )

    orchestrator.bind_published_content(
        story.run_id,
        {"instagram": {story_pub.content_id: {"format_name": "story_comedy"}}},
    )
    orchestrator.bind_published_content(
        dance.run_id,
        {"instagram": {dance_pub.content_id: {"format_name": "dance"}}},
    )

    synthetic = {
        story_pub.content_id: PlatformMetricSnapshot(
            content_id=story_pub.content_id,
            platform="instagram",
            views=10000,
            likes=900,
            comments=120,
            shares=700,
            saves=350,
            retention_pct=92,
            avg_watch_seconds=7.36,
            duration_seconds=8,
            followers_gained=180,
        ),
        dance_pub.content_id: PlatformMetricSnapshot(
            content_id=dance_pub.content_id,
            platform="instagram",
            views=10000,
            likes=250,
            comments=20,
            shares=35,
            saves=10,
            retention_pct=18,
            avg_watch_seconds=1.44,
            duration_seconds=8,
            followers_gained=5,
        ),
    }

    def fake_collect(content_map):
        rows = []
        for mapping in content_map.values():
            for content_id in mapping:
                rows.append(synthetic[content_id])
        return rows, {"connectors": [{"platform": "simulation", "status": "synthetic_test"}], "records": len(rows)}

    monkeypatch.setattr(orch, "collect_platform_metrics", fake_collect)

    # WAIT FOR METRICS -> ANALYTICS -> LEARNING.
    story_after = orchestrator.refresh_metrics_and_learning(story.run_id)
    dance_after = orchestrator.refresh_metrics_and_learning(dance.run_id)

    policy = dance_after.payload["next_loop_plan"]["policy"]
    assert story_after.payload["platform_metric_records"] == 1
    assert dance_after.payload["platform_metric_records"] >= 1
    assert policy["preferred_formats"][0] == "story_comedy"
    assert policy["preferred_formats"] == ["story_comedy"]

    # EXPERIMENT #2: Idea Generator must consume the learned policy.
    candidates = [
        {
            "idea_id": "story-next",
            "title": "Learned Story",
            "format_name": "story_comedy",
            "hook_pattern": "curiosity",
            "emotion": "amusement",
            "character": "MOMO",
            "beats": [],
            "trend_score": 0.5,
        },
        {
            "idea_id": "dance-next",
            "title": "Old Dance",
            "format_name": "dance",
            "hook_pattern": "challenge",
            "emotion": "cute",
            "character": "PIP",
            "beats": [],
            "trend_score": 0.5,
        },
    ]
    monkeypatch.setattr(orch, "generate_ideas", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(orch, "select_portfolio", lambda rows, memory=None, limit=60: [dict(x) for x in rows])

    next_store = control.CartoonRunStore("sim-next-002")
    next_run = next_store.create(
        {
            "topic": "next experiment",
            "ranked_signals_v2": [{"title": "test trend", "hook_patterns": ["curiosity"]}],
            "format_candidates": [],
            "learning_policy": policy,
        }
    )
    orchestrator.plane.store = next_store
    result = orchestrator._idea_generator(next_run)
    ranked = next_run.payload["candidates"]

    assert result["count"] == 2
    assert ranked[0]["format_name"] == "story_comedy"
    assert ranked[0]["learning_adjustment"] > ranked[1]["learning_adjustment"]
    assert ranked[0]["learning_adjustment"] == 0.10
    assert ranked[1]["learning_adjustment"] == 0.0
    assert ranked[1]["learning_adjustment"] == 0.0
