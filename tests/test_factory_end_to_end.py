"""Offline end-to-end proof: all 15 factory stages run in order and ship an episode.

Unlike the unit tests, nothing here is injected mid-pipeline: research returns
fixture trends (no network), then every real stage runs — idea generation, jury,
cheap animatic, story, script, storyboard, assets, local render, audio, captions,
video QA, publish outbox, analytics and learning. Persistent stores are redirected
to tmp_path so the test cannot touch real NOVA history.
"""
import json
import shutil
import subprocess

import pytest

import app.agent_memory as agent_memory
import app.cartoon_control as control
import app.cartoon_factory_orchestrator as orch
import app.experiment_system as experiment_system
import app.learning as legacy_learning
import app.learning_v2 as learning
import app.trend_history as trend_history
from app.agents.cartoon_research import ResearchSignal

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg")
    or not shutil.which("ffprobe")
    or not (shutil.which("say") or shutil.which("espeak-ng") or shutil.which("espeak")),
    reason="end-to-end render needs ffmpeg, ffprobe and a local TTS engine",
)


def _isolate(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    monkeypatch.setattr(control, "RUNS_DIR", root / "runs")
    monkeypatch.setattr(orch, "ARTIFACTS", root)
    monkeypatch.setattr(orch, "OUTBOX", root / "publish_queue.json")
    monkeypatch.setattr(learning, "METRICS_FILE", root / "metrics.json")
    monkeypatch.setattr(learning, "METRICS_HISTORY_FILE", root / "metrics_history.json")
    monkeypatch.setattr(learning, "STRATEGY_FILE", root / "strategy_memory.json")
    monkeypatch.setattr(legacy_learning, "METRICS_FILE", root / "metrics.json")
    monkeypatch.setattr(legacy_learning, "LEARNING_FILE", root / "learning.json")
    monkeypatch.setattr(agent_memory, "ARTIFACTS", root)
    monkeypatch.setattr(agent_memory, "EXPERIMENTS_FILE", root / "experiments.json")
    monkeypatch.setattr(agent_memory, "FAILURES_FILE", root / "failure_memory.json")
    monkeypatch.setattr(agent_memory, "PROVIDERS_FILE", root / "provider_memory.json")
    monkeypatch.setattr(agent_memory, "NEXT_LOOP_FILE", root / "next_loop_plan.json")
    monkeypatch.setattr(trend_history, "HISTORY_FILE", root / "trend_history.json")
    monkeypatch.setattr(experiment_system, "REGISTRY_FILE", root / "experiment_registry.json")
    monkeypatch.delenv("NOVA_CARTOON_ANIMATION_BACKEND", raising=False)
    monkeypatch.delenv("NOVA_VOICE_BACKEND", raising=False)
    return root


def _offline_research(monkeypatch):
    signals = [
        ResearchSignal("Robot vacuum fails dance challenge", "fixture", "tech", score=0.8,
                       published_at="2026-10-09T09:00:00+00:00"),
        ResearchSignal("Cats refusing to get up on Monday mornings", "fixture", "pets", score=0.7,
                       published_at="2026-10-09T10:00:00+00:00"),
    ]
    monkeypatch.setattr(orch, "discover", lambda limit=30: signals)
    monkeypatch.setattr(orch, "discover_social_trends", lambda: ([], {"fixture": "offline"}))


def _probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def test_factory_runs_every_stage_and_queues_a_reviewable_episode(monkeypatch, tmp_path):
    root = _isolate(monkeypatch, tmp_path)
    _offline_research(monkeypatch)

    factory = orch.CartoonFactoryOrchestrator()
    run = factory.run("offline end-to-end")

    failed = {k: v.error for k, v in run.stages.items() if v.status != "passed"}
    assert run.status == "completed", failed
    assert set(run.stages) == {stage.value for stage in control.STAGES}

    payload = run.payload
    candidate = payload["winner"]["candidate"]

    # A real vertical video with audio was rendered and passed video QA.
    video = orch.Path(payload["video"])
    probe = _probe(video)
    streams = {s["codec_type"]: s for s in probe["streams"]}
    assert (streams["video"]["width"], streams["video"]["height"]) == (1080, 1920)
    assert "audio" in streams
    assert 5.0 <= float(probe["format"]["duration"]) <= 15.0
    assert payload["final_qa"]["publish_gate"]["decision"] == "allow"

    # The spoken lines are the Script Engine's, not the raw idea text.
    script_lines = [b["voice_text"] for b in payload["script_contract"]["beats"]]
    motion = json.loads(orch.Path(payload["animation_metadata"]).read_text())
    assert payload["caption_metadata"]["captions"] == script_lines
    assert script_lines[0] == candidate["hook"]

    # Director AI planned exactly the rendered beats, and the renderer used its plans.
    shots = payload["storyboard"]["shots"]
    assert [s["action"] for s in shots] == [b["asset_action"] for b in candidate["beats"]]
    animation = run.stages[control.CartoonStage.ANIMATION_ENGINE.value].artifact
    assert animation["director_shot_specs"] == len(shots)
    assert all("shot_spec" in perf for perf in motion["audio_sync"]["voice_performance"])

    # Editor AI conformed the rendered reel to the edit plan before QA.
    assert payload["editor"]["passed"] and payload["editor"]["mode"] == "conformed"
    assert len(payload["editor"]["measured_cut_points_seconds"]) == len(shots) - 1

    # The episode sits in the approval-first outbox, disclosed as AI-generated.
    queue = json.loads((root / "publish_queue.json").read_text())
    assert len(queue) == 1
    assert queue[0]["status"] == "ready_for_review" and queue[0]["ai_generated"] is True

    # Learning persisted a policy, and the next run starts with it.
    assert (root / "next_loop_plan.json").exists()
    next_run = factory.create_run("next")
    assert next_run.payload["next_loop_plan"] == json.loads((root / "next_loop_plan.json").read_text())
