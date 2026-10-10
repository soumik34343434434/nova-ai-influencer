from pathlib import Path

import pytest

from app.cartoon_factory_orchestrator import CartoonFactoryOrchestrator
from app.cartoon_control import CartoonRun


class FakeQuality:
    def __init__(self):
        self.motion_calls = 0

    def render_check(self, path: Path):
        return {"passed": True, "reasons": []}

    def motion_check(self, path: Path):
        self.motion_calls += 1
        if self.motion_calls == 1:
            return {
                "passed": False,
                "score": 0.4,
                "reasons": ["insufficient_character_motion"],
                "repair_shots": [2],
            }
        return {"passed": True, "score": 0.8, "reasons": []}

    def evaluate(self, *args, **kwargs):
        raise AssertionError("idea evaluation is not part of this repair test")


def test_video_qa_triggers_one_targeted_rerender(monkeypatch, tmp_path: Path):
    import app.cartoon_factory_orchestrator as factory_module

    quality = FakeQuality()
    orchestrator = CartoonFactoryOrchestrator(quality=quality)
    run = orchestrator.create_run("repair-test")
    candidate = {
        "premise_slug": "repair_test",
        "character": "PIP",
        "beats": [
            {"beat": "setup", "voice": "One.", "sfx": "none", "duration": 2.0},
            {"beat": "escalation", "voice": "Two.", "sfx": "none", "duration": 2.0},
            {"beat": "reaction", "voice": "Three.", "sfx": "none", "duration": 2.0},
            {"beat": "punchline", "voice": "Four.", "sfx": "none", "duration": 2.0},
        ],
    }
    orchestrator.plane.store.set_payload(
        run,
        winner={"candidate": candidate},
        asset_paths=["a.png", "b.png", "c.png", "d.png"],
        video=str(tmp_path / "initial.mp4"),
    )

    calls = []

    def fake_rerender(source, images, captions, durations, beats, output, target_shots, **kwargs):
        calls.append(
            {
                "source": str(source),
                "images": [str(x) for x in images],
                "durations": durations,
                "beats": beats,
                "target_shots": target_shots,
                "tuning": kwargs["render_tuning"],
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"x" * 100_000)
        output.with_suffix(".motion.json").write_text("{}")
        return output

    monkeypatch.setattr(factory_module, "rerender_cartoon_shots", fake_rerender)
    monkeypatch.setattr(
        orchestrator,
        "quality",
        quality,
    )

    result = orchestrator._video_qa(run)

    assert result["passed"]
    assert result["repaired"] is True
    assert result["repair_attempts"] == 1
    assert len(calls) == 1
    assert calls[0]["tuning"]["repair_attempt"] == 1
    assert calls[0]["tuning"]["motion_gain"] > 1.0
    assert calls[0]["target_shots"] == [2]
    assert calls[0]["images"] == run.payload["asset_paths"]
    assert run.payload["winner"]["candidate"]["premise_slug"] == "repair_test"
    assert run.payload["motion_repair_attempts"] == 1
    # Video QA records the aggregate result, but missing publish-only domains
    # must not prevent this focused motion-repair stage from succeeding.
    assert run.payload["final_qa"]["publish_gate"]["decision"] == "block"
    assert "story_payoff:story_quality_evidence_missing" in run.payload["final_qa"]["blocking_failures"]
    # The same missing evidence remains fail-closed once the run reaches publish.
    with pytest.raises(RuntimeError, match="Final QA blocked publish"):
        orchestrator._publish(run)


def test_runway_repair_request_carries_expanded_plan_without_provider_call():
    from app.providers.ai_performance import build_runway_repair_request
    from app.agents.cartoon_motion_qa import CartoonMotionQA

    plan = CartoonMotionQA.plan_repair(["hand_deformation", "prop_drift"], [1])
    request = build_runway_repair_request(
        {"prompt": "Character holds a cup."},
        {"reasons": plan["reasons"], "repair_shots": [1], "plan": plan},
    )
    assert request["repair"]["target_shots"] == [1]
    assert set(request["repair"]["domains"]) == {"hands", "props"}
    assert request["repair"]["confidence"] == plan["confidence"]
    assert len(request["repair"]["instructions"]) == 2


class ScriptedQuality:
    """Returns pre-scripted (media, motion) QA results, one pair per QA pass."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.passes = 0

    def render_check(self, path: Path):
        media, _ = self.outcomes[min(self.passes, len(self.outcomes) - 1)]
        if not media["passed"]:
            self.passes += 1
        return media

    def motion_check(self, path: Path):
        _, motion = self.outcomes[min(self.passes, len(self.outcomes) - 1)]
        self.passes += 1
        return motion


OK_MEDIA = {"passed": True, "reasons": []}
BAD_MOTION = {"passed": False, "score": 0.4, "reasons": ["possible_pose_popping"], "repair_shots": [1]}
GOOD_MOTION = {"passed": True, "score": 0.85, "reasons": []}


def _repair_run(orchestrator, tmp_path):
    run = orchestrator.create_run("repair-loop")
    candidate = {
        "premise_slug": "repair_loop",
        "character": "PIP",
        "beats": [{"beat": b, "voice": b, "sfx": "none", "duration": 2.0} for b in ("setup", "escalation", "reaction", "punchline")],
    }
    orchestrator.plane.store.set_payload(
        run, winner={"candidate": candidate}, asset_paths=["a.png", "b.png", "c.png", "d.png"],
        video=str(tmp_path / "initial.mp4"),
    )
    return run


def _fake_rerender(calls):
    def fake(source, images, captions, durations, beats, output, target_shots, **kwargs):
        calls.append(kwargs["render_tuning"]["repair_attempt"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"x" * 100_000)
        output.with_suffix(".motion.json").write_text("{}")
        return output
    return fake


def test_repair_loop_retries_until_motion_qa_passes(monkeypatch, tmp_path: Path):
    import app.cartoon_factory_orchestrator as factory_module

    quality = ScriptedQuality([(OK_MEDIA, BAD_MOTION), (OK_MEDIA, BAD_MOTION), (OK_MEDIA, GOOD_MOTION)])
    orchestrator = CartoonFactoryOrchestrator(quality=quality)
    run = _repair_run(orchestrator, tmp_path)
    calls = []
    monkeypatch.setattr(factory_module, "rerender_cartoon_shots", _fake_rerender(calls))

    result = orchestrator._video_qa(run)

    assert result["passed"] and result["repair_attempts"] == 2
    assert calls == [1, 2]
    assert len(run.payload["motion_repair_history"]) == 2


def test_repair_loop_stops_at_its_budget(monkeypatch, tmp_path: Path):
    import app.cartoon_factory_orchestrator as factory_module

    monkeypatch.setenv("NOVA_MAX_REPAIR_ATTEMPTS", "2")
    quality = ScriptedQuality([(OK_MEDIA, BAD_MOTION)])
    orchestrator = CartoonFactoryOrchestrator(quality=quality)
    run = _repair_run(orchestrator, tmp_path)
    calls = []
    monkeypatch.setattr(factory_module, "rerender_cartoon_shots", _fake_rerender(calls))

    with pytest.raises(RuntimeError, match="Motion QA failed after 2 repair attempt"):
        orchestrator._video_qa(run)
    assert calls == [1, 2]


def test_technical_failure_gets_a_full_rerender(monkeypatch, tmp_path: Path):
    quality = ScriptedQuality([({"passed": False, "reasons": ["missing_audio"]}, None), (OK_MEDIA, GOOD_MOTION)])
    orchestrator = CartoonFactoryOrchestrator(quality=quality)
    run = _repair_run(orchestrator, tmp_path)
    rerendered = []

    def fake_animation(target_run):
        rerendered.append(True)
        orchestrator.plane.store.set_payload(target_run, video=str(tmp_path / "full.mp4"))
        return {}

    monkeypatch.setattr(orchestrator, "_animation_engine", fake_animation)
    result = orchestrator._video_qa(run)

    assert result["passed"] and result["repair_attempts"] == 1
    assert result["repair"]["strategy"] == "full_rerender"
    assert rerendered == [True]
    assert run.payload["video_repair_attempts"] == 1
