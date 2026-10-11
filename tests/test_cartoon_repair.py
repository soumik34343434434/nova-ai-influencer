import json
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


def test_technical_failure_on_a_hosted_reel_recuts_without_new_credits(monkeypatch, tmp_path: Path):
    quality = ScriptedQuality([({"passed": False, "reasons": ["missing_audio"]}, None), (OK_MEDIA, GOOD_MOTION)])
    orchestrator = CartoonFactoryOrchestrator(quality=quality)
    run = _repair_run(orchestrator, tmp_path)
    orchestrator.plane.store.set_payload(run, shot_clips=["s0.mp4", "s1.mp4", "s2.mp4", "s3.mp4"])
    recut = []

    def no_render(target_run):
        raise AssertionError("a hosted reel must not be re-rendered or regenerated")

    def fake_editor(target_run):
        recut.append(True)
        orchestrator.plane.store.set_payload(target_run, video=str(tmp_path / "recut.mp4"))
        return {"mode": "assembled"}

    monkeypatch.setattr(orchestrator, "_animation_engine", no_render)
    monkeypatch.setattr(orchestrator, "_editor_engine", fake_editor)
    result = orchestrator._video_qa(run)

    assert result["passed"] and result["repair"]["strategy"] == "reassemble_hosted_clips"
    assert recut == [True]



def test_hosted_reel_qa_failure_spends_nothing_unless_paid_repairs_are_on(monkeypatch, tmp_path: Path):
    import app.cartoon_factory_orchestrator as factory_module

    monkeypatch.delenv("NOVA_RUNWAY_PAID_REPAIRS", raising=False)
    quality = ScriptedQuality([(OK_MEDIA, BAD_MOTION)])
    orchestrator = CartoonFactoryOrchestrator(quality=quality)
    run = _repair_run(orchestrator, tmp_path)
    orchestrator.plane.store.set_payload(run, shot_clips=["s0.mp4", "s1.mp4", "s2.mp4", "s3.mp4"])

    def no_spend(*args, **kwargs):
        raise AssertionError("Runway must not be called when paid repairs are off")

    monkeypatch.setattr(factory_module.runway_video, "execute_plan", no_spend)
    monkeypatch.setattr(factory_module, "rerender_cartoon_shots", no_spend)
    with pytest.raises(RuntimeError, match="paid repairs are off"):
        orchestrator._video_qa(run)


RUNWAY_SHOT_INDEXES = (5, 7, 9, 11)  # plan shot numbers need not be 0..n-1
REPAIR_POSITION_2 = {"passed": False, "score": 0.3, "reasons": ["possible_pose_popping"], "repair_shots": [2]}


@pytest.fixture(autouse=True)
def _isolated_runs(monkeypatch, tmp_path: Path):
    """No test may write into the real artifacts/ tree or spend Runway credits."""
    import app.cartoon_control as control
    import app.cartoon_factory_orchestrator as factory_module
    import app.providers.runway_video as runway_video

    monkeypatch.setattr(control, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(factory_module, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(runway_video, "LEDGER_FILE", tmp_path / "ledger.json")
    monkeypatch.setattr(runway_video, "host_reachable", lambda host, timeout=8.0: True)
    for name in ("NOVA_RUNWAY_CLIPS_DIR", "NOVA_RUNWAY_PAID_REPAIRS", "NOVA_RUNWAY_EXECUTE", "RUNWAYML_API_SECRET",
                 "NOVA_RUNWAY_MAX_CREDITS", "NOVA_RUNWAY_DAILY_CREDITS", "NOVA_CARTOON_ANIMATION_BACKEND",
                 "NOVA_RUNWAY_MODEL_MAP", "NOVA_RUNWAY_DOWNLOAD_HOSTS", "NOVA_RUNWAY_LEDGER_RESTORED",
                 "GITHUB_ACTIONS"):
        monkeypatch.delenv(name, raising=False)


def _enable_paid_repairs(monkeypatch, budget="200"):
    monkeypatch.setenv("NOVA_RUNWAY_PAID_REPAIRS", "true")
    monkeypatch.setenv("NOVA_RUNWAY_EXECUTE", "true")
    monkeypatch.setenv("RUNWAYML_API_SECRET", "test-secret")
    monkeypatch.setenv("NOVA_RUNWAY_MAX_CREDITS", budget)


def _fake_runway(monkeypatch):
    """Route the factory's Runway client to an in-memory fake; returns the list of paid POSTs."""
    import app.providers.runway_video as runway_video

    posts = []

    def transport(method, url, payload):
        if method == "POST":
            posts.append(payload)
            return {"id": f"task-{len(posts)}"}
        return {"status": "SUCCEEDED", "output": [f"https://cdn.example/{url.rsplit('/', 1)[-1]}.mp4"]}

    def download(url, path):
        path.write_bytes(url.encode())

    real_client = runway_video.RunwayVideoClient
    monkeypatch.setattr(runway_video, "RunwayVideoClient",
                        lambda api_secret, **kwargs: real_client(api_secret, transport=transport, downloader=download))
    return posts


def _hosted_run(orchestrator, tmp_path: Path, *, runway_plan: bool = True):
    run = _repair_run(orchestrator, tmp_path)
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG fake")
    shots = [
        {"shot_index": index, "model": "gen-4-turbo", "image": str(image), "duration": 2.0,
         "aspect_ratio": "9:16", "prompt": f"Action: beat {index}."}
        for index in RUNWAY_SHOT_INDEXES
    ]
    plan_path = tmp_path / "runway_plan.json"
    plan_path.write_text(json.dumps({"provider": "runway", "shots": shots}))
    orchestrator.plane.store.set_payload(
        run, runway_plan=str(plan_path) if runway_plan else None,
        shot_clips=[str(tmp_path / f"s{i}.mp4") for i in range(4)], edit_plan={"shots": []}, audio_metadata={},
    )
    return run


def _assembler(monkeypatch, outcomes):
    """Fake editor: returns each outcome in turn (the last one repeats); records the clip order it saw."""
    import app.cartoon_factory_orchestrator as factory_module

    seen = []
    queue = list(outcomes)

    def assemble(clips, plan, output, audio=None):
        seen.append([Path(c).name for c in clips])
        return {"passed": queue.pop(0) if len(queue) > 1 else queue[0], "reasons": []}

    monkeypatch.setattr(factory_module, "assemble_reel", assemble)
    return seen


def test_paid_repair_regenerates_only_the_failing_shot_and_records_it_as_repair(monkeypatch, tmp_path: Path):
    import app.providers.runway_video as runway_video

    _enable_paid_repairs(monkeypatch)
    posts = _fake_runway(monkeypatch)
    seen = _assembler(monkeypatch, [True])
    orchestrator = CartoonFactoryOrchestrator(quality=ScriptedQuality([(OK_MEDIA, BAD_MOTION)]))
    run = _hosted_run(orchestrator, tmp_path)

    result = orchestrator._rerender_for_motion_qa(run, REPAIR_POSITION_2)

    assert len(posts) == 1 and posts[0]["promptText"].startswith("Animate the existing character. Timing:")
    rows = json.loads(runway_video.LEDGER_FILE.read_text())
    assert [(row["kind"], row["run_id"], row["shot"]) for row in rows] == [("repair", run.run_id, 9)]
    clips = run.payload["shot_clips"]
    assert "shot_09_" in Path(clips[2]).name  # shot 9 is plan position 2
    assert [Path(c).name for c in clips[:2] + clips[3:]] == ["s0.mp4", "s1.mp4", "s3.mp4"]
    assert seen == [["s0.mp4", "s1.mp4", Path(clips[2]).name, "s3.mp4"]]
    assert run.payload["motion_repair_attempts"] == 1 and run.payload["motion_repair_pending"] is None
    assert run.payload["video"] == result["video"] and result["target_shots"] == [2]


def test_paid_repair_counts_what_the_episode_already_spent(monkeypatch, tmp_path: Path):
    import app.providers.runway_video as runway_video

    _enable_paid_repairs(monkeypatch, budget="110")
    posts = _fake_runway(monkeypatch)
    _assembler(monkeypatch, [True])
    orchestrator = CartoonFactoryOrchestrator(quality=ScriptedQuality([(OK_MEDIA, BAD_MOTION)]))
    run = _hosted_run(orchestrator, tmp_path)
    runway_video.record_spend(run_id=run.run_id, shot=5, task_id="generated", credits=100, kind="generate",
                              model="gen4_turbo")

    with pytest.raises(runway_video.RunwayExecutionBlocked, match="already spent 100"):
        orchestrator._rerender_for_motion_qa(run, REPAIR_POSITION_2)
    assert posts == []
    assert "motion_repair_attempts" not in run.payload
    assert run.payload["shot_clips"][2] == str(tmp_path / "s2.mp4")
    assert run.payload["video"] == str(tmp_path / "initial.mp4")


def test_paid_repair_keeps_paid_clips_when_the_reel_fails_and_retries_without_paying(monkeypatch, tmp_path: Path):
    import app.providers.runway_video as runway_video

    _enable_paid_repairs(monkeypatch)
    posts = _fake_runway(monkeypatch)
    seen = _assembler(monkeypatch, [False, True])
    orchestrator = CartoonFactoryOrchestrator(quality=ScriptedQuality([(OK_MEDIA, BAD_MOTION)]))
    run = _hosted_run(orchestrator, tmp_path)
    initial_video = run.payload["video"]

    with pytest.raises(RuntimeError, match="Hosted repair: reel does not match the edit plan"):
        orchestrator._rerender_for_motion_qa(run, REPAIR_POSITION_2)
    assert run.payload["video"] == initial_video  # a failed reel is never adopted
    assert "shot_09_" in Path(run.payload["shot_clips"][2]).name  # the paid clip is kept
    assert run.payload["motion_repair_pending"]["attempt"] == 1

    result = orchestrator._rerender_for_motion_qa(run, REPAIR_POSITION_2)  # the stage retry
    assert len(posts) == 1  # the retry reuses the manifest and pays nothing
    assert len(json.loads(runway_video.LEDGER_FILE.read_text())) == 1
    assert result["attempt"] == 1 and run.payload["motion_repair_attempts"] == 1
    assert run.payload["motion_repair_pending"] is None and run.payload["video"] == result["video"]
    assert len(seen) == 2


def test_hosted_repair_without_a_repair_plan_spends_nothing(monkeypatch, tmp_path: Path):
    _enable_paid_repairs(monkeypatch)
    posts = _fake_runway(monkeypatch)
    orchestrator = CartoonFactoryOrchestrator(quality=ScriptedQuality([(OK_MEDIA, BAD_MOTION)]))
    run = _hosted_run(orchestrator, tmp_path, runway_plan=False)
    with pytest.raises(RuntimeError, match="no repair plan could be built"):
        orchestrator._rerender_for_motion_qa(run, REPAIR_POSITION_2)
    assert posts == []


def test_local_render_discards_hosted_clips_from_an_earlier_attempt(monkeypatch, tmp_path: Path):
    """A pillow re-render after a Runway run must not assemble the old hosted shots."""
    import app.cartoon_factory_orchestrator as factory_module

    monkeypatch.setenv("NOVA_CARTOON_ANIMATION_BACKEND", "pillow")
    rendered = []

    def fake_render(images, captions, durations, beats, output, **kwargs):
        rendered.append(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"x" * 1000)
        return output

    monkeypatch.setattr(factory_module, "render_cartoon_motion", fake_render)
    orchestrator = CartoonFactoryOrchestrator(quality=ScriptedQuality([(OK_MEDIA, GOOD_MOTION)]))
    run = _repair_run(orchestrator, tmp_path)
    orchestrator.plane.store.set_payload(run, shot_clips=["old-1.mp4"], runway_execution={"provider": "runway"},
                                         animation_backend="runway")

    orchestrator._animation_engine(run)

    assert len(rendered) == 1
    assert not run.payload.get("shot_clips")
    assert run.payload["runway_execution"] is None and run.payload["animation_backend"] == "pillow"

