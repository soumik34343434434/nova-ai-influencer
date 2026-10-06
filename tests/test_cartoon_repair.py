from pathlib import Path

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
