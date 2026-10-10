"""Editor AI assembles hosted shot clips and conforms local reels to the edit plan."""
import shutil
import subprocess

import pytest

from app.agents.edit_decision import build_edit_plan
from app.providers.reel_editor import assemble_reel, verify_reel

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")

BEATS = [
    {"beat": "hook", "story_role": "hook", "voice": "PIP pressed one button. Then everything went wrong!"},
    {"beat": "escalation", "story_role": "escalation", "voice": "Then twelve sandwiches appear!"},
    {"beat": "reaction", "story_role": "reaction", "voice": "Wait, why? Turns out it misunderstood?"},
    {"beat": "punchline", "story_role": "payoff", "voice": "Of course. Still one job."},
]
DURATIONS = [1.8, 2.4, 2.5, 2.2]


def _clip(path, seconds, size="720x1280"):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24:duration={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_assembles_hosted_clips_to_the_planned_cut(tmp_path):
    # Clips arrive at hosted lengths/sizes (5s, landscape) and must be cut to plan.
    clips = [_clip(tmp_path / f"c{i}.mp4", 5, size="1280x720" if i == 1 else "720x1280") for i in range(4)]
    audio = tmp_path / "voice.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=9", str(audio)], check=True)
    plan = build_edit_plan(BEATS, DURATIONS)

    report = assemble_reel(clips, plan, tmp_path / "reel.mp4", audio=audio)

    assert report["passed"], report["reasons"]
    assert abs(report["duration_seconds"] - sum(DURATIONS)) < 0.15
    for planned, measured in zip(report["planned_cut_points_seconds"], report["measured_cut_points_seconds"]):
        assert abs(planned - measured) < 0.05


def test_assembly_needs_one_clip_per_shot(tmp_path):
    plan = build_edit_plan(BEATS, DURATIONS)
    with pytest.raises(ValueError, match="one clip per planned shot"):
        assemble_reel([_clip(tmp_path / "c.mp4", 1)], plan, tmp_path / "reel.mp4")


def test_verify_flags_a_reel_that_drifts_from_the_plan(tmp_path):
    plan = build_edit_plan(BEATS, DURATIONS)
    wrong = _clip(tmp_path / "wrong.mp4", 4, size="720x1280")  # wrong size, too short, silent
    report = verify_reel(wrong, plan, shot_frames=[24, 24, 24, 24], fps=24)
    assert not report["passed"]
    joined = " ".join(report["reasons"])
    assert "not_vertical_1080x1920" in joined
    assert "missing_audio" in joined
    assert "duration_off_plan" in joined
    assert "cut_off_plan" in joined


def test_runway_backend_runs_hosted_shots_through_the_editor(monkeypatch, tmp_path):
    import json as _json

    import app.agent_memory as agent_memory
    import app.cartoon_control as control
    import app.cartoon_factory_orchestrator as orch
    from app.agents.action_packs import action_file
    from app.agents.storyboard import build_storyboard

    monkeypatch.setattr(control, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(orch, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(agent_memory, "EXPERIMENTS_FILE", tmp_path / "experiments.json")
    monkeypatch.setenv("NOVA_CARTOON_ANIMATION_BACKEND", "runway")

    actions = ["press_button", "shock", "overwhelmed", "deadpan"]
    beats = [
        {"beat": role, "asset_action": action, "voice": line, "sfx": "impact", "duration": 2.2}
        for role, action, line in zip(("hook", "escalation", "reaction", "punchline"), actions,
                                      ("PIP pressed one button. Then everything went wrong!", "Then twelve sandwiches appear!",
                                       "Wait, why? Turns out it misunderstood?", "Of course. Still one job."))
    ]
    factory = orch.CartoonFactoryOrchestrator()
    run = factory.create_run("hosted")
    factory.plane.store = control.CartoonRunStore(run.run_id)
    storyboard = build_storyboard({"title": "t", "hook": beats[0]["voice"], "character": "PIP", "script": "",
                                   "beats": [dict(b, dialogue=b["voice"]) for b in beats]}).to_dict()
    factory.plane.store.set_payload(
        run, winner={"candidate": {"premise_slug": "hosted", "character": "PIP", "hook": beats[0]["voice"], "beats": beats}},
        asset_paths=[str(action_file("PIP", a)) for a in actions], storyboard=storyboard,
    )

    def fake_local_render(images, captions, durations, beats_, output, **kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        _clip(output, 9, size="1080x1920")
        frames = output.parent / f"{output.stem}_frames"
        frames.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=9",
                        str(frames / "episode_audio.wav")], check=True)
        output.with_suffix(".motion.json").write_text(_json.dumps({"audio_sync": {"effective_shot_durations": DURATIONS}}))
        return output

    executed = {}

    def fake_execute(plan, output_dir, **kwargs):
        executed["shots"] = len(plan["shots"])
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"clips": {i: str(_clip(output_dir / f"s{i}.mp4", 5)) for i in range(len(plan["shots"]))},
                "tasks": [], "estimated_credits": 100.0, "budget": 100.0}

    monkeypatch.setattr(orch, "render_cartoon_motion", fake_local_render)
    monkeypatch.setattr(orch.runway_video, "execute_plan", fake_execute)

    animation = factory._animation_engine(run)
    assert animation["hosted_shots"] == 4 and executed["shots"] == 4
    factory._audio_engine(run)
    edited = factory._editor_engine(run)

    assert edited["mode"] == "assembled"
    assert run.payload["video"].endswith("hosted_edited.mp4")
    streams = {s["codec_type"]: s for s in _json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", run.payload["video"]],
        check=True, capture_output=True, text=True).stdout)["streams"]}
    assert (streams["video"]["width"], streams["video"]["height"]) == (1080, 1920) and "audio" in streams
