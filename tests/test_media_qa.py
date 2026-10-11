"""Media QA: parsers, pure checks and ffmpeg-measured fixtures (synthetic lavfi clips)."""
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

import app.agents.media_qa as media_qa
from app.agents.final_qa import FinalQAAggregator
from app.agents.media_qa import MediaQA, caption_checks, parse_freezes, parse_loudness, parse_scene_cuts, parse_silences
from app.agents.cartoon_motion_qa import CartoonMotionQA

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg and ffprobe are required"
)

SAFE_ZONE = [0.06, 0.12, 0.84, 0.76]
CAPTION = {"text": "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen",
           "start": 0.0, "end": 6.0, "box": [0.1, 0.4, 0.8, 0.6], "words_total": 15, "words_shown_max": 15}
DISCLOSURE = {"text": "AI-generated • fictional cartoon", "box": [0.3, 0.7, 0.8, 0.74], "start": 0.0, "end": 6.0}
GOOD_METADATA = {
    "fps": 24,
    "captions": [CAPTION],
    "disclosure": DISCLOSURE,
    "safe_zone": SAFE_ZONE,
    "shots": [{"beat": "setup", "frames": 144, "duration": 6.0}],
}
# 1 kHz-ish tone at this gain measures about -14 LUFS (checked in the fixture test below).
TONE = "sine=frequency=440:sample_rate=48000:duration={seconds}"
LOUD_TONE = TONE + ",volume=2.45"
# A light flat background with a centred pattern: no near-black pixels reach the edges.
GOOD_VIDEO = (
    "color=c=0x9aa0b0:s=1080x1920:r=24:d=6[bg];testsrc=size=720x720:rate=24:duration=6[fg];"
    "[bg][fg]overlay=180:600,format=yuv420p[v]"
)


def _encode(path: Path, video_graph: str, audio_graph: str | None) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    graph = video_graph if audio_graph is None else f"{video_graph};{audio_graph}"
    command = [ffmpeg, "-y", "-loglevel", "error", "-filter_complex", graph, "-map", "[v]"]
    if audio_graph is not None:
        command += ["-map", "[a]", "-c:a", "aac", "-b:a", "160k"]
    command += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p", "-b:v", "4M", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    return path


def _good_audio(seconds: float = 6.0) -> str:
    return LOUD_TONE.format(seconds=seconds) + "[a]"


@pytest.fixture(scope="module")
def good_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "good.mp4"
    return _encode(path, GOOD_VIDEO, _good_audio())


def _evaluate(path: Path, metadata: dict | None = None) -> dict:
    return MediaQA().evaluate(path, metadata=dict(GOOD_METADATA) if metadata is None else metadata)


# ---------------------------------------------------------------------------
# Pure parsers and pixel rules
# ---------------------------------------------------------------------------

LOUDNESS_LOG = """
[Parsed_ebur128_0 @ 0x1] Summary:

  Integrated loudness:
    I:         -37.6 LUFS
    Threshold: -47.6 LUFS

  True peak:
    Peak:      -21.2 dBFS
"""


def test_parse_loudness_reads_integrated_and_true_peak_summary():
    assert parse_loudness(LOUDNESS_LOG) == (-37.6, -21.2)


def test_parse_loudness_returns_none_when_summary_is_absent():
    assert parse_loudness("no summary here") == (None, None)


def test_parse_loudness_reports_silent_audio_as_non_finite_loudness():
    integrated, peak = parse_loudness("Integrated loudness:\n    I:  -inf LUFS\n  True peak:\n    Peak:  -inf dBFS")
    assert integrated == float("-inf") and peak == float("-inf")


def test_parse_silences_pairs_intervals_and_closes_trailing_silence_at_eof():
    log = "silence_start: 1.5\nsilence_end: 2.5 | silence_duration: 1.0\nsilence_start: 4.0\n"
    assert parse_silences(log, duration=6.0) == [(1.5, 2.5), (4.0, 6.0)]


def test_parse_freezes_closes_open_freeze_at_eof():
    log = "freeze_start: 2.0\nfreeze_end: 3.0\nfreeze_start: 5.0\n"
    assert parse_freezes(log, duration=6.0) == [(2.0, 3.0), (5.0, 6.0)]


def test_parse_scene_cuts_reads_showinfo_timestamps():
    log = "[Parsed_showinfo_1] n:0 pts_time:2.166667 pos:1\n[Parsed_showinfo_1] n:1 pts_time:4.541667 pos:2\n"
    assert parse_scene_cuts(log) == [2.167, 4.542]


def test_black_edge_fractions_detect_only_the_dark_border():
    gray = np.full((1920, 1080), 200, dtype=np.uint8)
    gray[:10] = 0
    fractions = media_qa.black_edge_fractions(gray)
    assert fractions["top"] == 1.0
    assert fractions["bottom"] == 0.0
    # The black top rows touch the side bands only in their corners, far below the 2% band rule.
    assert fractions["left"] < media_qa.EDGE_BAND_BLACK_FRACTION and fractions["right"] < media_qa.EDGE_BAND_BLACK_FRACTION


def test_caption_checks_accept_full_caption_inside_safe_zone():
    failures, measured = caption_checks(GOOD_METADATA)
    assert failures == {"captions": [], "disclosure": []}
    assert measured == {"captions": 1, "words_total": 15}


def test_caption_checks_flag_truncated_caption_and_box_outside_zone():
    metadata = dict(GOOD_METADATA, captions=[dict(CAPTION, words_shown_max=9, box=[0.1, 0.8, 0.9, 0.9])])
    failures, _ = caption_checks(metadata)
    assert "caption_truncated (caption 1 shows 9 of 15 words)" in failures["captions"]
    assert "caption_outside_safe_zone (caption 1)" in failures["captions"]


def test_caption_checks_fail_closed_without_metadata():
    failures, _ = caption_checks({})
    assert "safe_zone_evidence_missing" in failures["captions"]
    assert "caption_evidence_missing" in failures["captions"]
    assert failures["disclosure"] == ["disclosure_evidence_missing"]


def test_disclosure_is_not_passed_when_the_safe_zone_is_missing():
    metadata = {k: v for k, v in GOOD_METADATA.items() if k != "safe_zone"}
    failures, _ = caption_checks(metadata)
    assert failures["disclosure"] == ["safe_zone_evidence_missing"]


def test_disclosure_outside_safe_zone_is_flagged():
    metadata = dict(GOOD_METADATA, disclosure=dict(DISCLOSURE, box=[0.3, 0.95, 0.99, 0.99]))
    failures, _ = caption_checks(metadata)
    assert failures["disclosure"] == ["disclosure_outside_safe_zone"]


def test_combine_merges_render_reasons_into_technical_domain():
    render = {"passed": False, "reasons": ["wrong_video_dimensions"], "duration_seconds": 9.0}
    measured = {"passed": True, "reasons": [], "domains": {"technical": [], "audio": [], "video": [], "captions": []}}
    merged = MediaQA.combine(render, measured)
    assert merged["passed"] is False
    assert merged["reasons"] == ["wrong_video_dimensions"]
    assert merged["domains"]["technical"] == ["wrong_video_dimensions"]


def test_missing_file_fails_closed(tmp_path):
    result = MediaQA().evaluate(tmp_path / "missing.mp4", metadata={})
    assert result["passed"] is False
    assert result["reasons"] == ["media_file_missing_or_too_small"]


def test_missing_ffprobe_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"x" * 200_000)
    monkeypatch.setattr(media_qa, "probe_streams", lambda _path: None)
    result = MediaQA().evaluate(path, metadata={})
    assert result["passed"] is False
    assert result["reasons"] == ["ffprobe_failed"]


# ---------------------------------------------------------------------------
# ffmpeg-measured fixtures
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_good_episode_passes_every_measured_check(good_clip):
    result = _evaluate(good_clip)
    assert result["reasons"] == [], result["reasons"]
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert -15.5 <= result["checks"]["loudness"]["integrated_lufs"] <= -12.5
    assert result["checks"]["speech_rate"]["words_per_minute"] == pytest.approx(150.0, rel=0.02)


@needs_ffmpeg
def test_quiet_mix_is_blocked_on_loudness(tmp_path):
    clip = _encode(tmp_path / "quiet.mp4", GOOD_VIDEO, TONE.format(seconds=6) + ",volume=0.3[a]")
    result = _evaluate(clip)
    assert result["passed"] is False
    assert any(reason.startswith("loudness_out_of_range") for reason in result["domains"]["audio"])


@needs_ffmpeg
def test_hot_true_peak_is_blocked(tmp_path):
    clip = _encode(tmp_path / "hot.mp4", GOOD_VIDEO, TONE.format(seconds=6) + ",volume=12[a]")
    result = _evaluate(clip)
    assert any(reason.startswith("true_peak_too_hot") for reason in result["domains"]["audio"])


@needs_ffmpeg
def test_long_silent_gap_is_blocked_on_silence_and_speech_rate(tmp_path):
    audio = (
        f"{TONE.format(seconds=2)},volume=2.45[s1];"
        "anullsrc=r=48000:cl=mono,atrim=0:2[s2];"
        f"{TONE.format(seconds=2)},volume=2.45[s3];"
        "[s1][s2][s3]concat=n=3:v=0:a=1[a]"
    )
    clip = _encode(tmp_path / "gap.mp4", GOOD_VIDEO, audio)
    result = _evaluate(clip)
    assert any(reason.startswith("silence_ratio_too_high") for reason in result["domains"]["audio"])
    assert result["checks"]["silence"]["silence_ratio"] == pytest.approx(0.333, abs=0.03)


@needs_ffmpeg
def test_black_border_band_is_blocked(tmp_path):
    video = (
        "color=c=black:s=1080x1920:r=24:d=6[bg];testsrc2=size=1080x1800:rate=24:duration=6[fg];"
        "[bg][fg]overlay=0:60,format=yuv420p[v]"
    )
    clip = _encode(tmp_path / "black.mp4", video, _good_audio())
    result = _evaluate(clip)
    assert any(reason.startswith("black_edges") for reason in result["domains"]["video"])
    assert result["checks"]["black_edges"]["worst_edge"] == "top"


@needs_ffmpeg
def test_frozen_picture_is_blocked(tmp_path):
    clip = _encode(tmp_path / "frozen.mp4", "color=c=gray:s=1080x1920:r=24:d=6,format=yuv420p[v]", _good_audio())
    result = _evaluate(clip)
    assert any(reason.startswith("frozen_video") for reason in result["domains"]["video"])


@needs_ffmpeg
def test_wrong_resolution_and_codec_are_blocked(tmp_path):
    clip = _encode(tmp_path / "small.mp4", "testsrc2=size=720x1280:rate=24:duration=6,format=yuv420p[v]", _good_audio())
    result = _evaluate(clip)
    assert "wrong_video_dimensions" in result["domains"]["technical"]
    assert result["checks"]["format"]["video_codec"] == "h264"


@needs_ffmpeg
def test_missing_audio_stream_is_blocked(tmp_path):
    clip = _encode(tmp_path / "mute.mp4", GOOD_VIDEO, None)
    result = _evaluate(clip)
    assert "missing_audio" in result["domains"]["technical"]
    assert "audio_evidence_missing" in result["domains"]["audio"]


@needs_ffmpeg
def test_expected_cut_must_be_visible_in_the_picture(tmp_path):
    video = (
        "testsrc2=size=1080x1920:rate=24:duration=3[x];testsrc=size=1080x1920:rate=24:duration=3[y];"
        "[x][y]concat=n=2:v=1:a=0,format=yuv420p[v]"
    )
    clip = _encode(tmp_path / "cut.mp4", video, _good_audio())
    visible = dict(GOOD_METADATA, shots=[{"beat": "setup", "frames": 72}, {"beat": "punchline", "frames": 72}])
    assert "shot_cuts" not in _failed_checks(_evaluate(clip, visible))
    hidden = dict(GOOD_METADATA, shots=[{"beat": "setup", "frames": 24}, {"beat": "punchline", "frames": 120}])
    result = _evaluate(clip, hidden)
    assert any(reason.startswith("shot_cut_not_visible") for reason in result["domains"]["video"])


@needs_ffmpeg
def test_metadata_gaps_fail_closed_on_real_picture(good_clip):
    result = _evaluate(good_clip, metadata={})
    assert result["passed"] is False
    assert "safe_zone_evidence_missing" in result["domains"]["captions"]
    assert "caption_evidence_missing" in result["domains"]["captions"]
    assert "disclosure_evidence_missing" in result["domains"]["captions"]
    assert "speech_rate_evidence_missing" in result["domains"]["audio"]
    assert "shot_cut_evidence_missing" in result["domains"]["video"]


@needs_ffmpeg
def test_truncated_rendered_caption_is_blocked(good_clip):
    metadata = dict(GOOD_METADATA, captions=[dict(CAPTION, words_shown_max=10)])
    result = _evaluate(good_clip, metadata)
    assert any(reason.startswith("caption_truncated") for reason in result["domains"]["captions"])


@needs_ffmpeg
def test_media_qa_meets_the_fifteen_second_budget(tmp_path):
    video = (
        "color=c=0x9aa0b0:s=1080x1920:r=24:d=15[bg];testsrc=size=720x720:rate=24:duration=15[fg];"
        "[bg][fg]overlay=180:600,format=yuv420p[v]"
    )
    clip = _encode(tmp_path / "reel.mp4", video, _good_audio(15.0))
    metadata = dict(GOOD_METADATA, captions=[dict(CAPTION, words_total=37, words_shown_max=37)],
                    shots=[{"beat": "setup", "frames": 360}])
    started = time.monotonic()
    result = MediaQA().evaluate(clip, metadata=metadata)
    elapsed = time.monotonic() - started
    assert elapsed < 10.0, elapsed
    assert _failed_checks(result) == set(), result["reasons"]
    assert result["duration_seconds"] == pytest.approx(15.0, abs=0.1)


def _failed_checks(result: dict) -> set[str]:
    return {name for name, check in result["checks"].items() if not check["passed"]}


# ---------------------------------------------------------------------------
# Wiring: final QA, motion QA per-shot blocking, and the video QA stage
# ---------------------------------------------------------------------------

class _Quality:
    def evaluate(self, candidate, recent_titles=None):
        class _Result:
            score = 0.95
            passed = True
            reasons = ()
        return _Result()


def _final_payload(tmp_path, media):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x" * 100_001)
    return {
        "video": str(video),
        "candidate": {"title": "A", "hook": "PIP finds one more task!", "character": "PIP", "beats": [{"asset_action": "x"}] * 4},
        "final_media_qa": media,
        "final_motion_qa": {"passed": True, "score": 0.9, "reasons": [], "audio_sync_score": 0.98},
        "story_quality": {"passed": True, "score": 9.0, "blockers": []},
        "script_contract": {"quality": {"dimensions": [{"name": "caption_readiness", "score": 9.0}]}},
    }


def test_final_qa_blocks_on_measured_audio_and_caption_failures(tmp_path):
    media = {
        "passed": False,
        "reasons": ["loudness_out_of_range (-37.6 LUFS; need -15.5 to -12.5)", "caption_truncated (caption 1 shows 9 of 15 words)"],
        "domains": {
            "technical": [], "video": [],
            "audio": ["loudness_out_of_range (-37.6 LUFS; need -15.5 to -12.5)"],
            "captions": ["caption_truncated (caption 1 shows 9 of 15 words)"],
        },
    }
    result = FinalQAAggregator(_Quality()).evaluate(_final_payload(tmp_path, media))
    assert result["passed"] is False
    assert "audio_sync:loudness_out_of_range (-37.6 LUFS; need -15.5 to -12.5)" in result["blocking_failures"]
    assert "captions:caption_truncated (caption 1 shows 9 of 15 words)" in result["blocking_failures"]
    assert result["publish_gate"]["decision"] == "block"


def test_final_qa_allows_clean_measured_media(tmp_path):
    media = {"passed": True, "reasons": [], "domains": {"technical": [], "audio": [], "video": [], "captions": []}}
    result = FinalQAAggregator(_Quality()).evaluate(_final_payload(tmp_path, media))
    assert result["passed"] is True


def test_shot_motion_failures_block_except_calm_setup_and_punchline(tmp_path):
    """A static picture makes every shot dead; only the non-calm shot should block."""
    path = tmp_path / "still.mp4"
    noise = np.random.default_rng(7).integers(0, 255, size=(1920, 1080, 3), dtype=np.uint8)
    import cv2

    still = tmp_path / "still.png"
    cv2.imwrite(str(still), noise)
    ffmpeg = shutil.which("ffmpeg")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-framerate", "24", "-t", "3", "-i", str(still),
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True)
    path.with_suffix(".motion.json").write_text(
        '{"fps": 24, "width": 1080, "height": 1920, "shots": ['
        '{"beat": "setup", "frames": 36, "duration": 1.5},'
        '{"beat": "escalation", "frames": 36, "duration": 1.5}]}'
    )
    result = CartoonMotionQA().inspect(path)
    assert "insufficient_character_motion_shot_1" in result["reasons"]
    assert "insufficient_character_motion_shot_0" not in result["reasons"]
    assert result["passed"] is False
    assert result["shot_results"][0]["advisory_reasons"] == ["insufficient_character_motion"]
    # Shot 0 is advisory for low mean motion, but a static picture also makes it a
    # near-duplicate shot, which is a blocking per-shot failure.
    assert "too_many_near_duplicate_frames_shot_0" in result["reasons"]
    assert result["repair_shots"] == [0, 1]


def test_repair_planner_amplifies_motion_for_suffixed_per_shot_failures():
    plan = CartoonMotionQA.plan_repair(["insufficient_character_motion_shot_1"], [1])
    assert plan["profile"]["motion_gain"] == 1.35
    assert plan["target_shots"] == [1]
    assert plan["strategy"] == "targeted-shot-rerender"


@needs_ffmpeg
def test_video_qa_records_measured_media_failure_and_blocks_publish(tmp_path, monkeypatch):
    import app.cartoon_factory_orchestrator as factory_module
    from app.cartoon_factory_orchestrator import CartoonFactoryOrchestrator

    class _Quality:
        def render_check(self, path: Path):
            return {"passed": True, "reasons": []}

        def motion_check(self, path: Path):
            return {"passed": True, "score": 0.9, "reasons": []}

        def evaluate(self, *args, **kwargs):
            raise AssertionError("idea evaluation is not part of this test")

    class _StubMediaQA:
        def evaluate(self, path):
            reason = "loudness_out_of_range (-37.6 LUFS; need -15.5 to -12.5)"
            return {"passed": False, "reasons": [reason], "domains": {"technical": [], "audio": [reason], "video": [], "captions": []}}

    monkeypatch.setattr(factory_module, "record_provider_observation", lambda **kwargs: None)
    orchestrator = CartoonFactoryOrchestrator(quality=_Quality())
    orchestrator.media_qa = _StubMediaQA()
    run = orchestrator.create_run("media-qa")
    orchestrator.plane.store.set_payload(run, winner={"candidate": {"premise_slug": "media", "title": "T", "character": "PIP"}},
                                         asset_paths=[], video=str(tmp_path / "reel.mp4"))
    (tmp_path / "reel.mp4").write_bytes(b"x" * 100_001)

    result = orchestrator._video_qa(run)

    assert result["media"]["passed"] is False
    assert "audio_sync:loudness_out_of_range (-37.6 LUFS; need -15.5 to -12.5)" in run.payload["final_qa"]["blocking_failures"]
    with pytest.raises(RuntimeError, match="Final QA blocked publish"):
        orchestrator._publish(run)
