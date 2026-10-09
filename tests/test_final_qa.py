from app.agents.final_qa import FinalQAAggregator


class _Quality:
    def __init__(self, passed=True, score=0.95, reasons=()):
        self.passed = passed
        self.score = score
        self.reasons = tuple(reasons)

    def evaluate(self, candidate, recent_titles=None):
        return self


def _payload(tmp_path, *, media_passed=True, quality=None):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x" * 100_001)
    return {
        "video": str(video),
        "candidate": {"title": "A", "hook": "PIP finds one more task!", "character": "PIP", "beats": [{"asset_action": "x"}] * 4},
        "final_media_qa": {"passed": media_passed, "reasons": [] if media_passed else ["wrong_video_dimensions"]},
        "final_motion_qa": {"passed": True, "score": 0.9, "reasons": [], "audio_sync_score": 0.98},
        "story_quality": {"passed": True, "score": 9.0, "blockers": []},
        "script_contract": {"quality": {"dimensions": [{"name": "caption_readiness", "score": 9.0}]}},
    }, quality


def test_final_qa_passes_with_complete_evidence(tmp_path):
    payload, quality = _payload(tmp_path, quality=_Quality())
    result = FinalQAAggregator(quality).evaluate(payload)
    assert result["passed"] is True
    assert result["publish_gate"] == {"decision": "allow", "passed": True, "summary": "publish_ready", "blocking_count": 0, "advisory_count": 0}
    assert set(result["domains"]) == {"visual_character", "motion", "technical_validity", "story_payoff", "captions", "audio_sync", "artifact_integrity"}


def test_final_qa_blocks_hard_technical_failure(tmp_path):
    payload, quality = _payload(tmp_path, media_passed=False, quality=_Quality())
    result = FinalQAAggregator(quality).evaluate(payload)
    assert not result["passed"]
    assert "technical_validity:wrong_video_dimensions" in result["blocking_failures"]


def test_final_qa_keeps_creative_warning_advisory(tmp_path):
    payload, _ = _payload(tmp_path, quality=_Quality(False, 0.70, ("weak_hook", "score_below_threshold")))
    result = FinalQAAggregator(_Quality(False, 0.70, ("weak_hook", "score_below_threshold"))).evaluate(payload)
    assert result["passed"] is True
    assert "visual_character:weak_hook" in result["advisory_failures"]
    assert result["publish_gate"]["decision"] == "allow"


def test_final_qa_malformed_input_fails_closed():
    result = FinalQAAggregator().evaluate(["not", "an", "object"])
    assert result["passed"] is False
    assert result["publish_gate"]["decision"] == "block"
    assert result["blocking_failures"] == ["payload_must_be_object"]


def test_final_qa_missing_critical_evidence_fails_closed(tmp_path):
    payload, _ = _payload(tmp_path, quality=_Quality())
    payload.pop("final_motion_qa")
    result = FinalQAAggregator(_Quality()).evaluate(payload)
    assert not result["passed"]
    assert "motion:motion_qa_evidence_missing" in result["blocking_failures"]
