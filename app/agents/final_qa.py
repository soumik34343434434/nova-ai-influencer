"""Provider-neutral publish gate composed from the production QA checks."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _domain(score: float | None, reasons: list[str], *, blocking: bool, missing: bool = False) -> dict[str, Any]:
    return {
        "score": round(float(score), 3) if score is not None else None,
        "passed": score is not None and not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "failures": [{"reason": reason, "severity": "blocking" if blocking else "advisory"} for reason in dict.fromkeys(reasons)],
        "evidence_missing": missing,
    }


def _media_domains(media: dict[str, Any]) -> dict[str, list[str]]:
    """Group measured media reasons by domain; legacy media reports count as technical."""
    domains = media.get("domains")
    if not isinstance(domains, dict):
        return {"technical": list(media.get("reasons") or []), "audio": [], "video": [], "captions": []}
    return {name: list(domains.get(name) or []) for name in ("technical", "audio", "video", "captions")}


def _with_blocking(result: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    """Attach measured blocking reasons to a domain result."""
    extra = [reason for reason in dict.fromkeys(reasons) if reason not in result["reasons"]]
    if extra:
        result["reasons"] = [*result["reasons"], *extra]
        result["failures"] = [*result["failures"], *({"reason": reason, "severity": "blocking"} for reason in extra)]
        result["passed"] = False
    return result


class FinalQAAggregator:
    """Combine established checks into one stable, explicit publish decision."""

    def __init__(self, quality: Any | None = None, story_engine: Any | None = None):
        self.quality = quality
        self.story_engine = story_engine

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return self._malformed("payload_must_be_object")

        candidate = payload.get("winner", {}).get("candidate") if isinstance(payload.get("winner"), dict) else payload.get("candidate")
        if not isinstance(candidate, dict):
            return self._malformed("candidate_missing_or_malformed")

        domains: dict[str, dict[str, Any]] = {}
        quality = self.quality
        try:
            if quality is None:
                from app.agents.cartoon_quality import CartoonQuality
                quality = CartoonQuality()
            visual = quality.evaluate(candidate, payload.get("recent_titles", []))
            visual_reasons = list(visual.reasons)
            fatal = [reason for reason in visual_reasons if reason in {"forbidden_or_unsafe_signal", "unknown_character"}]
            domains["visual_character"] = _domain(visual.score, visual_reasons, blocking=False)
            domains["visual_character"]["failures"] = [
                {"reason": reason, "severity": "blocking" if reason in fatal else "advisory"}
                for reason in dict.fromkeys(visual_reasons)
            ]
            domains["visual_character"]["passed"] = bool(visual.passed)
        except Exception as exc:
            domains["visual_character"] = _domain(None, [f"visual_character_check_error:{type(exc).__name__}"], blocking=True, missing=True)

        motion = payload.get("final_motion_qa") or payload.get("video_qa", {}).get("motion")
        if isinstance(motion, dict):
            domains["motion"] = _domain(motion.get("score", 1.0 if motion.get("passed") else 0.0), list(motion.get("reasons") or []), blocking=True)
        else:
            domains["motion"] = _domain(None, ["motion_qa_evidence_missing"], blocking=True, missing=True)

        media = payload.get("final_media_qa") or payload.get("video_qa", {}).get("media")
        media_domains = _media_domains(media) if isinstance(media, dict) else {"technical": [], "audio": [], "video": [], "captions": []}
        if isinstance(media, dict):
            technical_reasons = [*media_domains["technical"], *media_domains["video"]]
            if not media.get("passed") and not any(media_domains.values()):
                technical_reasons = ["media_check_failed"]
            domains["technical_validity"] = _domain(1.0 if media.get("passed") else 0.0, technical_reasons, blocking=True)
        else:
            domains["technical_validity"] = _domain(None, ["media_qa_evidence_missing"], blocking=True, missing=True)

        story = payload.get("story_quality")
        if story is None and isinstance(payload.get("story"), dict):
            story = payload["story"].get("quality")
        if isinstance(story, dict):
            reasons = list(story.get("blockers") or [])
            passed = story.get("passed")
            score = story.get("score")
            if passed is False and not reasons:
                reasons = ["story_quality_failed"]
            domains["story_payoff"] = _domain(score, reasons, blocking=False)
            domains["story_payoff"]["passed"] = bool(passed)
            if passed is False and not reasons:
                domains["story_payoff"]["passed"] = False
        else:
            domains["story_payoff"] = _domain(None, ["story_quality_evidence_missing"], blocking=True, missing=True)

        contract = payload.get("script_contract")
        script_quality = contract.get("quality") if isinstance(contract, dict) else None
        if isinstance(script_quality, dict):
            dimensions = script_quality.get("dimensions") or []
            caption = next((row for row in dimensions if isinstance(row, dict) and row.get("name") == "caption_readiness"), None)
            if caption:
                caption_score = caption.get("score")
                reasons = ["caption_quality_below_target"] if caption_score is not None and float(caption_score) < 8.5 else []
                domains["captions"] = _domain(caption_score, reasons, blocking=False)
            else:
                domains["captions"] = _domain(None, ["caption_quality_evidence_missing"], blocking=True, missing=True)
        else:
            domains["captions"] = _domain(None, ["caption_quality_evidence_missing"], blocking=True, missing=True)
        # The script score never sees the rendered text, so truncated or off-zone
        # captions and the disclosure are checked on the mp4 and block here.
        domains["captions"] = _with_blocking(domains["captions"], media_domains["captions"])

        audio_score = motion.get("audio_sync_score") if isinstance(motion, dict) else None
        audio_reasons = list(motion.get("reasons") or []) if isinstance(motion, dict) else []
        sync_reasons = [reason for reason in audio_reasons if "voice_" in reason or "audio_video" in reason]
        if audio_score is None:
            domains["audio_sync"] = _domain(None, ["audio_sync_evidence_missing"], blocking=True, missing=True)
        else:
            domains["audio_sync"] = _domain(audio_score, sync_reasons, blocking=True)
        # Loudness, true peak, silence and speech rate are measured on the mp4.
        domains["audio_sync"] = _with_blocking(domains["audio_sync"], media_domains["audio"])

        video = payload.get("video")
        artifact_reasons = []
        if not isinstance(video, str) or not video:
            artifact_reasons.append("video_artifact_path_missing")
        elif not Path(video).is_file():
            artifact_reasons.append("video_artifact_missing")
        elif Path(video).stat().st_size < 100_000:
            artifact_reasons.append("video_artifact_too_small")
        domains["artifact_integrity"] = _domain(0.0 if artifact_reasons else 1.0, artifact_reasons, blocking=True)

        runtime = payload.get("runtime_qa")
        if runtime is not None:
            if isinstance(runtime, dict) and runtime.get("passed") is True:
                domains["runtime"] = _domain(runtime.get("score", 1.0), list(runtime.get("reasons") or []), blocking=True)
            else:
                reasons = list(runtime.get("reasons") or []) if isinstance(runtime, dict) else ["runtime_qa_malformed"]
                if not reasons:
                    reasons = ["runtime_qa_failed"]
                domains["runtime"] = _domain(runtime.get("score", 0.0) if isinstance(runtime, dict) else None, reasons, blocking=True)

        blocking = sorted(f"{name}:{failure['reason']}" for name, result in domains.items() for failure in result["failures"] if failure["severity"] == "blocking")
        advisory = sorted(f"{name}:{failure['reason']}" for name, result in domains.items() for failure in result["failures"] if failure["severity"] == "advisory")
        passed = not blocking
        return {
            "passed": passed,
            "status": "pass" if passed else "fail",
            "domains": domains,
            "blocking_failures": blocking,
            "advisory_failures": advisory,
            "publish_gate": {"decision": "allow" if passed else "block", "passed": passed, "summary": "publish_ready" if passed else "blocked_by_qa", "blocking_count": len(blocking), "advisory_count": len(advisory)},
        }

    @staticmethod
    def _malformed(reason: str) -> dict[str, Any]:
        return {"passed": False, "status": "fail", "domains": {}, "blocking_failures": [reason], "advisory_failures": [], "publish_gate": {"decision": "block", "passed": False, "summary": "blocked_by_qa", "blocking_count": 1, "advisory_count": 0}}
