"""Pre-media and post-render quality gates for cartoon episodes."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from app.agents.cartoon_motion_qa import CartoonMotionQA


@dataclass(frozen=True)
class CartoonEvaluation:
    score: float
    hook: float
    comedy: float
    novelty: float
    visual: float
    safety: float
    passed: bool
    reasons: tuple[str, ...]


class CartoonQuality:
    FORBIDDEN = (
        "celebrity", "lookalike", "private person", "real child", "minor",
        "repost", "face swap", "weapon", "self harm", "sexual", "hate",
    )

    def evaluate(self, draft: dict, recent_titles: list[str] | None = None) -> CartoonEvaluation:
        recent_titles = recent_titles or []
        text = json.dumps(draft, ensure_ascii=False).lower()
        reasons: list[str] = []
        hook_text = " ".join(str(draft.get("hook", "")).split())
        beats = draft.get("beats") if isinstance(draft.get("beats"), list) else []
        hook = 1.0 if 3 <= len(hook_text.split()) <= 12 else 0.55
        comedy_markers = sum(x in text for x in ("why", "?!", "apparently", "oops", "wait", "one", "too many"))
        comedy = min(1.0, 0.55 + comedy_markers * 0.08 + (0.12 if len(beats) == 4 else 0.0))
        title = str(draft.get("title", ""))
        tokens = set(re.findall(r"[a-z0-9']+", (title + " " + hook_text).lower()))
        novelty = 1.0
        for old in recent_titles:
            other = set(re.findall(r"[a-z0-9']+", old.lower()))
            union = tokens | other
            if union and len(tokens & other) / len(union) > 0.65:
                novelty = 0.35
                reasons.append("too_similar_to_recent_title")
                break
        visual = 1.0 if len(beats) == 4 and all(isinstance(b, dict) and b.get("asset_action") for b in beats) else 0.55
        safety = 0.0 if any(term in text for term in self.FORBIDDEN) else 1.0
        if safety == 0.0:
            reasons.append("forbidden_or_unsafe_signal")
        if hook < 0.9:
            reasons.append("weak_hook")
        if comedy < 0.75:
            reasons.append("weak_comedy_structure")
        if visual < 0.9:
            reasons.append("incomplete_visual_plan")
        if str(draft.get("character", "")) not in {"PIP", "MOMO", "BUBU"}:
            reasons.append("unknown_character")
            safety = 0.0
        score = round(hook * 0.28 + comedy * 0.30 + novelty * 0.15 + visual * 0.17 + safety * 0.10, 3)
        passed = score >= 0.76 and safety > 0 and not reasons
        if not passed and "score_below_threshold" not in reasons:
            reasons.append("score_below_threshold")
        return CartoonEvaluation(score, hook, comedy, novelty, visual, safety, passed, tuple(reasons))

    def render_check(self, path: Path, max_seconds: float = 15.0) -> dict:
        reasons: list[str] = []
        if not path.exists() or path.stat().st_size < 100_000:
            reasons.append("missing_or_too_small")
            return {"passed": False, "reasons": reasons}
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return {"passed": False, "reasons": ["ffprobe_missing"]}
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(probe.stdout)
        streams = data.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
        if not video or video.get("width") != 1080 or video.get("height") != 1920:
            reasons.append("wrong_video_dimensions")
        if not audio:
            reasons.append("missing_audio")
        if not 6.0 <= duration <= max_seconds:
            reasons.append("duration_outside_range")
        return {
            "passed": not reasons,
            "reasons": reasons,
            "duration_seconds": round(duration, 3),
            "width": video.get("width") if video else None,
            "height": video.get("height") if video else None,
            "video_codec": video.get("codec_name") if video else None,
            "audio_codec": audio.get("codec_name") if audio else None,
        }

    def motion_check(self, path: Path) -> dict:
        """Run the deterministic motion-performance quality gate."""
        return CartoonMotionQA().inspect(path)


def quality_to_dict(result: CartoonEvaluation) -> dict:
    return asdict(result)
