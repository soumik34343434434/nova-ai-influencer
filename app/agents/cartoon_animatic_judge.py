"""Cheap machine-checkable judge for a rendered animatic."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageStat


class AnimaticJudge:
    """Rejects animatics with broken media, dead shots, or insufficient variation."""

    def inspect(self, path: Path, expected_shots: int = 4) -> dict:
        reasons: list[str] = []
        ffprobe = shutil.which("ffprobe")
        ffmpeg = shutil.which("ffmpeg")
        if not path.exists():
            return {"passed": False, "score": 0.0, "reasons": ["missing_animatic"]}
        if not ffprobe or not ffmpeg:
            return {"passed": False, "score": 0.0, "reasons": ["ffmpeg_tools_missing"]}

        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        import json
        data = json.loads(probe.stdout)
        video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
        if not video or video.get("width") != 540 or video.get("height") != 960:
            reasons.append("wrong_animatic_dimensions")
        if not 2.0 <= duration <= 6.0:
            reasons.append("animatic_duration_outside_range")

        work = path.parent / (path.stem + "_judge_frames")
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-vf", "fps=3,scale=270:480", str(work / "f_%03d.jpg")],
            capture_output=True, text=True, check=True,
        )
        frames = sorted(work.glob("f_*.jpg"))
        if len(frames) < expected_shots:
            reasons.append("too_few_animatic_frames")
            motion_score = 0.0
            black_score = 0.0
        else:
            stats = []
            brightness = []
            previous = None
            for frame_path in frames:
                image = Image.open(frame_path).convert("L")
                brightness.append(ImageStat.Stat(image).mean[0])
                if previous is not None:
                    a = list(previous.getdata())
                    b = list(image.getdata())
                    diff = sum(abs(x - y) for x, y in zip(a, b)) / max(1, len(a) * 255)
                    stats.append(diff)
                previous = image
            motion_score = min(1.0, (sum(stats) / max(1, len(stats))) * 8.0)
            bright = sum(1 for value in brightness if value > 7)
            black_score = bright / max(1, len(brightness))
            if motion_score < 0.12:
                reasons.append("insufficient_visual_change")
            if black_score < 0.98:
                reasons.append("possible_black_frames")

        score = round(
            0.55 * motion_score
            + 0.25 * black_score
            + 0.20 * (1.0 if len(frames) >= expected_shots else 0.0),
            3,
        )
        passed = score >= 0.55 and not reasons
        if not passed and "animatic_below_threshold" not in reasons:
            reasons.append("animatic_below_threshold")
        return {
            "passed": passed,
            "score": score,
            "reasons": reasons,
            "duration_seconds": round(duration, 3),
            "motion_score": round(motion_score, 3),
            "non_black_frame_ratio": round(black_score, 3),
            "sample_frames": len(frames),
        }
