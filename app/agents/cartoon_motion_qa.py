"""Deterministic post-render motion QA for provider-independent cartoon videos."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class CartoonMotionQA:
    """Cheap video-level motion gate; no generative model required."""

    @staticmethod
    def plan_repair(
        reasons: list[str],
        target_shots: list[int] | None = None,
    ) -> dict[str, Any]:
        """Translate QA failures into one bounded, deterministic repair profile."""
        reasons = list(dict.fromkeys(reasons))
        target_shots = sorted({int(index) for index in (target_shots or []) if int(index) >= 0})
        plan: dict[str, Any] = {
            "motion_gain": 1.0,
            "camera_gain": 1.0,
            "shake_gain": 1.0,
            "transition_seconds": 0.20,
            "caption_y": 1515,
            "caption_zone": [40, 1480, 1040, 1885],
            "jpeg_quality": 91,
            "crf": 23,
        }
        if any(reason in reasons for reason in ("insufficient_character_motion", "too_many_near_duplicate_frames")):
            plan["motion_gain"] = 1.35
            plan["camera_gain"] = 1.25
        if "possible_pose_popping" in reasons:
            plan["motion_gain"] = min(plan["motion_gain"], 0.84)
            plan["camera_gain"] = min(plan["camera_gain"], 0.92)
            plan["shake_gain"] = 0.60
            plan["transition_seconds"] = 0.26
        if "excessive_transition_spike" in reasons:
            plan["camera_gain"] = min(plan["camera_gain"], 0.90)
            plan["shake_gain"] = 0.55
            plan["transition_seconds"] = 0.14
        if any(reason in reasons for reason in ("excessive_blur", "video_decode_failed", "possible_black_frames")):
            plan["jpeg_quality"] = 94
            plan["crf"] = 18
        if any(reason.startswith("caption_character_collision_") for reason in reasons):
            plan["caption_y"] = 1320
            plan["caption_zone"] = [40, 1285, 1040, 1510]
        if all(reason in {"motion_qa_below_threshold"} for reason in reasons):
            plan["motion_gain"] = 1.15
            plan["camera_gain"] = 1.10
        return {
            "reasons": reasons,
            "target_shots": target_shots,
            "profile": plan,
            "strategy": "targeted-shot-rerender",
        }

    def __init__(
        self,
        min_motion: float = 0.012,
        max_internal_p95: float = 0.11,
        min_laplacian_median: float = 600.0,
        max_transition_spike: float = 0.24,
        sample_fps: float = 6.0,
    ) -> None:
        self.min_motion = min_motion
        self.max_internal_p95 = max_internal_p95
        self.min_laplacian_median = min_laplacian_median
        self.max_transition_spike = max_transition_spike
        self.sample_fps = sample_fps

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, Any]:
        metadata = path.with_suffix(".motion.json")
        if not metadata.exists():
            return {}
        try:
            raw = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _shot_boundaries(metadata: dict[str, Any]) -> list[tuple[float, float, str]]:
        rows = metadata.get("shots")
        if not isinstance(rows, list):
            return []
        out: list[tuple[float, float, str]] = []
        elapsed = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            duration = float(row.get("duration", 0.0) or 0.0)
            beat = str(row.get("beat", ""))
            out.append((elapsed, elapsed + duration, beat))
            elapsed += duration
        return out

    @staticmethod
    def _near_boundary(timestamp: float, boundaries: list[tuple[float, float, str]], window: float) -> bool:
        for start, end, _ in boundaries[:-1]:
            if abs(timestamp - end) <= window or abs(timestamp - start) <= window:
                return True
        return False

    @staticmethod
    def _in_punchline_hold(timestamp: float, boundaries: list[tuple[float, float, str]]) -> bool:
        for start, end, beat in boundaries:
            if beat == "punchline" and start <= timestamp <= end:
                progress = (timestamp - start) / max(0.001, end - start)
                return progress >= 0.72
        return False

    @staticmethod
    def _shot_frame_ranges(
        metadata: dict[str, Any],
    ) -> list[tuple[int, int, str]]:
        """Return half-open frame ranges for the encoded shots."""
        rows = metadata.get("shots")
        if not isinstance(rows, list):
            return []
        out: list[tuple[int, int, str]] = []
        cursor = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            count = int(row.get("frames", 0) or 0)
            if count <= 0:
                duration = float(row.get("duration", 0.0) or 0.0)
                count = max(1, round(duration * float(metadata.get("fps", 24.0) or 24.0)))
            out.append((cursor, cursor + count, str(row.get("beat", ""))))
            cursor += count
        return out

    @staticmethod
    def _shot_for_frame(frame_index: int, ranges: list[tuple[int, int, str]]) -> int | None:
        for index, (start, end, _) in enumerate(ranges):
            if start <= frame_index < end:
                return index
        return None

    @staticmethod
    def _caption_collisions(metadata: dict[str, Any]) -> list[str]:
        zone = metadata.get("caption_zone")
        boxes = metadata.get("character_bboxes")
        if not isinstance(zone, list) or len(zone) != 4 or not isinstance(boxes, list):
            return []
        zx0, zy0, zx1, zy1 = [int(v) for v in zone]
        reasons: list[str] = []
        for index, box in enumerate(boxes):
            if not isinstance(box, list) or len(box) != 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in box]
            overlap_w = max(0, min(x1, zx1) - max(x0, zx0))
            overlap_h = max(0, min(y1, zy1) - max(y0, zy0))
            overlap = overlap_w * overlap_h
            area = max(1, (x1 - x0) * (y1 - y0))
            if overlap / area > 0.12:
                reasons.append(f"caption_character_collision_shot_{index}")
        return reasons

    def inspect(self, path: Path, expected_fps: float = 24.0, max_seconds: float = 15.0) -> dict[str, Any]:
        reasons: list[str] = []
        if not path.exists() or path.stat().st_size < 100_000:
            return {"passed": False, "score": 0.0, "reasons": ["missing_or_too_small"]}

        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return {"passed": False, "score": 0.0, "reasons": ["ffprobe_missing"]}

        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(probe.stdout)
        streams = data.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
        fps_text = str(video.get("r_frame_rate", "0/1")) if video else "0/1"
        try:
            num, den = fps_text.split("/", 1)
            fps = float(num) / max(1.0, float(den))
        except (ValueError, ZeroDivisionError):
            fps = 0.0

        if not video or video.get("width") != 1080 or video.get("height") != 1920:
            reasons.append("wrong_video_dimensions")
        if not audio:
            reasons.append("missing_audio")
        if abs(fps - expected_fps) > 0.15:
            reasons.append("wrong_frame_rate")
        if not 6.0 <= duration <= max_seconds:
            reasons.append("duration_outside_range")

        metadata = self._load_metadata(path)
        if metadata:
            if metadata.get("interpolation") not in {"none", None}:
                reasons.append("unexpected_frame_interpolation")
            reasons.extend(self._caption_collisions(metadata))

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            reasons.append("video_decode_failed")
            return {"passed": False, "score": 0.0, "reasons": reasons, "duration_seconds": round(duration, 3), "fps": round(fps, 3)}

        frame_step = max(1, round(fps / self.sample_fps))
        frames: list[tuple[int, float, np.ndarray, float, float]] = []
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % frame_step == 0:
                small = cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness = float(gray.mean())
                frames.append((index, index / max(1.0, fps), gray, laplacian, brightness))
            index += 1
        cap.release()

        if len(frames) < 8:
            reasons.append("too_few_motion_samples")

        diffs: list[float] = []
        internal_diffs: list[float] = []
        transition_diffs: list[float] = []
        boundaries = self._shot_boundaries(metadata)

        for previous, current in zip(frames, frames[1:]):
            diff = float(np.mean(np.abs(previous[2].astype(np.float32) - current[2].astype(np.float32))) / 255.0)
            diffs.append(diff)
            if self._near_boundary(current[1], boundaries, 0.34):
                transition_diffs.append(diff)
            elif not self._in_punchline_hold(current[1], boundaries):
                internal_diffs.append(diff)

        if not internal_diffs:
            internal_diffs = diffs.copy()

        mean_motion = float(np.mean(internal_diffs)) if internal_diffs else 0.0
        p95_motion = float(np.percentile(internal_diffs, 95)) if internal_diffs else 0.0
        transition_peak = max(transition_diffs) if transition_diffs else 0.0
        lap_median = float(np.median([x[3] for x in frames])) if frames else 0.0
        dark_ratio = sum(1 for x in frames if x[4] < 7.0) / max(1, len(frames))
        near_duplicate_ratio = sum(1 for x in internal_diffs if x < 0.003) / max(1, len(internal_diffs))

        shot_ranges = self._shot_frame_ranges(metadata)
        shot_results: list[dict[str, Any]] = []
        repair_shots: set[int] = set()

        for shot_index, (shot_start, shot_end, shot_beat) in enumerate(shot_ranges):
            shot_samples = [row for row in frames if shot_start <= row[0] < shot_end]
            shot_diffs: list[float] = []
            shot_transition_diffs: list[float] = []
            for previous, current in zip(frames, frames[1:]):
                current_shot = self._shot_for_frame(current[0], shot_ranges)
                if current_shot != shot_index:
                    continue
                diff = float(
                    np.mean(
                        np.abs(
                            previous[2].astype(np.float32) - current[2].astype(np.float32)
                        )
                    )
                    / 255.0
                )
                if self._near_boundary(current[1], boundaries, 0.34):
                    shot_transition_diffs.append(diff)
                elif not self._in_punchline_hold(current[1], boundaries):
                    shot_diffs.append(diff)

            shot_mean = float(np.mean(shot_diffs)) if shot_diffs else 0.0
            shot_p95 = float(np.percentile(shot_diffs, 95)) if shot_diffs else 0.0
            shot_transition = max(shot_transition_diffs) if shot_transition_diffs else 0.0
            shot_lap = (
                float(np.median([row[3] for row in shot_samples]))
                if shot_samples else 0.0
            )
            shot_dark = (
                sum(1 for row in shot_samples if row[4] < 7.0)
                / max(1, len(shot_samples))
            )
            shot_duplicate = (
                sum(1 for value in shot_diffs if value < 0.003)
                / max(1, len(shot_diffs))
            )
            shot_reasons: list[str] = []
            if shot_mean < self.min_motion:
                shot_reasons.append("insufficient_character_motion")
            if shot_p95 > self.max_internal_p95:
                shot_reasons.append("possible_pose_popping")
            if shot_transition > self.max_transition_spike:
                shot_reasons.append("excessive_transition_spike")
            if shot_lap and shot_lap < self.min_laplacian_median:
                shot_reasons.append("excessive_blur")
            if shot_dark > 0.03:
                shot_reasons.append("possible_black_frames")
            if shot_duplicate > 0.55:
                shot_reasons.append("too_many_near_duplicate_frames")
            if shot_reasons:
                repair_shots.add(shot_index)
            shot_results.append({
                "shot": shot_index,
                "beat": shot_beat,
                "frame_start": shot_start,
                "frame_end": shot_end,
                "mean_internal_motion": round(shot_mean, 5),
                "p95_internal_motion": round(shot_p95, 5),
                "max_transition_motion": round(shot_transition, 5),
                "median_laplacian": round(shot_lap, 2),
                "dark_frame_ratio": round(shot_dark, 3),
                "near_duplicate_ratio": round(shot_duplicate, 3),
                "reasons": shot_reasons,
            })

        for reason in self._caption_collisions(metadata):
            suffix = reason.rsplit("_", 1)[-1]
            if suffix.isdigit():
                repair_shots.add(int(suffix))

        global_structural = {
            "wrong_video_dimensions",
            "wrong_frame_rate",
            "video_decode_failed",
            "missing_audio",
            "duration_outside_range",
        }
        if any(reason in reasons for reason in global_structural):
            repair_shots.update(range(len(shot_ranges)))

        if not repair_shots and reasons:
            worst = min(
                shot_results,
                key=lambda row: (
                    row["mean_internal_motion"],
                    -row["median_laplacian"],
                ),
                default=None,
            )
            if worst is not None:
                repair_shots.add(int(worst["shot"]))

        if mean_motion < self.min_motion:
            reasons.append("insufficient_character_motion")
        if p95_motion > self.max_internal_p95:
            reasons.append("possible_pose_popping")
        if transition_peak > self.max_transition_spike:
            reasons.append("excessive_transition_spike")
        if lap_median < self.min_laplacian_median:
            reasons.append("excessive_blur")
        if dark_ratio > 0.03:
            reasons.append("possible_black_frames")
        if near_duplicate_ratio > 0.55:
            reasons.append("too_many_near_duplicate_frames")

        motion_score = min(1.0, mean_motion / max(0.001, self.min_motion))
        continuity_score = max(0.0, 1.0 - p95_motion / max(0.001, self.max_internal_p95))
        blur_score = min(1.0, lap_median / max(1.0, self.min_laplacian_median))
        frame_score = 1.0 - dark_ratio
        transition_score = max(0.0, 1.0 - transition_peak / max(0.001, self.max_transition_spike))

        score = round(
            0.30 * motion_score
            + 0.25 * continuity_score
            + 0.18 * blur_score
            + 0.12 * frame_score
            + 0.15 * transition_score,
            3,
        )
        passed = not reasons and score >= 0.68
        if not passed:
            reasons.append("motion_qa_below_threshold")

        return {
            "passed": passed,
            "score": score,
            "reasons": reasons,
            "duration_seconds": round(duration, 3),
            "fps": round(fps, 3),
            "sample_frames": len(frames),
            "mean_internal_motion": round(mean_motion, 5),
            "p95_internal_motion": round(p95_motion, 5),
            "max_transition_motion": round(transition_peak, 5),
            "median_laplacian": round(lap_median, 2),
            "dark_frame_ratio": round(dark_ratio, 3),
            "near_duplicate_ratio": round(near_duplicate_ratio, 3),
            "motion_score": round(motion_score, 3),
            "continuity_score": round(continuity_score, 3),
            "blur_score": round(blur_score, 3),
            "transition_score": round(transition_score, 3),
            "shot_results": shot_results,
            "repair_shots": sorted(repair_shots),
        }
