"""Runtime proof checks for rendered NOVA episodes.

This gate is deliberately deterministic: it proves the encoded asset can be
opened, has the expected delivery contract, and does not contain long runs of
black/duplicate frames. It is evidence for runtime health, not a substitute for
human visual review.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class RuntimeMetrics:
    decode_ok: bool
    dimensions_ok: bool
    fps_ok: bool
    duration_ok: bool
    audio_ok: bool
    black_frame_rate: float
    duplicate_frame_rate: float
    sharpness_median: float

    @property
    def passed(self) -> bool:
        return (
            self.decode_ok
            and self.dimensions_ok
            and self.fps_ok
            and self.duration_ok
            and self.audio_ok
            and self.black_frame_rate <= 0.02
            and self.duplicate_frame_rate <= 0.18
            and self.sharpness_median >= 80.0
        )


def score_metrics(metrics: RuntimeMetrics) -> float:
    """Convert runtime observations to an auditable 0..1 evidence score."""
    gates = [
        metrics.decode_ok,
        metrics.dimensions_ok,
        metrics.fps_ok,
        metrics.duration_ok,
        metrics.audio_ok,
    ]
    score = sum(gates) / len(gates)
    score -= min(0.12, metrics.black_frame_rate * 3.0)
    score -= min(0.10, metrics.duplicate_frame_rate * 0.35)
    if metrics.sharpness_median < 80.0:
        score -= min(0.10, (80.0 - metrics.sharpness_median) / 800.0)
    return round(max(0.0, min(1.0, score)), 3)


class CartoonRuntimeQA:
    def __init__(self, expected_width: int = 1080, expected_height: int = 1920, expected_fps: float = 24.0) -> None:
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.expected_fps = expected_fps

    def inspect(self, path: Path, max_seconds: float = 15.0) -> dict[str, Any]:
        if not path.exists() or path.stat().st_size < 100_000:
            return {"passed": False, "score": 0.0, "reasons": ["missing_or_too_small"]}
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return {"passed": False, "score": 0.0, "reasons": ["ffprobe_missing"]}
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(probe.stdout)
        streams = data.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
        fps = 0.0
        if video:
            try:
                num, den = str(video.get("r_frame_rate", "0/1")).split("/", 1)
                fps = float(num) / max(1.0, float(den))
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        reasons: list[str] = []
        dimensions_ok = bool(video and video.get("width") == self.expected_width and video.get("height") == self.expected_height)
        fps_ok = abs(fps - self.expected_fps) <= 0.15
        duration_ok = 6.0 <= duration <= max_seconds
        audio_ok = audio is not None
        if not dimensions_ok:
            reasons.append("wrong_video_dimensions")
        if not fps_ok:
            reasons.append("wrong_frame_rate")
        if not duration_ok:
            reasons.append("duration_outside_range")
        if not audio_ok:
            reasons.append("missing_audio")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            metrics = RuntimeMetrics(False, dimensions_ok, fps_ok, duration_ok, audio_ok, 1.0, 1.0, 0.0)
            return {"passed": False, "score": score_metrics(metrics), "reasons": reasons + ["video_decode_failed"]}
        frames: list[np.ndarray] = []
        sharpness: list[float] = []
        sample_every = max(1, round(max(fps, 1.0) / 6.0))
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % sample_every == 0:
                small = cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
                sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            index += 1
        cap.release()

        if len(frames) < 8:
            reasons.append("too_few_runtime_samples")
        black = sum(float(np.mean(frame)) < 8.0 for frame in frames) / max(1, len(frames))
        duplicates = 0
        comparisons = 0
        for previous, current in zip(frames, frames[1:]):
            diff = float(np.mean(np.abs(previous.astype(np.float32) - current.astype(np.float32))) / 255.0)
            comparisons += 1
            if diff < 0.002:
                duplicates += 1
        duplicate_rate = duplicates / max(1, comparisons)
        sharpness_median = float(np.median(sharpness)) if sharpness else 0.0
        metrics = RuntimeMetrics(
            decode_ok=True,
            dimensions_ok=dimensions_ok,
            fps_ok=fps_ok,
            duration_ok=duration_ok,
            audio_ok=audio_ok,
            black_frame_rate=black,
            duplicate_frame_rate=duplicate_rate,
            sharpness_median=sharpness_median,
        )
        if black > 0.02:
            reasons.append("possible_black_frame_runs")
        if duplicate_rate > 0.18:
            reasons.append("excessive_duplicate_frames")
        if sharpness_median < 80.0:
            reasons.append("low_runtime_sharpness")
        return {
            "passed": metrics.passed,
            "score": score_metrics(metrics),
            "reasons": reasons,
            "metrics": {
                "decode_ok": metrics.decode_ok,
                "dimensions_ok": metrics.dimensions_ok,
                "fps": round(fps, 3),
                "duration_seconds": round(duration, 3),
                "audio_ok": metrics.audio_ok,
                "black_frame_rate": round(black, 4),
                "duplicate_frame_rate": round(duplicate_rate, 4),
                "sharpness_median": round(sharpness_median, 2),
                "sample_count": len(frames),
            },
        }
