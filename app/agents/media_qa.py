"""Measured QA for the rendered episode file (media contracts #7 and #10).

Every check measures the encoded mp4 (ffprobe and ffmpeg filters) or the
motion metadata written beside it. Missing evidence fails closed: a check that
cannot measure its target reports an ``*_evidence_missing`` reason instead of
passing. Reasons are readable strings that name the measured value and the
limit, and ``domains`` groups them for the publish gate.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from app.agents.cartoon_motion_qa import CartoonMotionQA

WIDTH, HEIGHT = 1080, 1920
EXPECTED_FPS = 24.0
FPS_TOLERANCE = 0.15
LOUDNESS_RANGE_LUFS = (-15.5, -12.5)
TRUE_PEAK_MAX_DBTP = -1.0
SILENCE_DB = -50.0
SILENCE_MIN_SECONDS = 0.3
SILENCE_MAX_RATIO = 0.15
SPEECH_WPM_RANGE = (110.0, 210.0)
EDGE_BAND_PX = 10
EDGE_BLACK_LUMA = 24
EDGE_BAND_BLACK_FRACTION = 0.02
EDGE_SAMPLE_FPS = 12.0
EDGE_BLACK_FRAME_MAX_RATIO = 0.02
FREEZE_MAX_SECONDS = 1.5
CUT_SCENE_THRESHOLD = 0.12
CUT_TOLERANCE_SECONDS = 0.25
SAFE_ZONE_TOLERANCE = 0.002
FFMPEG_TIMEOUT_SECONDS = 60
MIN_VIDEO_BYTES = 100_000

DOMAINS = ("technical", "audio", "video", "captions")


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number(raw: str) -> float | None:
    """Parse an ffmpeg measurement; -inf is a real reading (silence), not a missing one."""
    try:
        return float(raw)
    except ValueError:
        return None


def _run_ffmpeg(args: list[str]) -> str | None:
    """Return ffmpeg's log output, or None when ffmpeg cannot run."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        proc = subprocess.run(
            [ffmpeg, "-nostats", "-hide_banner", "-loglevel", "info", *args],
            capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stderr


def probe_streams(path: Path) -> dict[str, Any] | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS,
        )
        data = json.loads(proc.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Parsers for ffmpeg log output. Kept pure so each can be unit-tested.
# ---------------------------------------------------------------------------

_LOUDNESS_RE = re.compile(r"Integrated loudness:\s*I:\s*(-?[\d.]+|-inf)\s*LUFS", re.S)
_TRUE_PEAK_RE = re.compile(r"True peak:\s*Peak:\s*(-?[\d.]+|-inf)\s*dBFS", re.S)
_SILENCE_RE = re.compile(r"silence_(start|end): (-?[\d.]+)")
_FREEZE_RE = re.compile(r"freeze_(start|end): (-?[\d.]+)")
_SCENE_RE = re.compile(r"pts_time:([\d.]+)")


def parse_loudness(log: str) -> tuple[float | None, float | None]:
    """Integrated loudness (LUFS) and true peak (dBTP) from an ebur128 summary."""
    integrated = _LOUDNESS_RE.search(log)
    peak = _TRUE_PEAK_RE.search(log)
    return (
        _number(integrated.group(1)) if integrated else None,
        _number(peak.group(1)) if peak else None,
    )


def parse_silences(log: str, duration: float) -> list[tuple[float, float]]:
    """Silence intervals reported by silencedetect; an open interval ends at EOF."""
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for kind, raw in _SILENCE_RE.findall(log):
        value = _number(raw)
        if value is None:
            continue
        if kind == "start":
            start = value
        elif start is not None:
            intervals.append((start, value))
            start = None
    if start is not None:
        intervals.append((start, duration))
    return intervals


def parse_freezes(log: str, duration: float) -> list[tuple[float, float]]:
    """Frozen spans (start, end) reported by freezedetect; an open span ends at EOF."""
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for kind, raw in _FREEZE_RE.findall(log):
        value = _number(raw)
        if value is None:
            continue
        if kind == "start":
            start = value
        elif start is not None:
            spans.append((start, value))
            start = None
    if start is not None:
        spans.append((start, duration))
    return spans


def parse_scene_cuts(log: str) -> list[float]:
    """Timestamps of frames whose scene score passed the cut threshold (showinfo)."""
    return sorted({round(float(value), 3) for value in _SCENE_RE.findall(log)})


def black_edge_fractions(gray: np.ndarray, band: int = EDGE_BAND_PX) -> dict[str, float]:
    """Near-black pixel fraction inside each outer border band of a grayscale frame."""
    return {
        "top": float(np.mean(gray[:band] <= EDGE_BLACK_LUMA)),
        "bottom": float(np.mean(gray[-band:] <= EDGE_BLACK_LUMA)),
        "left": float(np.mean(gray[:, :band] <= EDGE_BLACK_LUMA)),
        "right": float(np.mean(gray[:, -band:] <= EDGE_BLACK_LUMA)),
    }


def _sample_black_edges(path: Path) -> tuple[int, int, dict[str, int]] | None:
    """Count sampled frames with a black border band, by edge. None when ffmpeg is unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    command = [
        ffmpeg, "-nostats", "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
        "-vf", f"fps={EDGE_SAMPLE_FPS:g},scale={WIDTH}:{HEIGHT},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    frame_bytes = WIDTH * HEIGHT
    total = 0
    flagged = 0
    by_edge = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
            assert proc.stdout is not None
            while True:
                buffer = proc.stdout.read(frame_bytes)
                if len(buffer) < frame_bytes:
                    break
                gray = np.frombuffer(buffer, dtype=np.uint8).reshape(HEIGHT, WIDTH)
                total += 1
                edges = [edge for edge, fraction in black_edge_fractions(gray).items() if fraction >= EDGE_BAND_BLACK_FRACTION]
                if edges:
                    flagged += 1
                for edge in edges:
                    by_edge[edge] += 1
            proc.wait(timeout=FFMPEG_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return total, flagged, by_edge


# ---------------------------------------------------------------------------
# Checks. Each returns (failures, measured); failures are readable reasons.
# ---------------------------------------------------------------------------

def _inside(box: Any, zone: list[float]) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    values = [_float(v) for v in box]
    if any(v is None for v in values):
        return False
    x0, y0, x1, y1 = values
    zx0, zy0, zx1, zy1 = zone
    return (
        x0 >= zx0 - SAFE_ZONE_TOLERANCE and y0 >= zy0 - SAFE_ZONE_TOLERANCE
        and x1 <= zx1 + SAFE_ZONE_TOLERANCE and y1 <= zy1 + SAFE_ZONE_TOLERANCE
    )


def caption_checks(metadata: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Rendered captions and disclosure against the metadata safe zone and full text."""
    failures: dict[str, list[str]] = {"captions": [], "disclosure": []}
    measured: dict[str, Any] = {"captions": 0, "words_total": 0}
    zone_value = metadata.get("safe_zone")
    zone: list[float] | None = None
    if isinstance(zone_value, list) and len(zone_value) == 4 and all(_float(v) is not None for v in zone_value):
        zone = [float(v) for v in zone_value]
    else:
        failures["captions"].append("safe_zone_evidence_missing")
    rows = metadata.get("captions")
    if not isinstance(rows, list) or not rows:
        failures["captions"].append("caption_evidence_missing")
    else:
        measured["captions"] = len(rows)
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not str(row.get("text", "")).strip():
                failures["captions"].append(f"caption_evidence_missing (caption {index + 1} has no text)")
                continue
            total = row.get("words_total")
            shown = row.get("words_shown_max")
            if not isinstance(total, int) or total <= 0 or not isinstance(shown, int):
                failures["captions"].append(f"caption_evidence_missing (caption {index + 1} has no word counts)")
                continue
            measured["words_total"] += total
            if shown < total:
                failures["captions"].append(f"caption_truncated (caption {index + 1} shows {shown} of {total} words)")
            if zone is not None and not _inside(row.get("box"), zone):
                failures["captions"].append(f"caption_outside_safe_zone (caption {index + 1})")
    disclosure = metadata.get("disclosure")
    if not isinstance(disclosure, dict) or not str(disclosure.get("text", "")).strip():
        failures["disclosure"].append("disclosure_evidence_missing")
    elif zone is None:
        failures["disclosure"].append("safe_zone_evidence_missing")
    elif not _inside(disclosure.get("box"), zone):
        failures["disclosure"].append("disclosure_outside_safe_zone")
    return failures, measured


class MediaQA:
    """Measure the rendered mp4 and return a publish-gate report."""

    def __init__(self, expected_fps: float = EXPECTED_FPS) -> None:
        self.expected_fps = expected_fps

    def evaluate(self, path: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        path = Path(path)
        if metadata is None:
            metadata = CartoonMotionQA._load_metadata(path)
        domains: dict[str, list[str]] = {name: [] for name in DOMAINS}
        checks: dict[str, dict[str, Any]] = {}

        def record(name: str, domain: str, failures: list[str], **measured: Any) -> None:
            domains[domain].extend(failures)
            checks[name] = {"passed": not failures, "domain": domain, "failures": list(failures), **measured}

        if not path.is_file() or path.stat().st_size < MIN_VIDEO_BYTES:
            record("file", "technical", ["media_file_missing_or_too_small"])
            return self._report(domains, checks, 0.0)

        data = probe_streams(path)
        if data is None:
            record("probe", "technical", ["ffprobe_failed"])
            return self._report(domains, checks, 0.0)
        streams = data.get("streams", []) if isinstance(data.get("streams"), list) else []
        video = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"), None)
        duration = _float((data.get("format") or {}).get("duration")) or 0.0

        failures = []
        if not video or (video.get("width"), video.get("height")) != (WIDTH, HEIGHT):
            failures.append("wrong_video_dimensions")
        if video and video.get("codec_name") != "h264":
            failures.append(f"video_codec_not_h264 ({video.get('codec_name')})")
        fps = self._fps(video)
        if fps is None or abs(fps - self.expected_fps) > FPS_TOLERANCE:
            failures.append(f"wrong_frame_rate ({fps if fps is not None else 'unknown'} fps; need {self.expected_fps:g})")
        record("format", "technical", failures, width=(video or {}).get("width"), height=(video or {}).get("height"),
               fps=fps, video_codec=(video or {}).get("codec_name"), duration_seconds=round(duration, 3))

        if not audio:
            record("audio_stream", "technical", ["missing_audio"])
            record("loudness", "audio", ["audio_evidence_missing"])
            record("true_peak", "audio", ["audio_evidence_missing"])
            record("silence", "audio", ["audio_evidence_missing"])
            record("speech_rate", "audio", ["audio_evidence_missing"])
        else:
            record("audio_stream", "technical", [])
            self._audio_checks(path, duration, metadata, record)

        self._video_checks(path, duration, metadata, record)
        self._caption_checks(metadata, record)
        return self._report(domains, checks, duration)

    @staticmethod
    def _fps(video: dict[str, Any] | None) -> float | None:
        if not video:
            return None
        num, _, den = str(video.get("r_frame_rate", "0/1")).partition("/")
        try:
            denominator = float(den or 1)
            return float(num) / denominator if denominator else None
        except (TypeError, ValueError):
            return None

    def _audio_checks(self, path: Path, duration: float, metadata: dict[str, Any], record) -> None:
        log = _run_ffmpeg([
            "-i", str(path), "-map", "0:a:0",
            "-af", f"ebur128=peak=true:framelog=quiet,silencedetect=n={SILENCE_DB:g}dB:d={SILENCE_MIN_SECONDS:g}",
            "-f", "null", "-",
        ])
        if log is None:
            record("loudness", "audio", ["loudness_evidence_missing (ffmpeg unavailable)"])
            record("true_peak", "audio", ["true_peak_evidence_missing (ffmpeg unavailable)"])
            record("silence", "audio", ["silence_evidence_missing (ffmpeg unavailable)"])
            record("speech_rate", "audio", ["speech_rate_evidence_missing (ffmpeg unavailable)"])
            return

        integrated, true_peak = parse_loudness(log)
        low, high = LOUDNESS_RANGE_LUFS
        failures = []
        if integrated is None:
            failures.append("loudness_evidence_missing")
        elif not low <= integrated <= high:
            failures.append(f"loudness_out_of_range ({integrated:.1f} LUFS; need {low:g} to {high:g})")
        record("loudness", "audio", failures, integrated_lufs=integrated)

        failures = []
        if true_peak is None:
            failures.append("true_peak_evidence_missing")
        elif true_peak > TRUE_PEAK_MAX_DBTP:
            failures.append(f"true_peak_too_hot ({true_peak:.1f} dBTP; max {TRUE_PEAK_MAX_DBTP:g})")
        record("true_peak", "audio", failures, true_peak_dbtp=true_peak)

        silences = parse_silences(log, duration)
        silent_seconds = sum(end - start for start, end in silences)
        ratio = silent_seconds / duration if duration > 0 else 1.0
        failures = []
        if ratio > SILENCE_MAX_RATIO:
            failures.append(f"silence_ratio_too_high ({ratio:.0%} digital silence; max {SILENCE_MAX_RATIO:.0%})")
        record("silence", "audio", failures, silence_ratio=round(ratio, 3), silent_seconds=round(silent_seconds, 3))

        words = metadata_words(metadata)
        voiced = duration - silent_seconds
        failures = []
        wpm = None
        if words <= 0 or voiced <= 0:
            failures.append("speech_rate_evidence_missing")
        else:
            wpm = words / voiced * 60.0
            low, high = SPEECH_WPM_RANGE
            if not low <= wpm <= high:
                failures.append(f"speech_rate_out_of_range ({wpm:.0f} words per minute; need {low:g} to {high:g})")
        record("speech_rate", "audio", failures, words=words, voiced_seconds=round(voiced, 3),
               words_per_minute=round(wpm, 1) if wpm is not None else None)

    def _video_checks(self, path: Path, duration: float, metadata: dict[str, Any], record) -> None:
        freeze_log = _run_ffmpeg([
            "-i", str(path), "-map", "0:v:0",
            "-vf", f"freezedetect=n={SILENCE_DB:g}dB:d={FREEZE_MAX_SECONDS:g}", "-f", "null", "-",
        ])
        if freeze_log is None:
            record("frozen_video", "video", ["freeze_evidence_missing (ffmpeg unavailable)"])
        else:
            spans = parse_freezes(freeze_log, duration)
            longest = max((end - start for start, end in spans), default=0.0)
            failures = []
            if longest >= FREEZE_MAX_SECONDS:
                failures.append(f"frozen_video ({longest:.1f} s still; max {FREEZE_MAX_SECONDS:g} s)")
            record("frozen_video", "video", failures, longest_freeze_seconds=round(longest, 3))

        edges = _sample_black_edges(path)
        if edges is None:
            record("black_edges", "video", ["black_edge_evidence_missing (ffmpeg unavailable)"])
        else:
            total, flagged, by_edge = edges
            ratio = flagged / total if total else 0.0
            failures = []
            if total == 0:
                failures.append("black_edge_evidence_missing")
            elif ratio > EDGE_BLACK_FRAME_MAX_RATIO:
                failures.append(
                    f"black_edges ({ratio:.0%} of sampled frames have a near-black {max(by_edge, key=by_edge.get)} "
                    f"border; max {EDGE_BLACK_FRAME_MAX_RATIO:.0%})"
                )
            worst = max(by_edge, key=by_edge.get)
            record("black_edges", "video", failures, sampled_frames=total, flagged_frames=flagged,
                   flagged_ratio=round(ratio, 3), worst_edge=worst)

        scene_log = _run_ffmpeg([
            "-i", str(path), "-map", "0:v:0",
            "-vf", f"select='gt(scene,{CUT_SCENE_THRESHOLD:g})',showinfo", "-f", "null", "-",
        ])
        ranges = CartoonMotionQA._shot_frame_ranges(metadata)
        planned = [end / self.expected_fps for _, end, _ in ranges[:-1]]
        if scene_log is None or not ranges:
            record("shot_cuts", "video", ["shot_cut_evidence_missing"])
        else:
            detected = parse_scene_cuts(scene_log)
            failures = [
                f"shot_cut_not_visible (no cut at {boundary:.2f} s)"
                for boundary in planned
                if not any(abs(boundary - found) <= CUT_TOLERANCE_SECONDS for found in detected)
            ]
            record("shot_cuts", "video", failures, planned_cuts_seconds=planned, detected_cuts_seconds=detected)

    def _caption_checks(self, metadata: dict[str, Any], record) -> None:
        failures, measured = caption_checks(metadata)
        record("captions", "captions", failures["captions"], captions=measured["captions"])
        record("disclosure", "captions", failures["disclosure"])

    @staticmethod
    def _report(domains: dict[str, list[str]], checks: dict[str, dict[str, Any]], duration: float) -> dict[str, Any]:
        reasons = list(dict.fromkeys(reason for name in DOMAINS for reason in domains[name]))
        passed_checks = sum(1 for check in checks.values() if check["passed"])
        score = round(passed_checks / len(checks), 3) if checks else 0.0
        return {
            "passed": not reasons and bool(checks),
            "score": score,
            "reasons": reasons,
            "domains": {name: list(dict.fromkeys(domains[name])) for name in DOMAINS},
            "checks": checks,
            "duration_seconds": round(duration, 3),
        }

    @staticmethod
    def combine(render: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
        """Merge the basic render check with the measured report into one media verdict."""
        render_reasons = list(render.get("reasons") or [])
        domains = {name: list(values) for name, values in (measured.get("domains") or {}).items()}
        domains["technical"] = list(dict.fromkeys([*render_reasons, *domains.get("technical", [])]))
        reasons = list(dict.fromkeys([*render_reasons, *measured.get("reasons", [])]))
        passed = bool(render.get("passed")) and bool(measured.get("passed"))
        return {**render, **measured, "passed": passed, "reasons": reasons, "domains": domains}


def metadata_words(metadata: dict[str, Any]) -> int:
    """Total spoken words across the rendered captions."""
    rows = metadata.get("captions") if isinstance(metadata, dict) else None
    if not isinstance(rows, list):
        return 0
    return sum(int(row.get("words_total", 0) or 0) for row in rows if isinstance(row, dict))
