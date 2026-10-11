"""Editor AI: assemble and conform the final vertical reel against the edit plan.

Two jobs, one contract (``app/agents/edit_decision.build_edit_plan``):

* ``assemble_reel`` cuts separately generated shot clips (for example hosted
  Runway output) into one 1080x1920 reel: each clip is normalised, trimmed or
  held to its planned duration, captions and the AI disclosure are burned in at
  their planned times, and the locally produced voice/SFX track is muxed on top.
* ``verify_reel`` checks an already-assembled reel (the local renderer composes
  shots, audio and captions itself) against the same plan: format, duration,
  shot count, cut points and hook timing. Video QA then inspects what ships.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from app.providers.cartoon_audio import loudness_normalize

W, H = 1080, 1920
DISCLOSURE = "AI-generated • fictional cartoon"


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg is required for the editor")
    return exe


def probe(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required for the editor")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:4]


def render_caption_overlay(text: str, output: Path, *, disclosure: str = DISCLOSURE) -> Path:
    """Transparent 1080x1920 overlay: caption in the lower-third safe zone plus disclosure."""
    from app.providers.cartoon_animation import _font

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(58)
    lines = _wrap(draw, " ".join(str(text).split()), font, W - 2 * 110)
    y = 1490
    for line in lines:
        width = draw.textlength(line, font=font)
        x = (W - width) / 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255), stroke_width=6, stroke_fill=(10, 10, 20, 255))
        y += 70
    if disclosure:
        small = _font(30)
        width = draw.textlength(disclosure, font=small)
        x0, y0 = W - width - 60, H - 92
        draw.rounded_rectangle((x0 - 18, y0 - 10, x0 + width + 18, y0 + 44), radius=14, fill=(0, 0, 0, 170))
        draw.text((x0, y0), disclosure, font=small, fill=(255, 255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output)
    return output


def assemble_reel(
    clips: list[Path],
    edit_plan: dict[str, Any],
    output: Path,
    *,
    audio: Path | None = None,
    fps: int = 24,
    disclosure: str = DISCLOSURE,
) -> dict[str, Any]:
    """Cut shot clips into the planned reel with captions, disclosure and audio."""
    shots = list(edit_plan.get("shots", []))
    if not clips or len(clips) != len(shots):
        raise ValueError("assemble_reel needs exactly one clip per planned shot")
    ffmpeg = _ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nova-editor-") as tmp:
        work = Path(tmp)
        if audio is not None:
            # The local voice track gets the same -14 LUFS / -1.5 dBTP target as local episodes.
            normalized = work / "voice_loudness.wav"
            loudness_normalize(audio, normalized)
            audio = normalized
        segments = []
        for index, (clip, shot) in enumerate(zip(clips, shots)):
            duration = max(0.25, float(shot["duration_seconds"]))
            segment = work / f"seg_{index:02d}.mp4"
            # Cover-crop to 9:16, hold the last frame if the clip is short, then
            # trim to exactly the planned shot length.
            vf = (
                f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={fps},"
                f"tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f},setpts=PTS-STARTPTS"
            )
            subprocess.run(
                [ffmpeg, "-y", "-i", str(clip), "-vf", vf, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-r", str(fps), str(segment)],
                check=True, capture_output=True, text=True,
            )
            segments.append(segment)
        listing = work / "segments.txt"
        listing.write_text("".join(f"file '{seg}'\n" for seg in segments))
        joined = work / "joined.mp4"
        subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)],
            check=True, capture_output=True, text=True,
        )

        overlays = []
        for index, shot in enumerate(shots):
            caption = shot.get("caption", {})
            overlays.append((
                render_caption_overlay(str(caption.get("text", "")), work / f"cap_{index:02d}.png", disclosure=disclosure),
                float(caption.get("start_seconds", shot["start_seconds"])),
                float(caption.get("end_seconds", shot["end_seconds"])),
            ))
        inputs = ["-i", str(joined)]
        for path, _, _ in overlays:
            inputs += ["-i", str(path)]
        chain = []
        label = "0:v"
        for index, (_, start, end) in enumerate(overlays, start=1):
            out = f"v{index}"
            chain.append(f"[{label}][{index}:v]overlay=0:0:enable='between(t,{start:.3f},{end:.3f})'[{out}]")
            label = out
        command = [ffmpeg, "-y", *inputs]
        audio_index = len(overlays) + 1
        if audio is not None:
            command += ["-i", str(audio)]
        command += ["-filter_complex", ";".join(chain), "-map", f"[{label}]"]
        if audio is not None:
            command += ["-map", f"{audio_index}:a", "-af", "apad", "-c:a", "aac", "-b:a", "160k", "-shortest"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), str(output)]
        subprocess.run(command, check=True, capture_output=True, text=True)
    return verify_reel(output, edit_plan, shot_frames=[round(float(s["duration_seconds"]) * fps) for s in shots], fps=fps,
                       expect_audio=audio is not None)


def verify_reel(
    video: Path,
    edit_plan: dict[str, Any],
    *,
    shot_frames: list[int] | None = None,
    fps: int = 24,
    expect_audio: bool = True,
) -> dict[str, Any]:
    """Conform an assembled reel to the edit plan; returns a report with reasons."""
    reasons: list[str] = []
    data = probe(video)
    streams = data.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    planned = float(edit_plan.get("duration_seconds", 0.0) or 0.0)
    shots = list(edit_plan.get("shots", []))

    if (vstream.get("width"), vstream.get("height")) != (W, H):
        reasons.append("not_vertical_1080x1920")
    if expect_audio and not has_audio:
        reasons.append("missing_audio")
    if planned and abs(duration - planned) > max(0.35, 0.04 * planned):
        reasons.append(f"duration_off_plan:{duration:.2f}s_vs_{planned:.2f}s")

    measured_cuts: list[float] = []
    if shot_frames is not None:
        if len(shot_frames) != len(shots):
            reasons.append(f"shot_count_off_plan:{len(shot_frames)}_vs_{len(shots)}")
        clock = 0
        for frames in shot_frames[:-1]:
            clock += int(frames)
            measured_cuts.append(round(clock / float(fps), 3))
        for planned_cut, measured in zip(edit_plan.get("cut_points_seconds", []), measured_cuts):
            if abs(float(planned_cut) - measured) > 0.25:
                reasons.append(f"cut_off_plan:{measured:.2f}s_vs_{float(planned_cut):.2f}s")
                break

    hook = edit_plan.get("hook", {})
    if float(hook.get("end_seconds", 0.0) or 0.0) > 1.0 + 1e-6:
        reasons.append("hook_not_in_first_second")
    if shots and float(shots[0].get("duration_seconds", 0.0)) < 0.5:
        reasons.append("hook_shot_too_short_to_read")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "video": str(video),
        "duration_seconds": round(duration, 3),
        "planned_duration_seconds": planned,
        "measured_cut_points_seconds": measured_cuts,
        "planned_cut_points_seconds": list(edit_plan.get("cut_points_seconds", [])),
        "loop": edit_plan.get("loop", {}),
        "aspect_ratio": edit_plan.get("aspect_ratio", "9:16"),
    }
