"""Free-first Animation Engine 2.0 for original cartoon shorts.

The engine does not attempt full generative video on the local 8 GB Mac. Instead it
treats each approved action image as an animation keyframe and adds deliberate
performance: anticipation, acceleration, impact/recoil, settle, camera momentum,
breathing-scale, beat-aware punch-ins, shake, and caption timing.

It is intentionally deterministic so QA and learning can compare renders.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.providers.cartoon_decomposer import load_action_layers
from app.providers.cartoon_performance import arm_motion, head_motion
from app.providers.cartoon_rig import CharacterRig, get_rig

W, H = 1080, 1920


@dataclass(frozen=True)
class MotionProfile:
    beat: str
    energy: float
    zoom_start: float
    zoom_end: float
    pan_x: float
    pan_y: float
    rotation: float
    shake: float
    pause: float = 0.0


BEAT_PROFILES = {
    "setup": MotionProfile("setup", 0.65, 1.01, 1.065, -0.018, 0.006, -0.8, 0.0),
    "escalation": MotionProfile("escalation", 1.25, 1.035, 1.16, 0.026, -0.010, 1.4, 9.0),
    "reaction": MotionProfile("reaction", 1.05, 1.12, 1.025, -0.030, 0.012, -2.0, 4.5),
    "punchline": MotionProfile("punchline", 0.42, 1.055, 1.105, 0.008, 0.000, 0.35, 0.0, 0.20),
}


def _font(size: int):
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default()


def _cover(image: Image.Image) -> Image.Image:
    src = image.convert("RGB")
    scale = max(W / src.width, H / src.height)
    src = src.resize((int(src.width * scale), int(src.height * scale)), Image.Resampling.LANCZOS)
    x = max(0, (src.width - W) // 2)
    y = max(0, (src.height - H) // 2)
    return src.crop((x, y, x + W, y + H))


@lru_cache(maxsize=64)
def _load_cover(path: str) -> Image.Image:
    with Image.open(path) as img:
        return _cover(img).copy()


def _ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def _ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _spring(t: float, damping: float = 6.2, frequency: float = 8.0) -> float:
    # Critically damped-looking overshoot used for recoil/settle.
    return 1.0 - math.exp(-damping * t) * math.cos(frequency * t)


def _performance_accent(frame: Image.Image, progress: float, beat: str, shot_index: int) -> Image.Image:
    """Apply a feathered character-centric performance accent.

    The source art is flattened, so we cannot rig limbs. Instead we animate a
    soft upper-body/head window over the already animated camera frame. The
    generated action assets are character-centered, making this a useful,
    deterministic proxy for a tiny lean/bob/tilt without inventing pixels.
    """
    energy = BEAT_PROFILES.get(beat, BEAT_PROFILES["setup"]).energy
    p = max(0.0, min(1.0, progress))
    if beat == "setup":
        amp = 3.0 * energy
    elif beat == "escalation":
        amp = 10.0 * energy
    elif beat == "reaction":
        amp = 8.0 * energy
    else:
        amp = 2.5 * energy

    # Subject window: upper-middle character area used by the factory's
    # consistent portrait action assets. Keep the mask broad and feathered so
    # background seams remain unobtrusive.
    cx, cy = int(W * 0.50), int(H * 0.40)
    rw, rh = int(W * 0.36), int(H * 0.31)
    local = frame.crop((max(0, cx-rw), max(0, cy-rh), min(W, cx+rw), min(H, cy+rh))).copy()

    phase = p * math.tau
    if beat == "escalation":
        dx = amp * math.sin(phase * 1.35 + shot_index)
        dy = -amp * 0.55 * math.sin(phase * 1.8)
        angle = 2.0 * math.sin(phase * 1.15)
        scale = 1.0 + 0.018 * math.sin(phase * 1.6)
    elif beat == "reaction":
        dx = -amp * math.sin(phase * 1.05)
        dy = amp * 0.65 * math.sin(phase * 1.7)
        angle = -3.0 * math.sin(phase * 1.1)
        scale = 1.0 + 0.026 * math.sin(phase * 1.25)
    elif beat == "punchline":
        dx = 0.9 * amp * math.sin(phase * 0.7)
        dy = -0.35 * amp * math.sin(phase)
        angle = 0.8 * math.sin(phase * 0.8)
        scale = 1.0 + 0.008 * math.sin(phase)
    else:
        dx = 0.6 * amp * math.sin(phase)
        dy = -0.35 * amp * math.cos(phase)
        angle = 0.7 * math.sin(phase)
        scale = 1.0 + 0.006 * math.sin(phase)

    local = local.resize((max(2, int(local.width * scale)), max(2, int(local.height * scale))), Image.Resampling.BICUBIC)
    local = local.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    lx = int(cx - local.width / 2 + dx)
    ly = int(cy - local.height / 2 + dy)
    layer.alpha_composite(local.convert("RGBA"), (lx, ly))

    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse((cx-rw, cy-rh, cx+rw, cy+rh), fill=225)
    mask = mask.filter(ImageFilter.GaussianBlur(42))
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), mask))
    frame_rgba = frame.convert("RGBA")
    frame_rgba.alpha_composite(layer)
    return frame_rgba.convert("RGB")


def _transition_direction(previous: CharacterRig | None, current: CharacterRig | None) -> int:
    """Choose transition direction from actual character pose movement."""
    if previous is None or current is None:
        return 1
    dx = current.center[0] - previous.center[0]
    if abs(dx) < W * 0.035:
        return 1 if current.center[1] >= previous.center[1] else -1
    return 1 if dx > 0 else -1


def _transition_frame(previous: Image.Image, current: Image.Image, progress: float, direction: int) -> Image.Image:
    """Snap between poses with directional overlap and a tiny settle overshoot."""
    t = max(0.0, min(1.0, progress))
    # Fast arrival: the new pose should appear decisively, not dissolve.
    arrival = _ease_out(t)
    out = previous.convert("RGBA")
    inc = current.convert("RGBA")
    out_scale = 1.0 - 0.025 * arrival
    inc_scale = 0.975 + 0.025 * arrival
    out = out.resize((int(W*out_scale), int(H*out_scale)), Image.Resampling.BICUBIC)
    inc = inc.resize((int(W*inc_scale), int(H*inc_scale)), Image.Resampling.BICUBIC)
    out_canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    inc_canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    # Outgoing pose gives way in the direction of the incoming pose.
    out_dx = int(direction * 28 * arrival)
    in_dx = int(direction * 54 * (1.0 - arrival))
    out_canvas.alpha_composite(out, ((W-out.width)//2 + out_dx, (H-out.height)//2))
    inc_canvas.alpha_composite(inc, ((W-inc.width)//2 - in_dx, (H-inc.height)//2))

    # Slightly curved wipe edge avoids the sterile PowerPoint-slide look.
    edge = int(W * (0.06 + 0.88 * arrival))
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    if direction > 0:
        md.rectangle((0, 0, edge, H), fill=255)
    else:
        md.rectangle((W-edge, 0, W, H), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(13))
    result = Image.composite(inc_canvas, out_canvas, mask)

    # One-frame-ish impact veil at the handoff makes the new pose read as an
    # action beat rather than a hidden dissolve.
    impact = math.sin(math.pi * max(0.0, min(1.0, (t - 0.72) / 0.28)))
    if impact > 0.0:
        veil = Image.new("RGBA", (W, H), (255, 255, 255, int(18 * impact)))
        result.alpha_composite(veil)
    return result.convert("RGB")


def _visible_caption(text: str, progress: float, punch: bool = False) -> str:
    words = " ".join(text.split()).split()
    if not words:
        return ""
    n = max(1, min(len(words), math.ceil(len(words) * max(0.0, min(1.0, progress)))))
    shown = words[:n]
    if punch and shown:
        shown[-1] = shown[-1].upper()
    return " ".join(shown)


def _caption(draw: ImageDraw.ImageDraw, text: str, progress: float, punch: bool = False) -> None:
    shown = _visible_caption(text, progress, punch)
    if not shown:
        return
    font = _font(68)
    max_width = 920
    words = shown.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if draw.textbbox((0, 0), candidate, font=font, stroke_width=2)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    lines = lines[:3]
    line_h = 78
    y0 = H - 405
    for i, line in enumerate(lines):
        box = draw.textbbox((0, 0), line, font=font, stroke_width=2)
        tw = box[2] - box[0]
        draw.text(
            ((W - tw) / 2, y0 + i * line_h),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=8,
            stroke_fill=(0, 0, 0, 235),
        )


def _disclosure(draw: ImageDraw.ImageDraw) -> None:
    text = "AI-generated • fictional cartoon"
    font = _font(28)
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    draw.rounded_rectangle(
        (W - width - 50, H - 68, W - 22, H - 22),
        radius=12,
        fill=(8, 10, 15, 180),
    )
    draw.text((W - width - 36, H - 61), text, font=font, fill=(245, 245, 245, 235))


def _impact_graphics(frame: Image.Image, intensity: float) -> None:
    if intensity <= 0:
        return
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    cx, cy = int(W * 0.50), int(H * 0.47)
    count = 12
    for i in range(count):
        angle = (math.tau * i / count) + 0.16
        r0 = 145
        r1 = 245 + intensity * 9
        x0 = cx + math.cos(angle) * r0
        y0 = cy + math.sin(angle) * r0
        x1 = cx + math.cos(angle) * r1
        y1 = cy + math.sin(angle) * r1
        draw.line((x0, y0, x1, y1), fill=(255, 255, 255, int(110 * intensity)), width=7)
    overlay = overlay.filter(ImageFilter.GaussianBlur(0.35))
    frame.alpha_composite(overlay)


def _apply_rig_performance(frame: Image.Image, rig: CharacterRig, progress: float, beat: str) -> Image.Image:
    dx, dy, scale, rotation = head_motion(progress, beat)
    x0, y0, x1, y1 = rig.head_bbox
    crop = rig.head_layer.crop((x0, y0, x1, y1))
    crop = crop.resize((max(2, int(crop.width * scale)), max(2, int(crop.height * scale))), Image.Resampling.BICUBIC)
    crop = crop.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=False)
    hx, hy = rig.center
    cover = frame.filter(ImageFilter.GaussianBlur(11))
    base = frame.copy()
    # Feather only the head region, then paste the transformed head crop into
    # the existing RGB frame. Avoid constructing a full-frame transparent layer.
    base = Image.composite(cover, base, rig.cover_mask).convert("RGB")
    rgba = crop.convert("RGBA")
    mask = rgba.getchannel("A")
    base.paste(rgba.convert("RGB"), (int(hx - crop.width / 2 + dx), int(hy - crop.height / 2 + dy)), mask)
    return base


def _apply_torso_layer(frame: Image.Image, layer: Image.Image, progress: float, beat: str) -> Image.Image:
    """Animate the character core with a small beat-aware lean/bounce."""
    alpha = layer.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return frame
    x0, y0, x1, y1 = bbox
    crop = layer.crop((x0, y0, x1, y1))
    energy = BEAT_PROFILES.get(beat, BEAT_PROFILES["setup"]).energy
    anticipation = math.sin(math.pi * min(1.0, progress / 0.28))
    settle = max(0.0, min(1.0, (progress - 0.68) / 0.32))
    dx = math.sin(progress * math.pi * 1.35) * (3.0 + 5.0 * energy)
    dy = -anticipation * (3.0 + 8.0 * energy) + settle * 3.0
    scale = 1.0 + math.sin(math.pi * progress) * 0.010 * energy
    rotation = math.sin(math.pi * progress) * {"setup": -0.7, "escalation": 1.8, "reaction": -2.2, "punchline": 0.4}.get(beat, 0.0)
    crop = crop.resize((max(2, int(crop.width * scale)), max(2, int(crop.height * scale))), Image.Resampling.BICUBIC)
    crop = crop.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=False)
    ox = int(x0 + dx - (crop.width - (x1 - x0)) / 2)
    oy = int(y0 + dy - (crop.height - (y1 - y0)) / 2)
    base = frame.copy()
    base.paste(crop.convert("RGB"), (ox, oy), crop.getchannel("A"))
    return base


def _apply_arm_layer(frame: Image.Image, layer: Image.Image, progress: float, beat: str, side: str) -> Image.Image:
    alpha = layer.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return frame
    x0, y0, x1, y1 = bbox
    crop = layer.crop((x0, y0, x1, y1))
    dx, dy, scale, rotation = arm_motion(progress, beat, side)
    crop = crop.resize((max(2, int(crop.width * scale)), max(2, int(crop.height * scale))), Image.Resampling.BICUBIC)
    crop = crop.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=False)
    ox = int(x0 + dx - (crop.width - (x1 - x0)) / 2)
    oy = int(y0 + dy - (crop.height - (y1 - y0)) / 2)
    base = frame.copy()
    base.paste(crop.convert("RGB"), (ox, oy), crop.getchannel("A"))
    return base


def motion_frame(
    image: Path,
    caption: str,
    progress: float,
    shot_index: int,
    duration: float,
    beat: str,
    rig: CharacterRig | None = None,
) -> Image.Image:
    profile = BEAT_PROFILES.get(beat, BEAT_PROFILES["setup"])
    base = _load_cover(str(image))
    p = max(0.0, min(1.0, progress))
    # Punchlines should land and then hold. Freeze the camera at the final
    # performance pose so the joke reads as a deliberate comedic beat rather
    # than a continued Ken-Burns move.
    if beat == "punchline" and p >= 0.72:
        p = 0.72
    if rig is not None:
        try:
            rig_dir = Path(__file__).resolve().parents[2] / "artifacts" / "cartoon" / "pip_rig_v1"
            layers = load_action_layers(image, rig_dir)
            base = layers["body_plate"].convert("RGB")
            base = _apply_torso_layer(base, layers["torso"], p, beat)
            base = _apply_arm_layer(base, layers["left_arm"], p, beat, "left")
            base = _apply_arm_layer(base, layers["right_arm"], p, beat, "right")
        except (OSError, ValueError):
            pass
        base = _apply_rig_performance(base, rig, p, beat)
    eased = _ease_in_out(p)
    energy = profile.energy

    # First 20% is anticipation, middle is action, last 25% settles.
    anticipation = min(1.0, p / 0.20) if p < 0.20 else 1.0
    settle = max(0.0, min(1.0, (p - 0.72) / 0.28))
    spring = _spring(max(0.0, (p - 0.12) / 0.88))

    zoom = profile.zoom_start + (profile.zoom_end - profile.zoom_start) * eased
    zoom += 0.012 * math.sin(p * math.tau) * energy
    if beat == "reaction":
        zoom += 0.022 * math.sin((p * 1.2) * math.tau) * (1.0 - settle)

    crop_w = max(2, int(base.width / zoom))
    crop_h = max(2, int(base.height / zoom))
    max_x = max(0, base.width - crop_w)
    max_y = max(0, base.height - crop_h)

    drift_x = profile.pan_x * max_x * math.sin(math.pi * eased)
    drift_y = profile.pan_y * max_y * math.sin(math.pi * eased)
    x = int(max_x * 0.5 + drift_x)
    y = int(max_y * 0.5 + drift_y)
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))

    frame = base.crop((x, y, x + crop_w, y + crop_h)).resize((W, H), Image.Resampling.LANCZOS)
    angle = profile.rotation * math.sin(math.pi * eased) * (1.0 - 0.55 * settle)
    frame = frame.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

    # Squash/stretch proxy gives action frames a small animated rhythm without
    # deforming the generated character pixels.
    sx = 1.0 + 0.010 * math.sin(math.pi * spring) * energy
    sy = 1.0 - 0.006 * math.sin(math.pi * spring) * energy
    if beat == "escalation":
        sx += 0.014 * max(0.0, math.sin(p * math.pi * 2.0))
        sy -= 0.010 * max(0.0, math.sin(p * math.pi * 2.0))
    frame = frame.resize((int(W * sx), int(H * sy)), Image.Resampling.BICUBIC)
    if frame.size != (W, H):
        left = max(0, (frame.width - W) // 2)
        top = max(0, (frame.height - H) // 2)
        frame = frame.crop((left, top, left + W, top + H))

    # For rigged hero assets the head layer already supplies the localized
    # performance motion. Keep the older broad accent as the generic fallback
    # for unknown flattened art so we do not double-transform the subject.
    if rig is None:
        frame = _performance_accent(frame, p, beat, shot_index)

    # Deterministic impact shake around escalation/reaction peaks. The window
    # is deliberately short: the eye should read an event, not camera jitter.
    peak_start = 0.34 if beat == "escalation" else 0.36 if beat == "reaction" else 1.0
    peak_end = 0.52 if beat == "escalation" else 0.50 if beat == "reaction" else 1.0
    shake_window = math.sin(math.pi * max(0.0, min(1.0, (p - peak_start) / max(0.01, peak_end - peak_start))))
    shake = profile.shake * shake_window
    if shake:
        dx = int(math.sin(shot_index * 8 + p * 70) * shake)
        dy = int(math.cos(shot_index * 5 + p * 63) * shake * 0.6)
        frame = frame.transform((W, H), Image.AFFINE, (1, 0, -dx, 0, 1, -dy), resample=Image.Resampling.BICUBIC)

    # Soft contrast/sharpness polish is cheap and deterministic.
    if beat in {"escalation", "reaction"}:
        frame = ImageEnhance.Contrast(frame).enhance(1.035)
        frame = ImageEnhance.Sharpness(frame).enhance(1.08)

    rgba = frame.convert("RGBA")

    # White impact flash makes the escalation read as an event, not just a zoom.
    flash = 0.0
    if beat == "escalation":
        flash = max(0.0, math.sin(math.pi * max(0.0, min(1.0, (p - 0.35) / 0.12))))
    if flash > 0.0:
        veil = Image.new("RGBA", (W, H), (255, 255, 255, int(45 * flash)))
        rgba.alpha_composite(veil)
        _impact_graphics(rgba, flash)

    draw = ImageDraw.Draw(rgba, "RGBA")
    reveal = (p - 0.04) / max(0.15, 0.72 + profile.pause)
    if beat == "punchline":
        reveal = min(1.0, max(0.0, (p - profile.pause) / max(0.15, 1.0 - profile.pause)))
    punch = beat == "punchline" and p > 0.58
    _caption(draw, caption, reveal, punch=punch)
    _disclosure(draw)

    return rgba.convert("RGB")


def _audio_for_shots(
    work: Path,
    voice_lines: list[str],
    durations: list[float],
    sfx_kinds: list[str],
    ffmpeg: str,
    sfx_peak_fractions: list[float] | None = None,
) -> Path | None:
    from app.providers.cartoon_media import _audio_for_shots as build_audio
    return build_audio(work, voice_lines, durations, sfx_kinds, ffmpeg, sfx_peak_fractions)


def render_cartoon_motion(
    images: list[Path],
    captions: list[str],
    durations: list[float],
    beats: list[str],
    output: Path,
    voice_lines: list[str] | None = None,
    sfx_kinds: list[str] | None = None,
    fps: int = 24,
) -> Path:
    if not images or len(images) != len(captions) or len(images) != len(durations) or len(images) != len(beats):
        raise ValueError("images, captions, durations and beats must have equal non-zero lengths")
    if fps < 12 or fps > 30:
        raise ValueError("fps must be between 12 and 30")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")

    voice_lines = voice_lines or captions
    sfx_kinds = sfx_kinds or ["none"] * len(images)
    if len(voice_lines) != len(images) or len(sfx_kinds) != len(images):
        raise ValueError("voice_lines and sfx_kinds must match image count")

    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f"{output.stem}_frames"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    # Render at the requested delivery cadence directly. The old implementation
    # rendered 12fps and used minterpolate to synthesize 24fps, which could create
    # rubbery motion around captions and pose changes. Deterministic 24fps frames
    # are more faithful and avoid an expensive interpolation pass.
    source_fps = fps
    frame_index = 0
    frame_map = []
    previous_shot_frame: Image.Image | None = None
    previous_shot_rig: CharacterRig | None = None
    transition_frames = max(1, round(0.16 * source_fps))
    transition_directions: list[int] = []
    for shot, (image, caption, duration, beat) in enumerate(zip(images, captions, durations, beats)):
        count = max(2, round(duration * source_fps))
        frame_map.append({"shot": shot, "beat": beat, "frames": count, "duration": duration})
        for i in range(count):
            progress = i / max(1, count - 1)
            try:
                rig = get_rig(str(image))
            except ValueError:
                rig = None
            frame = motion_frame(image, caption, progress, shot, duration, beat, rig=rig)
            if previous_shot_frame is not None and i < transition_frames:
                if i == 0:
                    transition_directions.append(_transition_direction(previous_shot_rig, rig))
                transition_p = i / max(1, transition_frames - 1)
                direction = transition_directions[-1] if transition_directions else 1
                frame = _transition_frame(previous_shot_frame, frame, transition_p, direction)
            frame.save(work / f"frame_{frame_index:05d}.jpg", quality=91, optimize=True)
            frame_index += 1
        previous_shot_frame = frame
        previous_shot_rig = rig

    video_silent = output.with_suffix(".silent.mp4")
    subprocess.run(
        [
            ffmpeg, "-y",
            "-framerate", str(source_fps),
            "-i", str(work / "frame_%05d.jpg"),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-vf", "scale=1080:1920:flags=lanczos,format=yuv420p",
            "-r", str(source_fps),
            str(video_silent),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    video_for_mux = video_silent

    peak_by_beat = {
        "setup": 0.62,
        "escalation": 0.43,
        "reaction": 0.30,
        "punchline": 0.18,
    }
    audio = _audio_for_shots(
        work, voice_lines, durations, sfx_kinds, ffmpeg,
        [peak_by_beat.get(beat, 0.40) for beat in beats],
    )
    if audio:
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(video_for_mux), "-i", str(audio), "-shortest",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        video_for_mux.replace(output)
    video_for_mux.unlink(missing_ok=True)

    metadata = output.with_suffix(".motion.json")
    metadata.write_text(
        json.dumps(
            {
                "engine": "cartoon-motion-3.1",
                "rig_mode": "flattened-art-performance-proxy",
                "fps": fps,
                "source_fps": source_fps,
                "interpolation": "none",
                "performance": {
                    "action_transitions": True,
                    "pose_aware_transition_direction": True,
                    "character_micro_motion": True,
                    "anticipation_recoil_settle": True,
                    "punchline_hold": True,
                },
                "frames": frame_index,
                "shots": frame_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
