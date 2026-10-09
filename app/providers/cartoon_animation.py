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
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.providers.cartoon_decomposer import load_action_layers
from app.providers.cartoon_performance import arm_motion, head_motion, kinematic_state
from app.providers.cartoon_rig import CharacterRig, get_rig
from app.providers.cartoon_voice import prepare_voice_performance

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


def _camera_geometry(progress: float, beat: str, voice_performance: dict | None = None) -> tuple[int, int, int, int, float, float, float]:
    """Return the camera crop/rotation/squash used by motion_frame."""
    profile = BEAT_PROFILES.get(beat, BEAT_PROFILES["setup"])
    p = max(0.0, min(1.0, progress))
    shared_state = kinematic_state(p, beat, voice_performance)
    camera_state = kinematic_state(max(0.0, p - 0.045), beat, voice_performance)
    if beat == "punchline" and p >= 0.72:
        p = 0.72
    eased = _ease_in_out(p)
    energy = profile.energy
    settle = max(0.0, min(1.0, (p - 0.72) / 0.28))
    spring = _spring(max(0.0, (p - 0.12) / 0.88))
    zoom = profile.zoom_start + (profile.zoom_end - profile.zoom_start) * eased
    zoom += 0.012 * math.sin(p * math.tau) * energy
    if beat == "reaction":
        zoom += 0.022 * math.sin((p * 1.2) * math.tau) * (1.0 - settle)
    zoom += camera_state["camera_zoom"]

    crop_w = max(2, int(W / zoom))
    crop_h = max(2, int(H / zoom))
    max_x = max(0, W - crop_w)
    max_y = max(0, H - crop_h)
    drift_x = profile.pan_x * max_x * math.sin(math.pi * eased)
    drift_y = profile.pan_y * max_y * math.sin(math.pi * eased)
    # Camera is a follower, not another independent oscillator. Root motion is
    # filtered down heavily so the viewer feels the character's momentum without
    # getting seasick.
    follow_x = camera_state["camera_dx"]
    follow_y = camera_state["camera_dy"]
    focus_x = camera_state["camera_focus_x"] * max_x * 0.10
    focus_y = camera_state["camera_focus_y"] * max_y * 0.10
    x = max(0, min(int(max_x * 0.5 + drift_x - follow_x * 2.2 + focus_x), max_x))
    y = max(0, min(int(max_y * 0.5 + drift_y - follow_y * 2.2 + focus_y), max_y))
    angle = (profile.rotation * math.sin(math.pi * eased) * (1.0 - 0.55 * settle)) + camera_state["camera_rot"]
    sx = 1.0 + 0.010 * math.sin(math.pi * spring) * energy
    sy = 1.0 - 0.006 * math.sin(math.pi * spring) * energy
    if beat == "escalation":
        sx += 0.014 * max(0.0, math.sin(p * math.pi * 2.0))
        sy -= 0.010 * max(0.0, math.sin(p * math.pi * 2.0))
    return x, y, crop_w, crop_h, angle, sx, sy


def _impact_shake_envelope(progress: float, event_fraction: float, beat: str) -> float:
    """Return a brief shake envelope centered on the choreographed impact."""
    if beat not in {"escalation", "reaction"}:
        return 0.0
    start = max(0.0, event_fraction - 0.10)
    end = min(1.0, event_fraction + 0.10)
    if end <= start or progress <= start or progress >= end:
        return 0.0
    return math.sin(math.pi * (progress - start) / (end - start))


def _transform_part(
    layer: Image.Image,
    dx: float,
    dy: float,
    scale: float,
    rotation: float,
    pivot: tuple[float, float] | None = None,
) -> Image.Image:
    """Transform a character part while preserving a physical attachment pivot.

    Arms rotate around the shoulder, the head around the neck, and the backpack
    around its strap pivot. That small geometric detail removes a major source
    of the "floating sprite" look in flattened source artwork.
    """
    alpha = layer.getchannel("A")
    bbox = alpha.getbbox()
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if not bbox:
        return out
    x0, y0, x1, y1 = bbox
    crop = layer.crop((x0, y0, x1, y1))
    original_center = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
    pivot = pivot or original_center

    crop = crop.resize(
        (max(2, int(crop.width * scale)), max(2, int(crop.height * scale))),
        Image.Resampling.BICUBIC,
    )
    rel_x = (pivot[0] - original_center[0]) * scale
    rel_y = (pivot[1] - original_center[1]) * scale
    theta = math.radians(rotation)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotated_rel_x = cos_t * rel_x - sin_t * rel_y
    rotated_rel_y = sin_t * rel_x + cos_t * rel_y
    crop = crop.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=False)

    target_pivot = (pivot[0] + dx, pivot[1] + dy)
    crop_center = (crop.width * 0.5, crop.height * 0.5)
    ox = int(target_pivot[0] - rotated_rel_x - crop_center[0])
    oy = int(target_pivot[1] - rotated_rel_y - crop_center[1])
    out.alpha_composite(crop, (ox, oy))
    return out


def _project_part_to_camera(layer: Image.Image, progress: float, beat: str, voice_performance: dict | None = None) -> Image.Image:
    """Apply the same camera transform used for the finished motion frame."""
    x, y, crop_w, crop_h, angle, sx, sy = _camera_geometry(progress, beat, voice_performance)
    crop = layer.crop((x, y, x + crop_w, y + crop_h))
    crop = crop.resize((W, H), Image.Resampling.LANCZOS)
    crop = crop.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    crop = crop.resize((int(W * sx), int(H * sy)), Image.Resampling.BICUBIC)
    if crop.size != (W, H):
        left = max(0, (crop.width - W) // 2)
        top = max(0, (crop.height - H) // 2)
        crop = crop.crop((left, top, left + W, top + H))
    return crop


def _endpoint_character_parts(image: Path, rig: CharacterRig, beat: str, progress: float, voice_performance: dict | None = None) -> dict[str, Image.Image]:
    """Build endpoint character parts in final-camera coordinates."""
    if beat == "punchline" and progress >= 0.72:
        progress = 0.72
    rig_dir = Path(__file__).resolve().parents[2] / "artifacts" / "cartoon" / "pip_rig_v2"
    layers = load_action_layers(image, rig_dir)

    torso = layers["torso"]
    state = kinematic_state(progress, beat, voice_performance)
    torso_dx = state["root_dx"]
    torso_dy = state["root_dy"]
    torso_scale = state["root_scale"]
    torso_rot = state["root_rot"]
    torso = _transform_part(torso, torso_dx, torso_dy, torso_scale, torso_rot)

    left_dx, left_dy, left_scale, left_rot = arm_motion(progress, beat, "left", voice_performance)
    right_dx, right_dy, right_scale, right_rot = arm_motion(progress, beat, "right", voice_performance)
    left_arm = _transform_part(layers["left_arm"], left_dx, left_dy, left_scale, left_rot, pivot=_joint(image.name, "left_shoulder"))
    right_arm = _transform_part(layers["right_arm"], right_dx, right_dy, right_scale, right_rot, pivot=_joint(image.name, "right_shoulder"))

    head_dx, head_dy, head_scale, head_rot = head_motion(progress, beat, voice_performance)
    head = _transform_part(layers["head"], head_dx, head_dy, head_scale, head_rot, pivot=_joint(image.name, "neck"))
    face_dx, face_dy, face_sx, face_sy, _ = _face_motion(progress, beat, voice_performance)
    face = _transform_part(rig.face_layer, face_dx, face_dy, (face_sx + face_sy) * 0.5, 0.0)

    # Keep backpack attached to the torso with a restrained secondary motion.
    backpack = _transform_part(
        layers["backpack"],
        torso_dx * 0.78 + math.sin(progress * math.tau * 2.2) * 1.6,
        torso_dy * 0.84 + math.cos(progress * math.tau * 1.7) * 1.0,
        1.0 + (torso_scale - 1.0) * 0.6,
        torso_rot * 0.55,
        pivot=_joint(image.name, "backpack"),
    )

    parts: dict[str, Image.Image] = {}
    for name, layer in (
        ("torso", torso),
        ("backpack", backpack),
        ("head", head),
        ("left_arm", left_arm),
        ("right_arm", right_arm),
        ("face", face),
    ):
        parts[name] = _project_part_to_camera(layer, progress, beat, voice_performance)
    return parts



PIP_JOINTS: dict[str, dict[str, tuple[float, float]]] = {
    "pip_press_button.png": {
        "neck": (0.455, 0.355),
        "left_shoulder": (0.365, 0.515),
        "right_shoulder": (0.675, 0.405),
        "backpack": (0.185, 0.585),
    },
    "pip_shock.png": {
        "neck": (0.765, 0.505),
        "left_shoulder": (0.580, 0.615),
        "right_shoulder": (0.885, 0.585),
        "backpack": (0.900, 0.685),
    },
    "pip_overwhelmed.png": {
        "neck": (0.355, 0.665),
        "left_shoulder": (0.170, 0.690),
        "right_shoulder": (0.500, 0.675),
        "backpack": (0.105, 0.765),
    },
    "pip_deadpan.png": {
        "neck": (0.555, 0.515),
        "left_shoulder": (0.410, 0.625),
        "right_shoulder": (0.715, 0.605),
        "backpack": (0.375, 0.625),
    },
}


def _joint(name: str, part: str) -> tuple[float, float] | None:
    row = PIP_JOINTS.get(name, {})
    frac = row.get(part)
    if not frac:
        return None
    return (frac[0] * W, frac[1] * H)

POSE_SWITCH_FRACTIONS = {
    "left_arm": 0.40,
    "right_arm": 0.47,
    "torso": 0.44,
    "backpack": 0.46,
    "head": 0.52,
    "face": 0.54,
}


def _morph_transparent_layers(
    previous: Image.Image,
    current: Image.Image,
    progress: float,
    direction: int,
    part_name: str = "body",
) -> Image.Image:
    """Travel between endpoint geometry, then stagger each part into the new pose."""
    t = max(0.0, min(1.0, progress))
    a = previous.convert("RGBA")
    b = current.convert("RGBA")
    ab = a.getchannel("A").getbbox()
    bb = b.getchannel("A").getbbox()
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if not ab and not bb:
        return out

    def crop_info(layer: Image.Image, box: tuple[int, int, int, int] | None):
        if not box:
            return None
        x0, y0, x1, y1 = box
        crop = layer.crop(box)
        return crop, ((x0 + x1) * 0.5, (y0 + y1) * 0.5), (max(2, x1 - x0), max(2, y1 - y0))

    ai = crop_info(a, ab)
    bi = crop_info(b, bb)
    eased = _ease_in_out(t)

    if ai and bi:
        _, ac, asz = ai
        _, bc, bsz = bi
        cx = ac[0] + (bc[0] - ac[0]) * eased + direction * 10.0 * math.sin(math.pi * t)
        cy = ac[1] + (bc[1] - ac[1]) * eased - 4.0 * math.sin(math.pi * t)
        width = int(asz[0] + (bsz[0] - asz[0]) * eased)
        height = int(asz[1] + (bsz[1] - asz[1]) * eased)
        threshold = POSE_SWITCH_FRACTIONS.get(part_name, 0.50)
        chosen = ai[0] if t < threshold else bi[0]
        crop = chosen.resize((width, height), Image.Resampling.BICUBIC)

        # A very small trailing echo sells travel without creating a duplicate
        # character. It is deliberately blurred and capped at low opacity.
        travel = math.sin(math.pi * t)
        trail = crop.filter(ImageFilter.GaussianBlur(1.2))
        trail.putalpha(trail.getchannel("A").point(lambda v: int(v * 0.12 * travel)))
        out.alpha_composite(
            trail,
            (int(cx - width / 2 - direction * 10 * travel), int(cy - height / 2)),
        )
        out.alpha_composite(crop, (int(cx - width / 2), int(cy - height / 2)))
    elif ai:
        crop, center, size = ai
        crop = crop.copy()
        crop.putalpha(crop.getchannel("A").point(lambda v: int(v * (1.0 - eased))))
        out.alpha_composite(crop, (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2)))
    else:
        crop, center, size = bi
        crop = crop.copy()
        crop.putalpha(crop.getchannel("A").point(lambda v: int(v * eased)))
        out.alpha_composite(crop, (int(center[0] - size[0] / 2), int(center[1] - size[1] / 2)))

    return out


def _pose_transition_frame(
    previous_frame: Image.Image,
    current_frame: Image.Image,
    previous_image: Path,
    current_image: Path,
    previous_rig: CharacterRig,
    current_rig: CharacterRig,
    previous_beat: str,
    current_beat: str,
    progress: float,
    direction: int,
) -> Image.Image:
    """Move the character itself between action poses during a short bridge."""
    t = max(0.0, min(1.0, progress))
    # Keep the scene transition as a directional cut rather than a full-frame
    # dissolve. The character itself is what we interpolate; dissolving the
    # entire kitchen creates duplicate sandwich stacks/background geometry.
    base = _transition_frame(
        previous_frame.convert("RGB"),
        current_frame.convert("RGB"),
        t,
        direction,
    )

    # Remove the ghosted character from the endpoint base and rebuild it from
    # independently moving parts. Keep the caption zone untouched.
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    for rig in (previous_rig, current_rig):
        x0, y0, x1, y1 = rig.bbox
        y1 = min(y1, H - 440)
        if y1 > y0:
            md.rounded_rectangle((max(0, x0 - 35), max(0, y0 - 35), min(W, x1 + 35), y1), radius=70, fill=235)
    mask = mask.filter(ImageFilter.GaussianBlur(24))
    blurred = base.filter(ImageFilter.GaussianBlur(7.5))
    base = Image.composite(blurred, base, mask)

    prev_parts = _endpoint_character_parts(previous_image, previous_rig, previous_beat, 1.0)
    curr_parts = _endpoint_character_parts(current_image, current_rig, current_beat, 0.0)
    result = base.convert("RGBA")
    for name in ("torso", "backpack", "head", "left_arm", "right_arm", "face"):
        result.alpha_composite(
            _morph_transparent_layers(prev_parts[name], curr_parts[name], t, direction, name)
        )

    # A restrained one-beat flash right after the midpoint sells the pose snap.
    impact = max(0.0, math.sin(math.pi * max(0.0, min(1.0, (t - 0.58) / 0.26))))
    if impact > 0.0:
        result.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(13 * impact))))
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


def _caption(draw: ImageDraw.ImageDraw, text: str, progress: float, punch: bool = False, y0: int = H - 405) -> None:
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


def _speech_pulse(progress: float, performance: dict | None) -> float:
    if not performance:
        return 0.0
    activity = performance.get("activity") or []
    if not activity:
        return 0.0
    p = max(0.0, min(1.0, progress))
    idx = min(len(activity) - 1, int(round(p * (len(activity) - 1))))
    a = float(activity[idx])
    # Suppress tiny room-tone fluctuations; keep articulation pulses visible.
    return max(0.0, min(1.0, (a - 0.18) / 0.62))


def _gaze_dart(progress: float, event_fraction: float, energy: float, direction: float = 1.0) -> float:
    """Lead an action with a compact glance, then release with a counter-glance."""
    p = max(0.0, min(1.0, progress))
    def envelope(center: float, width: float) -> float:
        distance = (p - center) / width
        return math.exp(-2.8 * distance * distance)

    return direction * 3.2 * energy * (
        envelope(event_fraction - 0.055, 0.045)
        - 0.55 * envelope(event_fraction + 0.105, 0.070)
    )


def _face_motion(progress: float, beat: str, performance: dict | None = None) -> tuple[float, float, float, float, float]:
    """Return face dx/dy, x/y scale and blink strength."""
    p = max(0.0, min(1.0, progress))
    energy = BEAT_PROFILES.get(beat, BEAT_PROFILES["setup"]).energy
    speech = _speech_pulse(p, performance)
    shared_state = kinematic_state(p, beat, performance)
    # A quick look toward the event leads the body reaction, then releases
    # with a smaller counter-glance. Keeping this tied to the director's event
    # timing makes the face read as attentive instead of riding the body sine.
    gaze_dx = 0.0
    if beat in {"setup", "escalation", "reaction"}:
        attention_direction = -1.0 if shared_state["attention_x"] < 0 else 1.0
        gaze_dx = _gaze_dart(p, shared_state["event_fraction"], energy, attention_direction)
    if beat == "escalation":
        snap = math.sin(math.pi * max(0.0, min(1.0, (p - 0.16) / 0.34)))
        dx, dy = 7.0 * snap * energy, -4.0 * snap * energy
        sx, sy = 1.0 + 0.010 * snap + 0.010 * speech, 1.0 + 0.030 * snap + 0.014 * speech
        blink = max(0.0, math.sin(math.pi * max(0.0, min(1.0, (p - 0.10) / 0.08))))
    elif beat == "reaction":
        snap = math.sin(math.pi * max(0.0, min(1.0, (p - 0.10) / 0.30)))
        dx, dy = -8.0 * snap * energy, 3.0 * snap * energy
        sx, sy = 1.0 + 0.016 * snap + 0.009 * speech, 1.0 + 0.038 * snap + 0.018 * speech
        blink = max(0.0, math.sin(math.pi * max(0.0, min(1.0, (p - 0.18) / 0.09))))
    elif beat == "punchline":
        sway = math.sin(math.pi * p * 0.8)
        dx, dy = 2.0 * sway, -1.0 * sway
        sx, sy = 1.0 + 0.004 * speech, 1.0 + 0.012 * sway + 0.010 * speech
        blink = 0.72 * max(0.0, math.sin(math.pi * max(0.0, min(1.0, (p - 0.25) / 0.08))))
    else:
        breathe = math.sin(math.tau * p)
        dx, dy = 1.5 * breathe * energy, -0.8 * breathe * energy
        sx, sy = 1.0 + 0.002 * speech, 1.0 + 0.008 * breathe * energy + 0.008 * speech
        blink = 0.65 * max(0.0, math.sin(math.pi * max(0.0, min(1.0, (p - 0.32) / 0.08))))
    dx += shared_state["face_target_dx"] + gaze_dx
    dy += shared_state["face_target_dy"]
    return dx, dy, sx, sy, blink


def _apply_face_performance(frame: Image.Image, rig: CharacterRig, progress: float, beat: str, motion_gain: float = 1.0, performance: dict | None = None) -> Image.Image:
    fx0, fy0, fx1, fy1 = rig.face_bbox
    face = rig.face_layer.crop((fx0, fy0, fx1, fy1)).convert("RGBA")
    dx, dy, sx, sy, blink = _face_motion(progress, beat, performance)
    dx *= motion_gain
    dy *= motion_gain
    sx = 1.0 + (sx - 1.0) * motion_gain
    sy = 1.0 + (sy - 1.0) * motion_gain
    face = face.resize(
        (max(2, int(face.width * sx)), max(2, int(face.height * sy))),
        Image.Resampling.BICUBIC,
    )

    # Briefly compress the LED expression vertically for a blink. We deliberately
    # do not invent eye/pupil pixels; the original generated face remains the art.
    if blink > 0.02:
        fd = ImageDraw.Draw(face, "RGBA")
        band_y = int(face.height * 0.43)
        band_h = max(4, int(face.height * 0.09 * blink))
        fd.rounded_rectangle(
            (int(face.width * 0.08), band_y - band_h, int(face.width * 0.92), band_y + band_h),
            radius=max(3, band_h),
            fill=(16, 27, 38, int(205 * blink)),
        )

    base = frame.copy()
    ox = int(fx0 + dx - (face.width - (fx1 - fx0)) / 2)
    oy = int(fy0 + dy - (face.height - (fy1 - fy0)) / 2)
    base.paste(face.convert("RGB"), (ox, oy), face.getchannel("A"))
    return base


def _apply_rig_performance(
    frame: Image.Image,
    rig: CharacterRig,
    progress: float,
    beat: str,
    motion_gain: float = 1.0,
    performance: dict | None = None,
) -> Image.Image:
    dx, dy, scale, rotation = head_motion(progress, beat, performance)
    root = kinematic_state(progress, beat, performance)
    speech = _speech_pulse(progress, performance)
    dx = (root["root_dx"] + dx + 1.8 * math.sin(progress * math.tau * 3.0) * speech) * motion_gain
    dy = (root["root_dy"] + dy - 1.5 * speech) * motion_gain
    scale = 1.0 + ((root["root_scale"] * scale) - 1.0) * motion_gain
    scale += 0.008 * speech * motion_gain
    rotation = (root["root_rot"] + rotation) * motion_gain
    cover = frame.filter(ImageFilter.GaussianBlur(11))
    base = Image.composite(cover, frame, rig.cover_mask).convert("RGB")
    rig_dir = Path(__file__).resolve().parents[2] / "artifacts" / "cartoon" / "pip_rig_v2"
    try:
        head_layer = load_action_layers(Path(rig.source), rig_dir)["head"]
    except (OSError, ValueError, KeyError):
        head_layer = rig.head_layer
    transformed = _transform_part(head_layer, dx, dy, scale, rotation)
    base.paste(transformed.convert("RGB"), (0, 0), transformed.getchannel("A"))
    return _apply_face_performance(base, rig, progress, beat, motion_gain=motion_gain, performance=performance)


def _apply_torso_layer(frame: Image.Image, layer: Image.Image, progress: float, beat: str, motion_gain: float = 1.0, performance: dict | None = None, source_name: str | None = None) -> Image.Image:
    """Move the body root with the same impulse used by head/arms/camera."""
    alpha = layer.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return frame
    x0, y0, x1, y1 = bbox
    crop = layer.crop((x0, y0, x1, y1))
    state = kinematic_state(progress, beat, performance)
    dx = state["root_dx"] * motion_gain
    dy = state["root_dy"] * motion_gain
    scale = 1.0 + (state["root_scale"] - 1.0) * motion_gain
    rotation = state["root_rot"] * motion_gain
    crop = crop.resize((max(2, int(crop.width * scale)), max(2, int(crop.height * scale))), Image.Resampling.BICUBIC)
    crop = crop.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=False)
    ox = int(x0 + dx - (crop.width - (x1 - x0)) / 2)
    oy = int(y0 + dy - (crop.height - (y1 - y0)) / 2)
    base = frame.copy()
    base.paste(crop.convert("RGB"), (ox, oy), crop.getchannel("A"))
    return base


def _apply_arm_layer(frame: Image.Image, layer: Image.Image, progress: float, beat: str, side: str, motion_gain: float = 1.0, performance: dict | None = None, source_name: str | None = None) -> Image.Image:
    alpha = layer.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return frame
    x0, y0, x1, y1 = bbox
    crop = layer.crop((x0, y0, x1, y1))
    dx, dy, scale, rotation = arm_motion(progress, beat, side, performance)
    dx *= motion_gain
    dy *= motion_gain
    scale = 1.0 + (scale - 1.0) * motion_gain
    rotation *= motion_gain
    crop = crop.resize((max(2, int(crop.width * scale)), max(2, int(crop.height * scale))), Image.Resampling.BICUBIC)
    crop = crop.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=False)
    pivot = _joint(source_name, "left_shoulder" if side == "left" else "right_shoulder") if source_name else None
    ox = int(x0 + dx - (crop.width - (x1 - x0)) / 2) if pivot is None else int(pivot[0] + dx - crop.width / 2)
    oy = int(y0 + dy - (crop.height - (y1 - y0)) / 2) if pivot is None else int(pivot[1] + dy - crop.height / 2)
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
    render_tuning: dict | None = None,
    voice_performance: dict | None = None,
) -> Image.Image:
    tuning = render_tuning or {}
    motion_gain = float(tuning.get("motion_gain", 1.0))
    camera_gain = float(tuning.get("camera_gain", 1.0))
    shake_gain = float(tuning.get("shake_gain", 1.0))
    caption_y = int(tuning.get("caption_y", H - 405))
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
            rig_dir = Path(__file__).resolve().parents[2] / "artifacts" / "cartoon" / "pip_rig_v2"
            layers = load_action_layers(image, rig_dir)
            base = layers["body_plate"].convert("RGB")
            base = _apply_torso_layer(base, layers["torso"], p, beat, motion_gain=motion_gain, performance=voice_performance, source_name=image.name)
            base = _apply_arm_layer(base, layers["left_arm"], p, beat, "left", motion_gain=motion_gain, performance=voice_performance, source_name=image.name)
            base = _apply_arm_layer(base, layers["right_arm"], p, beat, "right", motion_gain=motion_gain, performance=voice_performance, source_name=image.name)
        except (OSError, ValueError):
            pass
        base = _apply_rig_performance(base, rig, p, beat, motion_gain=motion_gain, performance=voice_performance)
    eased = _ease_in_out(p)
    energy = profile.energy * motion_gain

    # First 20% is anticipation, middle is action, last 25% settles.
    anticipation = min(1.0, p / 0.20) if p < 0.20 else 1.0
    settle = max(0.0, min(1.0, (p - 0.72) / 0.28))
    spring = _spring(max(0.0, (p - 0.12) / 0.88))

    # Keep the delivery camera coupled to the same delayed root motion used
    # when character layers are projected into camera space. Without this
    # follower, layered poses and the final frame use different camera paths.
    camera_state = kinematic_state(max(0.0, p - 0.045), beat, voice_performance)
    zoom_start = 1.0 + (profile.zoom_start - 1.0) * camera_gain
    zoom_delta = (profile.zoom_end - profile.zoom_start) * camera_gain
    zoom = zoom_start + zoom_delta * eased
    zoom += 0.012 * math.sin(p * math.tau) * energy
    zoom += camera_state["camera_zoom"] * camera_gain
    if beat == "reaction":
        zoom += 0.022 * math.sin((p * 1.2) * math.tau) * (1.0 - settle) * camera_gain

    crop_w = max(2, int(base.width / zoom))
    crop_h = max(2, int(base.height / zoom))
    max_x = max(0, base.width - crop_w)
    max_y = max(0, base.height - crop_h)

    drift_x = profile.pan_x * max_x * math.sin(math.pi * eased)
    drift_y = profile.pan_y * max_y * math.sin(math.pi * eased)
    focus_x = camera_state["camera_focus_x"] * max_x * 0.10 * camera_gain
    focus_y = camera_state["camera_focus_y"] * max_y * 0.10 * camera_gain
    x = int(max_x * 0.5 + drift_x - camera_state["camera_dx"] * 2.2 * camera_gain + focus_x)
    y = int(max_y * 0.5 + drift_y - camera_state["camera_dy"] * 2.2 * camera_gain + focus_y)
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))

    frame = base.crop((x, y, x + crop_w, y + crop_h)).resize((W, H), Image.Resampling.LANCZOS)
    angle = profile.rotation * math.sin(math.pi * eased) * (1.0 - 0.55 * settle)
    angle += camera_state["camera_rot"] * camera_gain
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
    shake_window = _impact_shake_envelope(p, camera_state["event_fraction"], beat)
    shake = profile.shake * shake_window * shake_gain
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
        peak = float((voice_performance or {}).get("peak_fraction", 0.46))
        start = max(0.18, peak - 0.10)
        flash = max(0.0, math.sin(math.pi * max(0.0, min(1.0, (p - start) / 0.12))))
    if flash > 0.0:
        veil = Image.new("RGBA", (W, H), (255, 255, 255, int(45 * flash)))
        rgba.alpha_composite(veil)
        _impact_graphics(rgba, flash)

    draw = ImageDraw.Draw(rgba, "RGBA")
    reveal = (p - 0.04) / max(0.15, 0.72 + profile.pause)
    if beat == "punchline":
        reveal = min(1.0, max(0.0, (p - profile.pause) / max(0.15, 1.0 - profile.pause)))
    punch = beat == "punchline" and p > 0.58
    _caption(draw, caption, reveal, punch=punch, y0=caption_y)
    _disclosure(draw)

    return rgba.convert("RGB")


def _audio_for_shots(
    work: Path,
    voice_lines: list[str],
    durations: list[float],
    sfx_kinds: list[str],
    ffmpeg: str,
    sfx_peak_fractions: list[float] | None = None,
    voice_performance: list[dict] | None = None,
) -> Path | None:
    from app.providers.cartoon_media import _audio_for_shots as build_audio
    return build_audio(
        work,
        voice_lines,
        durations,
        sfx_kinds,
        ffmpeg,
        sfx_peak_fractions,
        [0.04, 0.03, 0.03, 0.06],
        voice_performance=voice_performance,
    )


def _performance_durations(
    voice_performance: list[dict],
    requested: list[float],
    beats: list[str],
) -> list[float]:
    """Retime shots around natural speech instead of forcing speech into long fixed cards."""
    tails = {
        "setup": 0.58,       # thought / setup breath
        "escalation": 0.62, # visual reveal + shock read
        "reaction": 0.62,   # body reaction after the line
        "punchline": 0.86,  # deadpan hold after delivery
    }
    minimums = {
        "setup": 1.55,
        "escalation": 1.20,
        "reaction": 1.50,
        "punchline": 2.00,
    }
    result = []
    for i, beat in enumerate(beats):
        perf = voice_performance[i] if i < len(voice_performance) else {}
        spoken = float(perf.get("duration", 0.0) or 0.0)
        requested_duration = float(requested[i])
        target = max(minimums.get(beat, 1.35), spoken + tails.get(beat, 0.55))
        # Never collapse a pre-authored beat just because TTS is extremely short.
        result.append(round(min(requested_duration, target) if target < requested_duration else target, 3))
    return result


def render_cartoon_motion(
    images: list[Path],
    captions: list[str],
    durations: list[float],
    beats: list[str],
    output: Path,
    voice_lines: list[str] | None = None,
    sfx_kinds: list[str] | None = None,
    fps: int = 24,
    render_tuning: dict | None = None,
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

    tuning = {
        "motion_gain": float((render_tuning or {}).get("motion_gain", 1.0)),
        "camera_gain": float((render_tuning or {}).get("camera_gain", 1.0)),
        "shake_gain": float((render_tuning or {}).get("shake_gain", 1.0)),
        "transition_seconds": float((render_tuning or {}).get("transition_seconds", 0.20)),
        "caption_y": int((render_tuning or {}).get("caption_y", 1515)),
        "caption_zone": list((render_tuning or {}).get("caption_zone", [40, 1480, 1040, 1885])),
        "jpeg_quality": int((render_tuning or {}).get("jpeg_quality", 91)),
        "crf": int((render_tuning or {}).get("crf", 23)),
        "repair_attempt": int((render_tuning or {}).get("repair_attempt", 0)),
        "repair_reasons": list((render_tuning or {}).get("repair_reasons", [])),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f"{output.stem}_frames"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    # Voice performance is generated before frames so visual acting can lock to
    # real speech onset/energy/peak rather than an assumed beat fraction.
    voice_performance = prepare_voice_performance(work, voice_lines, beats, durations)
    effective_durations = _performance_durations(voice_performance, durations, beats)

    # Render at the requested delivery cadence directly. The old implementation
    # rendered 12fps and used minterpolate to synthesize 24fps, which could create
    # rubbery motion around captions and pose changes. Deterministic 24fps frames
    # are more faithful and avoid an expensive interpolation pass.
    source_fps = fps
    frame_index = 0
    frame_map = []
    previous_shot_frame: Image.Image | None = None
    previous_shot_rig: CharacterRig | None = None
    previous_shot_image: Path | None = None
    previous_shot_beat: str | None = None
    # Five frames at 24fps (~0.21s) is long enough to see the pose travel while
    # still reading as a snappy comedic cut.
    transition_frames = max(2, round(tuning["transition_seconds"] * source_fps))
    transition_directions: list[int] = []
    character_bboxes: list[list[int] | None] = []
    character_frame_bboxes: list[list[int] | None] = []
    for shot, (image, caption, duration, beat) in enumerate(zip(images, captions, effective_durations, beats)):
        count = max(2, round(duration * source_fps))
        frame_map.append({"shot": shot, "beat": beat, "frames": count, "duration": duration})
        for i in range(count):
            progress = i / max(1, count - 1)
            try:
                rig = get_rig(str(image))
            except ValueError:
                rig = None
            if rig is not None and i == 0:
                # Use the performance/head bbox for subtitle safety. The lower
                # body can sit behind captions without obscuring the dialogue.
                character_bboxes.append(list(rig.head_bbox))
            elif i == 0:
                character_bboxes.append(None)
            if rig is not None:
                character_frame_bboxes.append(list(rig.head_bbox))
            else:
                character_frame_bboxes.append(None)
            frame = motion_frame(
                image, caption, progress, shot, duration, beat, rig=rig,
                render_tuning=tuning, voice_performance=voice_performance[shot],
            )
            if (
                previous_shot_frame is not None
                and previous_shot_image is not None
                and previous_shot_rig is not None
                and previous_shot_beat is not None
                and rig is not None
                and i < transition_frames
            ):
                if i == 0:
                    transition_directions.append(_transition_direction(previous_shot_rig, rig))
                transition_p = i / max(1, transition_frames - 1)
                direction = transition_directions[-1] if transition_directions else 1
                try:
                    frame = _pose_transition_frame(
                        previous_shot_frame,
                        frame,
                        previous_shot_image,
                        image,
                        previous_shot_rig,
                        rig,
                        previous_shot_beat,
                        beat,
                        transition_p,
                        direction,
                    )
                except (OSError, ValueError, KeyError):
                    frame = _transition_frame(previous_shot_frame, frame, transition_p, direction)
            elif previous_shot_frame is not None and i < transition_frames:
                transition_p = i / max(1, transition_frames - 1)
                direction = transition_directions[-1] if transition_directions else 1
                frame = _transition_frame(previous_shot_frame, frame, transition_p, direction)
            frame.save(work / f"frame_{frame_index:05d}.jpg", quality=tuning["jpeg_quality"], optimize=True)
            frame_index += 1
        previous_shot_frame = frame
        previous_shot_rig = rig
        previous_shot_image = image
        previous_shot_beat = beat

    video_silent = output.with_suffix(".silent.mp4")
    subprocess.run(
        [
            ffmpeg, "-y",
            "-framerate", str(source_fps),
            "-i", str(work / "frame_%05d.jpg"),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(tuning["crf"]), "-pix_fmt", "yuv420p",
            "-vf", "scale=1080:1920:flags=lanczos,format=yuv420p",
            "-r", str(source_fps),
            str(video_silent),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    video_for_mux = video_silent

    # These are the centers of the visual performance peaks, not arbitrary
    # audio offsets. SFX are generated around these fractions and therefore
    # share the same beat clock as the character motion.
    peak_by_beat = {
        "setup": 0.18,
        "escalation": 0.46,
        "reaction": 0.40,
        "punchline": 0.72,
    }
    measured_peaks = [
        max(0.10, min(0.90, float(row.get("performance_peak_fraction", row.get("peak_fraction", peak_by_beat.get(beat, 0.40))))))
        for row, beat in zip(voice_performance, beats)
    ]
    audio = _audio_for_shots(
        work, voice_lines, effective_durations, sfx_kinds, ffmpeg,
        measured_peaks,
        voice_performance=voice_performance,
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
                "engine": "cartoon-motion-3.6",
                "rig_mode": "flattened-art-performance-proxy",
                "performance_engine": "character-performance-v1.6",
                "attention_model": "targeted-gaze-and-camera-follow",
                "attachment_model": "clean-semantic-limbs-center-root",
                "joint_pivot_model": "neck-and-shoulder-anchors",
                "shared_kinematic_clock": True,
                "secondary_motion": ["root", "head", "arms", "backpack", "camera"],
                "fps": fps,
                "source_fps": source_fps,
                "interpolation": "none",
                "width": 1080,
                "height": 1920,
                "character_bboxes": character_bboxes,
                "character_frame_bboxes": character_frame_bboxes,
                "transition_seconds": round(transition_frames / source_fps, 3),
                "caption_y": tuning["caption_y"],
                "caption_zone": tuning["caption_zone"],
                "repair": {
                    "attempt": tuning["repair_attempt"],
                    "reasons": tuning["repair_reasons"],
                    "strategy": "targeted-single-rerender" if tuning["repair_attempt"] else "initial-render",
                },
                "performance": {
                    "action_transitions": True,
                    "pose_aware_transition_direction": True,
                    "character_micro_motion": True,
                    "anticipation_recoil_settle": True,
                    "punchline_hold": True,
                    "pose_morph_bridge": True,
                    "pose_interpolation_mode": "staggered-geometric-pose",
                    "audio_performance_clock": True,
                    "per_shot_voice": True,
                    "sfx_peak_sync": True,
                },
                "audio_sync": {
                    "sfx_peak_fractions": measured_peaks,
                    "voice_start_fractions": [0.0 for _ in voice_performance],
                    "sfx_mode": "short_peak_accent",
                    "voice_engine": "local-mac-say-character-performance-v1",
                    "voice_performance": voice_performance,
                    "effective_shot_durations": effective_durations,
                },
                "frames": frame_index,
                "shots": frame_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output



def rerender_cartoon_shots(
    source_video: Path,
    images: list[Path],
    captions: list[str],
    durations: list[float],
    beats: list[str],
    output: Path,
    target_shots: list[int],
    render_tuning: dict | None = None,
    fps: int = 24,
) -> Path:
    """Surgically replace failed shots and dependent outgoing transitions.

    Untargeted frames are copied from the source video byte-for-byte at the
    decoded-frame level. Audio is copied from the source container without a
    second mix or TTS pass.
    """
    if not target_shots:
        raise ValueError("target_shots must be non-empty")
    if not source_video.exists():
        raise FileNotFoundError(source_video)
    if not (len(images) == len(captions) == len(durations) == len(beats)):
        raise ValueError("images, captions, durations and beats must match")
    if any(index < 0 or index >= len(images) for index in target_shots):
        raise IndexError("target shot index out of range")
    if fps < 12 or fps > 30:
        raise ValueError("fps must be between 12 and 30")

    from app.agents.cartoon_motion_qa import CartoonMotionQA

    metadata_path = source_video.with_suffix(".motion.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Motion metadata missing: {metadata_path}")
    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shot_ranges = CartoonMotionQA._shot_frame_ranges(source_metadata)
    source_voice_performance = list(source_metadata.get("audio_sync", {}).get("voice_performance", []))
    effective_durations = list(source_metadata.get("audio_sync", {}).get("effective_shot_durations", durations))
    if len(shot_ranges) != len(images):
        raise ValueError("Motion metadata shot count does not match render inputs")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")

    tuning = {
        "motion_gain": float((render_tuning or {}).get("motion_gain", 1.0)),
        "camera_gain": float((render_tuning or {}).get("camera_gain", 1.0)),
        "shake_gain": float((render_tuning or {}).get("shake_gain", 1.0)),
        "transition_seconds": float((render_tuning or {}).get("transition_seconds", 0.20)),
        "caption_y": int((render_tuning or {}).get("caption_y", 1515)),
        "caption_zone": list((render_tuning or {}).get("caption_zone", [40, 1480, 1040, 1885])),
        "jpeg_quality": int((render_tuning or {}).get("jpeg_quality", 91)),
        "crf": int((render_tuning or {}).get("crf", 23)),
        "repair_attempt": int((render_tuning or {}).get("repair_attempt", 0)),
        "repair_reasons": list((render_tuning or {}).get("repair_reasons", [])),
    }

    full_targets = sorted(set(int(index) for index in target_shots))
    bridge_targets = sorted(
        {
            index + 1
            for index in full_targets
            if index + 1 < len(images) and index + 1 not in full_targets
        }
    )
    replace_shots = set(full_targets) | set(bridge_targets)
    transition_frames = max(2, round(tuning["transition_seconds"] * fps))

    with tempfile.TemporaryDirectory(prefix="nova-targeted-repair-") as tmp_name:
        work = Path(tmp_name)
        original_dir = work / "original"
        output_dir = work / "output"
        original_dir.mkdir()
        output_dir.mkdir()

        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(source_video),
                "-vsync", "0", "-q:v", "2",
                str(original_dir / "frame_%05d.jpg"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        original_frames = sorted(original_dir.glob("frame_*.jpg"))
        expected_frames = sum(end - start for start, end, _ in shot_ranges)
        actual_frames = len(original_frames)
        if actual_frames != expected_frames:
            # CFR encoding can round the final frame boundary down by one frame.
            # Preserve all earlier shot offsets and reconcile only the final range.
            if abs(actual_frames - expected_frames) > 1 or not shot_ranges:
                raise RuntimeError(
                    f"Decoded frame count mismatch: expected {expected_frames}, got {actual_frames}"
                )
            start, _, beat = shot_ranges[-1]
            if actual_frames <= start:
                raise RuntimeError("Encoded video ended before the final shot")
            shot_ranges[-1] = (start, actual_frames, beat)
        for frame_path in original_frames:
            shutil.copy2(frame_path, output_dir / frame_path.name)

        def save_frame(frame: Image.Image, frame_index: int) -> None:
            frame.save(
                output_dir / f"frame_{frame_index + 1:05d}.jpg",
                quality=tuning["jpeg_quality"],
                optimize=True,
            )

        def frame_path(frame_index: int) -> Path:
            return output_dir / f"frame_{frame_index + 1:05d}.jpg"

        def final_frame_for_shot(shot_index: int) -> Image.Image:
            _, end, _ = shot_ranges[shot_index]
            with Image.open(frame_path(end - 1)) as image:
                return image.convert("RGB").copy()

        def make_transition_frame(
            shot_index: int,
            local_index: int,
            current_frame: Image.Image,
        ) -> Image.Image:
            if shot_index == 0 or local_index >= transition_frames:
                return current_frame
            previous = shot_index - 1
            previous_frame = final_frame_for_shot(previous)
            previous_image = images[previous]
            current_image = images[shot_index]
            previous_rig = get_rig(str(previous_image))
            current_rig = get_rig(str(current_image))
            direction = _transition_direction(previous_rig, current_rig)
            transition_p = local_index / max(1, transition_frames - 1)
            return _pose_transition_frame(
                previous_frame,
                current_frame,
                previous_image,
                current_image,
                previous_rig,
                current_rig,
                beats[previous],
                beats[shot_index],
                transition_p,
                direction,
            )

        # Full replacement for failed shots. Processing in index order means a
        # previous repaired shot's final frame is available to the next bridge.
        for shot_index in full_targets:
            start, end, beat = shot_ranges[shot_index]
            try:
                rig = get_rig(str(images[shot_index]))
            except ValueError:
                rig = None
            for local_index, frame_index in enumerate(range(start, end)):
                progress = local_index / max(1, (end - start) - 1)
                frame = motion_frame(
                    images[shot_index],
                    captions[shot_index],
                    progress,
                    shot_index,
                    durations[shot_index],
                    beat,
                    rig=rig,
                    render_tuning=tuning,
                    voice_performance=source_voice_performance[shot_index] if shot_index < len(source_voice_performance) else None,
                )
                frame = make_transition_frame(shot_index, local_index, frame)
                save_frame(frame, frame_index)

        # Only the transition prefix is regenerated for shots immediately after
        # a repaired shot; their remaining frames stay untouched.
        for shot_index in bridge_targets:
            start, end, beat = shot_ranges[shot_index]
            try:
                rig = get_rig(str(images[shot_index]))
            except ValueError:
                rig = None
            prefix = min(transition_frames, end - start)
            for local_index, frame_index in enumerate(range(start, start + prefix)):
                progress = local_index / max(1, (end - start) - 1)
                frame = motion_frame(
                    images[shot_index],
                    captions[shot_index],
                    progress,
                    shot_index,
                    durations[shot_index],
                    beat,
                    rig=rig,
                    render_tuning=tuning,
                    voice_performance=source_voice_performance[shot_index] if shot_index < len(source_voice_performance) else None,
                )
                frame = make_transition_frame(shot_index, local_index, frame)
                save_frame(frame, frame_index)

        silent = work / "silent.mp4"
        subprocess.run(
            [
                ffmpeg, "-y",
                "-framerate", str(fps),
                "-i", str(output_dir / "frame_%05d.jpg"),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", str(tuning["crf"]),
                "-pix_fmt", "yuv420p",
                "-vf", "scale=1080:1920:flags=lanczos,format=yuv420p",
                "-r", str(fps),
                str(silent),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        muxed = work / "muxed.mp4"
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise RuntimeError("ffprobe is required")
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index",
             "-of", "csv=p=0", str(source_video)],
            capture_output=True,
            text=True,
            check=True,
        )
        has_audio = bool(probe.stdout.strip())
        if has_audio:
            subprocess.run(
                [
                    ffmpeg, "-y",
                    "-i", str(silent),
                    "-i", str(source_video),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "copy",
                    "-shortest", str(muxed),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            shutil.copy2(silent, muxed)

        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(muxed, output)

    new_metadata = dict(source_metadata)
    new_metadata["engine"] = "cartoon-motion-3.6"
    new_metadata["transition_seconds"] = round(transition_frames / fps, 3)
    new_metadata["caption_y"] = tuning["caption_y"]
    new_metadata["caption_zone"] = tuning["caption_zone"]
    new_metadata["performance_engine"] = "character-performance-v1"
    repair = dict(new_metadata.get("repair", {}))
    repair.update(
        {
            "attempt": tuning["repair_attempt"],
            "reasons": tuning["repair_reasons"],
            "strategy": "targeted-shot-rerender",
            "target_shots": full_targets,
            "bridge_shots": bridge_targets,
            "source_video": str(source_video),
            "audio_preserved": True,
            "replaced_frame_count": sum(
                shot_ranges[index][1] - shot_ranges[index][0]
                for index in full_targets
            ) + sum(
                min(
                    transition_frames,
                    shot_ranges[index][1] - shot_ranges[index][0],
                )
                for index in bridge_targets
            ),
        }
    )
    new_metadata["repair"] = repair
    if effective_durations:
        new_metadata.setdefault("audio_sync", {})["effective_shot_durations"] = effective_durations

    new_metadata["shots"] = [
        {**row, "repaired": index in full_targets}
        for index, row in enumerate(new_metadata.get("shots", []))
    ]
    output.with_suffix(".motion.json").write_text(
        json.dumps(new_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output
