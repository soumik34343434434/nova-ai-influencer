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

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.providers.cartoon_decomposer import load_action_layers
from app.providers.cartoon_performance import arm_motion, head_motion, kinematic_state
from app.providers.cartoon_rig import CharacterRig, get_rig
from app.providers.cartoon_voice import prepare_voice_performance

W, H = 1080, 1920
MOTION_ENGINE = "cartoon-motion-3.7"
# Every frame is rendered on a canvas this much larger than the frame and then
# center-cropped, so rotation, squash and shake can never expose the edge.
OVERSCAN = 1.10
PAD_X = round(W * (OVERSCAN - 1.0) / 2.0)
PAD_Y = round(H * (OVERSCAN - 1.0) / 2.0)
# Platform UI covers the outer frame; all text must stay inside this box
# (x0, y0, x1, y1) as fractions of the 1080x1920 frame.
SAFE_ZONE = (0.06, 0.12, 0.84, 0.76)
REACTION_MOTION_GAIN = 2.0  # extra movement on reaction beats (close-up shots read as static otherwise)
IDLE_BREATH_PX = 12.0  # whole-character rise/fall (px) so holds keep changing frame to frame
IDLE_BREATH_HZ = 2.0


def _idle_bob(progress: float, shot_seconds: float) -> float:
    return IDLE_BREATH_PX * math.sin(math.tau * IDLE_BREATH_HZ * progress * shot_seconds)
CAPTION_TOP = 200  # top band above the characters' heads and clear of platform UI
CAPTION_ZONE = (90, 170, 990, 500)  # reserved text area that must not touch a character (close-up heads start near y=520)
CAPTION_MAX_WIDTH = 720  # centred on W/2, so the block stays inside SAFE_ZONE
CAPTION_SIZES = (64, 58, 52, 46)
DISCLOSURE_TOP = int(H * SAFE_ZONE[1]) + 12
HOOK_FULL_CAPTION_SECONDS = 0.8  # the hook caption is fully readable within this
HOOK_SNAP_ZOOM = 0.10  # punch-in that settles over the first hook beat
TRANSITION_OVERSCAN = 1.06
SCENE_SETTINGS = ("kitchen", "bedroom", "classroom", "park", "bakery", "living_room", "garden", "studio")
NOTO_EMOJI_FONT = Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf")
PROP_SIZE = 210
ENCLOSED_BACKDROP_MIN_AREA = 2500  # enclosed backdrop pockets; bright eyes are excluded by luminance instead
DISCLOSURE_TEXT = "AI-generated • fictional cartoon"


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


def _cover_window(frame: Image.Image, scale: float, dx: float, dy: float) -> Image.Image:
    """Return a W x H window of ``frame`` scaled by ``scale`` and shifted by (dx, dy).

    The scaled image always covers the window, so no transparent or black
    margin can appear. Shifts are clamped to the scaled margin.
    """
    sw, sh = max(W, round(W * scale)), max(H, round(H * scale))
    scaled = frame.convert("RGB").resize((sw, sh), Image.Resampling.BICUBIC)
    mx, my = (sw - W) // 2, (sh - H) // 2
    left = mx - round(max(-mx, min(mx, dx)))
    top = my - round(max(-my, min(my, dy)))
    return scaled.crop((left, top, left + W, top + H))


def _transition_frame(previous: Image.Image, current: Image.Image, progress: float, direction: int) -> Image.Image:
    """Snap between poses with directional overlap and a tiny settle overshoot."""
    t = max(0.0, min(1.0, progress))
    # Fast arrival: the new pose should appear decisively, not dissolve.
    arrival = _ease_out(t)
    # Both sides are cropped from an overscanned copy, so a sliding or
    # scaling frame never exposes transparent or black edges.
    out_canvas = _cover_window(previous, 1.0 + (TRANSITION_OVERSCAN - 1.0) * arrival, direction * 28 * arrival, 0.0).convert("RGBA")
    inc_canvas = _cover_window(current, 1.0 + (TRANSITION_OVERSCAN - 1.0) * (1.0 - arrival), -direction * 54 * (1.0 - arrival), 0.0).convert("RGBA")

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
    x = max(0, min(int(max_x * 0.5 + drift_x - follow_x * 2.2), max_x))
    y = max(0, min(int(max_y * 0.5 + drift_y - follow_y * 2.2), max_y))
    angle = (profile.rotation * math.sin(math.pi * eased) * (1.0 - 0.55 * settle)) + camera_state["camera_rot"]
    zoom += camera_state["camera_zoom"]
    sx = 1.0 + 0.010 * math.sin(math.pi * spring) * energy
    sy = 1.0 - 0.006 * math.sin(math.pi * spring) * energy
    if beat == "escalation":
        sx += 0.014 * max(0.0, math.sin(p * math.pi * 2.0))
        sy -= 0.010 * max(0.0, math.sin(p * math.pi * 2.0))
    return x, y, crop_w, crop_h, angle, sx, sy


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


def _overscan_source(image: Image.Image, transparent: bool = False) -> Image.Image:
    """Pad a W x H source by the overscan margin.

    RGB plates are padded by replicating their edges; RGBA parts are padded with
    transparency so they never gain phantom pixels.
    """
    arr = np.asarray(image.convert("RGBA" if transparent else "RGB"))
    pad = ((PAD_Y, PAD_Y), (PAD_X, PAD_X), (0, 0))
    padded = np.pad(arr, pad, mode="constant") if transparent else np.pad(arr, pad, mode="edge")
    return Image.fromarray(padded)


def _camera_canvas(
    padded: Image.Image,
    x: int,
    y: int,
    crop_w: int,
    crop_h: int,
    angle: float,
    sx: float,
    sy: float,
    shake: tuple[int, int] = (0, 0),
) -> Image.Image:
    """Render the camera move on an overscanned canvas and center-crop to W x H.

    ``(x, y, crop_w, crop_h)`` is the visible window in source coordinates. The
    canvas covers OVERSCAN times that window at the same scale, so rotation,
    squash and shake only ever move pixels that the final crop discards.
    """
    cw, ch = crop_w * OVERSCAN, crop_h * OVERSCAN
    left = x + crop_w / 2.0 - cw / 2.0 + PAD_X
    top = y + crop_h / 2.0 - ch / 2.0 + PAD_Y
    canvas_w, canvas_h = round(W * OVERSCAN), round(H * OVERSCAN)
    frame = padded.crop((round(left), round(top), round(left + cw), round(top + ch)))
    frame = frame.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
    frame = frame.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    frame = frame.resize((round(canvas_w * sx), round(canvas_h * sy)), Image.Resampling.BICUBIC)
    dx, dy = shake
    if dx or dy:
        frame = frame.transform(frame.size, Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy), resample=Image.Resampling.BICUBIC)
    left = (frame.width - W) // 2
    top = (frame.height - H) // 2
    return frame.crop((left, top, left + W, top + H))


def _project_part_to_camera(layer: Image.Image, progress: float, beat: str, voice_performance: dict | None = None) -> Image.Image:
    """Apply the same camera transform used for the finished motion frame."""
    x, y, crop_w, crop_h, angle, sx, sy = _camera_geometry(progress, beat, voice_performance)
    return _camera_canvas(_overscan_source(layer, transparent=True), x, y, crop_w, crop_h, angle, sx, sy)


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


def _visible_word_count(text: str, progress: float) -> int:
    """Number of caption words revealed at ``progress``; never zero for a non-empty line."""
    total = len(text.split())
    if not total:
        return 0
    return max(1, min(total, math.ceil(total * max(0.0, min(1.0, progress)))))


def _caption_progress(progress: float, beat: str, shot_index: int, duration: float) -> float:
    """Map shot progress to caption reveal progress.

    The reveal is computed from the unclamped shot progress: camera and pose
    freeze at the punchline, but the joke's last word must still arrive.
    """
    p = max(0.0, min(1.0, progress))
    if beat == "punchline":
        pause = BEAT_PROFILES["punchline"].pause
        return max(0.0, min(1.0, (p - pause) / max(0.15, 1.0 - pause)))
    if shot_index == 0:
        # The hook is read first: a third of it is up at frame 0 and the rest
        # lands within HOOK_FULL_CAPTION_SECONDS.
        return max(0.0, min(1.0, 0.35 + 0.65 * (p * max(0.0, duration)) / HOOK_FULL_CAPTION_SECONDS))
    return max(0.0, min(1.0, (p - 0.04) / 0.72))


def _caption_layout(draw: ImageDraw.ImageDraw, text: str) -> tuple[ImageFont.ImageFont, int, list[str]]:
    """Pick the largest caption size that wraps the whole line into at most four lines."""
    words = text.split()
    font, size, lines = None, CAPTION_SIZES[-1], []
    for size in CAPTION_SIZES:
        font = _font(size)
        lines = []
        current = ""
        for word in words:
            candidate = (current + " " + word).strip()
            if draw.textbbox((0, 0), candidate, font=font, stroke_width=2)[2] <= CAPTION_MAX_WIDTH:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        if len(lines) <= 3:
            break
    return font, size, lines


def _draw_caption(
    draw: ImageDraw.ImageDraw,
    text: str,
    shown_words: int,
    punch: bool,
    y0: int,
) -> tuple[int, int, int, int] | None:
    """Draw the first ``shown_words`` of ``text``, centred in the safe zone.

    The font size is fixed by the full line, so the block does not jump while
    words arrive. Returns the drawn pixel box, or None when nothing is shown.
    """
    words = text.split()
    if not words or shown_words <= 0:
        return None
    shown = words[:shown_words]
    if punch:
        shown[-1] = shown[-1].upper()
    font, size, _ = _caption_layout(draw, text)
    lines: list[str] = []
    current = ""
    for word in shown:
        candidate = (current + " " + word).strip()
        if draw.textbbox((0, 0), candidate, font=font, stroke_width=2)[2] <= CAPTION_MAX_WIDTH:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    stroke = max(3, round(size / 8))
    line_h = round(size * 1.16)
    box: list[int] | None = None
    for i, line in enumerate(lines):
        bb = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
        tw = bb[2] - bb[0]
        x = (W - tw) / 2
        y = y0 + i * line_h
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0, 235))
        line_box = [round(x) + bb[0], round(y) + bb[1], round(x) + bb[2], round(y) + bb[3]]
        if box is None:
            box = line_box
        else:
            box = [min(box[0], line_box[0]), min(box[1], line_box[1]), max(box[2], line_box[2]), max(box[3], line_box[3])]
    return tuple(box) if box else None


def _draw_disclosure(draw: ImageDraw.ImageDraw) -> tuple[int, int, int, int]:
    """Draw the AI disclosure centred at the top of the safe zone; returns its box."""
    text = DISCLOSURE_TEXT
    font = _font(28)
    bb = draw.textbbox((0, 0), text, font=font)
    width = bb[2] - bb[0]
    x0 = (W - width) // 2 - 16
    y0 = DISCLOSURE_TOP
    box = (x0, y0, x0 + width + 32, y0 + 46)
    draw.rounded_rectangle(box, radius=12, fill=(8, 10, 15, 180))
    draw.text((x0 + 16, y0 + 7), text, font=font, fill=(245, 245, 245, 235))
    return box


def _union_box(a: list[int] | None, b: tuple[int, int, int, int] | None) -> list[int] | None:
    if b is None:
        return a
    if a is None:
        return list(b)
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _fraction_box(box: list[int] | None) -> list[float] | None:
    if box is None:
        return None
    return [round(box[0] / W, 4), round(box[1] / H, 4), round(box[2] / W, 4), round(box[3] / H, 4)]


def _draw_shot_text(
    frame: Image.Image,
    caption: str,
    progress: float,
    beat: str,
    shot_index: int,
    duration: float,
    caption_y: int = CAPTION_TOP,
) -> tuple[Image.Image, dict]:
    """Burn the caption and disclosure onto a finished frame.

    Runs after transitions, so text is never cut by a wipe. Returns the frame
    and what was drawn, for the motion metadata.
    """
    rgba = frame.convert("RGBA")
    draw = ImageDraw.Draw(rgba, "RGBA")
    total = len(caption.split())
    p = max(0.0, min(1.0, progress))
    shown = _visible_word_count(caption, _caption_progress(progress, beat, shot_index, duration))
    punch = beat == "punchline" and p > 0.58
    box = _draw_caption(draw, caption, shown, punch, caption_y)
    disclosure_box = _draw_disclosure(draw)
    return rgba.convert("RGB"), {
        "words_shown": shown if box else 0,
        "words_total": total,
        "box": box,
        "disclosure_box": disclosure_box,
    }


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


def _face_motion(progress: float, beat: str, performance: dict | None = None) -> tuple[float, float, float, float, float]:
    """Return face dx/dy, x/y scale and blink strength."""
    p = max(0.0, min(1.0, progress))
    energy = BEAT_PROFILES.get(beat, BEAT_PROFILES["setup"]).energy
    speech = _speech_pulse(p, performance)
    shared_state = kinematic_state(p, beat, performance)
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
    dx += shared_state["face_target_dx"]
    dy += shared_state["face_target_dy"]
    return dx, dy, sx, sy, blink


def _apply_face_performance(
    frame: Image.Image,
    rig: CharacterRig,
    progress: float,
    beat: str,
    motion_gain: float = 1.0,
    performance: dict | None = None,
    background_key: Image.Image | None = None,
) -> Image.Image:
    fx0, fy0, fx1, fy1 = rig.face_bbox
    face = rig.face_layer.crop((fx0, fy0, fx1, fy1)).convert("RGBA")
    if background_key is not None:
        # The face selection is a rounded rectangle, so it can take in backdrop
        # above the brows. Keep that backdrop off the scene.
        face = _without_background(face, background_key.crop((fx0, fy0, fx1, fy1)))
    dx, dy, sx, sy, blink = _face_motion(progress, beat, performance)
    dx *= motion_gain
    dy *= motion_gain
    sx = 1.0 + (sx - 1.0) * motion_gain
    sy = 1.0 + (sy - 1.0) * motion_gain
    face = face.resize(
        (max(2, int(face.width * sx)), max(2, int(face.height * sy))),
        Image.Resampling.BICUBIC,
    )

    # PIP's face is an LED screen, so a dark band reads as a blink. Organic
    # characters have painted eyes; a dark band across them reads as a censor
    # bar, and we have no closed-eye art for them, so they do not blink here.
    if blink > 0.02 and rig.character == "PIP":
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
    background_key: Image.Image | None = None,
    shot_seconds: float | None = None,
) -> Image.Image:
    dx, dy, scale, rotation = head_motion(progress, beat, performance)
    root = kinematic_state(progress, beat, performance)
    speech = _speech_pulse(progress, performance)
    dx += root["root_dx"] + 1.8 * math.sin(progress * math.tau * 3.0) * speech
    dy += root["root_dy"] - 1.5 * speech
    scale *= root["root_scale"]
    rotation += root["root_rot"]
    scale += 0.008 * speech
    root = kinematic_state(progress, beat, performance)
    dx = (root["root_dx"] + dx) * motion_gain
    dy = (root["root_dy"] + dy) * motion_gain
    scale = 1.0 + ((root["root_scale"] * scale) - 1.0) * motion_gain
    rotation = (root["root_rot"] + rotation) * motion_gain
    # Idle breath: a slow 1.4 px rise and fall of the head through every shot.
    # Holds were near-identical frames (failing the duplicate-frame check) and
    # this keeps them alive without faking a pose change.
    if shot_seconds:
        dy += _idle_bob(progress, shot_seconds)
    cover = frame.filter(ImageFilter.GaussianBlur(11))
    base = Image.composite(cover, frame, rig.cover_mask).convert("RGB")
    rig_dir = Path(__file__).resolve().parents[2] / "artifacts" / "cartoon" / "pip_rig_v2"
    try:
        head_layer = load_action_layers(Path(rig.source), rig_dir)["head"]
    except (OSError, ValueError, KeyError):
        head_layer = rig.head_layer
    if background_key is not None:
        # The head layer is cut from the original plate, so its elliptical
        # selection can include studio background. Keep it off a scene backdrop.
        head_layer = _without_background(head_layer, background_key)
    transformed = _transform_part(head_layer, dx, dy, scale, rotation)
    base.paste(transformed.convert("RGB"), (0, 0), transformed.getchannel("A"))
    return _apply_face_performance(base, rig, progress, beat, motion_gain=motion_gain, performance=performance, background_key=background_key)


def _apply_torso_layer(frame: Image.Image, layer: Image.Image, progress: float, beat: str, motion_gain: float = 1.0, performance: dict | None = None, source_name: str | None = None, idle_dy: float = 0.0) -> Image.Image:
    """Move the body root with the same impulse used by head/arms/camera."""
    alpha = layer.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return frame
    x0, y0, x1, y1 = bbox
    crop = layer.crop((x0, y0, x1, y1))
    state = kinematic_state(progress, beat, performance)
    dx = state["root_dx"] * motion_gain
    dy = state["root_dy"] * motion_gain + idle_dy
    scale = 1.0 + (state["root_scale"] - 1.0) * motion_gain
    rotation = state["root_rot"] * motion_gain
    crop = crop.resize((max(2, int(crop.width * scale)), max(2, int(crop.height * scale))), Image.Resampling.BICUBIC)
    crop = crop.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=False)
    ox = int(x0 + dx - (crop.width - (x1 - x0)) / 2)
    oy = int(y0 + dy - (crop.height - (y1 - y0)) / 2)
    base = frame.copy()
    base.paste(crop.convert("RGB"), (ox, oy), crop.getchannel("A"))
    return base


def _apply_arm_layer(frame: Image.Image, layer: Image.Image, progress: float, beat: str, side: str, motion_gain: float = 1.0, performance: dict | None = None, source_name: str | None = None, idle_dy: float = 0.0) -> Image.Image:
    alpha = layer.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return frame
    x0, y0, x1, y1 = bbox
    crop = layer.crop((x0, y0, x1, y1))
    dx, dy, scale, rotation = arm_motion(progress, beat, side, performance)
    dx *= motion_gain
    dy = dy * motion_gain + idle_dy
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


def _normalise_scene(scene: dict | None) -> dict | None:
    """Validate the story scene contract (setting, prop, prop_beats); None when absent."""
    if not isinstance(scene, dict) or not scene:
        return None
    setting = str(scene.get("setting") or "studio").strip().lower()
    if setting not in SCENE_SETTINGS:
        setting = "studio"
    prop = str(scene.get("prop") or "").strip()
    if not prop or len(prop) > 8:
        prop = ""
    beats = []
    for value in scene.get("prop_beats") or []:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            beats.append(int(value))
    return {
        "setting": setting,
        "prop": prop,
        "prop_label": str(scene.get("prop_label") or "")[:60],
        "prop_beats": sorted(set(beats)),
    }


@lru_cache(maxsize=8)
def _scene_backdrop(setting: str) -> Image.Image:
    """Stylised set for a setting, drawn once per process at frame size."""
    palettes = {
        "kitchen": ((240, 232, 214), (196, 160, 118)),
        "bedroom": ((226, 220, 240), (190, 160, 140)),
        "classroom": ((214, 234, 214), (170, 130, 90)),
        "park": ((176, 220, 246), (118, 186, 108)),
        "bakery": ((252, 238, 222), (214, 170, 120)),
        "living_room": ((236, 226, 210), (166, 128, 98)),
        "garden": ((200, 236, 250), (124, 196, 116)),
        "studio": ((240, 240, 244), (214, 216, 226)),
    }
    wall, floor = palettes.get(setting, palettes["studio"])
    img = Image.new("RGB", (W, H), wall)
    d = ImageDraw.Draw(img)
    horizon = int(H * 0.66)
    d.rectangle((0, horizon, W, H), fill=floor)
    if setting == "kitchen":
        for x in range(0, W, 120):
            d.line((x, 0, x, horizon), fill=(226, 214, 192), width=3)
        d.rectangle((0, 300, W, 520), fill=(120, 150, 170))
        d.rectangle((0, horizon - 60, W, horizon + 40), fill=(176, 132, 96))
    elif setting == "bedroom":
        d.rectangle((120, horizon - 260, 960, horizon + 60), fill=(118, 138, 200))
        d.rounded_rectangle((180, horizon - 300, 900, horizon - 200), radius=40, fill=(250, 250, 252))
        d.rectangle((0, 180, 240, 560), fill=(206, 196, 230))
    elif setting == "classroom":
        d.rectangle((140, 240, 940, 720), fill=(46, 74, 60), outline=(120, 90, 60), width=18)
        d.rectangle((0, horizon - 40, W, horizon + 30), fill=(150, 110, 80))
    elif setting == "park":
        for cx, cy, r in ((160, 760, 150), (930, 720, 170), (560, 640, 120)):
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(96, 170, 96))
        d.rectangle((120, 760, 150, horizon), fill=(120, 84, 56))
        d.rectangle((900, 720, 930, horizon), fill=(120, 84, 56))
    elif setting == "bakery":
        for i in range(0, W, 108):
            d.rectangle((i, 180, i + 54, 330), fill=(230, 96, 96) if (i // 108) % 2 else (250, 250, 250))
        d.rectangle((90, 900, 990, 930), fill=(170, 120, 80))
        d.rectangle((90, 760, 990, 790), fill=(170, 120, 80))
    elif setting == "living_room":
        d.rounded_rectangle((120, horizon - 230, 960, horizon + 40), radius=60, fill=(90, 110, 140))
        d.ellipse((880, 260, 980, 360), fill=(255, 226, 150))
    elif setting == "garden":
        for x in range(0, W, 90):
            d.ellipse((x - 60, horizon - 120, x + 60, horizon + 10), fill=(72, 150, 84))
        for x in range(60, W, 170):
            d.ellipse((x, 300, x + 26, 326), fill=(250, 170, 190))
    return img


@lru_cache(maxsize=64)
def _background_key(path: str) -> Image.Image | None:
    """Mask (255 = plain studio background) for an action image, or None.

    Only near-uniform backgrounds are keyed. A background colour is sampled
    from the frame border, and only pixels connected to the border count, so a
    character-coloured region enclosed by the character is never replaced.
    """
    cover = np.asarray(_load_cover(path).convert("RGB")).astype(np.int16)
    border = np.concatenate([
        cover[:24].reshape(-1, 3), cover[-24:].reshape(-1, 3),
        cover[:, :24].reshape(-1, 3), cover[:, -24:].reshape(-1, 3),
    ])
    # A studio backdrop is near-neutral and bright, with a soft vignette, so the
    # key is chroma + luminance rather than one exact colour.
    chroma = cover.max(axis=2) - cover.min(axis=2)
    luma = cover.mean(axis=2)
    border_chroma = border.max(axis=1) - border.min(axis=1)
    if float(np.median(border_chroma)) > 20.0:
        return None
    border_luma = float(np.median(border.mean(axis=1)))
    near = ((chroma <= 20) & (luma >= 140)).astype(np.uint8)
    count, labels = cv2.connectedComponents(near, connectivity=4)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    # Plain backdrop that a character encloses (for example a light patch behind
    # the head) is keyed too, but only when it is large and backdrop-coloured.
    # Small enclosed whites such as eyes are left alone.
    areas = np.bincount(labels.ravel(), minlength=count)
    lumas = np.bincount(labels.ravel(), weights=luma.ravel(), minlength=count) / np.maximum(areas, 1)
    enclosed_backdrop = (areas >= ENCLOSED_BACKDROP_MIN_AREA) & (np.abs(lumas - border_luma) <= 36)
    keep_labels = set(int(v) for v in edge_labels if v != 0) | set(int(v) for v in np.nonzero(enclosed_backdrop)[0] if v != 0)
    keep = np.isin(labels, np.array(sorted(keep_labels), dtype=labels.dtype)) if keep_labels else np.zeros(labels.shape, dtype=bool)
    coverage = float(keep.mean())
    if not 0.20 <= coverage <= 0.95:
        return None
    mask = cv2.GaussianBlur(keep.astype(np.uint8) * 255, (0, 0), 1.2)
    return Image.fromarray(mask, "L")


def _backdrop_over(base: Image.Image, key: Image.Image | None, setting: str | None) -> Image.Image:
    if key is None or not setting:
        return base
    return Image.composite(_scene_backdrop(setting), base.convert("RGB"), key)


def _without_background(layer: Image.Image, key: Image.Image) -> Image.Image:
    out = layer.copy()
    inverse = ImageChops.invert(key)
    out.putalpha(ImageChops.multiply(out.getchannel("A"), inverse))
    return out


@lru_cache(maxsize=32)
def _emoji_sprite(emoji: str) -> Image.Image | None:
    """Colour emoji rendered from Noto Color Emoji, or None when the font is unavailable."""
    if not emoji or not NOTO_EMOJI_FONT.exists():
        return None
    try:
        font = ImageFont.truetype(str(NOTO_EMOJI_FONT), 109)
        canvas = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(canvas).text((8, 8), emoji, font=font, embedded_color=True)
    except (OSError, ValueError, RuntimeError):
        return None
    bbox = canvas.getbbox()
    return canvas.crop(bbox) if bbox else None


def _prop_center(rig: CharacterRig | None) -> tuple[int, int] | None:
    """Where the prop sits: in the safe zone, clear of the character's head and face."""
    half = PROP_SIZE // 2 + 10
    zone_x0, zone_y0 = SAFE_ZONE[0] * W, SAFE_ZONE[1] * H
    zone_x1, zone_y1 = SAFE_ZONE[2] * W, SAFE_ZONE[3] * H
    avoid = None
    if rig is not None:
        hx0, hy0, hx1, hy1 = rig.head_bbox
        avoid = (hx0 - 30, hy0 - 30, hx1 + 30, hy1 + 30)
    # Headroom above the character first, then the sides of the body.
    candidates = (
        (W * 0.5, H * 0.25), (W * 0.27, H * 0.25), (W * 0.73, H * 0.25),
        (W * 0.5, H * 0.31), (W * 0.22, H * 0.50), (W * 0.78, H * 0.50),
    )
    for cx, cy in candidates:
        box = (cx - half, cy - half, cx + half, cy + half)
        if box[0] < zone_x0 or box[2] > zone_x1 or box[1] < zone_y0 or box[3] > zone_y1:
            continue
        if avoid and not (box[2] <= avoid[0] or box[0] >= avoid[2] or box[3] <= avoid[1] or box[1] >= avoid[3]):
            continue
        return (round(cx), round(cy))
    return None


def _draw_prop(frame: Image.Image, sprite: Image.Image, center: tuple[int, int], progress: float) -> None:
    """Drop the prop into place over the first third of its shot, then let it bob."""
    enter = _ease_out(min(1.0, max(0.0, progress) / 0.35))
    scale = (PROP_SIZE / max(sprite.size)) * (0.55 + 0.45 * enter)
    size = (max(2, round(sprite.width * scale)), max(2, round(sprite.height * scale)))
    layer = sprite.resize(size, Image.Resampling.LANCZOS)
    layer.putalpha(layer.getchannel("A").point(lambda v: int(v * enter)))
    bob = math.sin(max(0.0, progress) * math.tau * 1.5) * 5.0 * enter
    drop = -40.0 * (1.0 - enter)
    x = int(center[0] - size[0] / 2)
    y = int(center[1] - size[1] / 2 + drop + bob)
    frame.alpha_composite(layer, (x, y))


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
    scene: dict | None = None,
    draw_text: bool = True,
) -> Image.Image:
    """Render one animated frame of a shot.

    Text is burned in last, by ``_draw_shot_text``, unless ``draw_text`` is False,
    so transitions blend clean pictures and captions never slice through a wipe.
    """
    tuning = render_tuning or {}
    motion_gain = float(tuning.get("motion_gain", 1.0))
    if beat == "reaction":
        # Close-up reactions have little body to move; give them a clear read.
        motion_gain *= REACTION_MOTION_GAIN
    camera_gain = float(tuning.get("camera_gain", 1.0))
    shake_gain = float(tuning.get("shake_gain", 1.0))
    caption_y = int(tuning.get("caption_y", CAPTION_TOP))
    profile = BEAT_PROFILES.get(beat, BEAT_PROFILES["setup"])
    scene = _normalise_scene(scene)
    setting = scene["setting"] if scene else None
    background_key = _background_key(str(image)) if setting else None
    base = _backdrop_over(_load_cover(str(image)), background_key, setting)
    p = max(0.0, min(1.0, progress))
    # Punchlines should land and then hold. Freeze the camera at the final
    # performance pose so the joke reads as a deliberate comedic beat rather
    # than a continued Ken-Burns move. The caption keeps the unclamped progress.
    if beat == "punchline" and p >= 0.72:
        p = 0.72
    if rig is not None:
        try:
            rig_dir = Path(__file__).resolve().parents[2] / "artifacts" / "cartoon" / "pip_rig_v2"
            layers = load_action_layers(image, rig_dir)
            base = _backdrop_over(layers["body_plate"].convert("RGB"), background_key, setting)
            bob = _idle_bob(p, duration)
            base = _apply_torso_layer(base, layers["torso"], p, beat, motion_gain=motion_gain, performance=voice_performance, source_name=image.name, idle_dy=bob)
            base = _apply_arm_layer(base, layers["left_arm"], p, beat, "left", motion_gain=motion_gain, performance=voice_performance, source_name=image.name, idle_dy=bob)
            base = _apply_arm_layer(base, layers["right_arm"], p, beat, "right", motion_gain=motion_gain, performance=voice_performance, source_name=image.name, idle_dy=bob)
        except (OSError, ValueError):
            pass
        base = _apply_rig_performance(base, rig, p, beat, motion_gain=motion_gain, performance=voice_performance, background_key=background_key, shot_seconds=duration)
    eased = _ease_in_out(p)
    energy = profile.energy * motion_gain

    # First 20% is anticipation, middle is action, last 25% settles.
    anticipation = min(1.0, p / 0.20) if p < 0.20 else 1.0
    settle = max(0.0, min(1.0, (p - 0.72) / 0.28))
    spring = _spring(max(0.0, (p - 0.12) / 0.88))

    zoom_start = 1.0 + (profile.zoom_start - 1.0) * camera_gain
    zoom_delta = (profile.zoom_end - profile.zoom_start) * camera_gain
    zoom = zoom_start + zoom_delta * eased
    zoom += 0.012 * math.sin(p * math.tau) * energy
    if beat == "reaction":
        zoom += 0.022 * math.sin((p * 1.2) * math.tau) * (1.0 - settle) * camera_gain
    if shot_index == 0:
        # The opening is a punch-in that settles inside the first beat, so the
        # first second has motion rather than a static picture.
        hook_snap = 1.0 - _ease_out(min(1.0, progress * max(0.0, duration) / 0.6))
        zoom += HOOK_SNAP_ZOOM * hook_snap

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

    angle = profile.rotation * math.sin(math.pi * eased) * (1.0 - 0.55 * settle)

    # Squash/stretch proxy gives action frames a small animated rhythm without
    # deforming the generated character pixels.
    sx = 1.0 + 0.010 * math.sin(math.pi * spring) * energy
    sy = 1.0 - 0.006 * math.sin(math.pi * spring) * energy
    if beat == "escalation":
        sx += 0.014 * max(0.0, math.sin(p * math.pi * 2.0))
        sy -= 0.010 * max(0.0, math.sin(p * math.pi * 2.0))

    # Deterministic impact shake around escalation/reaction peaks. The window
    # is deliberately short: the eye should read an event, not camera jitter.
    peak_start = 0.34 if beat == "escalation" else 0.36 if beat == "reaction" else 1.0
    peak_end = 0.52 if beat == "escalation" else 0.50 if beat == "reaction" else 1.0
    shake_window = math.sin(math.pi * max(0.0, min(1.0, (p - peak_start) / max(0.01, peak_end - peak_start))))
    shake = profile.shake * shake_window * shake_gain
    shake_px = (0, 0)
    if shake:
        shake_px = (int(math.sin(shot_index * 8 + p * 70) * shake), int(math.cos(shot_index * 5 + p * 63) * shake * 0.6))

    frame = _camera_canvas(_overscan_source(base), x, y, crop_w, crop_h, angle, sx, sy, shake=shake_px)

    # For rigged hero assets the head layer already supplies the localized
    # performance motion. Keep the older broad accent as the generic fallback
    # for unknown flattened art so we do not double-transform the subject.
    if rig is None:
        frame = _performance_accent(frame, p, beat, shot_index)

    # Soft contrast/sharpness polish is cheap and deterministic.
    if beat in {"escalation", "reaction"}:
        frame = ImageEnhance.Contrast(frame).enhance(1.035)
        frame = ImageEnhance.Sharpness(frame).enhance(1.08)

    rgba = frame.convert("RGBA")

    # The story prop is placed in final-frame coordinates, clear of the face.
    if scene and scene["prop"] and shot_index in scene["prop_beats"]:
        sprite = _emoji_sprite(scene["prop"])
        center = _prop_center(rig)
        if sprite is not None and center is not None:
            _draw_prop(rgba, sprite, center, progress)

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

    frame = rgba.convert("RGB")
    if draw_text:
        frame, _ = _draw_shot_text(frame, caption, progress, beat, shot_index, duration, caption_y)
    return frame



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
    # Tails are the dead air after each line. They were 0.6-0.9 s (about 38% of
    # the runtime); the punchline keeps its hold because the joke needs it.
    tails = {
        "setup": 0.30,       # thought / setup breath
        "escalation": 0.32, # visual reveal + shock read
        "reaction": 0.32,   # body reaction after the line
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


def _render_tuning(render_tuning: dict | None) -> dict:
    raw = render_tuning or {}
    style = str(raw.get("transition_style", "hard_cut"))
    if style not in {"hard_cut", "pose_morph"}:
        style = "hard_cut"
    return {
        "motion_gain": float(raw.get("motion_gain", 1.0)),
        "camera_gain": float(raw.get("camera_gain", 1.0)),
        "shake_gain": float(raw.get("shake_gain", 1.0)),
        # Shots are hard cuts unless a caller opts into the pose bridge; the
        # bridge length (transition_seconds) only applies to pose_morph.
        "transition_style": style,
        "transition_seconds": float(raw.get("transition_seconds", 0.20)),
        "caption_y": int(raw.get("caption_y", CAPTION_TOP)),
        "caption_zone": list(raw.get("caption_zone", CAPTION_ZONE)),
        "crf": int(raw.get("crf", 18)),
        "repair_attempt": int(raw.get("repair_attempt", 0)),
        "repair_reasons": list(raw.get("repair_reasons", [])),
    }


def _transition_frame_count(tuning: dict, fps: int) -> int:
    if tuning["transition_style"] != "pose_morph":
        return 0
    return max(2, round(tuning["transition_seconds"] * fps))


def _rig_or_none(image: Path) -> CharacterRig | None:
    try:
        return get_rig(str(image))
    except ValueError:
        return None


def _shot_transition(
    previous_frame: Image.Image,
    previous_image: Path,
    previous_rig: CharacterRig | None,
    previous_beat: str,
    frame: Image.Image,
    image: Path,
    rig: CharacterRig | None,
    beat: str,
    progress: float,
    direction: int,
) -> Image.Image:
    """Bridge two clean shot frames; falls back to a directional wipe without a rig."""
    if previous_rig is not None and rig is not None:
        try:
            return _pose_transition_frame(previous_frame, frame, previous_image, image, previous_rig, rig, previous_beat, beat, progress, direction)
        except (OSError, ValueError, KeyError):
            pass
    return _transition_frame(previous_frame, frame, progress, direction)


def _encode_master(ffmpeg: str, frames: Path, fps: int, output: Path, crf: int) -> None:
    """Encode lossless PNG frames to the delivery master.

    High profile, yuv420p, BT.709 matrix and primaries, and faststart so the
    file starts playing before it is fully downloaded.
    """
    subprocess.run(
        [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", str(frames),
            "-vf", "scale=1080:1920:flags=lanczos:out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-r", str(fps), "-movflags", "+faststart",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _seam_diff(first: Path, last: Path) -> float:
    """Mean luma difference (0..1) between the first and last frame; low means a clean loop."""
    a = np.asarray(Image.open(first).convert("L").resize((270, 480)), dtype=np.float32)
    b = np.asarray(Image.open(last).convert("L").resize((270, 480)), dtype=np.float32)
    return round(float(np.abs(a - b).mean()) / 255.0, 4)


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
    shot_specs: list[dict | None] | None = None,
    character: str | None = None,
    scene: dict | None = None,
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

    tuning = _render_tuning(render_tuning)
    scene = _normalise_scene(scene)

    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f"{output.stem}_frames"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    # Voice performance is generated before frames so visual acting can lock to
    # real speech onset/energy/peak rather than an assumed beat fraction.
    voice_performance = prepare_voice_performance(work, voice_lines, beats, durations, character=character)
    # Director AI's per-shot acting plan (anticipation, eye lead, head follow,
    # impact...) rides on each shot's performance entry. It is persisted with the
    # motion metadata, so a targeted rerender reuses the same plan.
    director_specs = 0
    for perf, spec in zip(voice_performance, shot_specs or []):
        if isinstance(perf, dict) and isinstance(spec, dict) and spec:
            perf["shot_spec"] = spec
            director_specs += 1
    effective_durations = _performance_durations(voice_performance, durations, beats)

    # Frames are rendered at the delivery cadence directly; no interpolation.
    source_fps = fps
    transition_frames = _transition_frame_count(tuning, source_fps)
    frame_index = 0
    frame_map = []
    previous_shot_frame: Image.Image | None = None
    previous_shot_rig: CharacterRig | None = None
    previous_shot_image: Path | None = None
    previous_shot_beat: str | None = None
    character_bboxes: list[list[int] | None] = []
    character_frame_bboxes: list[list[int] | None] = []
    caption_rows: list[dict] = []
    transitions: list[dict] = []
    disclosure_box = None
    backdrop_shots: list[int] = []
    prop_shots: list[int] = []
    for shot, (image, caption, duration, beat) in enumerate(zip(images, captions, effective_durations, beats)):
        count = max(2, round(duration * source_fps))
        shot_start = round(frame_index / source_fps, 3)
        frame_map.append({"shot": shot, "beat": beat, "frames": count, "duration": duration})
        rig = _rig_or_none(image)
        if scene and _background_key(str(image)) is not None:
            backdrop_shots.append(shot)
        if scene and scene["prop"] and shot in scene["prop_beats"] and _emoji_sprite(scene["prop"]) is not None and _prop_center(rig) is not None:
            prop_shots.append(shot)
        # Use the performance/head bbox for subtitle safety. The lower body can
        # sit behind captions without obscuring the dialogue.
        character_bboxes.append(list(rig.head_bbox) if rig is not None else None)
        shot_bridge = transition_frames > 0 and previous_shot_frame is not None
        if previous_shot_frame is not None:
            kind = "pose_morph" if shot_bridge and previous_shot_rig is not None and rig is not None else ("directional_wipe" if shot_bridge else "hard_cut")
            transitions.append({"at": shot_start, "kind": kind})
        direction = _transition_direction(previous_shot_rig, rig) if previous_shot_frame is not None else 1
        text_box = None
        words_shown_max = 0
        clean: Image.Image | None = None
        for i in range(count):
            progress = i / max(1, count - 1)
            clean = motion_frame(
                image, caption, progress, shot, duration, beat, rig=rig,
                render_tuning=tuning, voice_performance=voice_performance[shot],
                scene=scene, draw_text=False,
            )
            character_frame_bboxes.append(list(rig.head_bbox) if rig is not None else None)
            frame = clean
            if shot_bridge and i < transition_frames:
                transition_p = i / max(1, transition_frames - 1)
                frame = _shot_transition(
                    previous_shot_frame, previous_shot_image, previous_shot_rig, previous_shot_beat,
                    clean, image, rig, beat, transition_p, direction,
                )
            # Text is burned in after the transition, so no wipe can slice it.
            frame, info = _draw_shot_text(frame, caption, progress, beat, shot, duration, tuning["caption_y"])
            text_box = _union_box(text_box, info["box"])
            words_shown_max = max(words_shown_max, info["words_shown"])
            disclosure_box = _union_box(disclosure_box, info["disclosure_box"])
            frame.save(work / f"frame_{frame_index:05d}.png", compress_level=1)
            frame_index += 1
        caption_rows.append({
            "text": caption,
            "start": shot_start,
            "end": round(frame_index / source_fps, 3),
            "box": _fraction_box(text_box),
            "words_total": len(caption.split()),
            "words_shown_max": words_shown_max,
        })
        previous_shot_frame = clean
        previous_shot_rig = rig
        previous_shot_image = image
        previous_shot_beat = beat

    video_silent = output.with_suffix(".silent.mp4")
    _encode_master(ffmpeg, work / "frame_%05d.png", source_fps, video_silent, tuning["crf"])
    loop_seam = _seam_diff(work / "frame_00000.png", work / f"frame_{frame_index - 1:05d}.png")
    for frame_path in work.glob("frame_*.png"):
        frame_path.unlink()

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
                ffmpeg, "-y", "-i", str(video_for_mux), "-i", str(audio),
                # Explicit duration, not -shortest: a voice track shorter than the
                # picture must not drop frames, or shot frame ranges stop matching.
                "-t", f"{frame_index / source_fps:.3f}",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(output),
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
                "transition_style": tuning["transition_style"],
                "transitions": transitions,
                "captions": caption_rows,
                "disclosure": {
                    "text": DISCLOSURE_TEXT,
                    "box": _fraction_box(disclosure_box),
                    "start": 0.0,
                    "end": round(frame_index / source_fps, 3),
                },
                "safe_zone": list(SAFE_ZONE),
                "overscan": OVERSCAN,
                "scene": _scene_metadata(scene, backdrop_shots, prop_shots),
                "loop": {"match_opening": False, "seam_mean_abs_diff": loop_seam},
                "encode": _encode_metadata(tuning["crf"]),
                "caption_y": tuning["caption_y"],
                "caption_zone": tuning["caption_zone"],
                "repair": {
                    "attempt": tuning["repair_attempt"],
                    "reasons": tuning["repair_reasons"],
                    "strategy": "targeted-single-rerender" if tuning["repair_attempt"] else "initial-render",
                },
                "performance": {
                    "action_transitions": tuning["transition_style"] == "pose_morph",
                    "pose_aware_transition_direction": True,
                    "character_micro_motion": True,
                    "anticipation_recoil_settle": True,
                    "punchline_hold": True,
                    "pose_morph_bridge": tuning["transition_style"] == "pose_morph",
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


def _scene_metadata(scene: dict | None, backdrop_shots: list[int], prop_shots: list[int]) -> dict | None:
    """What the scene actually rendered; the QA gate checks these, not the request."""
    if not scene:
        return None
    return {
        **scene,
        "backdrop_shots": backdrop_shots,
        "prop_shots": prop_shots,
        "prop_font_available": bool(scene["prop"]) and _emoji_sprite(scene["prop"]) is not None,
    }


def _encode_metadata(crf: int) -> dict:
    return {
        "codec": "libx264",
        "preset": "slow",
        "crf": crf,
        "profile": "high",
        "pix_fmt": "yuv420p",
        "colorspace": "bt709",
        "color_primaries": "bt709",
        "color_trc": "bt709",
        "faststart": True,
        "intermediates": "png",
    }


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

    Untargeted frames are copied from the source video at the decoded-frame
    level. Audio is copied from the source container without a second mix or
    TTS pass.
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
    scene = _normalise_scene(source_metadata.get("scene"))

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")

    tuning = _render_tuning(render_tuning)
    full_targets = sorted(set(int(index) for index in target_shots))
    bridge_targets = sorted(
        {
            index + 1
            for index in full_targets
            if index + 1 < len(images) and index + 1 not in full_targets
        }
    )
    transition_frames = _transition_frame_count(tuning, fps)

    def performance(shot_index: int) -> dict | None:
        return source_voice_performance[shot_index] if shot_index < len(source_voice_performance) else None

    def clean_frame(shot_index: int, progress: float) -> Image.Image:
        return motion_frame(
            images[shot_index], captions[shot_index], progress, shot_index,
            effective_durations[shot_index], beats[shot_index],
            rig=_rig_or_none(images[shot_index]), render_tuning=tuning,
            voice_performance=performance(shot_index), scene=scene, draw_text=False,
        )

    with tempfile.TemporaryDirectory(prefix="nova-targeted-repair-") as tmp_name:
        work = Path(tmp_name)
        original_dir = work / "original"
        output_dir = work / "output"
        original_dir.mkdir()
        output_dir.mkdir()

        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(source_video),
                "-vsync", "0",
                str(original_dir / "frame_%05d.png"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        original_frames = sorted(original_dir.glob("frame_*.png"))
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

        def frame_path(frame_index: int) -> Path:
            return output_dir / f"frame_{frame_index + 1:05d}.png"

        def save_frame(frame: Image.Image, frame_index: int) -> None:
            frame.save(frame_path(frame_index), compress_level=1)

        def render_replaced(shot_index: int, local_index: int, progress: float) -> Image.Image:
            frame = clean_frame(shot_index, progress)
            if shot_index > 0 and transition_frames and local_index < transition_frames:
                previous = shot_index - 1
                previous_rig = _rig_or_none(images[previous])
                current_rig = _rig_or_none(images[shot_index])
                transition_p = local_index / max(1, transition_frames - 1)
                frame = _shot_transition(
                    clean_frame(previous, 1.0), images[previous], previous_rig, beats[previous],
                    frame, images[shot_index], current_rig, beats[shot_index],
                    transition_p, _transition_direction(previous_rig, current_rig),
                )
            frame, _ = _draw_shot_text(
                frame, captions[shot_index], progress, beats[shot_index], shot_index,
                effective_durations[shot_index], tuning["caption_y"],
            )
            return frame

        # Full replacement for failed shots. Processing in index order means a
        # previous repaired shot's final frame is available to the next bridge.
        for shot_index in full_targets:
            start, end, _ = shot_ranges[shot_index]
            for local_index, frame_index in enumerate(range(start, end)):
                progress = local_index / max(1, (end - start) - 1)
                save_frame(render_replaced(shot_index, local_index, progress), frame_index)

        # Only the transition prefix is regenerated for shots immediately after
        # a repaired shot; their remaining frames stay untouched.
        for shot_index in bridge_targets:
            start, end, _ = shot_ranges[shot_index]
            prefix = min(transition_frames, end - start)
            for local_index, frame_index in enumerate(range(start, start + prefix)):
                progress = local_index / max(1, (end - start) - 1)
                save_frame(render_replaced(shot_index, local_index, progress), frame_index)

        silent = work / "silent.mp4"
        _encode_master(ffmpeg, output_dir / "frame_%05d.png", fps, silent, tuning["crf"])

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
                    "-t", f"{actual_frames / fps:.3f}", "-movflags", "+faststart", str(muxed),
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
    new_metadata["transition_style"] = tuning["transition_style"]
    new_metadata["caption_y"] = tuning["caption_y"]
    new_metadata["caption_zone"] = tuning["caption_zone"]
    new_metadata["safe_zone"] = list(SAFE_ZONE)
    new_metadata["overscan"] = OVERSCAN
    new_metadata["encode"] = _encode_metadata(tuning["crf"])
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
