from pathlib import Path

from PIL import Image

from app.providers.cartoon_animation import BEAT_PROFILES, motion_frame, render_cartoon_motion


def test_motion_frame_changes_by_performance(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (768, 1376), "white").save(source)
    setup = motion_frame(source, "One sandwich, please.", 0.2, 0, 2.3, "setup")
    escalation = motion_frame(source, "TWELVE?!", 0.55, 1, 2.3, "escalation")
    assert setup.size == (1080, 1920)
    assert escalation.size == (1080, 1920)
    assert setup.tobytes() != escalation.tobytes()


def test_profiles_cover_core_comedy_beats():
    assert set(("setup", "escalation", "reaction", "punchline")) <= set(BEAT_PROFILES)


def test_impact_paper_burst_is_deterministic_and_contact_localized():
    from PIL import ImageChops
    from app.providers.cartoon_animation import _impact_graphics

    blank = Image.new("RGBA", (1080, 1920), (30, 40, 50, 255))
    no_impact = blank.copy()
    _impact_graphics(no_impact, 0.0)
    early = blank.copy()
    _impact_graphics(early, 0.25)
    first = blank.copy()
    repeated = blank.copy()
    _impact_graphics(first, 0.8)
    _impact_graphics(repeated, 0.8)

    assert ImageChops.difference(blank.convert("RGB"), no_impact.convert("RGB")).getbbox() is None
    assert first.tobytes() == repeated.tobytes()
    assert early.tobytes() != first.tobytes()
    difference = ImageChops.difference(blank.convert("RGB"), first.convert("RGB"))
    assert difference.crop((300, 700, 780, 1100)).getbbox()
    assert difference.crop((0, 0, 220, 500)).getbbox() is None


def test_render_cartoon_motion_smoke(tmp_path: Path):
    images = []
    for i in range(4):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (768, 1376), (220 - i * 10, 230, 240)).save(path)
        images.append(path)
    audio = tmp_path / "empty.wav"
    import wave
    with wave.open(str(audio), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\\x00\\x00" * 4410)
    output = tmp_path / "out.mp4"
    render_cartoon_motion(
        images,
        ["One.", "Two.", "Three.", "Four."],
        [0.25, 0.25, 0.25, 0.25],
        ["setup", "escalation", "reaction", "punchline"],
        output,
        voice_lines=["One.", "Two.", "Three.", "Four."],
        sfx_kinds=["none"] * 4,
        fps=12,
    )
    assert output.exists()
    assert output.stat().st_size > 0


def test_punchline_holds_final_performance_pose(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (768, 1376), "white").save(source)
    held = motion_frame(source, "THE END.", 0.72, 3, 2.8, "punchline")
    later = motion_frame(source, "THE END.", 0.98, 3, 2.8, "punchline")
    assert held.tobytes() == later.tobytes()


def test_motion_metadata_declares_delivery_profile(tmp_path: Path):
    images = []
    for i in range(4):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (768, 1376), (220 - i * 10, 230, 240)).save(path)
        images.append(path)
    output = tmp_path / "out.mp4"
    import app.providers.cartoon_animation as animation
    original_audio = animation._audio_for_shots
    animation._audio_for_shots = lambda *args, **kwargs: None
    try:
        render_cartoon_motion(
            images,
            ["One.", "Two.", "Three.", "Four."],
            [0.25, 0.25, 0.25, 0.25],
            ["setup", "escalation", "reaction", "punchline"],
            output,
            voice_lines=["One.", "Two.", "Three.", "Four."],
            sfx_kinds=["none"] * 4,
            fps=24,
        )
    finally:
        animation._audio_for_shots = original_audio
    metadata = output.with_suffix(".motion.json").read_text()
    assert '"engine": "cartoon-motion-3.6"' in metadata
    assert '"fps": 24' in metadata
    assert '"interpolation": "none"' in metadata


def test_transition_direction_follows_pose_centers():
    from app.providers.cartoon_animation import _transition_direction
    from app.providers.cartoon_rig import get_rig

    root = Path("assets/cartoon/actions")
    press = get_rig(str(root / "pip_press_button.png"))
    shock = get_rig(str(root / "pip_shock.png"))
    overwhelmed = get_rig(str(root / "pip_overwhelmed.png"))
    deadpan = get_rig(str(root / "pip_deadpan.png"))

    assert _transition_direction(press, shock) == 1
    assert _transition_direction(shock, overwhelmed) == -1
    assert _transition_direction(overwhelmed, deadpan) == 1


def test_delivery_camera_follows_delayed_character_root():
    import math
    from app.providers.cartoon_animation import BEAT_PROFILES, _camera_geometry
    from app.providers.cartoon_performance import kinematic_state

    progress = 0.30
    beat = "escalation"
    camera = _camera_geometry(progress, beat)
    delayed_root = kinematic_state(progress - 0.045, beat)
    profile = BEAT_PROFILES[beat]
    eased = progress * progress * (3 - 2 * progress)
    zoom = profile.zoom_start + (profile.zoom_end - profile.zoom_start) * eased
    zoom += 0.012 * math.sin(progress * math.tau) * profile.energy
    zoom += delayed_root["camera_zoom"]
    crop_w = int(1080 / zoom)
    crop_h = int(1920 / zoom)
    max_x, max_y = 1080 - crop_w, 1920 - crop_h
    expected_x = int(max_x * 0.5 + profile.pan_x * max_x * math.sin(math.pi * eased) - delayed_root["camera_dx"] * 2.2 + delayed_root["camera_focus_x"] * max_x * 0.10)
    expected_y = int(max_y * 0.5 + profile.pan_y * max_y * math.sin(math.pi * eased) - delayed_root["camera_dy"] * 2.2 + delayed_root["camera_focus_y"] * max_y * 0.10)

    assert camera[:4] == (expected_x, expected_y, crop_w, crop_h)


def test_transition_is_not_identical_at_midpoint(tmp_path: Path):
    from app.providers.cartoon_animation import _transition_frame

    a = Image.new("RGB", (1080, 1920), "white")
    b = Image.new("RGB", (1080, 1920), "black")
    first = _transition_frame(a, b, 0.15, 1)
    middle = _transition_frame(a, b, 0.50, 1)
    last = _transition_frame(a, b, 0.95, 1)
    assert first.tobytes() != middle.tobytes()
    assert middle.tobytes() != last.tobytes()


def test_pose_morph_moves_character_part_between_endpoints():
    from app.providers.cartoon_animation import _morph_transparent_layers

    previous = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    current = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    previous.paste(Image.new("RGBA", (100, 200), (10, 20, 30, 255)), (150, 700))
    current.paste(Image.new("RGBA", (200, 100), (40, 50, 60, 255)), (700, 900))

    mid = _morph_transparent_layers(previous, current, 0.5, 1)
    bbox = mid.getchannel("A").getbbox()
    assert bbox is not None
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    assert 350 < cx < 700
    assert 750 < cy < 950


def test_pose_transition_metadata_is_declared(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (768, 1376), "white").save(source)
    from app.providers.cartoon_animation import _pose_transition_frame
    from app.providers.cartoon_rig import get_rig

    a = Path("assets/cartoon/actions/pip_press_button.png")
    b = Path("assets/cartoon/actions/pip_shock.png")
    ar = get_rig(str(a))
    br = get_rig(str(b))
    pa = motion_frame(a, "One", 1.0, 0, 2.3, "setup", rig=ar)
    pb = motion_frame(b, "Two", 0.0, 1, 2.3, "escalation", rig=br)
    out = _pose_transition_frame(pa, pb, a, b, ar, br, "setup", "escalation", 0.5, 1)
    assert out.size == (1080, 1920)


def test_pose_switches_are_staggered_for_follow_through():
    from app.providers.cartoon_animation import POSE_SWITCH_FRACTIONS
    assert POSE_SWITCH_FRACTIONS["left_arm"] < POSE_SWITCH_FRACTIONS["torso"]
    assert POSE_SWITCH_FRACTIONS["torso"] < POSE_SWITCH_FRACTIONS["head"]
    assert POSE_SWITCH_FRACTIONS["head"] < POSE_SWITCH_FRACTIONS["face"]
