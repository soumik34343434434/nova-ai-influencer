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
    assert '"engine": "cartoon-motion-3.1"' in metadata
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


def test_transition_is_not_identical_at_midpoint(tmp_path: Path):
    from app.providers.cartoon_animation import _transition_frame

    a = Image.new("RGB", (1080, 1920), "white")
    b = Image.new("RGB", (1080, 1920), "black")
    first = _transition_frame(a, b, 0.15, 1)
    middle = _transition_frame(a, b, 0.50, 1)
    last = _transition_frame(a, b, 0.95, 1)
    assert first.tobytes() != middle.tobytes()
    assert middle.tobytes() != last.tobytes()
