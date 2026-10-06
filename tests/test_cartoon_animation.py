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
