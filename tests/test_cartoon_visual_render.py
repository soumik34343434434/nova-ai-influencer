"""Visual render contract: captions, blink, overscan, transitions, scene, encode, metadata."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import app.providers.cartoon_animation as animation
from app.providers.cartoon_rig import get_rig

ACTIONS = Path(__file__).resolve().parents[1] / "assets" / "cartoon" / "actions"
LINE = "Of course. Still one job. Apparently."


def _dark_edge_fraction(frame: Image.Image, width: int = 12) -> float:
    arr = np.asarray(frame.convert("RGB")).astype(int)
    edge = np.concatenate([
        arr[:width].reshape(-1, 3), arr[-width:].reshape(-1, 3),
        arr[:, :width].reshape(-1, 3), arr[:, -width:].reshape(-1, 3),
    ])
    return float((edge.max(axis=1) < 30).mean())


def test_punchline_caption_reveals_every_word_at_the_end_of_the_shot():
    words = LINE.split()
    progress = animation._caption_progress(1.0, "punchline", 3, 2.8)
    assert animation._visible_word_count(LINE, progress) == len(words)
    frame = Image.new("RGB", (animation.W, animation.H), (200, 200, 200))
    _, info = animation._draw_shot_text(frame, LINE, 1.0, "punchline", 3, 2.8)
    assert info["words_shown"] == info["words_total"] == len(words)


def test_hook_caption_is_mostly_readable_from_the_first_frame():
    progress = animation._caption_progress(0.0, "setup", 0, 2.3)
    assert progress >= 0.35
    assert animation._caption_progress(0.4, "setup", 0, 2.3) == 1.0


def test_caption_and_disclosure_stay_inside_the_safe_zone():
    frame = Image.new("RGB", (animation.W, animation.H), (200, 200, 200))
    long_line = "Turns out the help was literally the problem and then the problem had a problem of its own."
    _, info = animation._draw_shot_text(frame, long_line, 1.0, "escalation", 1, 2.0)
    x0, y0, x1, y1 = animation.SAFE_ZONE
    for box in (info["box"], info["disclosure_box"]):
        assert box is not None
        assert box[0] >= x0 * animation.W and box[2] <= x1 * animation.W
        assert box[1] >= y0 * animation.H and box[3] <= y1 * animation.H


def test_blink_bar_is_drawn_only_for_pip(monkeypatch):
    bubu = get_rig(str(ACTIONS / "bubu_wave_hello.png"))
    pip = get_rig(str(ACTIONS / "pip_shock.png"))
    assert bubu.character == "BUBU" and pip.character == "PIP"
    base = Image.new("RGB", (animation.W, animation.H), (200, 200, 200))

    def blink(progress, beat, performance=None):
        return 0.0, 0.0, 1.0, 1.0, 0.9

    def still(progress, beat, performance=None):
        return 0.0, 0.0, 1.0, 1.0, 0.0

    monkeypatch.setattr(animation, "_face_motion", blink)
    bubu_blink = animation._apply_face_performance(base, bubu, 0.5, "escalation")
    pip_blink = animation._apply_face_performance(base, pip, 0.5, "escalation")
    monkeypatch.setattr(animation, "_face_motion", still)
    bubu_still = animation._apply_face_performance(base, bubu, 0.5, "escalation")
    pip_still = animation._apply_face_performance(base, pip, 0.5, "escalation")
    assert bubu_blink.tobytes() == bubu_still.tobytes()
    assert pip_blink.tobytes() != pip_still.tobytes()


@pytest.mark.parametrize("progress", [0.0, 0.45, 1.0])
@pytest.mark.parametrize("beat", ["escalation", "reaction"])
def test_camera_moves_never_expose_frame_edges(progress, beat):
    frame = animation.motion_frame(
        ACTIONS / "bubu_wave_hello.png", "", progress, 1, 2.3, beat,
        rig=None, draw_text=False,
    )
    assert frame.size == (animation.W, animation.H)
    assert _dark_edge_fraction(frame) < 0.001


def test_transition_frames_never_expose_black_margins():
    light = Image.new("RGB", (animation.W, animation.H), (210, 210, 214))
    dark_scene = Image.new("RGB", (animation.W, animation.H), (60, 120, 180))
    for t in (0.0, 0.3, 0.7, 1.0):
        for direction in (1, -1):
            frame = animation._transition_frame(light, dark_scene, t, direction)
            assert _dark_edge_fraction(frame, width=4) < 0.001


def test_hard_cut_is_the_default_and_pose_bridge_is_opt_in():
    assert animation._transition_frame_count(animation._render_tuning(None), 24) == 0
    bridged = animation._render_tuning({"transition_style": "pose_morph", "transition_seconds": 0.2})
    assert animation._transition_frame_count(bridged, 24) == 5


def test_performance_tails_are_short_except_the_punchline_hold():
    perf = [{"duration": 2.0}] * 4
    durations = animation._performance_durations(perf, [5.0] * 4, ["setup", "escalation", "reaction", "punchline"])
    assert durations[0] == pytest.approx(2.3)
    assert durations[1] == pytest.approx(2.32)
    assert durations[3] == pytest.approx(2.86)


def test_scene_contract_is_validated():
    assert animation._normalise_scene(None) is None
    scene = animation._normalise_scene({"setting": "moon", "prop": "\U0001F96A", "prop_label": "sandwich", "prop_beats": [0, "x", 1, 1]})
    assert scene == {"setting": "studio", "prop": "\U0001F96A", "prop_label": "sandwich", "prop_beats": [0, 1]}
    assert animation._normalise_scene({"setting": "park", "prop": "a very long string of text"})["prop"] == ""


def test_backdrop_is_keyed_only_for_plain_studio_art():
    assert animation._background_key(str(ACTIONS / "bubu_wave_hello.png")) is not None
    assert animation._background_key(str(ACTIONS / "pip_press_button.png")) is None


def test_backdrop_key_takes_enclosed_backdrop_but_keeps_eyes():
    key = animation._background_key(str(ACTIONS / "bubu_wave_hello.png"))
    assert key is not None
    assert key.getpixel((320, 880)) > 127  # light backdrop pocket above the head
    assert key.getpixel((400, 1000)) < 127  # eye white stays with the character


def test_background_removal_masks_layer_alpha():
    layer = Image.new("RGBA", (40, 40), (10, 20, 30, 255))
    key = Image.new("L", (40, 40), 0)
    ImageDraw.Draw(key).rectangle((0, 0, 19, 39), fill=255)
    out = animation._without_background(layer, key)
    assert out.getchannel("A").getpixel((5, 5)) == 0
    assert out.getchannel("A").getpixel((30, 5)) == 255


def test_prop_placement_never_covers_the_face():
    rig = get_rig(str(ACTIONS / "bubu_wave_hello.png"))
    center = animation._prop_center(rig)
    assert center is not None
    half = animation.PROP_SIZE // 2
    hx0, hy0, hx1, hy1 = rig.head_bbox
    box = (center[0] - half, center[1] - half, center[0] + half, center[1] + half)
    assert box[2] <= hx0 - 30 or box[0] >= hx1 + 30 or box[3] <= hy0 - 30 or box[1] >= hy1 + 30


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg required")
def test_delivery_master_is_high_profile_bt709_and_faststart(tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(6):
        Image.new("RGB", (animation.W, animation.H), (40 + i * 20, 120, 200)).save(frames / f"frame_{i:05d}.png")
    output = tmp_path / "master.mp4"
    animation._encode_master("ffmpeg", frames / "frame_%05d.png", 24, output, 18)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=profile,pix_fmt,color_space,color_primaries,color_transfer,width,height",
         "-of", "json", str(output)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["profile"] == "High"
    assert stream["pix_fmt"] == "yuv420p"
    assert (stream["width"], stream["height"]) == (1080, 1920)
    assert stream["color_space"] == "bt709"
    assert stream["color_primaries"] == "bt709"
    assert stream["color_transfer"] == "bt709"
    head = output.read_bytes()[:4096]
    assert head.find(b"moov") != -1 and head.find(b"moov") < head.find(b"mdat")


def test_render_writes_the_motion_metadata_contract(tmp_path: Path, monkeypatch):
    images = []
    for i in range(2):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (768, 1376), (210 - i * 10, 214, 220)).save(path)
        images.append(path)
    monkeypatch.setattr(animation, "_audio_for_shots", lambda *args, **kwargs: None)
    output = tmp_path / "episode.mp4"
    animation.render_cartoon_motion(
        images,
        ["Of course.", "Still one job."],
        [0.3, 0.3],
        ["setup", "punchline"],
        output,
        voice_lines=["Of course.", "Still one job."],
        sfx_kinds=["none", "none"],
        fps=24,
        scene={"setting": "kitchen", "prop": "\U0001F96A", "prop_label": "sandwich", "prop_beats": [1]},
    )
    meta = json.loads(output.with_suffix(".motion.json").read_text(encoding="utf-8"))
    assert meta["safe_zone"] == list(animation.SAFE_ZONE)
    assert meta["overscan"] == animation.OVERSCAN
    assert meta["transition_style"] == "hard_cut"
    assert [row["kind"] for row in meta["transitions"]] == ["hard_cut"]
    assert [row["text"] for row in meta["captions"]] == ["Of course.", "Still one job."]
    for row in meta["captions"]:
        assert row["words_shown_max"] == row["words_total"]
        assert row["box"] is not None and row["start"] < row["end"]
    assert meta["disclosure"]["text"] == animation.DISCLOSURE_TEXT
    assert meta["disclosure"]["box"] is not None
    assert meta["scene"]["setting"] == "kitchen"
    assert meta["encode"]["faststart"] is True
    assert meta["loop"]["seam_mean_abs_diff"] >= 0.0
    assert not list(output.parent.glob("episode_frames/frame_*.png"))


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe required")
def test_targeted_rerender_preserves_frame_count_and_records_repair(tmp_path: Path, monkeypatch):
    images = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (768, 1376), (200 + i * 8, 214, 224)).save(path)
        images.append(path)
    monkeypatch.setattr(animation, "_audio_for_shots", lambda *args, **kwargs: None)
    source = tmp_path / "source.mp4"
    lines = ["One.", "Two.", "Three."]
    beats = ["setup", "escalation", "punchline"]
    animation.render_cartoon_motion(
        images, lines, [0.3, 0.3, 0.3], beats, source,
        voice_lines=lines, sfx_kinds=["none"] * 3, fps=24,
    )
    source_meta = json.loads(source.with_suffix(".motion.json").read_text(encoding="utf-8"))
    output = tmp_path / "repaired.mp4"
    animation.rerender_cartoon_shots(
        source, images, lines, [0.3, 0.3, 0.3], beats, output,
        target_shots=[1], render_tuning={"repair_attempt": 1, "repair_reasons": ["motion_shot_1"]},
        fps=24,
    )
    frames_before = sum(row["frames"] for row in source_meta["shots"])
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", str(output)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert int(stream["nb_read_frames"]) == frames_before
    assert (int(stream["width"]), int(stream["height"])) == (1080, 1920)
    repaired = json.loads(output.with_suffix(".motion.json").read_text(encoding="utf-8"))
    assert repaired["repair"]["strategy"] == "targeted-shot-rerender"
    assert repaired["repair"]["target_shots"] == [1]
    assert [row["repaired"] for row in repaired["shots"]] == [False, True, False]
    assert repaired["encode"]["faststart"] is True


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe required")
def test_short_voice_track_does_not_drop_picture_frames(tmp_path: Path, monkeypatch):
    import wave

    images = []
    for i in range(2):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (768, 1376), (210, 214, 220 - i * 6)).save(path)
        images.append(path)
    voice = tmp_path / "short_voice.wav"
    with wave.open(str(voice), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\x00\x00" * 13230)  # 0.3 s, far shorter than the picture
    monkeypatch.setattr(animation, "_audio_for_shots", lambda *args, **kwargs: voice)
    output = tmp_path / "episode.mp4"
    animation.render_cartoon_motion(
        images, ["One.", "Two."], [0.3, 0.3], ["setup", "punchline"], output,
        voice_lines=["One.", "Two."], sfx_kinds=["none"] * 2, fps=24,
    )
    meta = json.loads(output.with_suffix(".motion.json").read_text(encoding="utf-8"))
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert int(json.loads(probe.stdout)["streams"][0]["nb_read_frames"]) == meta["frames"]
