from pathlib import Path

from app.providers.blender_animation import BlenderAnimationProvider


def test_blender_provider_contract():
    root = Path(__file__).resolve().parents[1]
    provider = BlenderAnimationProvider(blender_bin="/definitely/missing/blender")
    assert (root / "tools" / "blender" / "pip_cutout_rig_v2.py").exists()
    assert isinstance(provider.available(), bool)


def test_blender_rig_script_has_performance_features():
    root = Path(__file__).resolve().parents[1]
    text = (root / "tools" / "blender" / "pip_cutout_rig_v2.py").read_text(encoding="utf-8")
    for marker in ("Rig", "ARMATURE", "vertex_groups", "TARGETS", "Eye.L", "Pupil.L", "overshoot-and-settle-bezier"):
        assert marker in text


def test_blender_script_has_camera_performance_and_voice_clock():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert 'camera_performance_keys' in text
    assert 'voice_json' in text
    assert 'camera_performance_keys' in text


def test_blender_episode_metadata_contract_in_source():
    root=Path(__file__).resolve().parents[1]
    text=(root/'app'/'providers'/'blender_animation.py').read_text(encoding='utf-8')
    assert 'effective_shot_durations' in text
    assert 'sfx_peak_fractions' in text
    assert 'voice_performance' in text


def test_provider_source_supports_action_assets_and_background():
    root=Path(__file__).resolve().parents[1]
    text=(root/'app'/'providers'/'blender_animation.py').read_text(encoding='utf-8')
    assert 'action_assets' in text
    assert 'background_path' in text
    assert 'Missing Blender action assets' in text


def test_provider_has_on_demand_character_compiler():
    root=Path(__file__).resolve().parents[1]
    text=(root/'app'/'providers'/'blender_animation.py').read_text(encoding='utf-8')
    assert 'ensure_character_assets' in text
    assert 'compile_cartoon_character_asset.py' in text
    assert 'character: str = "PIP"' in text


def test_pose_library_is_action_driven():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert 'pose_for_index' in text
    assert "row.get('action')" in text
