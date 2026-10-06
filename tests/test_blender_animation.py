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
    for marker in ("Rig", "ARMATURE", "vertex_groups", "TARGETS", "Eye.L", "Pupil.L", "action-chain-anticipation-contact-overshoot-recoil-settle"):
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


def test_action_pose_chain_has_physical_phases():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    for phase in ('anticipation','contact','overshoot','recoil','settle'):
        assert f"'{phase}'" in text
    assert 'action_pose_chain' in text


def test_pose_chain_is_deterministic_and_overshoots():
    from app.providers.cartoon_pose_library import make_action_pose_chain
    base={"root":[0,0],"neck":[0,0],"head_center":[0,0],"left_shoulder":[0,0],"left_elbow":[0,0],"left_wrist":[0,0],"right_shoulder":[0,0],"right_elbow":[0,0],"right_wrist":[0,0],"head_rot":0,"root_rot":0,"eye_open":1,"gaze":[0,0],"mouth":"flat"}
    target={**base,"root":[100,50],"neck":[100,50],"head_center":[100,50]}
    chain=make_action_pose_chain(base,target,'shock',span=100)
    assert [x[2] for x in chain]==['approach','anticipation','contact_approach','contact','overshoot','recoil','settle']
    assert chain[1][1]['root'][0] < 0
    assert chain[4][1]['root'][0] > target['root'][0]
