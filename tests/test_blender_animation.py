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


def test_action_pose_chain_accepts_data_driven_transition_profile():
    from app.providers.cartoon_pose_library import make_action_pose_chain
    base={"root":[0,0],"neck":[0,0],"head_center":[0,0],"left_shoulder":[0,0],"left_elbow":[0,0],"left_wrist":[0,0],"right_shoulder":[0,0],"right_elbow":[0,0],"right_wrist":[0,0],"head_rot":0,"root_rot":0,"eye_open":1,"gaze":[0,0],"mouth":"flat"}
    target={**base,"root":[100,0],"neck":[100,0],"head_center":[100,0]}
    profiles={"default": {"anticipation_at":0.20,"anticipation_t":-0.10,"contact_approach_at":0.45,"contact_approach_t":0.70,"contact_at":0.60,"contact_t":1.0,"overshoot_at":0.72,"overshoot_t":1.08,"recoil_at":0.84,"recoil_t":0.94,"settle_at":1.0,"settle_t":1.0}, "press_button":{"anticipation_t":-0.35,"overshoot_t":1.22}}
    chain=make_action_pose_chain(base,target,'press_button',span=100,transition_profiles=profiles)
    assert chain[1][0] == 20
    assert chain[1][1]["root"][0] == -35
    assert chain[4][1]["root"][0] == 122


def test_transition_profiles_exist_for_all_cartoon_characters():
    import json
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'config'/'cartoon_blender_profiles.json').read_text(encoding='utf-8'))
    for character in ('PIP','MOMO','BUBU'):
        profile=data['characters'][character]['transition_profiles']
        assert 'default' in profile
        assert 'secondary' in profile['default']
        assert profile['default']['secondary']['head_lag_deg'] > 0
        assert profile['default']['secondary']['accessory_follow_deg'] > 0


def test_blender_source_declares_secondary_motion_and_profile_driven_chain():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    for marker in ('transition_profiles','secondary','head_lag','hand_follow','accessory_follow','impact_eye_open','_motion','_phase'):
        assert marker in text
    assert "'engine':'blender-cutout-performance-rig-v4'" in text


def test_blender_episode_audio_helper_creates_missing_work_directory(tmp_path):
    import shutil
    import wave
    from app.providers.cartoon_media import _audio_for_shots

    wav = tmp_path / "voice.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 2205)

    work = tmp_path / "missing" / "audio"
    out = _audio_for_shots(
        work, ["hello"], [1.0], ["none"], shutil.which("ffmpeg"),
        voice_performance=[{"path": str(wav)}],
    )
    assert out is not None
    assert out.exists()


def test_blender_episode_audio_uses_the_effective_voice_clock():
    root=Path(__file__).resolve().parents[1]
    text=(root/'app'/'providers'/'blender_animation.py').read_text(encoding='utf-8')
    assert 'reaction_tails = (0.48,0.58,0.60,0.82)' in text
    assert 'effective_shot_durations' in text
    assert '_audio_for_shots(work / "audio", voice_lines, effective_shot_durations' in text
    assert 'total_seconds = sum(effective_shot_durations)' in text


def test_character_specific_acting_poses_have_distinct_silhouettes():
    import json
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'config'/'cartoon_blender_profiles.json').read_text(encoding='utf-8'))
    from app.providers.cartoon_pose_library import build_pose_library
    momo=build_pose_library(data['characters']['MOMO']['base'],'MOMO')
    bubu=build_pose_library(data['characters']['BUBU']['base'],'BUBU')
    assert momo['present']['left_wrist'] != data['characters']['MOMO']['base']['left_wrist']
    assert bubu['shock']['right_wrist'] != data['characters']['BUBU']['base']['right_wrist']
    assert momo['present']['right_wrist'] != bubu['present']['right_wrist']


def test_action_profiles_define_impact_translation():
    import json
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'config'/'cartoon_blender_profiles.json').read_text(encoding='utf-8'))
    for character in ('PIP','MOMO','BUBU'):
        profile=data['characters'][character]['transition_profiles']
        assert profile['default']['impact_translate_px'] > 0
        assert profile['shock']['impact_translate_px'] > 0
