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


def test_camera_action_focus_tracks_performed_contact_hand():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert 'def _performed_action_focus(action, arm, fallback):' in text
    assert "'press_button': 'R'" in text
    assert "'give_homework': 'L'" in text
    assert "contact_focus=_performed_action_focus(action,arm,(fx,fy))" in text
    assert "'contact_approach':0.45,'contact':0.90,'overshoot':0.70,'recoil':0.25,'settle':0.0" in text
    assert 'point = arm.matrix_world @ hand.tail' in text
    assert 'point = arm.matrix_world @ hand.matrix.translation' not in text


def test_camera_reaction_focus_tracks_the_performed_head_pose():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert "if action in {'shock', 'overwhelmed', 'deadpan'}:" in text
    assert "head = arm.pose.bones.get('head')" in text
    assert 'point = arm.matrix_world @ head.tail' in text
    assert 'focus_x=fx+(contact_focus[0]-fx)*focus_mix' in text


def test_camera_contact_focus_peaks_at_impact_and_releases_on_settle():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert "'contact_approach':0.45,'contact':0.90,'overshoot':0.70,'recoil':0.25,'settle':0.0" in text
    assert 'focus_x=fx+(contact_focus[0]-fx)*focus_mix' in text
    assert 'focus_y=fy+(contact_focus[1]-fy)*focus_mix' in text


def test_camera_impact_shake_is_phase_locked_and_decays_after_contact():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert "'escalation':{'approach':0.0,'anticipation':0.0,'contact_approach':0.0,'contact':0.010,'overshoot':0.0065,'recoil':0.0025,'settle':0.0}" in text
    assert 'shake=shake_profile[beat][phase]' in text
    assert 'math.sin(frame*1.21)*shake_amp' not in text


def test_camera_escalation_push_peaks_after_contact_and_releases_on_recoil():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert "'contact_approach':-0.020,'contact':-0.105,'overshoot':-0.125,'recoil':-0.045,'settle':-0.010" in text
    assert "cam.data.ortho_scale=max(10.0,base_scale[beat]*action_scale.get(action,1.0)*(1.0+punch))" in text


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


def test_action_pose_chain_gaze_leads_body_without_erasing_extrapolation():
    from app.providers.cartoon_pose_library import make_action_pose_chain
    base={"root":[0,0],"neck":[0,0],"head_center":[0,0],"left_shoulder":[0,0],"left_elbow":[0,0],"left_wrist":[0,0],"right_shoulder":[0,0],"right_elbow":[0,0],"right_wrist":[0,0],"head_rot":0,"root_rot":0,"eye_open":1,"gaze":[0,0],"mouth":"flat"}
    target={**base,"root":[100,0],"gaze":[1,0]}
    chain=make_action_pose_chain(base,target,'default',span=100)
    approach=chain[0][1]
    assert approach["root"][0] == 0
    # At contact approach the body follows smoothstep (0.82), while the
    # gaze's ease-out is further along toward its target (0.9728).
    assert chain[2][1]["gaze"][0] > chain[2][1]["root"][0] / 100
    assert chain[1][1]["gaze"][0] < 0  # anticipation still looks with the pose
    assert chain[4][1]["gaze"][0] > 1  # overshoot remains physical extrapolation


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
    assert "'engine':'blender-cutout-performance-rig-v5'" in text


def test_secondary_hand_follow_through_emphasizes_the_acting_hand():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert "acting_hand={'press_button':'R','give_homework':'L'" in text
    assert 'follow_weight=1.35 if side==acting_hand else 0.65 if acting_hand else 1.0' in text


def test_impact_facial_reaction_is_asymmetric_toward_action_hand():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert "contact_side={'press_button':1,'give_homework':-1" in text
    assert 'side_weight=1.25 if contact_side and sign==contact_side else 0.75 if contact_side else 1.0' in text
    assert 'side_weight=1.25 if contact_side and side==contact_side else 0.75 if contact_side else 1.0' in text


def test_shock_brow_reaction_leads_at_contact_and_releases_through_recoil():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert "brow_reaction={'contact':0.055,'overshoot':0.040,'recoil':0.018,'settle':0.0}.get(phase,0.0) if action=='shock' else 0.0" in text
    assert 'left_brow_lift=lift + brow_reaction*(1.0 if contact_side < 0 else 0.35)' in text
    assert 'right_brow_lift=lift + brow_reaction*(1.0 if contact_side > 0 else 0.35)' in text
    assert 'brow_l.location.y += left_brow_lift; brow_r.location.y += right_brow_lift' in text


def test_organic_face_brows_pose_with_action_and_return_to_neutral():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert "brows=face.get('brows',())" in text
    assert "'shock':0.045,'overwhelmed':0.018,'deadpan':-0.012" in text
    assert "brow.keyframe_insert('location',frame=f)" in text
    assert "brow.keyframe_insert('location',frame=min(end,f+3))" in text
    assert "brow.keyframe_insert('rotation_euler',frame=min(end,f+3))" in text


def test_gaze_darts_return_to_the_evaluated_pose_without_accumulating():
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert 'bpy.context.scene.frame_set(df)' in text
    assert 'origin=eye.location.copy()' in text
    assert 'dart_delta=dart_mag*(1.0 if dart_i%2==0 else -0.7)' in text
    assert 'eye.location.x=origin.x + dart_delta' in text
    assert 'eye.location=origin' in text
    assert 'eye.keyframe_insert(\'location\',frame=df+2)' in text
    assert 'pupil.location.x=pupil_origin.x + dart_delta' in text
    assert 'pupil.keyframe_insert(\'location\',frame=df+2)' in text


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


def test_blender_provider_cleans_stale_frames_and_uses_effective_caption_clock():
    root=Path(__file__).resolve().parents[1]
    text=(root/'app'/'providers'/'blender_animation.py').read_text(encoding='utf-8')
    assert 'stale in (*frame_dir.glob("frame_*.png"), *frame_dir.glob("key_*.png"))' in text
    assert 'caption_durations = effective_shot_durations if effective_shot_durations else durations' in text


def test_blender_metadata_declares_cinematic_camera_and_world_stage():
    root=Path(__file__).resolve().parents[1]
    text=(root/'app'/'providers'/'blender_animation.py').read_text(encoding='utf-8')
    assert 'phase-driven-damped-follow-with-impact-push' in text
    assert 'procedural-desk-and-action-prop-kit' in text
    assert 'multi_blink' in text


def test_blender_social_hook_and_visual_prop_reveal_are_declared():
    root=Path(__file__).resolve().parents[1]
    provider=(root/'app'/'providers'/'blender_animation.py').read_text(encoding='utf-8')
    rig=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    orch=(root/'app'/'cartoon_factory_orchestrator.py').read_text(encoding='utf-8')
    assert 'hook_text: str = ""' in provider
    assert 'shot == 0 and hook_text' in provider
    assert 'PropFlyingPaper' in rig
    assert 'phase-driven-hook-reveal-push-pull-impact-shake' in rig
    assert 'hook_text=candidate.get("hook", "")' in orch


def test_shock_paper_burst_tracks_contact_overshoot_recoil_and_settle():
    root=Path(__file__).resolve().parents[1]
    rig=(root/'tools'/'blender'/'pip_cutout_rig_v2.py').read_text(encoding='utf-8')
    assert "frame_at(0.58)" in rig  # contact launch
    assert "frame_at(0.72)" in rig  # overshoot arc
    assert "frame_at(0.84)" in rig  # recoil
    assert "frame_at(1.0)" in rig  # settle
    assert "burst.keyframe_insert('rotation_euler',index=2,frame=frame_at(0.84))" in rig
