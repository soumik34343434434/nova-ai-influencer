from app.agents.cartoon_factory import CAST, coverage, list_episode_specs, quality_gate


def test_episode_factory_returns_multiple_original_episodes():
    specs = list_episode_specs(5)
    assert len(specs) == 5
    assert len({s.slug for s in specs}) == 5
    assert all(s.character in CAST for s in specs)


def test_pip_episode_has_full_action_coverage():
    spec = next(s for s in list_episode_specs(5) if s.slug == "smart_kitchen")
    assert coverage(spec) == 1.0
    allowed, reasons = quality_gate(spec)
    assert allowed, reasons


def test_unready_character_is_blocked_until_action_pack_exists():
    spec = next(s for s in list_episode_specs(5) if s.slug == "homework_robot")
    assert coverage(spec) == 0.0
    allowed, reasons = quality_gate(spec)
    assert not allowed
    assert "missing_action_asset_coverage" in reasons


def test_blender_backend_accepts_compiled_character_master(monkeypatch):
    from app.agents.cartoon_factory import action_asset
    monkeypatch.setenv("NOVA_CARTOON_ANIMATION_BACKEND", "blender")
    assert action_asset("MOMO", "give_homework") is not None
    assert action_asset("BUBU", "tap_alarm") is not None


def test_cartoon_quality_targets_preserve_strong_areas_and_raise_weak_areas():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'config'/'cartoon_quality_targets.json').read_text(encoding='utf-8'))
    areas=data['areas']
    weak={'animation_mechanics','anticipation_contact_overshoot','secondary_motion','facial_acting','camera_acting','background_visual_world','comedy_storytelling','social_video_potential'}
    assert all(areas[name]['minimum'] >= 8.5 for name in weak)
    for name,area in areas.items():
        if area['protected']:
            assert area['minimum'] >= area['baseline']
