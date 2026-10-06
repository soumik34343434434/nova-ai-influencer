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
