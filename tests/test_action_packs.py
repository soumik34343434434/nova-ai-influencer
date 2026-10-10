"""The action-pack registry must describe renderable assets for every cast member."""
import wave

from app.agents.action_packs import (
    ROLE_ORDER,
    action_file,
    character_actions,
    load_registry,
    regions_for_file,
    renderable_characters,
    role_actions,
)
from app.agents.cartoon_factory import action_asset
from app.providers.cartoon_audio import make_sfx
from app.providers.cartoon_rig import get_rig, rig_summary


def test_every_registered_action_image_exists():
    registry = load_registry()
    for character in registry["characters"]:
        for action in character_actions(character):
            assert action_file(character, action) is not None, f"{character}:{action} image missing"


def test_every_cast_member_covers_every_story_role():
    assert renderable_characters(("PIP", "MOMO", "BUBU")) == ("PIP", "MOMO", "BUBU")
    for character in ("PIP", "MOMO", "BUBU"):
        roles = role_actions(character)
        assert tuple(roles) == ROLE_ORDER
        # Four distinct poses keep every beat visually different.
        assert len(set(roles.values())) == len(ROLE_ORDER)


def test_action_asset_uses_registry_on_default_backend(monkeypatch):
    monkeypatch.delenv("NOVA_CARTOON_ANIMATION_BACKEND", raising=False)
    assert action_asset("MOMO", "walk_in").name == "momo_walk_in.png"
    assert action_asset("BUBU", "panic_run").name == "bubu_panic_run.png"
    assert action_asset("PIP", "press_button").name == "pip_press_button.png"
    # Legacy office actions were never drawn for the cartoon cast.
    assert action_asset("PIP", "checks_phone") is None


def test_generated_packs_carry_rig_regions_used_by_the_renderer():
    registry = load_registry()
    for character in ("MOMO", "BUBU"):
        for spec in registry["characters"][character]["actions"].values():
            regions = regions_for_file(spec["file"])
            assert regions is not None
            for x0, y0, x1, y1 in regions.values():
                assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
    rig = get_rig(str(action_file("MOMO", "deadpan")))
    summary = rig_summary(rig)
    assert summary["manifest"] is True
    assert rig.character == "MOMO"


def test_sound_cues_used_by_ideas_are_audible(tmp_path):
    for kind in ("notification", "pop", "whoosh", "tap", "beat", "countdown", "impact", "chaos", "overwhelmed", "reaction_hit", "punchline"):
        path = make_sfx(kind, 0.4, tmp_path / f"{kind}.wav")
        with wave.open(str(path)) as wav:
            frames = wav.readframes(wav.getnframes())
        assert any(frames[i] or frames[i + 1] for i in range(0, len(frames), 2)), kind
