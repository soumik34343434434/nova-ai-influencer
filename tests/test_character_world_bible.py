from copy import deepcopy

from app.agents.character_world_bible import (
    load_bible,
    validate_bible,
    validate_character_plan,
    validate_storyboard_continuity,
)
from app.agents.storyboard import build_storyboard


def test_canonical_bible_entries_are_complete_and_plan_valid():
    bible = load_bible()
    assert validate_bible(bible) == []
    for name in ("PIP", "MOMO", "BUBU"):
        assert validate_character_plan(name, {}, bible) == []


def test_identity_and_style_transformation_violation_is_detected():
    errors = validate_character_plan("PIP", {"transformations": ["recolor_identity"]})
    assert "identity_violation:PIP:recolor_identity" in errors


def test_missing_bible_fields_are_reported():
    bible = deepcopy(load_bible())
    del bible["characters"]["MOMO"]["signature_props"]
    assert "missing_field:characters.MOMO.signature_props" in validate_bible(bible)


def test_cross_shot_character_change_is_reported():
    errors = validate_storyboard_continuity([
        {"character": "PIP", "character_plan": {}},
        {"character": "MOMO", "character_plan": {}},
    ])
    assert "continuity_violation:shot_1:character_changed" in errors


def test_storyboard_embeds_canonical_character_and_validates():
    board = build_storyboard({
        "character": "PIP",
        "hook": "One sandwich, please.",
        "script": "PIP asks for a sandwich.\n\nPIP looks at the result.",
    }).to_dict()
    assert board["validation"] == {"passed": True, "errors": []}
    assert board["shots"][0]["world"]["identity_lock"]["signature_props"] == ["orange backpack"]
    assert board["shots"][0]["world"]["lighting"]
