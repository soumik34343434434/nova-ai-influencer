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


def _cover_and_bbox(character, action):
    from PIL import Image

    from app.agents.action_packs import action_file
    from app.providers.cartoon_rig import _cover, get_rig

    path = action_file(character, action)
    with Image.open(path) as image:
        return _cover(image), get_rig(str(path)).bbox


def test_every_action_image_matches_its_character_palette():
    from app.agents.action_packs import character_actions
    from app.agents.character_world_bible import validate_asset_identity

    for character in ("PIP", "MOMO", "BUBU"):
        for action in character_actions(character):
            image, bbox = _cover_and_bbox(character, action)
            assert validate_asset_identity(character, image, bbox) == [], (character, action)


def test_swapped_character_art_is_rejected():
    from app.agents.character_world_bible import validate_asset_identity

    momo, momo_box = _cover_and_bbox("MOMO", "deadpan")
    bubu, bubu_box = _cover_and_bbox("BUBU", "deadpan")
    pip, pip_box = _cover_and_bbox("PIP", "deadpan")
    assert validate_asset_identity("BUBU", momo, momo_box)
    assert validate_asset_identity("MOMO", bubu, bubu_box)
    # PIP and BUBU are both blue; PIP's orange backpack share separates them.
    assert validate_asset_identity("PIP", bubu, bubu_box)
    assert validate_asset_identity("BUBU", pip, pip_box)


def test_close_ups_may_crop_signature_props_but_wider_shots_may_not():
    from app.agents.character_world_bible import validate_storyboard_continuity

    plan = {"transformations": ["pose", "gaze"]}
    close = [{"character": "BUBU", "framing": "close_up", "character_plan": plan, "signature_props": []}]
    wide = [{"character": "BUBU", "framing": "wide", "character_plan": plan, "signature_props": []}]
    assert validate_storyboard_continuity(close) == []
    assert any("missing_signature_props" in e for e in validate_storyboard_continuity(wide))


def test_storyboard_records_real_bible_plans_per_shot():
    from app.agents.storyboard import build_storyboard

    beats = [
        {"beat": "hook", "asset_action": "wave_hello", "dialogue": "BUBU is trying to help. Then it happened!", "duration": 3.0},
        {"beat": "escalation", "asset_action": "panic_run", "dialogue": "Then the plan grows. Twelve problems now!", "duration": 3.0},
        {"beat": "reaction", "asset_action": "worried_close", "dialogue": "Wait, why? Turns out the help was the problem?", "duration": 3.0, "framing": "close_up"},
        {"beat": "punchline", "asset_action": "deadpan", "dialogue": "Still counts as helping. Apparently.", "duration": 2.5},
    ]
    board = build_storyboard({"title": "t", "hook": beats[0]["dialogue"], "character": "BUBU", "script": "", "beats": beats})
    data = board.to_dict()
    assert data["validation"]["passed"]
    assert [s.action for s in board.shots] == [b["asset_action"] for b in beats]
    assert board.shots[2].world["framing"] == "close_up"
    assert board.shots[0].world["signature_props"] == ["tiny yellow backpack"]
    assert all(s.visual.startswith("BUBU ") for s in board.shots)
