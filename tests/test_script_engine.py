from app.agents.script_engine import QUALITY_FLOOR, ScriptEngine


def _story():
    return {
        "premise": "PIP vs the Smart Kitchen",
        "hook": "PIP asked for one sandwich. Big mistake!",
        "character": "PIP",
        "duration_target": 9.0,
        "beats": [
            {"story_role": "setup", "asset_action": "press_button", "voice": "One sandwich, please.", "visual": "PIP presses the cooking button.", "story_intent": "establish the character's goal and visible context", "sfx": "tap"},
            {"story_role": "escalation", "asset_action": "shock", "voice": "TWELVE?! The kitchen made twelve.", "visual": "A huge stack of sandwiches appears.", "story_intent": "make the problem larger, stranger, or harder", "sfx": "impact"},
            {"story_role": "reaction", "asset_action": "overwhelmed", "voice": "PIP stares at the giant pile.", "visual": "PIP looks overwhelmed.", "story_intent": "show the character processing the consequence", "sfx": "reaction_hit"},
            {"story_role": "payoff", "asset_action": "deadpan", "voice": "PIP: The kitchen misunderstood one.", "visual": "PIP gives a deadpan look to camera.", "story_intent": "land the reversal and end decisively", "sfx": "punchline"},
        ],
    }


def test_script_engine_reaches_quality_floor():
    contract = ScriptEngine().build(_story()).to_dict()
    quality = contract["quality"]
    assert quality["score"] >= QUALITY_FLOOR
    assert quality["min_dimension_score"] >= QUALITY_FLOOR
    assert quality["all_dimensions_above_floor"] is True


def test_script_engine_outputs_production_contract():
    contract = ScriptEngine().build(_story()).to_dict()
    assert len(contract["beats"]) == 4
    assert contract["text"]
    assert contract["hook"] == "PIP asked for one sandwich. Big mistake!"
    assert abs(sum(row["duration"] for row in contract["beats"]) - 9.0) <= 0.01
    for row in contract["beats"]:
        assert row["dialogue"]
        assert row["voice_text"] == row["dialogue"]
        assert row["caption"]
        assert row["word_count"] <= 18
        assert len(row["caption"].split()) <= 8
        assert len(row["caption"]) <= 42
        assert row["sfx"]
        assert row["delivery"]
        assert row["performance"]["emotion"]
        assert row["performance"]["gaze"]
        assert row["timing_end"] > row["timing_start"]
    assert contract["beats"][-1]["story_role"] == "payoff"
    assert contract["beats"][-1]["caption_style"] == "punchline"


def test_script_engine_keeps_four_beat_story_roles():
    contract = ScriptEngine().build(_story()).to_dict()
    assert [row["story_role"] for row in contract["beats"]] == ["setup", "escalation", "reaction", "payoff"]


def test_script_engine_rejects_too_few_story_beats():
    story = _story()
    story["beats"] = story["beats"][:3]
    try:
        ScriptEngine().build(story)
    except ValueError as exc:
        assert "at least four beats" in str(exc)
    else:
        raise AssertionError("Expected Script Engine to reject an underspecified story")


def test_five_beats_fit_minimum_supported_duration():
    story = _story()
    story["duration_target"] = 5.0
    story["beats"].insert(0, {
        "story_role": "hook",
        "asset_action": "checks_phone",
        "voice": "Why is the kitchen texting me?",
        "visual": "PIP checks the phone.",
        "story_intent": "open with an immediate question",
        "sfx": "room",
    })

    contract = ScriptEngine().build(story).to_dict()
    beats = contract["beats"]

    assert len(beats) == 5
    assert abs(sum(beat["duration"] for beat in beats) - 5.0) < 0.001
    assert all(beat["duration"] >= 0.999 for beat in beats)
    assert all(abs(left["timing_end"] - right["timing_start"]) < 0.002
               for left, right in zip(beats, beats[1:]))
    assert beats[-1]["timing_end"] == 5.0
