import json

from app.agents.storyboard import build_storyboard
from app.agents.director_decision import validate_packet


def test_storyboard_creates_multiple_visual_beats():
    draft = {
        "hook": "Your office installed an AI agent and somehow created three new meetings.",
        "script": "Manager: Can the AI do this?\n\nNOVA checks the phone and sees a new calendar invite.\n\nNOVA: The work is finished.\n\nNOVA stares at camera.",
    }
    board = build_storyboard(draft)
    assert len(board.shots) >= 4
    assert 10 <= board.total_duration <= 18
    assert board.shots[0].camera == "hook_close_up"
    assert board.shots[-1].action == "deadpan_camera_reaction"
    assert abs(sum(s.duration for s in board.shots) - board.total_duration) < 0.1


def test_storyboard_emits_coherent_serializable_director_packets():
    board = build_storyboard({"hook": "A tiny hook", "script": "A tiny hook\n\nNOVA checks a phone."})
    data = board.to_dict()
    data = json.loads(json.dumps(data))
    shot = data["shots"][0]
    packet = shot["director_decision"]
    assert packet["version"] == "1.0"
    assert packet["camera"]["framing"] == shot["camera"]
    assert packet["timing"]["duration"] == shot["duration"]
    assert packet["story"]["intent"]
    assert packet["world"]["scene"]
    assert packet["priorities"]["camera_actor_world"] == "camera > actor > world"
    assert packet["continuity_locks"] and packet["negative_constraints"]
    assert packet["audio"]["emphasis"] == "dialogue-led"
    assert packet["timing"]["phases"]
    assert packet["handoff"]["contract"] == "nova.director_decision.v1"


def test_director_packet_rejects_camera_and_actor_conflict():
    board = build_storyboard({"hook": "Hook", "script": "Hook"})
    packet_data = board.shots[0].director_decision
    from dataclasses import fields
    from app.agents.director_decision import DirectorDecisionPacket
    packet_data["character_performance"]["camera"] = "wide_desk"
    packet = DirectorDecisionPacket(**{f.name: packet_data[f.name] for f in fields(DirectorDecisionPacket)})
    try:
        validate_packet(packet)
    except ValueError as exc:
        assert "camera framing" in str(exc)
    else:
        raise AssertionError("conflicting camera intent should fail validation")


def test_optional_director_directives_can_be_absent():
    board = build_storyboard({"hook": "Hook", "script": "Hook"})
    assert board.shots[0].director_decision["audio"]["sfx"] == ""
