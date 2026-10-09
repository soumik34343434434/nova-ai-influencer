from app.providers.runway_voice import build_runway_speech_request, render_runway_script


def test_runway_voice_uses_expressive_pause_timing_by_beat():
    assert render_runway_script("We have a problem", "escalation").endswith("[pause]")
    assert render_runway_script("Seriously", "reaction").startswith("[pause]")
    assert "[pause]" in render_runway_script("Apparently this is fine", "punchline")


def test_runway_speech_payload_is_router_compatible_and_dry_run_safe():
    payload = build_runway_speech_request("Hello", "voice-config", voice={"type": "runway-preset", "presetId": "demo"}, duration=2.0, dry_run=True)
    assert payload["configId"] == "voice-config"
    assert payload["dryRun"] is True
    assert payload["input"]["type"] == "speech"
    assert payload["input"]["promptText"] == "Hello"
    assert payload["input"]["duration"] == 2.0


def test_runway_script_enforces_current_expressive_length_contract():
    try:
        build_runway_speech_request("x" * 2501, "cfg")
    except ValueError as exc:
        assert "2500" in str(exc)
    else:
        raise AssertionError("expected script length guard")
