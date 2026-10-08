from app.cartoon_factory_orchestrator import CartoonFactoryOrchestrator
from app.cartoon_control import CartoonRun


class _Store:
    def __init__(self):
        self.payload_updates = []

    def set_payload(self, run, **values):
        run.payload.update(values)
        self.payload_updates.append(values)


def _run():
    return CartoonRun(
        run_id="integration-script-test",
        created_at="",
        current_stage="6_script_engine",
        status="running",
        payload={
            "winner": {
                "candidate": {
                    "title": "PIP vs the Smart Kitchen",
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
            }
        },
        stages={},
    )


def test_orchestrator_uses_structured_script_contract():
    orchestrator = CartoonFactoryOrchestrator()
    store = _Store()
    orchestrator.plane.store = store
    run = _run()

    artifact = orchestrator._script_engine(run)

    assert hasattr(orchestrator, "script_engine")
    assert "script_contract" in run.payload
    contract = run.payload["script_contract"]
    assert contract["quality"]["min_dimension_score"] >= 8.5
    assert contract["quality"]["all_dimensions_above_floor"] is True
    assert contract["text"] == run.payload["script"]
    assert contract["hook"] == run.payload["hook"]
    assert artifact["status"] == "passed"
    assert artifact["quality_score"] >= 8.5
