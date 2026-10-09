from app.agents.idea_prediction import predict_idea
from app.agents.cartoon_jury import CartoonIdeaJury


def _idea(strong=True):
    if strong:
        return {
            "title": "PIP and the smart kitchen", "hook": "PIP asked for one sandwich. Big mistake!",
            "character": "PIP", "format_name": "story_comedy", "emotion": "amusement",
            "duration_target": 8, "trend_score": .9, "saturation_score": .1, "production_cost": .3,
            "beats": [
                {"beat": "hook", "asset_action": "press", "voice": "One sandwich"},
                {"beat": "setup", "asset_action": "wait", "voice": "It starts"},
                {"beat": "escalation", "asset_action": "shock", "voice": "Twelve?!"},
                {"beat": "punchline", "asset_action": "deadpan", "voice": "Big mistake"},
            ],
        }
    return {"hook": "Something happened.", "duration_target": 20, "trend_score": .1,
            "saturation_score": .9, "production_cost": .9, "beats": [{"asset_action": "wait"}]}


def test_empty_history_is_explicitly_insufficient_and_not_analytics():
    result = predict_idea(_idea(), [])
    assert result["evidence_status"] == "insufficient_evidence"
    assert result["historical_observations"] == 0
    assert result["observed_analytics"] is False
    assert result["estimate_type"] == "pre_production_prediction"


def test_strong_idea_scores_above_weak_idea():
    strong, weak = predict_idea(_idea(), []), predict_idea(_idea(False), [])
    assert strong["predicted_retention"] > weak["predicted_retention"]
    assert strong["predicted_shareability"] > weak["predicted_shareability"]


def test_sparse_history_shrinks_estimate_and_keeps_low_confidence():
    row = {"content_id": "x", "views": 1000, "shares": 10, "retention_pct": 45,
           "format_name": "story_comedy", "character": "PIP", "emotion": "amusement"}
    result = predict_idea(_idea(), [row])
    assert result["historical_observations"] == 1
    assert result["evidence_status"] == "insufficient_evidence"
    assert result["confidence"] < .5


def test_prediction_values_and_uncertainty_are_bounded_and_jury_separate():
    result = predict_idea(_idea(), [])
    for key in ("predicted_retention", "predicted_shareability", "uncertainty", "confidence"):
        assert 0 <= result[key] <= 1
    unsafe = _idea()
    unsafe["title"] += " weapon"
    jury = CartoonIdeaJury().judge(unsafe)
    assert jury.prediction["estimate_type"] == "pre_production_prediction"
    assert not jury.passed
    assert "unsafe_or_reuse_signal" in jury.reasons
