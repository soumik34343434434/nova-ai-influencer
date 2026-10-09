from app.agents.subsystem_priority import SAFE_FALLBACK_ORDER, select_subsystem_priority


def test_competing_gaps_choose_highest_measured_impact_and_explain_score():
    progress = {"engines": {
        "performance_animation": {"status": "active", "score": 8.4,
                                  "evidence": "render review and regression results"},
        "analytics": {"status": "pending", "score": 3.0,
                      "evidence": "conversion tracking fails in 7/10 runs"},
    }}
    decision = select_subsystem_priority(progress, [
        {"name": "performance_animation", "failures": 1, "user_impact": 4,
         "evidence_confidence": 9},
        {"name": "analytics", "failures": 7, "user_impact": 10,
         "evidence_confidence": 9},
    ])
    assert decision["selected"] == "analytics"
    assert decision["priority_score"] > 0
    assert "failure_frequency" in decision["reason"]
    assert decision["ranked"][0]["components"]["quality_gap"] > 0


def test_good_subsystem_does_not_win_over_large_gap():
    progress = {"engines": {
        "audio": {"status": "active", "score": 9.5, "evidence": "verified"},
        "publishing": {"status": "pending", "score": 2.0, "evidence": "adapter failures"},
    }}
    decision = select_subsystem_priority(progress, [
        {"name": "audio", "user_impact": 5, "evidence_confidence": 10},
        {"name": "publishing", "failures": 5, "user_impact": 9,
         "evidence_confidence": 9},
    ])
    assert decision["selected"] == "publishing"


def test_incomplete_dependency_is_deprioritized_and_unblocker_is_explained():
    progress = {"engines": {
        "runtime_render_proof": {"status": "active"},
        "publishing": {"status": "pending"},
    }}
    decision = select_subsystem_priority(progress, [
        {"name": "publishing", "score": 2, "evidence": "blocked", "dependencies": ["runtime_render_proof"],
         "user_impact": 10, "evidence_confidence": 10},
        {"name": "runtime_render_proof", "score": 6, "evidence": "render smoke test absent",
         "user_impact": 9, "unblock_leverage": 10, "evidence_confidence": 7},
    ])
    assert decision["selected"] == "runtime_render_proof"
    publishing = next(row for row in decision["ranked"] if row["name"] == "publishing")
    assert publishing["blocked"] is True
    assert "runtime_render_proof" in publishing["unmet_dependencies"]


def test_missing_evidence_uses_safe_fallback_without_claiming_gap():
    decision = select_subsystem_priority({"engines": {}})
    assert decision["selected"] == SAFE_FALLBACK_ORDER[0]
    assert decision["fallback_order"] == list(SAFE_FALLBACK_ORDER)
    runtime = next(row for row in decision["ranked"] if row["name"] == "runtime_render_proof")
    assert runtime["evidence_present"] is False
    assert runtime["components"]["quality_gap"] == 0
    assert "evidence missing" in runtime["explanation"][-1]
