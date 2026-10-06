from app.agents.content_variants import generate_variants
from app.agents.content_evaluator import ContentEvaluator
from app.agents.trend_agent import Trend, rank_trends
from app.orchestrator import SafetyGate, select_best_variant


def test_best_variant_selection_returns_one_winner_and_alternatives():
    trend = rank_trends([Trend("AI workplace agents create meeting chaos", "test", category="ai")])[0]
    variants = generate_variants(trend.title)
    for v in variants:
        v.update(originality_score=0.91, risk_score=0.02)
    selected = select_best_variant(variants, trend, ContentEvaluator(), SafetyGate(), [])
    assert selected is not None
    score, evaluation, winner, alternatives = selected
    assert 0 <= score <= 1
    assert evaluation.passed
    assert winner["angle"] in {"workplace", "pov", "experiment"}
    assert len(alternatives) == 2
