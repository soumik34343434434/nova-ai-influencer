from app.agents.content_variants import generate_variants
from app.agents.content_evaluator import ContentEvaluator


def test_three_angles_are_distinct_and_evaluable():
    variants = generate_variants("AI agents are becoming everyone's new coworker")
    assert len(variants) == 3
    assert {v["angle"] for v in variants} == {"workplace", "pov", "experiment"}
    assert len({v["hook"] for v in variants}) == 3
    evaluator = ContentEvaluator()
    results = [evaluator.evaluate(v) for v in variants]
    assert all(r.passed for r in results)


def test_variant_engine_is_deterministically_reordered_per_topic():
    a = generate_variants("AI agents are becoming everyone's new coworker")
    b = generate_variants("AI agents are becoming everyone's new coworker")
    assert [x["angle"] for x in a] == [x["angle"] for x in b]
