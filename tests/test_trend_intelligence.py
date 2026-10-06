from app.agents.trend_agent import Trend, rank_trends


def test_trend_intelligence_populates_fit_dimensions():
    trend = Trend(
        "AI workplace meeting meme goes viral",
        "test",
        published_at="2026-10-06T00:00:00+00:00",
        category="ai",
    )
    ranked = rank_trends([trend])
    item = ranked[0]
    assert 0 <= item.score <= 1
    assert item.freshness_score > 0
    assert item.relevance_score > 0
    assert item.comedy_score > 0
    assert item.visual_score > 0
    assert "freshness=" in item.why_now
