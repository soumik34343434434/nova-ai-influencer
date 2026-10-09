from app.agents.trend_opportunity import analyze_trend_transfer, score_content_opportunity
from app.agents.trend_intelligence_v2 import TrendSignalV2


def _signal(**kwargs):
    base = dict(
        trend_id="x", title="Funny cat meme", source="reddit", source_type="rss", url="https://example.com",
        published_at="2026-10-07T00:00:00+00:00", region="IN", language="en", category="meme",
        description="cute cat reaction meme about work", hashtags=("cat", "meme"), music="", duration_seconds=8,
        views=10000, likes=1000, comments=100, shares=300, source_count=2, mention_count=5,
        hook_patterns=("reaction", "reveal"), formats=("story_comedy", "reaction", "cute_animal"),
        emotions=("amusement", "surprise"), freshness_score=.9, velocity_score=.8,
        cross_platform_score=.7, saturation_score=.2, novelty_score=.8, opportunity_score=.8,
        confidence=.8, evidence=(), capability_notes=()
    )
    base.update(kwargs)
    return TrendSignalV2(**base)


def test_opportunity_prefers_character_fit_and_feasible_format():
    s = score_content_opportunity(_signal())
    assert 0 <= s["content_opportunity_score"] <= 1
    assert s["best_character"] == "MOMO"
    assert s["best_format"] in {"story_comedy", "reaction", "cute_animal"}
    assert s["character_fit"]["MOMO"] > s["character_fit"]["PIP"]


def test_opportunity_penalizes_low_feasibility_and_weak_hook():
    s = score_content_opportunity(_signal(
        title="Obscure technical announcement",
        description="technical documentation only",
        formats=("transformation",),
        hook_patterns=(),
        emotions=(),
        novelty_score=.1,
        confidence=.2,
    ))
    assert s["production_feasibility_score"] < .8
    assert s["hook_strength_score"] < .8
    assert s["content_opportunity_score"] < .7


def test_trend_signal_defaults_remain_backward_compatible():
    s = _signal()
    assert s.content_opportunity_score == 0.0
    assert s.best_character == ""
    assert s.best_format == ""


def test_trend_transfer_strong_fit_explains_nova_match():
    result = analyze_trend_transfer(_signal())
    assert result["score"] > .55
    assert result["best_format"] in {"story_comedy", "reaction", "cute_animal"}
    assert result["best_character"] == "MOMO"
    assert result["observed_evidence"]
    assert "trend_to_character" in result["inferred_fit"]
    assert result["reasons"]


def test_trend_transfer_weak_fit_is_cautious():
    result = analyze_trend_transfer(_signal(
        title="Obscure regulatory filing", description="Dense technical appendix",
        formats=(), emotions=(), hook_patterns=(), source_count=0, saturation_score=.9,
    ))
    assert result["score"] < .55
    assert any("No format signal" in x for x in result["missing_evidence_warnings"])
    assert any("owned content" in x for x in result["missing_evidence_warnings"])


def test_trend_transfer_penalizes_contradictory_format_evidence():
    result = analyze_trend_transfer(_signal(
        title="Dance trend with no dance and without music", formats=("dance", "trend_audio"),
    ))
    assert result["dimensions"]["trend_to_format"] < .5
    assert any("Contradiction" in x for x in result["missing_evidence_warnings"])


def test_trend_transfer_marks_missing_signals_and_integrates_output():
    from app.agents.trend_intelligence_v2 import enrich_signals
    sparse = _signal(title="Something", description="", formats=(), emotions=(), hook_patterns=(), source_count=0)
    result = analyze_trend_transfer(sparse)
    assert len(result["missing_evidence_warnings"]) >= 4
    enriched = enrich_signals([{"title": "POV: cute cat office reaction", "source_type": "reddit"}])[0]
    assert enriched.trend_transfer_analysis
    assert enriched.trend_transfer_analysis["score"] >= 0
