from datetime import datetime, timezone

from app.agents.trend_intelligence_v2 import (
    TrendSignalV2,
    enrich_signals,
    extract_hook_patterns,
    infer_emotions,
    infer_formats,
)
from app.agents.format_engine_v2 import infer_format_patterns, rank_format_candidates
from app.agents.idea_generator_v2 import generate_ideas
from app.learning_v2 import build_strategy_memory, ingest, score_candidate, select_portfolio
from app.trend_history import attach_historical_signals


def _trend():
    return {
        "trend_id": "trend-test",
        "title": "POV: cute cat dance reaction to a viral song",
        "source": "test",
        "source_type": "test",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "category": "social",
        "views": 100000,
        "likes": 7000,
        "comments": 500,
        "shares": 3000,
        "source_count": 2,
        "mention_count": 3,
        "formats": ("dance", "reaction", "trend_audio"),
        "emotions": ("cute", "amusement", "surprise"),
        "hook_patterns": ("pov", "reaction"),
        "opportunity_score": 0.82,
        "velocity_score": 0.91,
        "saturation_score": 0.22,
    }


def test_hook_emotion_and_format_detection():
    text = _trend()["title"]
    assert "pov" in extract_hook_patterns(text)
    assert "dance" in infer_formats(text, "social")
    assert "cute" in infer_emotions(text)


def test_enrich_signals_builds_opportunity_and_saturation_proxy():
    rows = enrich_signals([_trend()])
    assert rows
    row = rows[0]
    assert 0 <= row.opportunity_score <= 1
    assert 0 <= row.saturation_score <= 1
    assert row.trend_id.startswith("trend-")


def test_trend_clustering_uses_shared_music_for_differently_worded_posts():
    rows = enrich_signals([
        {
            **_trend(),
            "title": "POV: cute cat dance reaction to a viral song",
            "music": "music-123",
            "hashtags": ("#catdance", "#viral"),
            "source_type": "youtube",
        },
        {
            **_trend(),
            "title": "Cat tries the viral dance reaction challenge",
            "music": "music-123",
            "hashtags": ("#catdance", "#challenge"),
            "source_type": "tiktok_research",
        },
    ])
    assert len(rows) == 1
    assert rows[0].mention_count == 2
    assert rows[0].source_count == 2


def test_trend_clustering_does_not_merge_generic_unrelated_posts():
    rows = enrich_signals([
        {**_trend(), "title": "Cute cat sleeps on a sofa", "music": "music-a"},
        {**_trend(), "title": "Cute cat wins a dance contest", "music": "music-b"},
    ])
    assert len(rows) == 2


def test_historical_snapshot_changes_only_with_matching_history():
    row = _trend()
    row["trend_id"] = "same"
    out = attach_historical_signals([row], [{
        "trend_id": "same",
        "views": 50000,
        "captured_at": "2026-10-06T00:00:00+00:00"
    }, {
        "trend_id": "same",
        "views": 60000,
        "captured_at": "2026-10-06T12:00:00+00:00"
    }])
    assert out[0]["historical_snapshot_count"] == 2
    assert out[0]["historical_growth_rate"] > 0


def test_historical_signals_recompute_opportunity_score():
    row = enrich_signals([_trend()])[0].__dict__
    history = [
        {"trend_id": row["trend_id"], "views": 100000, "captured_at": "2026-10-06T00:00:00+00:00"},
        {"trend_id": row["trend_id"], "views": 110000, "captured_at": "2026-10-06T12:00:00+00:00"},
        {"trend_id": row["trend_id"], "views": 112000, "captured_at": "2026-10-07T00:00:00+00:00"},
        {"trend_id": row["trend_id"], "views": 90000, "captured_at": "2026-10-07T12:00:00+00:00"},
        {"trend_id": row["trend_id"], "views": 95000, "captured_at": "2026-10-08T00:00:00+00:00"},
    ]
    baseline = row["opportunity_score"]
    updated = attach_historical_signals([row], history)[0]
    assert updated["historical_snapshot_count"] == 5
    assert updated["historical_growth_rate"] > 0
    assert updated["opportunity_score"] != baseline
    assert updated["content_opportunity_score"] != row["content_opportunity_score"]


def test_format_engine_uses_learned_memory():
    signal = _trend()
    memory = build_strategy_memory([])
    patterns = infer_format_patterns(signal, memory)
    assert patterns
    ranked = rank_format_candidates([signal], memory)
    assert ranked


def test_generator_produces_a_large_diverse_pool():
    ideas = generate_ideas([_trend()], count=60)
    assert len(ideas) >= 20
    assert len({x["format_name"] for x in ideas}) >= 3
    assert len({x["character"] for x in ideas}) == 3


def test_generator_draws_ideas_from_every_trend():
    topics = ("coffee", "deadline", "printer", "monday", "wifi", "meeting", "laundry", "alarm", "pizza", "homework")
    trends = [
        {**_trend(), "trend_id": f"t{i}", "title": f"viral {topic} chaos meme", "opportunity_score": 1 - i * 0.05}
        for i, topic in enumerate(topics)
    ]
    ideas = generate_ideas(trends, count=100)
    assert len(ideas) == 100
    assert {x["trend_id"] for x in ideas} == {f"t{i}" for i in range(10)}


def test_generator_only_stages_renderable_actions():
    pip_actions = ("press_button", "shock", "overwhelmed", "deadpan")
    ideas = generate_ideas([_trend()], count=60, renderable_actions={"PIP": pip_actions, "MOMO": ()})
    assert ideas
    # MOMO has no renderable actions and BUBU is absent from the mapping.
    assert {x["character"] for x in ideas} == {"PIP"}
    for idea in ideas:
        actions = [beat["asset_action"] for beat in idea["beats"]]
        assert set(actions) <= set(pip_actions)
        assert len(set(actions)) >= 3


def test_formats_stage_different_action_sequences():
    ideas = generate_ideas([_trend()], count=120)
    sequences = {x["format_name"]: tuple(b["asset_action"] for b in x["beats"]) for x in ideas}
    assert len(set(sequences.values())) >= 4


def test_learning_scores_and_portfolio_are_diversity_aware(tmp_path, monkeypatch):
    import app.learning_v2 as learning

    monkeypatch.setattr(learning, "METRICS_FILE", tmp_path / "metrics.json")
    monkeypatch.setattr(learning, "STRATEGY_FILE", tmp_path / "strategy.json")
    ingest([
        {"content_id":"a","platform":"instagram","views":10000,"likes":900,"shares":500,"retention_pct":82,"format_name":"dance","hook_pattern":"pov","emotion":"cute","character":"MOMO"},
        {"content_id":"b","platform":"instagram","views":10000,"likes":500,"shares":20,"retention_pct":40,"format_name":"story_comedy","hook_pattern":"reaction","emotion":"frustration","character":"PIP"},
        {"content_id":"c","platform":"instagram","views":10000,"likes":750,"shares":180,"retention_pct":72,"format_name":"dance","hook_pattern":"pov","emotion":"cute","character":"BUBU"},
    ])
    memory = build_strategy_memory()
    assert memory["observations"] == 3
    assert "dance" in memory["strategy"]["preferred_formats"]
    assert "interactions" in memory["posteriors"]
    assert "format_name__character" in memory["posteriors"]["interactions"]
    assert "dance|MOMO" in memory["posteriors"]["interactions"]["format_name__character"]
    assert score_candidate({"format_name":"dance","hook_pattern":"pov","emotion":"cute","character":"MOMO","trend_score":0.8}, memory) > 0.5
    selected = select_portfolio([
        {"idea_id":"1","format_name":"dance","character":"MOMO","trend_score":0.8,"saturation_score":0.1},
        {"idea_id":"2","format_name":"story_comedy","character":"PIP","trend_score":0.7,"saturation_score":0.1},
        {"idea_id":"3","format_name":"reaction","character":"BUBU","trend_score":0.7,"saturation_score":0.1},
    ], memory=memory, limit=3)
    assert len(selected) == 3
    assert len({(x["format_name"], x["character"]) for x in selected}) == 3
