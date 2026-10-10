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


def test_instagram_hashtag_trends_use_graph_hashtag_search(monkeypatch):
    from app.agents.trend_intelligence_v2 import discover_instagram_hashtag_trends, trend_source_status

    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN", raising=False)
    assert discover_instagram_hashtag_trends() == []
    assert trend_source_status()["instagram"]["active"] is False

    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "ig-token")
    monkeypatch.setenv("INSTAGRAM_USER_ID", "1784")
    monkeypatch.setenv("INSTAGRAM_TREND_HASHTAGS", "#cartoon")
    calls = []

    class Response:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self.data

    def fake_get(url, params):
        calls.append((url, params))
        if url.endswith("/ig_hashtag_search"):
            return Response({"data": [{"id": "h1"}]})
        return Response({"data": [{"id": "m1", "caption": "Robot tries a dance", "like_count": 900,
                                   "comments_count": 40, "timestamp": "2026-10-09T10:00:00+0000",
                                   "permalink": "https://instagram.example/p/m1"}]})

    rows = discover_instagram_hashtag_trends(http_get=fake_get)
    assert rows[0]["source"] == "instagram" and rows[0]["likes"] == 900
    assert calls[0][1]["q"] == "cartoon"
    assert calls[1][0].endswith("/h1/top_media")
    assert trend_source_status()["instagram"]["active"] is True
