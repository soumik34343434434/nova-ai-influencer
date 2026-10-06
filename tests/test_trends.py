from app.agents.trend_agent import parse_rss, rank_trends

RSS = """<rss><channel>
<item><title>New AI tool goes viral</title><link>https://example.com/1</link><pubDate>Mon, 05 Oct 2026 10:00:00 GMT</pubDate></item>
<item><title>Creator economy news</title><link>https://example.com/2</link><pubDate>Mon, 05 Oct 2026 09:00:00 GMT</pubDate></item>
</channel></rss>"""

def test_parse_and_rank_rss():
    ranked = rank_trends(parse_rss(RSS, "test", "ai"))
    assert len(ranked) == 2
    assert ranked[0].score >= ranked[1].score
    assert ranked[0].title == "New AI tool goes viral"
