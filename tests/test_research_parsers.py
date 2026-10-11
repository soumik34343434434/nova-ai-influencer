"""Parsers for the research sources, checked against realistic payload shapes.

Each fixture mirrors what the upstream service really returns: Google News RSS with a
publisher <source>, Reddit Atom (not RSS), GDELT DOC 2.0 JSON, GDELT's HTTP-200 text
errors, Google Trends RSS with the ht: namespace, and YouTube contentDetails. No test
touches the network or the operator's environment.
"""
import json

import httpx
import pytest

import app.agents.cartoon_research as research
import app.agents.trend_intelligence_v2 as trends
from app.agents.trend_intelligence_v2 import (
    enrich_signals,
    extract_hook_patterns,
    infer_emotions,
    infer_formats,
)

GOOGLE_NEWS_RSS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/"><channel>
<generator>NFE/5.0</generator><title>"robot" - Google News</title>
<link>https://news.google.com/search?q=robot</link><language>en-IN</language>
<item><title>Robot vacuum goes viral after chasing a cat around the house - Times of India</title>
<link>https://news.google.com/rss/articles/CBMiqgFBVV95cUxPb3JvYm90?oc=5</link>
<guid isPermaLink="false">CBMiqgFBVV95cUxPb3JvYm90</guid>
<pubDate>Sat, 10 Oct 2026 06:12:00 GMT</pubDate>
<description>&lt;a href="https://news.google.com/rss/articles/CBMiqgFB?oc=5" target="_blank"&gt;Robot vacuum goes viral after chasing a cat around the house&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font color="#6f6f6f"&gt;Times of India&lt;/font&gt;</description>
<source url="https://timesofindia.indiatimes.com">Times of India</source></item>
<item><title>Gunman kills 12 in school shooting - NDTV</title>
<link>https://news.google.com/rss/articles/CBMiqgFBVV95cUxPZ3Vubg?oc=5</link>
<pubDate>Sat, 10 Oct 2026 05:02:00 GMT</pubDate>
<description>&lt;a href="https://news.google.com/rss/articles/CBMiqgFB?oc=5"&gt;Gunman kills 12 in school shooting&lt;/a&gt;</description>
<source url="https://www.ndtv.com">NDTV</source></item>
</channel></rss>"""

REDDIT_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:media="http://search.yahoo.com/mrss/">
<category term="cats" label="r/cats"/><updated>2026-10-10T06:00:00+00:00</updated>
<id>/r/cats/.rss</id><link rel="self" href="https://www.reddit.com/r/cats/.rss" type="application/atom+xml" />
<title>cats</title>
<entry><author><name>/u/someone</name><uri>https://www.reddit.com/user/someone</uri></author>
<category term="cats" label="r/cats"/><content type="html">&lt;p&gt;my cat vs the new robot vacuum&lt;/p&gt;</content>
<id>t3_abc123</id><link href="https://www.reddit.com/r/cats/comments/abc123/x/" />
<updated>2026-10-10T05:00:00+00:00</updated><published>2026-10-10T05:00:00+00:00</published>
<title>My cat declared war on the new robot vacuum</title></entry>
<entry><id>t3_def456</id><link href="https://www.reddit.com/r/cats/comments/def456/y/" />
<updated>2026-10-10T04:00:00+00:00</updated><published>2026-10-10T04:00:00+00:00</published>
<title>POV: you tried to give your cat a bath</title></entry>
</feed>"""

EMPTY_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>robots</title></feed>"""

GDELT_JSON = json.dumps({"articles": [
    {"url": "https://example.in/a1", "title": "Smart home robot dog learns to open the fridge",
     "seendate": "20261010T051500Z", "domain": "example.in", "language": "English"},
    {"url": "https://example.in/a2", "title": "Robot cooks the best sandwich at a school fair",
     "seendate": "20261010T043000Z", "domain": "example.in", "language": "English"},
]})
GDELT_TEXT_ERROR = "Queries containing OR'd terms must be surrounded by ()."
GDELT_RATE_LIMIT = "Please limit requests to one every 5 seconds."

GOOGLE_TRENDS_RSS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<rss xmlns:atom="http://www.w3.org/2005/Atom" xmlns:ht="https://trends.google.com/trending/rss" version="2.0">
<channel><title>Daily Search Trends</title><description>Recent searches</description>
<link>https://trends.google.com/trending/rss?geo=IN</link>
<item><title>india vs australia</title><ht:approx_traffic>2000000+</ht:approx_traffic><description></description>
<link>https://trends.google.com/trending/rss?geo=IN</link><pubDate>Sat, 10 Oct 2026 03:30:00 -0700</pubDate>
<ht:picture>https://encrypted-tbn0.gstatic.com/x</ht:picture><ht:picture_source>ESPN</ht:picture_source>
<ht:news_item><ht:news_item_title>India vs Australia live score</ht:news_item_title><ht:news_item_snippet></ht:news_item_snippet>
<ht:news_item_url>https://espn.example/1</ht:news_item_url><ht:news_item_picture>x</ht:news_item_picture><ht:news_item_source>ESPN</ht:news_item_source></ht:news_item></item>
<item><title>salman khan</title><ht:approx_traffic>500000+</ht:approx_traffic><description></description>
<link>https://trends.google.com/trending/rss?geo=IN</link><pubDate>Sat, 10 Oct 2026 02:30:00 -0700</pubDate></item>
<item><title>earthquake today</title><ht:approx_traffic>200000+</ht:approx_traffic><description></description>
<link>https://trends.google.com/trending/rss?geo=IN</link><pubDate>Sat, 10 Oct 2026 01:30:00 -0700</pubDate></item>
</channel></rss>"""


class FakeResponse:
    def __init__(self, text: str = "", status_code: int = 200, content_type: str = "application/xml"):
        self.text = text
        self.status_code = status_code
        self.headers = {"content-type": content_type}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("GET", "https://example.test"),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        return json.loads(self.text)


def _route(monkeypatch, module, handler):
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None, **kwargs):
        calls.append({"url": url, "params": params, "headers": dict(headers or {})})
        return handler(url, headers or {})

    monkeypatch.setattr(module.httpx, "get", fake_get)
    return calls


def test_google_news_rss_strips_publisher_and_markup(monkeypatch):
    _route(monkeypatch, research, lambda url, headers: FakeResponse(GOOGLE_NEWS_RSS))
    signals = research._rss("https://news.google.com/rss/search?q=x", "ai_gadgets", timeout=1)
    assert [s.title for s in signals] == [
        "Robot vacuum goes viral after chasing a cat around the house",
        "Gunman kills 12 in school shooting",
    ]
    assert signals[0].published_at.startswith("2026-10-10T06:12:00")


def test_google_news_rows_drop_repeated_headline_description(monkeypatch):
    _route(monkeypatch, trends, lambda url, headers: FakeResponse(GOOGLE_NEWS_RSS))
    rows = trends._rss_items("https://news.google.com/rss/search?q=x", "google_social_search", "instagram_mentions")
    assert rows[0]["title"] == "Robot vacuum goes viral after chasing a cat around the house"
    assert rows[0]["description"] == ""
    assert rows[0]["source_type"] == "google_social_search"


def test_reddit_atom_entries_parse_with_href_links_and_dates(monkeypatch):
    calls = _route(monkeypatch, trends, lambda url, headers: FakeResponse(REDDIT_ATOM, content_type="application/atom+xml"))
    rows = trends._rss_items("https://www.reddit.com/r/cats/.rss", "reddit", "reddit_cats",
                             user_agent=trends.REDDIT_USER_AGENT)
    assert [row["title"] for row in rows] == [
        "My cat declared war on the new robot vacuum",
        "POV: you tried to give your cat a bath",
    ]
    assert rows[0]["url"] == "https://www.reddit.com/r/cats/comments/abc123/x/"
    assert rows[0]["published_at"] == "2026-10-10T05:00:00+00:00"
    assert rows[0]["description"] == "my cat vs the new robot vacuum"
    assert calls[0]["headers"]["User-Agent"] == trends.REDDIT_USER_AGENT


def test_reddit_connector_reports_each_subreddit_outcome(monkeypatch):
    def handler(url, headers):
        if "/r/aww/" in url:
            return FakeResponse("<html>blocked</html>", status_code=403, content_type="text/html")
        if "/r/robots/" in url:
            return FakeResponse(EMPTY_ATOM, content_type="application/atom+xml")
        return FakeResponse(REDDIT_ATOM, content_type="application/atom+xml")

    calls = _route(monkeypatch, trends, handler)
    status: dict[str, str] = {}
    rows = trends.discover_reddit_rss(timeout=1, status=status, pause_seconds=0)
    assert status["aww"] == "http_403"
    assert status["robots"] == "parsed_0"
    assert status["cats"] == "ok:2"
    # aww is blocked and robots is empty; the other three feeds give two entries each.
    assert len(rows) == 6
    assert len(calls) == len(trends.REDDIT_SUBREDDITS)


def test_http_errors_are_labelled_even_without_a_response():
    request = httpx.Request("GET", "https://www.reddit.com/r/aww/.rss")
    with_response = httpx.Response(403, request=request)
    assert trends._error_label(httpx.HTTPStatusError("403", request=request, response=with_response)) == "http_403"
    # A status error without a response must not crash the connector loop.
    assert trends._error_label(httpx.HTTPStatusError("403", request=None, response=None)) == "error:HTTPStatusError"
    assert trends._error_label(httpx.ConnectError("refused")) == "error:ConnectError"


def test_gdelt_json_parses_articles(monkeypatch):
    _route(monkeypatch, research, lambda url, headers: FakeResponse(GDELT_JSON, content_type="application/json"))
    rows = research._gdelt(research.GDELT_QUERY, timeout=1)
    assert [r.title for r in rows] == [
        "Smart home robot dog learns to open the fridge",
        "Robot cooks the best sandwich at a school fair",
    ]
    assert rows[0].published_at == "20261010T051500Z"


def test_gdelt_query_is_one_parenthesised_request(monkeypatch):
    calls = _route(monkeypatch, research, lambda url, headers: FakeResponse(GDELT_JSON, content_type="application/json"))
    research._gdelt(research.GDELT_QUERY, timeout=1)
    url = calls[0]["url"]
    assert url.count("api.gdeltproject.org") == 1
    assert "%28" in url and "%29" in url  # OR'd terms are wrapped in parentheses
    assert "sourcelang%3Aenglish" in url
    assert "maxrecords=25" in url
    assert len(research.GDELT_QUERIES) == 1


@pytest.mark.parametrize("body", [GDELT_TEXT_ERROR, GDELT_RATE_LIMIT])
def test_gdelt_text_answers_are_errors_not_silent_zero_rows(monkeypatch, body):
    _route(monkeypatch, research, lambda url, headers: FakeResponse(body, content_type="text/html"))
    with pytest.raises(ValueError, match="non_json:"):
        research._gdelt(research.GDELT_QUERY, timeout=1)


def test_gdelt_rate_limit_is_recorded_as_that_source_failing(monkeypatch):
    def handler(url, headers):
        if "news.google.com" in url:
            return FakeResponse(GOOGLE_NEWS_RSS)
        return FakeResponse(GDELT_RATE_LIMIT, status_code=429, content_type="text/html")

    _route(monkeypatch, research, handler)
    signals, status = research.discover_report(limit=30, timeout=1)
    assert status["gdelt:global_news"].startswith("error:HTTPStatusError")
    assert signals and all(s.source == "google_news" for s in signals)


def test_discover_report_records_every_source_outcome(monkeypatch):
    def handler(url, headers):
        if "news.google.com" in url:
            return FakeResponse(GOOGLE_NEWS_RSS)
        return FakeResponse(GDELT_TEXT_ERROR, content_type="text/html")

    _route(monkeypatch, research, handler)
    signals, status = research.discover_report(limit=30, timeout=1)
    gnews = [label for label in status if label.startswith("google_news:")]
    assert len(gnews) == len(research.GOOGLE_QUERIES)
    assert all(status[label].startswith("ok:") for label in gnews)
    assert status["gdelt:global_news"].startswith("error:ValueError:non_json:")
    assert {s.source for s in signals} == {"google_news"}


def test_google_trends_rss_keeps_traffic_and_news_context(monkeypatch):
    calls = _route(monkeypatch, trends, lambda url, headers: FakeResponse(GOOGLE_TRENDS_RSS))
    status: dict[str, str] = {}
    rows = trends.discover_google_trends_rss(timeout=1, status=status)
    assert len(calls) == 1  # one feed, not the same feed fetched twice
    assert [r["title"] for r in rows] == ["india vs australia", "salman khan", "earthquake today"]
    assert [r["views"] for r in rows] == [2000000, 500000, 200000]
    assert rows[0]["description"] == "India vs Australia live score"
    assert status["google_trends"] == "ok:3"


def test_youtube_keeps_shorts_length_videos_from_family_categories(monkeypatch):
    items = {
        "items": [
            {"id": "a1", "snippet": {"title": "Cat plays piano", "publishedAt": "2026-10-10T00:00:00Z"},
             "statistics": {"viewCount": "1000"}, "contentDetails": {"duration": "PT1M30S"}},
            {"id": "a2", "snippet": {"title": "Full movie", "publishedAt": "2026-10-10T00:00:00Z"},
             "statistics": {"viewCount": "9000"}, "contentDetails": {"duration": "PT4M"}},
            {"id": "a1", "snippet": {"title": "Cat plays piano", "publishedAt": "2026-10-10T00:00:00Z"},
             "statistics": {"viewCount": "1000"}, "contentDetails": {"duration": "PT1M30S"}},
        ]
    }
    calls = _route(monkeypatch, trends, lambda url, headers: FakeResponse(json.dumps(items), content_type="application/json"))
    status: dict[str, str] = {}
    rows = trends.discover_youtube_most_popular(api_key="test-key", timeout=1, status=status)
    categories = [c["params"]["videoCategoryId"] for c in calls]
    assert categories == list(trends.YOUTUBE_FAMILY_CATEGORIES)
    # The same video returned by several category charts is kept once; the feature film is dropped.
    assert [r["title"] for r in rows] == ["Cat plays piano"]
    assert rows[0]["duration_seconds"] == 90.0
    assert status["youtube"] == "ok:1"


@pytest.mark.parametrize("value, seconds", [("PT1M30S", 90.0), ("PT3M", 180.0), ("PT45S", 45.0), ("P1DT1H", 90000.0), ("", 0.0)])
def test_iso_duration_parsing(value, seconds):
    assert trends._iso_seconds(value) == seconds


def test_google_news_title_and_social_mention_count_as_one_upstream_source():
    base = {"title": "Robot vacuum goes viral after chasing a cat", "published_at": "2026-10-10T06:12:00+00:00",
            "views": 0, "description": ""}
    rows = [
        {**base, "source": "google_news", "category": "home_tech"},
        {**base, "source": "google_social_search", "source_type": "web_social_proxy", "category": "instagram_mentions"},
    ]
    signals = enrich_signals(rows)
    assert len(signals) == 1
    assert signals[0].source_count == 1


def test_keyword_matching_uses_whole_words():
    # Substrings that used to match: 'dance' in guidance, 'day' in today, 'wait' in Kuwait, 'cute' in prosecute.
    assert "dance" not in infer_formats("RBI guidance on loans today", "")
    assert "challenge" not in infer_formats("Officials discuss the budget today", "")
    assert infer_emotions("Court to prosecute officials over Kuwait border") == ()
    assert "cute_animal" in infer_formats("Cute cat video goes viral", "")
    assert "curiosity_question" in extract_hook_patterns("Why does my cat do that?")


def test_numbers_signal_a_list_only_with_a_list_noun():
    assert "listicle" in extract_hook_patterns("5 ways to fix your desk")
    assert "listicle" not in extract_hook_patterns("Season 5 finale and oc=5")
    assert infer_formats("Season 5 finale, oc=5", "") == ()


def test_character_fit_ignores_keywords_inside_other_words():
    from app.agents.trend_opportunity import score_content_opportunity

    inside = score_content_opportunity({"title": "Mumbai said hello"})["character_fit"]["PIP"]
    plain = score_content_opportunity({"title": "Plain words here"})["character_fit"]["PIP"]
    assert inside == plain  # 'ai' used to match inside 'Mumbai' and 'said'


def test_feed_descriptions_lose_markup_and_entities():
    raw = '&lt;a href="https://news.google.com/rss/articles/CBMiqgFB"&gt;Robot&lt;/a&gt;&amp;nbsp;&lt;font color="#6f6f6f"&gt;Times&lt;/font&gt;'
    assert trends._plain(raw) == "Robot Times"


def test_trend_source_status_reports_configuration_not_reachability(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    status = trends.trend_source_status()
    assert status["youtube"]["configured"] is False
    assert status["youtube"]["active"] is False
    assert status["reddit"]["configured"] is True
    assert "blocked" in status["reddit"]["note"]
