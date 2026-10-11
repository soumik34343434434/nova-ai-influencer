"""Refresh-token flows for YouTube and TikTok, shared by publishing and analytics.

Every HTTP call goes through a fake; no test touches the network or reads real
credentials.
"""
import json
import logging

import pytest

from app.providers import oauth_tokens
from app.providers.publisher import PublishError


class FakeOAuth:
    """Answers token-endpoint calls in order and records each request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append({"method": method, "url": url, "headers": headers, "body": body.decode()})
        status, payload = self.responses.pop(0)
        return status, {}, json.dumps(payload).encode()


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    oauth_tokens.clear_cache()
    for name in ("YOUTUBE_ACCESS_TOKEN", "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN",
                 "TIKTOK_ACCESS_TOKEN", "TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET", "TIKTOK_REFRESH_TOKEN",
                 "INSTAGRAM_ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    yield
    oauth_tokens.clear_cache()


@pytest.fixture
def youtube_refresh(monkeypatch):
    monkeypatch.setenv("YOUTUBE_CLIENT_ID", "client-id")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "yt-refresh")


@pytest.fixture
def tiktok_refresh(monkeypatch):
    monkeypatch.setenv("TIKTOK_CLIENT_KEY", "tt-key")
    monkeypatch.setenv("TIKTOK_CLIENT_SECRET", "tt-secret")
    monkeypatch.setenv("TIKTOK_REFRESH_TOKEN", "tt-refresh-1")


def test_youtube_refresh_grant_returns_a_fresh_access_token(youtube_refresh):
    http = FakeOAuth((200, {"access_token": "ya-1", "expires_in": 3599}))
    token = oauth_tokens.get_access_token("youtube", http, clock=Clock())
    assert token == "ya-1"
    call = http.calls[0]
    assert call["method"] == "POST" and call["url"] == oauth_tokens.GOOGLE_TOKEN_URL
    assert "grant_type=refresh_token" in call["body"] and "refresh_token=yt-refresh" in call["body"]
    assert call["headers"]["Content-Type"] == "application/x-www-form-urlencoded"


def test_cached_token_is_reused_until_it_nears_expiry(youtube_refresh):
    clock = Clock()
    http = FakeOAuth((200, {"access_token": "ya-1", "expires_in": 3600}),
                     (200, {"access_token": "ya-2", "expires_in": 3600}))
    assert oauth_tokens.get_access_token("youtube", http, clock=clock) == "ya-1"
    clock.now += 60
    assert oauth_tokens.get_access_token("youtube", http, clock=clock) == "ya-1"
    assert len(http.calls) == 1
    clock.now += 3600 - 60 - oauth_tokens.EXPIRY_SKEW_SECONDS + 1  # inside the refresh-ahead window
    assert oauth_tokens.get_access_token("youtube", http, clock=clock) == "ya-2"
    assert len(http.calls) == 2


def test_token_refreshes_lazily_not_at_job_start(youtube_refresh):
    from app.providers.platform_publishers import YouTubeShortsPublisher

    http = FakeOAuth((200, {"access_token": "ya-lazy", "expires_in": 3600}))
    publisher = YouTubeShortsPublisher(http)
    assert http.calls == []  # constructing an adapter exchanges nothing
    assert publisher._token() == "ya-lazy"
    assert len(http.calls) == 1


def test_access_token_env_is_the_fallback_when_refresh_is_not_configured(monkeypatch):
    monkeypatch.setenv("YOUTUBE_ACCESS_TOKEN", "manual-token")
    http = FakeOAuth()
    assert oauth_tokens.get_access_token("youtube", http) == "manual-token"
    assert http.calls == []


def test_refresh_wins_over_a_static_access_token(youtube_refresh, monkeypatch):
    monkeypatch.setenv("YOUTUBE_ACCESS_TOKEN", "stale-manual-token")
    http = FakeOAuth((200, {"access_token": "fresh", "expires_in": 3600}))
    assert oauth_tokens.get_access_token("youtube", http, clock=Clock()) == "fresh"


def test_refresh_failure_is_classified_and_names_no_secret(youtube_refresh):
    http = FakeOAuth((400, {"error": "invalid_grant", "error_description": "yt-refresh is revoked"}))
    with pytest.raises(PublishError) as raised:
        oauth_tokens.get_access_token("youtube", http, clock=Clock())
    assert raised.value.retryable is False
    assert "invalid_grant" in str(raised.value)
    assert "yt-refresh" not in str(raised.value)


def test_server_errors_during_refresh_are_retryable(youtube_refresh):
    http = FakeOAuth((503, {}))
    with pytest.raises(PublishError) as raised:
        oauth_tokens.get_access_token("youtube", http, clock=Clock())
    assert raised.value.retryable is True


def test_tiktok_refresh_rotates_the_refresh_token_and_warns(tiktok_refresh, caplog):
    clock = Clock()
    http = FakeOAuth(
        (200, {"access_token": "t-1", "expires_in": 86400, "refresh_token": "tt-refresh-2"}),
        (200, {"access_token": "t-2", "expires_in": 86400, "refresh_token": "tt-refresh-3"}),
    )
    with caplog.at_level(logging.WARNING, logger=oauth_tokens.logger.name):
        assert oauth_tokens.get_access_token("tiktok", http, clock=clock) == "t-1"
    assert "rotated its refresh token" in caplog.text
    assert "tt-refresh-2" not in caplog.text and "tt-refresh-1" not in caplog.text

    clock.now += 86400  # the cached token has expired, so the next refresh uses the rotated token
    assert oauth_tokens.get_access_token("tiktok", http, clock=clock) == "t-2"
    assert "refresh_token=tt-refresh-2" in http.calls[1]["body"]
    assert "client_key=tt-key" in http.calls[0]["body"]


def test_changing_the_configured_refresh_token_discards_the_cache(tiktok_refresh, monkeypatch):
    http = FakeOAuth((200, {"access_token": "t-1", "expires_in": 86400}),
                     (200, {"access_token": "t-2", "expires_in": 86400}))
    clock = Clock()
    assert oauth_tokens.get_access_token("tiktok", http, clock=clock) == "t-1"
    monkeypatch.setenv("TIKTOK_REFRESH_TOKEN", "tt-refresh-new")
    assert oauth_tokens.get_access_token("tiktok", http, clock=clock) == "t-2"
    assert "refresh_token=tt-refresh-new" in http.calls[1]["body"]


def test_instagram_token_is_used_as_is_and_never_refreshed(monkeypatch):
    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "ig-page-token")
    http = FakeOAuth()
    assert oauth_tokens.get_access_token("instagram", http) == "ig-page-token"
    assert http.calls == []
    monkeypatch.delenv("INSTAGRAM_ACCESS_TOKEN")
    with pytest.raises(PublishError, match="INSTAGRAM_ACCESS_TOKEN"):
        oauth_tokens.get_access_token("instagram", http)


def test_missing_env_names_what_to_set(monkeypatch):
    assert oauth_tokens.missing_env("youtube") == ["YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"]
    assert oauth_tokens.missing_env("tiktok") == ["TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET", "TIKTOK_REFRESH_TOKEN"]
    assert oauth_tokens.missing_env("instagram") == ["INSTAGRAM_ACCESS_TOKEN"]
    assert not oauth_tokens.has_credentials("youtube")


def test_try_access_token_never_raises_for_analytics(youtube_refresh):
    http = FakeOAuth((500, {}))
    assert oauth_tokens.try_access_token("youtube", http) == ""


def test_youtube_owner_analytics_sends_the_token_in_a_header(monkeypatch, youtube_refresh):
    import httpx

    from app.providers import platform_analytics

    token_reply = json.dumps({"access_token": "ya-metrics", "expires_in": 3600}).encode()
    monkeypatch.setattr(oauth_tokens, "_default_http", lambda *args: (200, {}, token_reply))
    seen = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"columnHeaders": [{"name": "views"}], "rows": [[42]]}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params or {}, headers=headers or {})
        return Response()

    monkeypatch.setattr(httpx, "get", fake_get)
    snapshot = platform_analytics.YouTubeClient().fetch_owner_analytics("vid1")
    assert snapshot is not None and snapshot.views == 42
    assert seen["headers"]["Authorization"] == "Bearer ya-metrics"
    assert "access_token" not in seen["params"]
