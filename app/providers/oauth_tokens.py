"""Access tokens for the YouTube and TikTok adapters, refreshed on demand.

Publishing (``platform_publishers``) and analytics (``platform_analytics``) both
call :func:`get_access_token`. A refresh-token exchange runs lazily, when no
token is cached or the cached one is about to expire, so a scheduled job never
starts with a stale token.

Credentials (names are the contract; never commit values):

* YouTube (Google OAuth, access tokens last about an hour):
  ``YOUTUBE_CLIENT_ID``, ``YOUTUBE_CLIENT_SECRET``, ``YOUTUBE_REFRESH_TOKEN``.
  ``YOUTUBE_ACCESS_TOKEN`` is still read when the refresh set is absent, for
  short manual runs. Public view and like counts need only ``YOUTUBE_API_KEY``,
  which does not expire.
* TikTok (access tokens last about 24 hours): ``TIKTOK_CLIENT_KEY``,
  ``TIKTOK_CLIENT_SECRET``, ``TIKTOK_REFRESH_TOKEN``. TikTok may rotate the
  refresh token on each refresh. The new one is used for the rest of the
  process and a warning asks the owner to store it, without logging its value.
  ``TIKTOK_ACCESS_TOKEN`` is the same manual fallback.
* Instagram: ``INSTAGRAM_ACCESS_TOKEN`` must be a non-expiring token, a Page
  access token or system-user token created from a long-lived user token. An
  Instagram-Login token expires after 60 days and must not be used unattended.
  There is no refresh flow for it, so :func:`get_access_token` returns the value
  as-is.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.parse
from typing import Any, Callable

from app.providers.publisher import PublishError

logger = logging.getLogger(__name__)

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
TIKTOK_TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
# Refresh this long before expiry so an upload started now is not cut off.
EXPIRY_SKEW_SECONDS = 120.0
DEFAULT_LIFETIME_SECONDS = 3600.0

# (method, url, headers, body) -> (status, response headers, response body)
Http = Callable[[str, str, dict[str, str], "bytes | None"], tuple[int, dict[str, str], bytes]]

REFRESH_ENV = {
    "youtube": ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"),
    "tiktok": ("TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET", "TIKTOK_REFRESH_TOKEN"),
}
STATIC_ENV = {
    "youtube": "YOUTUBE_ACCESS_TOKEN",
    "tiktok": "TIKTOK_ACCESS_TOKEN",
    "instagram": "INSTAGRAM_ACCESS_TOKEN",
}

# platform -> the last exchange: which credentials produced it, its token and expiry.
_STATE: dict[str, dict[str, Any]] = {}


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def refresh_configured(platform: str) -> bool:
    names = REFRESH_ENV.get(platform)
    return bool(names) and all(_env(name) for name in names)


def missing_env(platform: str) -> list[str]:
    """Names still needed before ``platform`` can authenticate; empty when configured."""
    if refresh_configured(platform) or _env(STATIC_ENV.get(platform, "")):
        return []
    if platform in REFRESH_ENV:
        return list(REFRESH_ENV[platform])
    return [STATIC_ENV[platform]] if platform in STATIC_ENV else [f"{platform.upper()}_ACCESS_TOKEN"]


def has_credentials(platform: str) -> bool:
    return not missing_env(platform)


def clear_cache() -> None:
    """Forget cached tokens; used by tests and long-lived processes that rotate credentials."""
    _STATE.clear()


def _parse(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body.decode() or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _default_http(method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, dict[str, str], bytes]:
    from app.providers.platform_publishers import urllib_http

    return urllib_http(method, url, headers, body)


def _source_key(platform: str) -> str:
    material = "|".join(_env(name) for name in REFRESH_ENV[platform])
    return hashlib.sha256(material.encode()).hexdigest()


def _exchange(platform: str, refresh_token: str, http: Http) -> dict[str, Any]:
    """POST the refresh grant; raises PublishError with a retry classification."""
    if platform == "youtube":
        client_id, client_secret, _ = (_env(name) for name in REFRESH_ENV["youtube"])
        url = GOOGLE_TOKEN_URL
        form = {"client_id": client_id, "client_secret": client_secret,
                "grant_type": "refresh_token", "refresh_token": refresh_token}
    else:
        client_key, client_secret, _ = (_env(name) for name in REFRESH_ENV["tiktok"])
        url = TIKTOK_TOKEN_URL
        form = {"client_key": client_key, "client_secret": client_secret,
                "grant_type": "refresh_token", "refresh_token": refresh_token}
    status, _, body = http("POST", url, {"Content-Type": "application/x-www-form-urlencoded",
                                         "Cache-Control": "no-cache"},
                           urllib.parse.urlencode(form).encode())
    data = _parse(body)
    if status >= 400 or not data.get("access_token"):
        # Report the OAuth error code only; the body can echo credentials.
        code = str(data.get("error", "")) or "no_access_token"
        raise PublishError(f"{platform} token refresh failed with HTTP {status} ({code})",
                           retryable=status in {408, 429} or status >= 500)
    return data


def get_access_token(platform: str, http: Http | None = None, *, clock: Callable[[], float] = time.time) -> str:
    """A usable access token for ``platform``, refreshing it when needed.

    Raises PublishError (non-retryable) when no credentials are configured, and
    PublishError (retryable for 5xx, 429 and timeouts) when a refresh fails.
    """
    if platform == "instagram":
        token = _env(STATIC_ENV["instagram"])
        if not token:
            raise PublishError("instagram access token missing (INSTAGRAM_ACCESS_TOKEN)", retryable=False)
        return token
    if not refresh_configured(platform):
        token = _env(STATIC_ENV.get(platform, ""))
        if not token:
            raise PublishError(f"{platform} credentials missing: {', '.join(missing_env(platform))}", retryable=False)
        return token

    now = clock()
    source = _source_key(platform)
    state = _STATE.get(platform)
    if state and state["source"] == source and state["expires_at"] - now > EXPIRY_SKEW_SECONDS:
        return state["access_token"]
    # After a rotation, the newest refresh token is the only one that still works.
    refresh_token = state["refresh_token"] if state and state["source"] == source else _env(REFRESH_ENV[platform][2])
    data = _exchange(platform, refresh_token, http or _default_http)
    new_refresh = str(data.get("refresh_token") or refresh_token)
    if new_refresh != refresh_token:
        logger.warning("%s rotated its refresh token; store the new value as %s before the old one expires",
                       platform, REFRESH_ENV[platform][2])
    lifetime = float(data.get("expires_in") or DEFAULT_LIFETIME_SECONDS)
    _STATE[platform] = {
        "source": source,
        "refresh_token": new_refresh,
        "access_token": str(data["access_token"]),
        "expires_at": now + lifetime,
    }
    return _STATE[platform]["access_token"]


def try_access_token(platform: str, http: Http | None = None) -> str:
    """Like get_access_token for read-only callers: returns "" instead of raising."""
    try:
        return get_access_token(platform, http)
    except (PublishError, OSError) as exc:
        logger.warning("%s access unavailable for analytics: %s", platform, exc)
        return ""
