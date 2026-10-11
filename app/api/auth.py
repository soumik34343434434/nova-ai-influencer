"""Shared-token guard for the NOVA HTTP API.

Every mutating or data-returning route requires the ``X-NOVA-API-Key`` header to
equal ``NOVA_API_TOKEN``. A custom header cannot be sent by a plain cross-site
form post, so a web page the operator visits cannot drive the API. Run uvicorn
on 127.0.0.1 (the default for local use) so the API is not reachable from the
network at all.
"""
from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status


def check_token(provided: str | None) -> None:
    expected = os.getenv("NOVA_API_TOKEN", "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NOVA_API_TOKEN is not configured",
        )
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid NOVA API token",
        )


def require_token(x_nova_api_key: str | None = Header(default=None, alias="X-NOVA-API-Key")) -> None:
    """FastAPI dependency: attach with ``dependencies=[Depends(require_token)]``."""
    check_token(x_nova_api_key)
