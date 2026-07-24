"""API hardening: optional API-key auth + rate limiting.

- Auth is opt-in: if ``settings.api_key`` is unset (the default), endpoints are open — convenient
  for local dev. Set ``DEEPFAKE_API_KEY`` in the environment to require an ``X-API-Key`` header.
- Rate limiting uses slowapi, keyed by client IP, with the limit from ``settings.rate_limit``.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.config import settings

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Validate the API key if one is configured; otherwise allow the request."""
    if settings.api_key is None:
        return  # auth disabled
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
