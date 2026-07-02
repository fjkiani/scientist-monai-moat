"""Shared HTTP helper for the L2 evidence tools.

Exponential backoff on 429 Too Many Requests. Public APIs like NCBI E-utils,
arXiv, and Europe PMC throttle anonymous callers; we retry a few times before
giving up so a burst of parallel tool calls in a reasoning loop does not
crash the loop.
"""
from __future__ import annotations

import asyncio

import httpx

MAX_RETRIES = 3
INITIAL_BACKOFF_S = 0.5


async def get_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    params: dict | None = None,
) -> httpx.Response:
    """GET with exponential backoff on 429. Raises on non-429 4xx/5xx."""
    delay = INITIAL_BACKOFF_S
    last_response: httpx.Response | None = None
    for attempt in range(MAX_RETRIES + 1):
        r = await client.get(url, params=params or {})
        if r.status_code != 429:
            r.raise_for_status()
            return r
        last_response = r
        if attempt < MAX_RETRIES:
            await asyncio.sleep(delay)
            delay *= 2
    assert last_response is not None
    last_response.raise_for_status()
    raise RuntimeError("unreachable")  # pragma: no cover
