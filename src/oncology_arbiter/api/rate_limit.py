"""Rate-limit key extractor.

The default ``slowapi.util.get_remote_address`` reads ``request.client.host``,
which is the immediate TCP peer. Behind Render + Cloudflare that peer is the
Render load balancer (or Cloudflare's edge node), so every real caller is
mapped to a handful of internal proxy IPs — the buckets *never* fan out, and
because slowapi's default in-memory store is per-process any single tenant can
burst well past the configured limit without ever tripping a 429.

Empirical observation on the 2026-07-05 v0.2 rollout: 180 real requests in
~11 s (well above ``60/minute``) all returned 200/401 with no 429. See
``/mnt/results/proofs/v0.2-enforcement/summary.json`` (``rate_limit_check``).

This module resolves that by extracting the *true* caller IP from Cloudflare
and Render forwarded headers, in a trust-aware order:

    1. ``CF-Connecting-IP`` — set by Cloudflare on ingress, stripped from any
       inbound value the client tries to spoof. Trustworthy iff the service is
       behind Cloudflare (which is the case for Render web services by
       default). This is the primary key we want for public traffic.

    2. ``X-Forwarded-For`` — set by Render's load balancer. First hop is the
       client IP; subsequent hops are proxies. We take ``xff.split(",")[0]``.

    3. ``request.client.host`` — the immediate TCP peer. Only used as a
       last-resort fallback (mainly for local dev where neither forwarded
       header is set).

Trust gate
----------
Trusting forwarded headers on a public HTTP endpoint that is NOT actually
behind a trusted proxy is worse than the status quo: any caller can forge
``X-Forwarded-For`` and rotate their apparent IP to defeat the bucket. So the
header path is opt-in via ``ONCOLOGY_ARBITER_TRUST_FORWARDED_FOR``:

    - "1", "true", "yes", "on" (case-insensitive) → trust the headers.
    - anything else (including unset) → fall back to
      ``get_remote_address`` behavior.

Production deployments on Render should set this env var to ``1``. Local dev
and CI leave it unset, so the tests exercise the fallback path.

Usage
-----
Passed directly to ``slowapi.Limiter(key_func=...)`` in
``oncology_arbiter.api.app.create_app``.

    from oncology_arbiter.api.rate_limit import make_key_func
    key_func = make_key_func()
    limiter = Limiter(key_func=key_func, default_limits=["60/minute"])
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from starlette.requests import Request


ENV_TRUST_FORWARDED = "ONCOLOGY_ARBITER_TRUST_FORWARDED_FOR"

_TRUTHY = {"1", "true", "yes", "on"}


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() in _TRUTHY


def _extract_forwarded_ip(request: Request) -> Optional[str]:
    """Pull the caller IP from forwarded headers, or return None.

    Header precedence:
      - ``CF-Connecting-IP`` (Cloudflare-signed; the canonical caller IP when
        behind CF; single value).
      - ``X-Forwarded-For`` (comma-separated hop list; take the first entry).
    """
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        v = cf.strip()
        if v:
            return v

    xff = request.headers.get("x-forwarded-for")
    if xff:
        # first hop = original client
        first = xff.split(",", 1)[0].strip()
        if first:
            return first
    return None


def _fallback_ip(request: Request) -> str:
    """Immediate TCP peer, or literal placeholder if the client info is None.

    Starlette test requests have ``request.client = None`` — return a stable
    placeholder so slowapi still gets a hashable key.
    """
    client = request.client
    if client is None:
        return "__no_client__"
    return client.host


def caller_ip(request: Request, *, trust_forwarded: Optional[bool] = None) -> str:
    """Resolve the caller IP for rate-limit bucketing.

    Args:
        request: Starlette/FastAPI request.
        trust_forwarded: Override the env var trust gate. When ``None`` (the
            default), read ``ONCOLOGY_ARBITER_TRUST_FORWARDED_FOR`` from env.
    """
    trust = _truthy(os.environ.get(ENV_TRUST_FORWARDED)) if trust_forwarded is None else trust_forwarded
    if trust:
        ip = _extract_forwarded_ip(request)
        if ip is not None:
            return ip
    return _fallback_ip(request)


def make_key_func(*, trust_forwarded: Optional[bool] = None) -> Callable[[Request], str]:
    """Return a slowapi-compatible ``key_func``.

    Args:
        trust_forwarded: If given, freeze the trust gate at construction time.
            If ``None`` (default), the env var is read on *every* call so a
            single running process can change behavior via env var without a
            restart (useful for smoke tests).
    """
    if trust_forwarded is None:
        def _dynamic(request: Request) -> str:
            return caller_ip(request)
        return _dynamic

    frozen: bool = bool(trust_forwarded)

    def _frozen(request: Request) -> str:
        return caller_ip(request, trust_forwarded=frozen)

    return _frozen


__all__ = [
    "ENV_TRUST_FORWARDED",
    "caller_ip",
    "make_key_func",
]
