"""Tests for the rate-limit key extractor.

Contract exercised (see src/oncology_arbiter/api/rate_limit.py):

  - When trust is off (env var unset or false): return request.client.host,
    ignoring any forwarded headers (a caller can't spoof their bucket).
  - When trust is on: prefer CF-Connecting-IP, then X-Forwarded-For (first
    hop), then fall back to request.client.host.
  - X-Forwarded-For is comma-separated hop list; only the FIRST hop is used
    (the client-most IP).
  - Missing / blank / whitespace-only header values are treated as absent.
  - request.client == None (Starlette TestClient sometimes) returns the
    literal ``__no_client__`` placeholder so slowapi still gets a key.
"""
from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from oncology_arbiter.api.rate_limit import (
    ENV_TRUST_FORWARDED,
    caller_ip,
    make_key_func,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_request(*, headers: dict[str, str] | None = None, client_host: str = "127.0.0.1") -> Request:
    """Build a bare Starlette Request with the given headers and client peer.

    We construct the ASGI scope by hand to avoid needing a full app for pure
    unit tests. Headers arrive as a list of (bytes, bytes) tuples in ASGI.
    """
    scope: dict = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
        ],
        "client": (client_host, 12345) if client_host is not None else None,
        "server": ("testserver", 80),
        "scheme": "http",
        "http_version": "1.1",
        "root_path": "",
    }
    return Request(scope)


# ---------------------------------------------------------------------------
# trust OFF (default): headers ignored, always use client.host
# ---------------------------------------------------------------------------


def test_no_trust_returns_client_host_ignoring_cf_header(monkeypatch):
    monkeypatch.delenv(ENV_TRUST_FORWARDED, raising=False)
    req = _make_request(
        headers={"cf-connecting-ip": "203.0.113.7"},
        client_host="10.0.0.1",
    )
    assert caller_ip(req) == "10.0.0.1"


def test_no_trust_returns_client_host_ignoring_xff_header(monkeypatch):
    monkeypatch.delenv(ENV_TRUST_FORWARDED, raising=False)
    req = _make_request(
        headers={"x-forwarded-for": "203.0.113.7, 10.0.0.5"},
        client_host="10.0.0.1",
    )
    assert caller_ip(req) == "10.0.0.1"


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "  ", "maybe"])
def test_env_falsey_disables_trust(monkeypatch, value):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, value)
    req = _make_request(
        headers={"cf-connecting-ip": "203.0.113.7"},
        client_host="10.0.0.1",
    )
    assert caller_ip(req) == "10.0.0.1"


# ---------------------------------------------------------------------------
# trust ON: CF-Connecting-IP first, then X-Forwarded-For, then client.host
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "YES", "on", "On"])
def test_env_truthy_enables_trust(monkeypatch, value):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, value)
    req = _make_request(headers={"cf-connecting-ip": "203.0.113.7"}, client_host="10.0.0.1")
    assert caller_ip(req) == "203.0.113.7"


def test_cf_connecting_ip_takes_precedence_over_xff(monkeypatch):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, "1")
    req = _make_request(
        headers={
            "cf-connecting-ip": "203.0.113.7",
            "x-forwarded-for": "198.51.100.9, 10.0.0.5",
        },
        client_host="10.0.0.1",
    )
    assert caller_ip(req) == "203.0.113.7"


def test_xff_used_when_cf_header_absent(monkeypatch):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, "1")
    req = _make_request(
        headers={"x-forwarded-for": "198.51.100.9, 10.0.0.5"},
        client_host="10.0.0.1",
    )
    assert caller_ip(req) == "198.51.100.9"


def test_xff_takes_only_first_hop(monkeypatch):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, "1")
    req = _make_request(
        headers={"x-forwarded-for": "203.0.113.7, 198.51.100.9, 10.0.0.5"},
        client_host="10.0.0.1",
    )
    assert caller_ip(req) == "203.0.113.7"


def test_xff_strips_whitespace(monkeypatch):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, "1")
    req = _make_request(
        headers={"x-forwarded-for": "  203.0.113.7 , 10.0.0.5"},
        client_host="10.0.0.1",
    )
    assert caller_ip(req) == "203.0.113.7"


def test_falls_back_to_client_host_when_headers_absent(monkeypatch):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, "1")
    req = _make_request(client_host="10.0.0.1")
    assert caller_ip(req) == "10.0.0.1"


def test_falls_back_when_cf_header_is_blank(monkeypatch):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, "1")
    req = _make_request(
        headers={"cf-connecting-ip": "   "},
        client_host="10.0.0.1",
    )
    assert caller_ip(req) == "10.0.0.1"


def test_falls_back_when_xff_first_hop_is_blank(monkeypatch):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, "1")
    req = _make_request(
        headers={"x-forwarded-for": ", 10.0.0.5"},
        client_host="10.0.0.1",
    )
    assert caller_ip(req) == "10.0.0.1"


# ---------------------------------------------------------------------------
# request.client == None edge case
# ---------------------------------------------------------------------------


def test_client_is_none_returns_placeholder(monkeypatch):
    monkeypatch.delenv(ENV_TRUST_FORWARDED, raising=False)
    req = _make_request(client_host=None)
    assert caller_ip(req) == "__no_client__"


def test_client_is_none_but_cf_header_present_and_trust_on(monkeypatch):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, "1")
    req = _make_request(
        headers={"cf-connecting-ip": "203.0.113.7"},
        client_host=None,
    )
    # header wins even without a client
    assert caller_ip(req) == "203.0.113.7"


# ---------------------------------------------------------------------------
# override at construction time
# ---------------------------------------------------------------------------


def test_explicit_trust_false_beats_env_true(monkeypatch):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, "1")
    req = _make_request(
        headers={"cf-connecting-ip": "203.0.113.7"},
        client_host="10.0.0.1",
    )
    assert caller_ip(req, trust_forwarded=False) == "10.0.0.1"


def test_explicit_trust_true_beats_env_false(monkeypatch):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, "off")
    req = _make_request(
        headers={"cf-connecting-ip": "203.0.113.7"},
        client_host="10.0.0.1",
    )
    assert caller_ip(req, trust_forwarded=True) == "203.0.113.7"


# ---------------------------------------------------------------------------
# make_key_func returns a callable that slowapi can consume
# ---------------------------------------------------------------------------


def test_make_key_func_returns_callable(monkeypatch):
    monkeypatch.delenv(ENV_TRUST_FORWARDED, raising=False)
    kf = make_key_func()
    req = _make_request(client_host="10.0.0.1")
    assert callable(kf)
    assert kf(req) == "10.0.0.1"


def test_make_key_func_frozen_true_ignores_env(monkeypatch):
    """A frozen key_func reads the header path regardless of env changes."""
    monkeypatch.setenv(ENV_TRUST_FORWARDED, "off")
    kf = make_key_func(trust_forwarded=True)
    req = _make_request(
        headers={"cf-connecting-ip": "203.0.113.7"},
        client_host="10.0.0.1",
    )
    assert kf(req) == "203.0.113.7"


def test_make_key_func_frozen_false_ignores_env(monkeypatch):
    monkeypatch.setenv(ENV_TRUST_FORWARDED, "1")
    kf = make_key_func(trust_forwarded=False)
    req = _make_request(
        headers={"cf-connecting-ip": "203.0.113.7"},
        client_host="10.0.0.1",
    )
    assert kf(req) == "10.0.0.1"


def test_make_key_func_dynamic_reads_env_each_call(monkeypatch):
    """When trust_forwarded=None, the env var is read fresh per request."""
    kf = make_key_func()  # dynamic
    req_with_cf = _make_request(
        headers={"cf-connecting-ip": "203.0.113.7"},
        client_host="10.0.0.1",
    )

    monkeypatch.setenv(ENV_TRUST_FORWARDED, "1")
    assert kf(req_with_cf) == "203.0.113.7"

    monkeypatch.setenv(ENV_TRUST_FORWARDED, "0")
    assert kf(req_with_cf) == "10.0.0.1"


# ---------------------------------------------------------------------------
# End-to-end integration with a small Starlette app + slowapi to confirm 429
# ---------------------------------------------------------------------------


def _build_limited_app(monkeypatch, *, trust: bool, limit: str = "3/second") -> TestClient:
    """Spin a minimal Starlette app with slowapi wired via our key_func."""
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from starlette.responses import JSONResponse

    from oncology_arbiter.api.rate_limit import make_key_func

    if trust:
        monkeypatch.setenv(ENV_TRUST_FORWARDED, "1")
    else:
        monkeypatch.delenv(ENV_TRUST_FORWARDED, raising=False)

    limiter = Limiter(key_func=make_key_func(), default_limits=[limit])

    async def ok(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/", ok)])
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    def _handler(request, exc):
        return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)

    return TestClient(app)


def test_end_to_end_bucket_shared_when_trust_off(monkeypatch):
    """
    With trust OFF, requests from the TestClient all look like the same
    peer (127.0.0.1) even though we vary CF-Connecting-IP. The 4th request
    within a second must 429.
    """
    client = _build_limited_app(monkeypatch, trust=False, limit="3/second")
    codes = []
    for i in range(6):
        r = client.get("/", headers={"cf-connecting-ip": f"203.0.113.{i}"})
        codes.append(r.status_code)
    assert 429 in codes, f"expected a 429 in {codes}"
    # first 3 should be 200
    assert codes[:3] == [200, 200, 200], codes


def test_end_to_end_bucket_split_when_trust_on(monkeypatch):
    """
    With trust ON, each distinct CF-Connecting-IP gets its own bucket, so
    6 rapid requests with 6 distinct client IPs should ALL be 200 under a
    3/second per-key limit.
    """
    client = _build_limited_app(monkeypatch, trust=True, limit="3/second")
    codes = []
    for i in range(6):
        r = client.get("/", headers={"cf-connecting-ip": f"203.0.113.{i}"})
        codes.append(r.status_code)
    assert all(c == 200 for c in codes), codes


def test_end_to_end_same_cf_ip_still_429s(monkeypatch):
    """With trust ON, hammering with the SAME CF-Connecting-IP still 429s."""
    client = _build_limited_app(monkeypatch, trust=True, limit="3/second")
    codes = []
    for i in range(6):
        r = client.get("/", headers={"cf-connecting-ip": "203.0.113.7"})
        codes.append(r.status_code)
    assert 429 in codes, f"expected a 429 in {codes}"
    assert codes[:3] == [200, 200, 200], codes
