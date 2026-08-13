"""`web_fetch` must not report a bot wall / interstitial as a successful fetch.

Discovered live: `https://pubmed.ncbi.nlm.nih.gov/22007042/` answered HTTP 203
("Non-Authoritative Information", RFC 9110 s15.3.4) with a 100-character
"Cookies must be enabled" interstitial. The tool's success gate was
`status_code >= 400`, so 203 passed, trafilatura extracted the interstitial
text, and the caller received `is_error=False` with a short plausible document
that was not the requested source. An evidence pipeline would then have cited
the cookie wall.

These tests are hermetic: they drive the status handling through a mock
transport, so they run without network access and cannot be masked by an
upstream that happens to answer 200 today.
"""
from __future__ import annotations

import httpx
import pytest

from oncology_arbiter.tools import WebFetchTool
from oncology_arbiter.tools.base import ToolCtx

INTERSTITIAL = (
    b"<html><body><p>Cookies must be enabled</p>"
    b"<p>Enable cookies for pubmed.ncbi.nlm.nih.gov and reload this page "
    b"to continue.</p></body></html>"
)
REAL_DOCUMENT = (
    b"<html><body><article><p>"
    b"Among women screened annually, the cumulative probability of a "
    b"false-positive recall was 61.3% and 41.6% after ten rounds, with a "
    b"7.0% cumulative probability of a false-positive biopsy recommendation."
    b"</p></article></body></html>"
)

URL = "https://pubmed.ncbi.nlm.nih.gov/22007042/"


def _patch_transport(monkeypatch: pytest.MonkeyPatch, status: int, body: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, content=body, headers={"Content-Type": "text/html; charset=utf-8"}
        )

    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = httpx.MockTransport(handler)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [201, 202, 203, 204, 206, 226, 300, 304])
async def test_non_200_is_never_a_successful_fetch(
    tmp_path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    _patch_transport(monkeypatch, status, INTERSTITIAL)
    result = await WebFetchTool().call(
        {"url": URL, "max_chars": 30000}, ToolCtx(artifacts_dir=tmp_path / "artifacts")
    )
    assert result.is_error is True, f"HTTP {status} was accepted as a document"
    assert "non-authoritative" in (result.error_message or "")
    assert str(status) in (result.error_message or "")
    assert result.content == {"url": URL, "status": status}


@pytest.mark.asyncio
async def test_the_exact_observed_bot_wall_is_refused(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the live observation: 203 + cookie interstitial."""
    _patch_transport(monkeypatch, 203, INTERSTITIAL)
    result = await WebFetchTool().call(
        {"url": URL, "max_chars": 30000}, ToolCtx(artifacts_dir=tmp_path / "artifacts")
    )
    assert result.is_error is True
    # The interstitial text must not leak to the caller as document content.
    assert result.content is not None
    assert "text" not in result.content
    assert "Cookies" not in str(result.content)


@pytest.mark.asyncio
async def test_200_still_extracts_the_document(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity: the same path succeeds when the origin answers 200."""
    _patch_transport(monkeypatch, 200, REAL_DOCUMENT)
    result = await WebFetchTool().call(
        {"url": URL, "max_chars": 30000}, ToolCtx(artifacts_dir=tmp_path / "artifacts")
    )
    assert result.is_error is False, result.error_message
    assert result.content is not None
    assert result.content["status"] == 200
    for statistic in ("61.3", "41.6", "7.0"):
        assert statistic in result.content["text"]


@pytest.mark.asyncio
async def test_4xx_keeps_its_original_error_shape(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_transport(monkeypatch, 404, b"not found")
    result = await WebFetchTool().call(
        {"url": URL, "max_chars": 30000}, ToolCtx(artifacts_dir=tmp_path / "artifacts")
    )
    assert result.is_error is True
    assert result.error_message == "HTTP 404"
