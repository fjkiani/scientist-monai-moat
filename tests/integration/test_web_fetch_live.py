"""LIVE integration test — web_fetch pulls the real Hubbard 2011 PubMed page.

Run with:  pytest -m integration tests/integration/test_web_fetch_live.py

We assert that the full-text extraction is good enough to preserve the
ground-truth statistics we anchor PLAN.md on. If web_fetch's HTML extraction
degrades (bad content-selector, PubMed layout change), this test catches it.
"""
from __future__ import annotations

import pytest

from oncology_arbiter.orchestrator.reflection import LoopResult
from oncology_arbiter.tools import WebFetchTool, filter_evidence_by_seen_urls
from oncology_arbiter.tools.base import ToolCtx

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_web_fetch_hubbard_pubmed_page(tmp_path) -> None:
    """Fetch the real PubMed page for Hubbard 2011 and extract the abstract."""
    tool = WebFetchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    result = await tool.call(
        {"url": "https://pubmed.ncbi.nlm.nih.gov/22007042/", "max_chars": 30000},
        ctx,
    )
    assert not result.is_error, f"web_fetch failed: {result.error_message}"
    payload = result.content

    assert payload["status"] == 200
    assert payload["url"].startswith("https://pubmed.ncbi.nlm.nih.gov/22007042")
    # trafilatura should extract >1kB of the abstract body.
    assert len(payload["text"]) > 1000, (
        f"extracted only {len(payload['text'])} chars — extractor may be broken"
    )
    # The ground-truth statistics must appear in the extracted text.
    text = payload["text"]
    assert "61.3" in text, "Hubbard's 61.3% figure missing from extraction"
    assert "41.6" in text, "Hubbard's 41.6% figure missing from extraction"
    assert "7.0" in text, "Hubbard's 7.0% biopsy figure missing from extraction"


@pytest.mark.asyncio
async def test_web_fetch_end_to_end_with_honesty_gate(tmp_path) -> None:
    """Full loop: fetch a URL, register it in LoopResult, and a downstream
    'evidence' claim referencing that URL must survive the honesty gate while
    a hallucinated URL is dropped.
    """
    tool = WebFetchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")

    real_url = "https://pubmed.ncbi.nlm.nih.gov/22007042/"
    result = await tool.call({"url": real_url, "max_chars": 5000}, ctx)
    assert not result.is_error
    fetched_url = result.content["url"]

    loop = LoopResult()
    loop.register_fetch(fetched_url)

    evidence = [
        {"url": fetched_url, "claim": "Hubbard 2011 cumulative FP recall = 61.3%"},
        {"url": "https://fake-hallucinated-source.example.com", "claim": "invented"},
    ]
    filtered = filter_evidence_by_seen_urls(evidence, loop.seen_urls)
    assert len(filtered) == 1
    assert filtered[0]["url"] == fetched_url


@pytest.mark.asyncio
async def test_web_fetch_cache_roundtrip(tmp_path) -> None:
    """First fetch hits the network; second call returns the cached artifact
    at effectively zero latency.
    """
    tool = WebFetchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts", session_id="s1")
    url = "https://pubmed.ncbi.nlm.nih.gov/22007042/"

    r1 = await tool.call({"url": url}, ctx)
    assert not r1.is_error
    first_ms = r1.duration_ms

    r2 = await tool.call({"url": url}, ctx)
    assert not r2.is_error
    second_ms = r2.duration_ms

    # Cache read should be at least 5x faster than the initial network fetch.
    assert second_ms * 5 < first_ms, (
        f"cache did not fire: first={first_ms}ms second={second_ms}ms"
    )
    # Content should match on the key fields.
    assert r1.content["url"] == r2.content["url"]
