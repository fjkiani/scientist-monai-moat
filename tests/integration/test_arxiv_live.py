"""LIVE integration tests — hits arXiv Atom API in real time.

Run with:  pytest -m integration tests/integration/test_arxiv_live.py

Ground truth (verified via WebFetch on the Google MedSigLIP model card):
    arXiv:2507.05201  "MedGemma Technical Report"
       Sellergren, Andrew et al. (2025)
       — This is the paper that documents BOTH MedGemma and MedSigLIP.
       MedSigLIP does not have a separate arXiv entry.

    arXiv:2504.06196  TxGemma (referenced verbatim in Google's TxGemma card)
"""
from __future__ import annotations

import pytest

from oncology_arbiter.tools import ArxivSearchTool
from oncology_arbiter.tools.base import ToolCtx

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_arxiv_returns_medgemma_paper(tmp_path) -> None:
    """arXiv:2507.05201 is MedGemma Technical Report (documents MedSigLIP too)."""
    tool = ArxivSearchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    result = await tool.call({"query": "2507.05201", "max_results": 3}, ctx)

    assert not result.is_error, f"arxiv failed: {result.error_message}"
    payload = result.content
    assert payload["n"] >= 1

    # Locate the exact record by arxiv_id — arXiv may return other 2507.xxxxx
    # ids for a bare-number query, so we filter.
    hit = next(
        (r for r in payload["results"] if r["arxiv_id"].startswith("2507.05201")),
        None,
    )
    assert hit is not None, (
        f"expected arxiv_id 2507.05201 in results, got: "
        f"{[r['arxiv_id'] for r in payload['results']]}"
    )
    # Content assertions
    assert "medgemma" in hit["title"].lower()
    # Sellergren is first author on the MedGemma paper
    assert any("Sellergren" in a for a in hit["authors"]), (
        f"Sellergren not in authors: {hit['authors']}"
    )
    # Publication year 2025 (arXiv preprint)
    assert hit["year"] == "2025"
    # abs_url points back to arXiv
    assert hit["abs_url"].startswith("http")
    assert "2507.05201" in hit["abs_url"]


@pytest.mark.asyncio
async def test_arxiv_title_query_finds_medgemma_technical_report(tmp_path) -> None:
    """Title-scoped search finds the MedGemma Technical Report reliably.

    Note: a bare all:MedSigLIP query is drowned out by other 2025 medical
    imaging papers because arXiv relevance ranks recency and citation graph.
    A title-scoped query is the deterministic way to retrieve it.
    """
    tool = ArxivSearchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    result = await tool.call(
        {"query": 'ti:"MedGemma Technical Report"', "max_results": 5}, ctx
    )
    assert not result.is_error
    assert result.content["n"] >= 1
    hit = next(
        (r for r in result.content["results"] if "medgemma" in r["title"].lower()),
        None,
    )
    assert hit is not None, (
        "title:MedGemma Technical Report returned no MedGemma paper — "
        f"got titles: {[r['title'][:60] for r in result.content['results']]}"
    )
    assert "2507.05201" in hit["abs_url"]


@pytest.mark.asyncio
async def test_arxiv_record_shape_and_categories(tmp_path) -> None:
    """Every record must have arxiv_id, abs_url, and non-empty categories."""
    tool = ArxivSearchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    result = await tool.call({"query": "2504.06196", "max_results": 1}, ctx)
    assert not result.is_error
    assert result.content["n"] >= 1
    hit = result.content["results"][0]
    assert hit["arxiv_id"]
    assert hit["abs_url"] and hit["abs_url"].startswith("http")
    assert isinstance(hit["categories"], list)
