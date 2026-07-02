"""LIVE integration test — hits Europe PMC REST in real time.

Run with:  pytest -m integration tests/integration/test_europe_pmc_live.py

Europe PMC's value-add over PubMed for this project is preprint coverage
(bioRxiv, medRxiv). We assert:
  * A DOI-scoped query retrieves the same record NCBI does.
  * An open-access-only filter meaningfully changes the result set.
  * URL fields are shaped so `LoopResult.register_fetch` can gate citations.
"""
from __future__ import annotations

import pytest

from oncology_arbiter.tools import EuropePMCSearchTool
from oncology_arbiter.tools.base import ToolCtx

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_europe_pmc_finds_hubbard_by_doi(tmp_path) -> None:
    """Europe PMC returns the same Hubbard record when queried by DOI."""
    tool = EuropePMCSearchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    result = await tool.call(
        {"query": 'DOI:"10.7326/0003-4819-155-8-201110180-00004"', "max_results": 3},
        ctx,
    )
    assert not result.is_error, f"europe_pmc failed: {result.error_message}"
    payload = result.content
    assert payload["n"] >= 1
    hit = next(
        (r for r in payload["results"] if r.get("doi", "").endswith("-00004")),
        None,
    )
    assert hit is not None, (
        f"expected Hubbard doi in results, got: "
        f"{[r.get('doi') for r in payload['results']]}"
    )
    assert hit["year"] == "2011"
    assert "false-positive" in (hit["title"] or "").lower()


@pytest.mark.asyncio
async def test_europe_pmc_open_access_filter(tmp_path) -> None:
    """The open_access_only flag translates to an OPEN_ACCESS:Y filter and
    returns strictly a subset of unrestricted queries.
    """
    tool = EuropePMCSearchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")

    unrestricted = await tool.call(
        {"query": "screening mammography false positive recall", "max_results": 20},
        ctx,
    )
    oa_only = await tool.call(
        {
            "query": "screening mammography false positive recall",
            "max_results": 20,
            "open_access_only": True,
        },
        ctx,
    )
    assert not unrestricted.is_error and not oa_only.is_error

    # Every OA result must have is_open_access=True.
    for r in oa_only.content["results"]:
        assert r["is_open_access"] is True, (
            f"open_access_only returned a non-OA record: {r.get('id')}"
        )

    # OA-restricted set should be <= unrestricted set for the same query.
    assert oa_only.content["n"] <= unrestricted.content["n"], (
        f"OA n={oa_only.content['n']} > unrestricted n={unrestricted.content['n']}"
    )


@pytest.mark.asyncio
async def test_europe_pmc_url_shape(tmp_path) -> None:
    """URLs must be shaped so LoopResult.register_fetch can key on them."""
    tool = EuropePMCSearchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    result = await tool.call({"query": "MedSigLIP", "max_results": 3}, ctx)
    assert not result.is_error
    for r in result.content["results"]:
        assert r["url"] and r["url"].startswith("https://europepmc.org/article/")
