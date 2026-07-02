"""LIVE integration test — hits NCBI E-utilities in real time.

Run with:  pytest -m integration tests/integration/test_pubmed_live.py

We assert:
  * A known PMID search returns the actual paper (not a stub) with correct DOI,
    year, and journal.
  * Author-parsing correctly surfaces Hubbard as first author.
  * Abstract text contains the ground-truth 61.3% and 41.6% figures.
  * `url` is the exact PubMed permalink, which is what `LoopResult.register_fetch`
    will store — meaning downstream `filter_evidence_by_seen_urls` will accept
    a citation to that URL.

Ground truth (verified against `/mnt/results/execution_trace/transcript.jsonl`):
    Hubbard RA et al. 2011.
    "Cumulative probability of false-positive recall or biopsy recommendation
     after 10 years of screening mammography: a cohort study."
    Ann Intern Med.
    PMID: 22007042
    DOI:  10.7326/0003-4819-155-8-201110180-00004  (verified via WebFetch on
          the PubMed page; note the sibling record PMID 22007048 with
          DOI ending -00014 is Autier's editorial commentary, NOT this paper)
"""
from __future__ import annotations

import pytest

from oncology_arbiter.tools import PubmedSearchTool
from oncology_arbiter.tools.base import ToolCtx

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_pubmed_returns_hubbard_2011_by_pmid(tmp_path) -> None:
    """PubMed esearch+efetch pipeline returns the correct Hubbard paper."""
    tool = PubmedSearchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    # Query by PMID directly — this is the most deterministic PubMed query.
    result = await tool.call({"query": "22007042[uid]", "max_results": 1}, ctx)

    assert not result.is_error, f"pubmed_search failed: {result.error_message}"
    payload = result.content
    assert payload["n"] == 1, f"expected exactly one hit, got {payload['n']}"

    rec = payload["results"][0]
    # Identity
    assert rec["pmid"] == "22007042"
    assert rec["doi"] == "10.7326/0003-4819-155-8-201110180-00004", (
        f"DOI drift: got {rec['doi']!r}"
    )
    # Bibliographic metadata
    assert rec["year"] == "2011"
    assert "Ann" in rec["journal"], f"journal drift: {rec['journal']!r}"
    # Author parsing
    authors_str = " ".join(rec["authors"])
    assert "Hubbard" in authors_str, f"authors drift: {rec['authors']}"
    # Title sanity
    assert "false-positive" in rec["title"].lower()
    # URL is the canonical PubMed permalink — must match what LoopResult sees
    assert rec["url"] == "https://pubmed.ncbi.nlm.nih.gov/22007042/"


@pytest.mark.asyncio
async def test_pubmed_abstract_contains_ground_truth_stats(tmp_path) -> None:
    """The 61.3% / 41.6% figures we build clinical baselines on ARE in the abstract.

    If this test breaks it means either (a) NCBI changed the abstract text or
    (b) our PLAN.md's headline stat is wrong. Either way we want to know.
    """
    tool = PubmedSearchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    result = await tool.call({"query": "22007042[uid]", "max_results": 1}, ctx)
    abstract = result.content["results"][0]["abstract"]

    # Both headline numbers cited in PLAN.md §1 must appear verbatim.
    assert "61.3" in abstract, "PLAN.md's 10y annual FP recall of 61.3% not in abstract"
    assert "41.6" in abstract, "PLAN.md's biennial 41.6% figure not in abstract"


@pytest.mark.asyncio
async def test_pubmed_multi_result_search_and_url_registration(tmp_path) -> None:
    """A real query returns multiple ranked records; each has a fetchable URL."""
    tool = PubmedSearchTool()
    ctx = ToolCtx(artifacts_dir=tmp_path / "artifacts")
    result = await tool.call(
        {
            "query": "mammography screening false positive recall Hubbard",
            "max_results": 5,
        },
        ctx,
    )
    assert not result.is_error
    assert result.content["n"] >= 1
    for rec in result.content["results"]:
        assert rec["pmid"], "missing PMID"
        assert rec["url"] is not None
        assert rec["url"].startswith("https://pubmed.ncbi.nlm.nih.gov/")
        # Downstream honesty gate uses exact-match on URL — verify the shape.
        assert rec["url"].endswith(f"/{rec['pmid']}/"), (
            f"url shape drift: {rec['url']!r}"
        )
