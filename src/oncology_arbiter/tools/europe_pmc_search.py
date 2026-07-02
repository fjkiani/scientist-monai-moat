"""Europe PMC search — covers PubMed + bioRxiv/medRxiv preprints.

Ported verbatim from `Co-Scientist/co_scientist/tools/builtins/europe_pmc.py`.
No config dependency to strip; this endpoint requires no key.

For a mammography reasoning system this matters because a lot of recent AI
prior-art (Kim 2020, McKinney 2020, Wu 2021, DBT-vs-DM comparisons) has
preprint versions on bioRxiv/medRxiv that only appear here.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from ._http import get_with_backoff
from .base import ToolCtx, ToolResult

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePMCSearchTool:
    name = "europe_pmc_search"
    description = (
        "Search Europe PMC (PubMed + bioRxiv/medRxiv + full-text where available). "
        "Returns {id, source, title, abstract, authors, journal, year, doi, url, "
        "pubmed_url, is_open_access}."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            "open_access_only": {"type": "boolean", "default": False},
        },
        "required": ["query"],
    }

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        t0 = time.monotonic()
        query = (args.get("query") or "").strip()
        n = int(args.get("max_results") or 10)
        oa = bool(args.get("open_access_only"))
        if not query:
            return ToolResult(is_error=True, error_message="empty query")
        q = f"({query}) AND OPEN_ACCESS:Y" if oa else query

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                r = await get_with_backoff(
                    client,
                    EUROPE_PMC_URL,
                    {
                        "query": q,
                        "format": "json",
                        "pageSize": n,
                        "resultType": "core",
                    },
                )
                data = r.json()
        except httpx.HTTPError as e:
            return ToolResult(is_error=True, error_message=f"europe_pmc failed: {e}")

        results: list[dict[str, Any]] = []
        for hit in data.get("resultList", {}).get("result", [])[:n]:
            pmid = hit.get("pmid")
            doi = hit.get("doi")
            results.append({
                "id": hit.get("id"),
                "source": hit.get("source"),
                "title": hit.get("title"),
                "abstract": hit.get("abstractText", ""),
                "authors": hit.get("authorString", ""),
                "journal": hit.get("journalTitle"),
                "year": hit.get("pubYear"),
                "doi": doi,
                "url": (
                    f"https://europepmc.org/article/{hit.get('source', 'MED')}/{hit.get('id', '')}"
                    if hit.get("id") else None
                ),
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                "is_open_access": hit.get("isOpenAccess") == "Y",
            })
        payload = {"query": q, "n": len(results), "results": results}
        return ToolResult(
            content=payload,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(str(payload)),
        )
