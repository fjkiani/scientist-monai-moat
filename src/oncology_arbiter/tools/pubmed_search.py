"""PubMed search via the NCBI E-utilities API.

Ported from `Co-Scientist/co_scientist/tools/builtins/pubmed.py`. The only
adaptation is removal of the `cfg.secrets.NCBI_API_KEY` indirection — we
read from `os.environ["NCBI_API_KEY"]` directly. Without a key, NCBI throttles
to 3 req/s; with one, 10 req/s.

Returns light records: pmid, title, abstract, journal, authors, year, doi, url.
Use `web_fetch` to pull full text.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from ._http import get_with_backoff
from .base import ToolCtx, ToolResult

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubmedSearchTool:
    name = "pubmed_search"
    description = (
        "Search PubMed. Returns up to N records with pmid, title, abstract, journal, "
        "authors, year, doi, url. Use for biomedical queries — mammography, oncology, "
        "pathology, radiology. For physics/CS/ML methods papers, prefer arxiv_search."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "PubMed query (E-utilities syntax allowed).",
            },
            "max_results": {
                "type": "integer", "minimum": 1, "maximum": 50, "default": 10,
            },
            "sort": {
                "type": "string", "enum": ["relevance", "pub_date"], "default": "relevance",
            },
        },
        "required": ["query"],
    }

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        t0 = time.monotonic()
        query = (args.get("query") or "").strip()
        n = int(args.get("max_results") or 10)
        sort = args.get("sort", "relevance")
        if not query:
            return ToolResult(is_error=True, error_message="empty query")

        api_key = os.environ.get("NCBI_API_KEY") or None
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                pmids = await _esearch(client, query, n, sort, api_key)
                if not pmids:
                    return ToolResult(
                        content={"query": query, "n": 0, "results": []},
                        duration_ms=int((time.monotonic() - t0) * 1000),
                    )
                records = await _efetch(client, pmids, api_key)
        except httpx.HTTPError as e:
            return ToolResult(is_error=True, error_message=f"pubmed failed: {e}")

        payload = {"query": query, "n": len(records), "results": records}
        return ToolResult(
            content=payload,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(str(payload)),
        )


async def _esearch(
    client: httpx.AsyncClient,
    query: str,
    n: int,
    sort: str,
    api_key: str | None,
) -> list[str]:
    params: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": n,
        "sort": "relevance" if sort == "relevance" else "pub+date",
    }
    if api_key:
        params["api_key"] = api_key
    r = await get_with_backoff(client, ESEARCH, params)
    return r.json().get("esearchresult", {}).get("idlist", [])


async def _efetch(
    client: httpx.AsyncClient, pmids: list[str], api_key: str | None
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key
    r = await get_with_backoff(client, EFETCH, params)
    return await asyncio.to_thread(_parse_pubmed_xml, r.text)


def _parse_pubmed_xml(xml: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml)
    out: list[dict[str, Any]] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = (art.findtext(".//ArticleTitle") or "").strip()
        journal = (art.findtext(".//Journal/Title") or "").strip()
        year_node = (
            art.findtext(".//PubDate/Year")
            or art.findtext(".//PubDate/MedlineDate")
            or ""
        )
        abstract_parts: list[str] = []
        for at in art.findall(".//Abstract/AbstractText"):
            label = at.get("Label")
            txt = "".join(at.itertext()).strip()
            abstract_parts.append(f"{label}: {txt}" if label else txt)
        abstract = "\n\n".join(p for p in abstract_parts if p)
        authors: list[str] = []
        for au in art.findall(".//Author")[:8]:
            last = au.findtext("LastName") or ""
            init = au.findtext("Initials") or ""
            if last:
                authors.append(f"{last} {init}".strip())
        doi = None
        for aid in art.findall(".//ArticleId"):
            if (aid.get("IdType") or "").lower() == "doi":
                doi = aid.text
                break
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        out.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "authors": authors,
            "year": year_node[:4] if year_node else None,
            "doi": doi,
            "url": url,
        })
    return out
