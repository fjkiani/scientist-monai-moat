"""arXiv search via the public Atom API.

Ported verbatim from `Co-Scientist/co_scientist/tools/builtins/arxiv.py` — no
config dependency to strip, this one was already clean. Kept here so oncology
methods-paper retrieval (MedSigLIP arXiv 2507.05201, TxGemma 2504.06196, EMBED
processing papers, etc.) works out of the box.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from ._http import get_with_backoff
from .base import ToolCtx, ToolResult

ARXIV_URL = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivSearchTool:
    name = "arxiv_search"
    description = (
        "Search arXiv. Returns up to N records with arxiv_id, title, summary, authors, "
        "year, pdf_url, abs_url, categories. Use for CS / ML / quantitative-biology "
        "methods papers — for biomedical clinical evidence prefer pubmed_search or europe_pmc_search."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            "sort": {
                "type": "string",
                "enum": ["relevance", "submitted", "lastUpdated"],
                "default": "relevance",
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

        sort_param = {
            "relevance": "relevance",
            "submitted": "submittedDate",
            "lastUpdated": "lastUpdatedDate",
        }[sort]

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                r = await get_with_backoff(
                    client,
                    ARXIV_URL,
                    {
                        "search_query": (
                            query
                            if any(query.startswith(p) for p in (
                                "all:", "ti:", "au:", "abs:", "cat:", "id:",
                            ))
                            else f"all:{query}"
                        ),
                        "max_results": n,
                        "sortBy": sort_param,
                        "sortOrder": "descending",
                    },
                )
                records = await asyncio.to_thread(_parse_atom, r.text)
        except httpx.HTTPError as e:
            return ToolResult(is_error=True, error_message=f"arxiv failed: {e}")

        payload = {"query": query, "n": len(records), "results": records}
        return ToolResult(
            content=payload,
            duration_ms=int((time.monotonic() - t0) * 1000),
            result_bytes=len(str(payload)),
        )


def _parse_atom(xml: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml)
    out: list[dict[str, Any]] = []
    for e in root.findall("a:entry", _NS):
        id_url = (e.findtext("a:id", default="", namespaces=_NS) or "").strip()
        arxiv_id = id_url.rsplit("/", 1)[-1]
        title = (e.findtext("a:title", default="", namespaces=_NS) or "").strip()
        summary = (e.findtext("a:summary", default="", namespaces=_NS) or "").strip()
        published = (e.findtext("a:published", default="", namespaces=_NS) or "")[:10]
        authors = [
            (a.findtext("a:name", default="", namespaces=_NS) or "").strip()
            for a in e.findall("a:author", _NS)
        ]
        categories = [
            c.get("term", "") for c in e.findall("a:category", _NS) if c.get("term")
        ]
        pdf_url = None
        for link in e.findall("a:link", _NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break
        out.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "summary": summary,
            "authors": authors,
            "year": published[:4] if published else None,
            "pdf_url": pdf_url,
            "abs_url": id_url,
            "categories": categories,
        })
    return out
