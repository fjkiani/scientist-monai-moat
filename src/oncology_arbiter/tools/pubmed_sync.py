"""Synchronous PubMed search for API routes (wraps E-utilities without asyncio)."""
from __future__ import annotations

import os
import urllib.parse
import urllib.request
from typing import Any
from xml.etree import ElementTree as ET

from oncology_arbiter.tools.pubmed_search import _parse_pubmed_xml

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def pubmed_search_sync(
    query: str,
    *,
    max_results: int = 3,
    timeout_s: float = 30.0,
) -> list[dict[str, Any]]:
    """Return up to ``max_results`` PubMed records for ``query``."""
    query = (query or "").strip()
    if not query:
        return []
    n = max(1, min(int(max_results), 50))
    api_key = os.environ.get("NCBI_API_KEY") or None
    params: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(n),
        "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key
    url = ESEARCH + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        payload = resp.read().decode()
    import json

    pmids = json.loads(payload).get("esearchresult", {}).get("idlist", []) or []
    if not pmids:
        return []
    fetch_params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if api_key:
        fetch_params["api_key"] = api_key
    fetch_url = EFETCH + "?" + urllib.parse.urlencode(fetch_params)
    with urllib.request.urlopen(fetch_url, timeout=timeout_s) as resp:
        xml = resp.read().decode()
    return _parse_pubmed_xml(xml)


def evidence_records_from_pubmed(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, str]]:
    """Map PubMed hits into ApiEnvelope EvidenceRecord dicts."""
    records = pubmed_search_sync(query, max_results=max_results)
    out: list[dict[str, str]] = []
    for rec in records[:max_results]:
        url = str(rec.get("url") or "").strip()
        if not url:
            continue
        title = str(rec.get("title") or "").strip()
        abstract = str(rec.get("abstract") or "").strip()
        quoted = (title + ("\n\n" + abstract[:800] if abstract else "")).strip()
        if not quoted:
            quoted = f"PubMed PMID {rec.get('pmid')}"
        out.append(
            {
                "url": url,
                "quoted_text": quoted[:1200],
                "source": "pubmed",
            }
        )
    return out
