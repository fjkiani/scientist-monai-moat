"""The honesty gate — copied verbatim in spirit from Co-Scientist's reflection agent.

The upstream Reflection agent's three-line filter is the single most important
pattern in the whole project:

    # Drop evidence entries whose URL we never saw — keep the review honest.
    seen = loop_result.seen_urls
    record["evidence"] = [
        e for e in record.get("evidence", [])
        if isinstance(e, dict) and e.get("url") in seen
    ]

We preserve it here as a first-class utility so every stage of the arbiter
platform can call it. Any arbiter that returns an `evidence[]` field MUST
funnel through this before being surfaced to the API caller.

Rationale: an LLM (or a reflection loop) can hallucinate URLs. If the tool
loop never actually fetched a URL, no citation should reference it. This
gate makes it structurally impossible to ship a fake citation.
"""
from __future__ import annotations

from typing import Any


def filter_evidence_by_seen_urls(
    evidence: list[dict[str, Any]],
    seen_urls: set[str] | frozenset[str],
) -> list[dict[str, Any]]:
    """Return only the evidence entries whose `url` field was actually fetched.

    Args:
        evidence: list of dicts. Each is expected to have at least a `url` key.
        seen_urls: set of URLs the tool-loop actually saw during this reasoning
            step (e.g., successful web_fetch / pubmed_search results).

    Returns:
        Filtered list — same shape, only honest citations.

    Notes:
        - Non-dict entries are dropped silently (they can't be verified).
        - Entries without a `url` key are dropped.
        - No trimming or URL normalization is applied. Exact-match only.
          If callers need normalization, do it BEFORE both populating
          seen_urls and calling this filter, so the sets stay consistent.
    """
    return [
        e for e in evidence
        if isinstance(e, dict) and e.get("url") in seen_urls
    ]
