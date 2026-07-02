"""Reflection / honesty gate.

This is the load-bearing primitive from Co-Scientist. Its whole job is:
    ─ record the set of URLs the tool-loop actually fetched
    ─ when the model returns a review with an `evidence` list, drop every
      entry whose URL we never saw
    ─ return an honest record

If we don't do this — the model can hallucinate references, cite them
persuasively, and no downstream consumer can tell the difference. The three
lines that do the filter are copied in spirit from
`Co-Scientist/co_scientist/agents/reflection.py` (upstream comment retained
below verbatim).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tools.honesty import filter_evidence_by_seen_urls


@dataclass
class LoopResult:
    """Envelope tracking one reasoning loop.

    `seen_urls` is the ONLY authority on what was actually fetched. Every
    tool that successfully returned content should extend this set with the
    URLs it saw (final redirected URL, not the caller's input URL).
    """

    seen_urls: set[str] = field(default_factory=set)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def register_fetch(self, url: str | None) -> None:
        if url:
            self.seen_urls.add(url)

    def register_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
    ) -> None:
        self.tool_calls.append({
            "tool": tool_name,
            "args": args or {},
            "result_summary": result_summary or {},
        })


def filter_evidence(
    record: dict[str, Any],
    loop_result: LoopResult,
) -> dict[str, Any]:
    """Rewrite `record['evidence']` to only include URLs we actually saw.

    Verbatim upstream comment (Co-Scientist/co_scientist/agents/reflection.py):
        # Drop evidence entries whose URL we never saw — keep the review honest.
    """
    # Drop evidence entries whose URL we never saw — keep the review honest.
    record = dict(record)  # shallow copy so caller can keep the original
    record["evidence"] = filter_evidence_by_seen_urls(
        record.get("evidence", []),
        loop_result.seen_urls,
    )
    return record


def reflect_and_filter(
    record: dict[str, Any],
    loop_result: LoopResult,
    *,
    require_evidence: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Filter `record` through the honesty gate and return (filtered, warnings).

    Warnings surface cases where the filter changed something meaningful — e.g.
    the model referenced 5 URLs but only 3 were actually fetched. Callers can
    log these, surface them in an audit trail, or block the response.

    Args:
        record: dict with an optional `evidence: list[{url, ...}]` field.
        loop_result: LoopResult tracking what the tool-loop actually fetched.
        require_evidence: if True (default), a record whose evidence list
            becomes empty after filtering will emit a warning.

    Returns:
        (filtered_record, warnings)
    """
    original_evidence = record.get("evidence", [])
    filtered = filter_evidence(record, loop_result)

    warnings: list[str] = []
    orig_urls = {e.get("url") for e in original_evidence if isinstance(e, dict)}
    kept_urls = {e.get("url") for e in filtered["evidence"]}
    dropped_urls = orig_urls - kept_urls - {None}
    if dropped_urls:
        warnings.append(
            f"dropped {len(dropped_urls)} hallucinated citation(s): "
            f"{sorted(u for u in dropped_urls if u)[:5]}"
        )
    if require_evidence and not filtered["evidence"]:
        if original_evidence:
            warnings.append(
                "record had evidence entries but none matched fetched URLs — "
                "likely fully hallucinated"
            )
        else:
            warnings.append("record has no evidence at all")

    return filtered, warnings
