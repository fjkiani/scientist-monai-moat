"""Tests for the evidence honesty gate.

This gate is the single load-bearing safety primitive of the whole platform.
If it's broken, the arbiters can ship hallucinated citations. It must have
tests before anything downstream depends on it.
"""
from __future__ import annotations

from oncology_arbiter.orchestrator.reflection import (
    LoopResult,
    filter_evidence,
    reflect_and_filter,
)
from oncology_arbiter.tools.honesty import filter_evidence_by_seen_urls


def test_filter_drops_unseen_urls() -> None:
    """The core three-line filter: only URLs actually fetched pass through."""
    evidence = [
        {"url": "https://pubmed.ncbi.nlm.nih.gov/22007042/", "text": "Hubbard 2011"},
        {"url": "https://fake.example.com/hallucinated", "text": "invented"},
        {"url": "https://arxiv.org/abs/2507.05201", "text": "MedSigLIP"},
    ]
    seen = {
        "https://pubmed.ncbi.nlm.nih.gov/22007042/",
        "https://arxiv.org/abs/2507.05201",
    }
    out = filter_evidence_by_seen_urls(evidence, seen)
    urls = {e["url"] for e in out}
    assert urls == seen
    assert len(out) == 2


def test_filter_drops_non_dict_entries() -> None:
    """Non-dict entries (e.g. bare strings) cannot be verified — must drop."""
    evidence = [
        {"url": "https://ok.example.com/x"},
        "https://raw-string.example.com/y",       # bare string — dropped
        None,                                     # None — dropped
        42,                                       # int — dropped
    ]
    seen = {"https://ok.example.com/x", "https://raw-string.example.com/y"}
    out = filter_evidence_by_seen_urls(evidence, seen)
    assert len(out) == 1
    assert out[0]["url"] == "https://ok.example.com/x"


def test_filter_drops_entries_without_url() -> None:
    """Evidence dicts missing a `url` key are unverifiable — drop them."""
    evidence = [
        {"url": "https://ok.example.com/x", "quote": "real"},
        {"quote": "orphan quote with no url"},
    ]
    seen = {"https://ok.example.com/x"}
    out = filter_evidence_by_seen_urls(evidence, seen)
    assert len(out) == 1


def test_empty_seen_urls_drops_everything() -> None:
    """If we fetched nothing, no citation can survive."""
    evidence = [{"url": "https://anywhere.example.com"}]
    assert filter_evidence_by_seen_urls(evidence, set()) == []


def test_frozenset_input_works() -> None:
    """seen_urls may be a frozenset (e.g. finalized loop result)."""
    evidence = [{"url": "https://x.example.com"}]
    out = filter_evidence_by_seen_urls(evidence, frozenset(["https://x.example.com"]))
    assert len(out) == 1


def test_filter_evidence_returns_copy_not_mutation() -> None:
    """The wrapper must not mutate the input record dict."""
    record = {
        "reasoning": "some hypothesis",
        "evidence": [
            {"url": "https://ok.example.com"},
            {"url": "https://bad.example.com"},
        ],
    }
    loop = LoopResult(seen_urls={"https://ok.example.com"})
    original_evidence = record["evidence"]
    filtered = filter_evidence(record, loop)
    # input record should be untouched (shallow copy semantics)
    assert record["evidence"] is original_evidence
    assert len(record["evidence"]) == 2
    # output is filtered
    assert len(filtered["evidence"]) == 1


def test_reflect_and_filter_flags_dropped_citations() -> None:
    """reflect_and_filter should emit a warning when it drops hallucinations."""
    record = {
        "claim": "MedSigLIP achieves 0.933 AUROC on invasive breast cancer",
        "evidence": [
            {"url": "https://huggingface.co/google/medsiglip-448"},
            {"url": "https://arxiv.org/abs/2507.05201"},
            {"url": "https://fake.example.com/invented-source"},
        ],
    }
    loop = LoopResult(seen_urls={
        "https://huggingface.co/google/medsiglip-448",
        "https://arxiv.org/abs/2507.05201",
    })
    filtered, warnings = reflect_and_filter(record, loop)
    assert len(filtered["evidence"]) == 2
    assert any("hallucinated" in w.lower() for w in warnings)


def test_reflect_and_filter_flags_fully_hallucinated_records() -> None:
    """A record that started with evidence but has none post-filter is flagged."""
    record = {
        "claim": "bogus",
        "evidence": [
            {"url": "https://all-fake-1.example.com"},
            {"url": "https://all-fake-2.example.com"},
        ],
    }
    loop = LoopResult(seen_urls=set())
    filtered, warnings = reflect_and_filter(record, loop)
    assert filtered["evidence"] == []
    # both the dropped-citations warning AND the fully-hallucinated warning
    joined = " ".join(warnings).lower()
    assert "hallucinated" in joined


def test_loop_result_register_fetch() -> None:
    """LoopResult.register_fetch adds URLs and ignores None."""
    loop = LoopResult()
    loop.register_fetch("https://a.example.com")
    loop.register_fetch("https://a.example.com")  # duplicate — set semantics
    loop.register_fetch(None)                      # None — ignore
    loop.register_fetch("")                        # empty — ignore
    assert loop.seen_urls == {"https://a.example.com"}


def test_loop_result_register_tool_call() -> None:
    """LoopResult accumulates tool invocations in order."""
    loop = LoopResult()
    loop.register_tool_call("pubmed_search", {"query": "mammography"}, {"n": 5})
    loop.register_tool_call("web_fetch", {"url": "https://x.example.com"})
    assert len(loop.tool_calls) == 2
    assert loop.tool_calls[0]["tool"] == "pubmed_search"
    assert loop.tool_calls[1]["args"]["url"] == "https://x.example.com"
