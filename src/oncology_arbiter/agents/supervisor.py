"""Co-Scientist supervisor — real LLM-driven 4-phase loop.

Phases (mirrored from the ``fjkiani/Co-Scientist`` reference implementation):

1. **GENERATE** — Gemma proposes ``n_hypotheses`` (default 6) plausible clinical
   claims from the patient context. No citations are attempted in this phase.
2. **EVIDENCE** — for each of the top-K (default 3) hypotheses by initial
   confidence, one of the 5 Co-Scientist tools is invoked to fetch supporting
   or contradicting evidence. Every URL that comes back is recorded in
   ``seen_urls``.
3. **REFLECT** — Gemma critiques each hypothesis given the retrieved evidence.
   Any hypothesis that cites a URL not in ``seen_urls`` is dropped through the
   :func:`tools.honesty.filter_evidence_by_seen_urls` gate.
4. **TOURNAMENT** — pairwise Gemma prompts rank surviving hypotheses. Elo (K=32)
   is updated deterministically (pairs are sorted by hypothesis id).
5. **META_REVIEW** — Gemma composes a clinician-facing narrative from the
   Elo-ranked list. Every sentence must be groundable in a hypothesis rationale
   or a fetched URL.

The old ``plan_stage()`` and ``run_placeholder()`` functions are kept for
back-compat with existing test suites and API-layer imports. The new
:func:`execute_stage` returns a :class:`StageResult` populated with real
evidence, real Elo scores, and a real meta-review.

Failure mode: if the LLM route ladder is completely exhausted,
:func:`execute_stage` degrades to ``run_placeholder()`` and marks
``model_state="llm_unavailable"``. Never fabricates.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER
from oncology_arbiter.models.llm_client import GemmaClient, LlmResponse, LlmUnavailable

SUPERVISOR_VERSION = "1.0.0"

_REQUIRED_TOOL_MODULES = (
    "oncology_arbiter.tools.pubmed_search",
    "oncology_arbiter.tools.arxiv_search",
    "oncology_arbiter.tools.europe_pmc_search",
    "oncology_arbiter.tools.web_fetch",
    "oncology_arbiter.tools.honesty",
)


class AgentPhase(str, Enum):
    """Four Co-Scientist phases (plus internal EVIDENCE step)."""
    GENERATE   = "generate"
    EVIDENCE   = "evidence"
    REFLECT    = "reflect"
    TOURNAMENT = "tournament"
    META_REVIEW = "meta_review"


@dataclass
class Hypothesis:
    """One clinical hypothesis produced by the loop."""
    hypothesis_id: str          # uuid4 hex
    claim: str                  # one sentence
    rationale: str              # 2-3 sentences
    source_scope: str           # imaging | molecular | clinical_history | literature
    initial_confidence: float   # 0..1, GENERATE-time self-assessment
    evidence_urls: List[str] = field(default_factory=list)
    reflection: str = ""
    evidence_alignment: str = "unknown"  # supports | partial | contradicts | unrelated | unknown
    elo: float = 1500.0
    tournament_wins: int = 0
    tournament_losses: int = 0
    tournament_ties: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "claim": self.claim,
            "rationale": self.rationale,
            "source_scope": self.source_scope,
            "initial_confidence": round(self.initial_confidence, 3),
            "evidence_urls": list(self.evidence_urls),
            "reflection": self.reflection,
            "evidence_alignment": self.evidence_alignment,
            "elo": round(self.elo, 1),
            "tournament_wins": self.tournament_wins,
            "tournament_losses": self.tournament_losses,
            "tournament_ties": self.tournament_ties,
        }


@dataclass
class StagePlan:
    """Deterministic per-stage execution plan."""
    stage: str
    phase_tools: Dict[AgentPhase, List[str]]
    seen_urls_policy: str = "reflect_and_prune"
    disclaimer: str = RUO_DISCLAIMER
    caveat: str = AUROC_CAVEAT


@dataclass
class StageResult:
    """Output of :func:`execute_stage`.

    ``model_state`` values in v1.0.0:

        - ``"executed"``           — real LLM loop ran end-to-end with fetched evidence
        - ``"llm_unavailable"``    — LLM ladder exhausted, degraded to placeholder
        - ``"llm_evidence_empty"`` — GENERATE ran but PubMed/EuropePMC returned
          no URLs; pairwise Elo is forbidden in this state
        - ``"llm_reflection_incomplete"`` — at least one evidence-backed
          hypothesis lacked a valid MedGemma reflection receipt
        - ``"llm_tournament_incomplete"`` — at least one Elo pair lacked a
          valid model judgment and request receipt
        - ``"placeholder"``        — legacy shim from :func:`run_placeholder`
    """
    stage: str
    model_state: str
    hypotheses: List[Hypothesis] = field(default_factory=list)
    meta_review: str = ""
    evidence: List[Mapping[str, Any]] = field(default_factory=list)
    seen_urls: List[str] = field(default_factory=list)
    seen_urls_count: int = 0
    evidence_kept: int = 0
    evidence_dropped: int = 0
    llm_calls: int = 0
    llm_total_tokens: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_cost_usd: float = 0.0
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_app_version: str | None = None
    llm_revision_sha256: str | None = None
    llm_request_ids: List[str] = field(default_factory=list)
    llm_latency_s: float = 0.0
    llm_honesty_warnings: List[str] = field(default_factory=list)
    reflection_calls_expected: int = 0
    reflection_calls_completed: int = 0
    reflection_receipts: List[Mapping[str, Any]] = field(default_factory=list)
    tournament_pairs_expected: int = 0
    tournament_pairs_completed: int = 0
    tournament_matches: List[Mapping[str, Any]] = field(default_factory=list)
    meta_review_request_id: str | None = None
    meta_review_response_sha256: str | None = None
    latency_s: float = 0.0
    notes: str = ""
    disclaimer: str = RUO_DISCLAIMER
    caveat: str = AUROC_CAVEAT

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "model_state": self.model_state,
            "hypotheses": [h.as_dict() for h in self.hypotheses],
            "meta_review": self.meta_review,
            "evidence": list(self.evidence),
            "seen_urls_count": self.seen_urls_count,
            "seen_urls": list(self.seen_urls),
            "seen_urls_sample": self.seen_urls[:10],
            "evidence_kept": self.evidence_kept,
            "evidence_dropped": self.evidence_dropped,
            "llm_calls": self.llm_calls,
            "llm_total_tokens": self.llm_total_tokens,
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "llm_cost_usd": round(self.llm_cost_usd, 6),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_app_version": self.llm_app_version,
            "llm_revision": self.llm_revision_sha256,
            "llm_request_ids": list(self.llm_request_ids),
            "llm_latency_s": round(self.llm_latency_s, 3),
            "llm_honesty_warnings": list(self.llm_honesty_warnings),
            "reflection_calls_expected": self.reflection_calls_expected,
            "reflection_calls_completed": self.reflection_calls_completed,
            "reflection_receipts": list(self.reflection_receipts),
            "tournament_pairs_expected": self.tournament_pairs_expected,
            "tournament_pairs_completed": self.tournament_pairs_completed,
            "tournament_matches": list(self.tournament_matches),
            "meta_review_request_id": self.meta_review_request_id,
            "meta_review_response_sha256": self.meta_review_response_sha256,
            "latency_s": round(self.latency_s, 2),
            "notes": self.notes,
            "disclaimer": self.disclaimer,
            "caveat": self.caveat,
            "supervisor_version": SUPERVISOR_VERSION,
        }


# ─── module-scope tool-availability check (build-time invariant) ───────────
_MISSING_TOOL_MODULES: tuple[str, ...] = tuple(
    m for m in _REQUIRED_TOOL_MODULES
    if importlib.util.find_spec(m) is None
)


# ─── Per-stage tool plans (unchanged from Phase 1) ─────────────────────────
_STAGE_PLANS: Dict[str, Dict[AgentPhase, List[str]]] = {
    "screening": {
        AgentPhase.GENERATE: ["oncology_arbiter.tools.pubmed_search",
                              "oncology_arbiter.tools.europe_pmc_search"],
        AgentPhase.EVIDENCE: ["oncology_arbiter.tools.pubmed_search",
                              "oncology_arbiter.tools.europe_pmc_search",
                              "oncology_arbiter.tools.web_fetch"],
        AgentPhase.REFLECT:  ["oncology_arbiter.tools.honesty"],
        AgentPhase.TOURNAMENT: ["oncology_arbiter.tools.honesty"],
        AgentPhase.META_REVIEW: ["oncology_arbiter.tools.web_fetch"],
    },
    "biopsy": {
        AgentPhase.GENERATE: ["oncology_arbiter.tools.pubmed_search",
                              "oncology_arbiter.tools.arxiv_search"],
        AgentPhase.EVIDENCE: ["oncology_arbiter.tools.pubmed_search",
                              "oncology_arbiter.tools.europe_pmc_search",
                              "oncology_arbiter.tools.arxiv_search"],
        AgentPhase.REFLECT:  ["oncology_arbiter.tools.honesty"],
        AgentPhase.TOURNAMENT: ["oncology_arbiter.tools.honesty"],
        AgentPhase.META_REVIEW: ["oncology_arbiter.tools.web_fetch"],
    },
    "therapy": {
        AgentPhase.GENERATE: ["oncology_arbiter.tools.pubmed_search",
                              "oncology_arbiter.tools.arxiv_search",
                              "oncology_arbiter.tools.europe_pmc_search"],
        AgentPhase.EVIDENCE: ["oncology_arbiter.tools.pubmed_search",
                              "oncology_arbiter.tools.europe_pmc_search",
                              "oncology_arbiter.tools.web_fetch"],
        AgentPhase.REFLECT:  ["oncology_arbiter.tools.honesty"],
        AgentPhase.TOURNAMENT: ["oncology_arbiter.tools.honesty"],
        AgentPhase.META_REVIEW: ["oncology_arbiter.tools.web_fetch"],
    },
    "case_full": {
        AgentPhase.GENERATE: ["oncology_arbiter.tools.pubmed_search",
                              "oncology_arbiter.tools.arxiv_search",
                              "oncology_arbiter.tools.europe_pmc_search"],
        AgentPhase.EVIDENCE: ["oncology_arbiter.tools.pubmed_search",
                              "oncology_arbiter.tools.europe_pmc_search",
                              "oncology_arbiter.tools.arxiv_search",
                              "oncology_arbiter.tools.web_fetch"],
        AgentPhase.REFLECT:  ["oncology_arbiter.tools.honesty"],
        AgentPhase.TOURNAMENT: ["oncology_arbiter.tools.honesty"],
        AgentPhase.META_REVIEW: ["oncology_arbiter.tools.web_fetch"],
    },
}


# ─── Prompt scaffolds ──────────────────────────────────────────────────────
_SYSTEM_GENERATE = (
    "You are the GENERATE phase of a Co-Scientist loop supporting an oncology "
    "clinician. Given the patient context, propose {n} plausible clinical "
    "hypotheses about diagnosis, treatment, or prognosis. Return valid JSON "
    "with a top-level key 'hypotheses' whose value is a list of objects, each "
    "with fields: claim (one sentence), rationale (2-3 sentences), source_scope "
    "(one of: imaging, molecular, clinical_history, literature), "
    "initial_confidence (float 0..1). DO NOT invent citations — evidence is "
    "retrieved in a later phase. Respond with JSON only, no prose wrapper."
)

_SYSTEM_REFLECT = (
    "You are the REFLECT phase. Given a clinical hypothesis and the evidence "
    "excerpt retrieved from a real URL, critique the hypothesis in 2-4 "
    "sentences. Note whether the evidence supports, partially supports, "
    "contradicts, or is unrelated to the claim. Return valid JSON: "
    "{critique: str, evidence_alignment: 'supports'|'partial'|'contradicts'|'unrelated'}. "
    "Respond with JSON only."
)

_SYSTEM_TOURNAMENT = (
    "You are the TOURNAMENT phase. Given two clinical hypotheses (A, B) and "
    "their reflections, decide which is stronger given the evidence retrieved "
    "so far. Return valid JSON: {winner: 'A'|'B'|'tie', margin: 'clear'|'moderate'|'marginal'|'tie', "
    "reason: one-sentence justification}. Respond with JSON only."
)

_SYSTEM_META_REVIEW = (
    "You are the META_REVIEW phase. Write a 4-8 sentence clinician-facing "
    "narrative summarizing the top-ranked hypotheses below. Every sentence "
    "must be grounded in one of the hypothesis rationales or evidence URLs — "
    "do NOT invent new claims. Prefer plain English over jargon. Respond with "
    "plain text, no headers, no JSON."
)


# ─── Public API ────────────────────────────────────────────────────────────

def plan_stage(stage: str) -> StagePlan:
    """Return the deterministic plan for the given stage.

    Raises ValueError on unknown stage; ImportError if a required tool module
    is missing.
    """
    if stage not in _STAGE_PLANS:
        raise ValueError(f"Unknown stage {stage!r}; allowed = {sorted(_STAGE_PLANS)}")
    if _MISSING_TOOL_MODULES:
        raise ImportError(
            "Supervisor cannot build a plan — required tool modules missing: "
            f"{_MISSING_TOOL_MODULES}"
        )
    plan = _STAGE_PLANS[stage]
    return StagePlan(stage=stage, phase_tools={ph: list(tools) for ph, tools in plan.items()})


def run_placeholder(stage: str) -> StageResult:
    """Legacy shim — Phase 1 API kept for backwards compatibility."""
    if stage not in _STAGE_PLANS:
        raise ValueError(f"Unknown stage {stage!r}; allowed = {sorted(_STAGE_PLANS)}")
    return StageResult(
        stage=stage,
        model_state="placeholder",
        notes=(
            "Legacy placeholder path — call execute_stage() for the real "
            f"Co-Scientist loop (supervisor v{SUPERVISOR_VERSION})."
        ),
    )


def execute_stage(
    stage: str,
    context: Mapping[str, Any],
    *,
    n_hypotheses: int = 6,
    n_evidence_top_k: int = 3,
    llm: Optional[GemmaClient] = None,
    seed_urls: Optional[Sequence[str]] = None,
) -> StageResult:
    """Run the real Co-Scientist loop for one stage.

    Args:
        stage: one of ``screening | biopsy | therapy | case_full``.
        context: patient context dict — will be summarized into GENERATE prompt.
            Expected keys (all optional): ``age``, ``sex``, ``findings``,
            ``receptor_panel``, ``prior_biopsy``, ``family_history``,
            ``clinical_question``. Extra keys are preserved and passed through.
        n_hypotheses: number generated in GENERATE (default 6).
        n_evidence_top_k: how many top hypotheses get evidence retrieval
            (default 3).
        llm: optional custom :class:`GemmaClient` — used for testing.
        seed_urls: optional pre-seed of URLs the caller already fetched.

    Returns:
        Fully populated :class:`StageResult`.

    Never raises for LLM failures — degrades to ``model_state="llm_unavailable"``.
    """
    if stage not in _STAGE_PLANS:
        raise ValueError(f"Unknown stage {stage!r}; allowed = {sorted(_STAGE_PLANS)}")
    if _MISSING_TOOL_MODULES:
        raise ImportError(f"Missing tool modules: {_MISSING_TOOL_MODULES}")

    t0 = time.time()
    llm = llm or GemmaClient()

    # We collect every URL the tool loop actually saw.
    seen_urls: set[str] = set(seed_urls or [])
    evidence: List[Mapping[str, Any]] = []
    evidence_errors: List[str] = []
    result = StageResult(stage=stage, model_state="executed")

    try:
        # ── GENERATE ────────────────────────────────────────────────────
        gen_user = _build_generate_prompt(stage, context, n_hypotheses)
        gen_resp, gen_json = _chat_json(
            llm,
            result,
            _SYSTEM_GENERATE.format(n=n_hypotheses),
            gen_user,
            max_tokens=1800,
            temperature=0.7,
        )
        if gen_resp is None:
            result.notes = "GENERATE unavailable: MedGemma chat failed"
            result.model_state = "llm_unavailable"
            result.latency_s = time.time() - t0
            return result
        hypotheses = _parse_hypotheses(
            gen_resp.text,
            expected_n=n_hypotheses,
            parsed=gen_json,
        )

        if not hypotheses:
            result.notes = "GENERATE returned no parseable hypotheses"
            result.model_state = "llm_parse_failed"
            result.latency_s = time.time() - t0
            return result

        # ── EVIDENCE ────────────────────────────────────────────────────
        # Every generated hypothesis must retrieve literature. Top-K is a
        # per-hypothesis hit cap, not a license to Elo-rank unreflected claims.
        by_conf = sorted(hypotheses, key=lambda h: h.initial_confidence, reverse=True)
        for h in by_conf:
            ev = _retrieve_evidence(
                stage, h, seen_urls, context=context, errors=evidence_errors,
                max_hits=n_evidence_top_k,
            )
            evidence.extend(ev)

        if not evidence:
            result.model_state = "llm_evidence_empty"
            result.notes = (
                "EVIDENCE retrieved 0 URLs; pairwise Elo is forbidden without "
                "fetched literature."
                + ((" Retrieval errors: " + "; ".join(evidence_errors[:8]))
                   if evidence_errors else "")
            )
            result.hypotheses = hypotheses
            result.evidence = []
            result.seen_urls = sorted(seen_urls)
            result.seen_urls_count = len(seen_urls)
            result.latency_s = time.time() - t0
            return result

        # ── REFLECT ────────────────────────────────────────────────────
        for h in hypotheses:
            result.reflection_calls_expected += 1
            if not h.evidence_urls:
                h.reflection = "No evidence retrieved for this hypothesis in EVIDENCE phase."
                h.evidence_alignment = "unknown"
                result.reflection_receipts.append({
                    "hypothesis_id": h.hypothesis_id,
                    "model_request_id": None,
                    "model_response_sha256": None,
                    "model_response_chars": 0,
                    "evidence_alignment": "unknown",
                    "completed": False,
                })
                continue
            ev_text = _summarize_evidence_for_hypothesis(evidence, h)
            ref_resp, r_json = _chat_json(
                llm,
                result,
                _SYSTEM_REFLECT,
                (
                    f"Hypothesis:\n{h.claim}\n\nRationale:\n{h.rationale}\n\n"
                    f"Retrieved evidence:\n{ev_text}"
                ),
                max_tokens=2048,
                temperature=0.3,
            )
            critique = str((r_json or {}).get("critique", "")).strip()
            alignment = str(
                (r_json or {}).get("evidence_alignment", "unknown")
            ).strip().lower()
            completed = (
                bool(critique)
                and alignment in {"supports", "partial", "contradicts", "unrelated"}
                and bool(ref_resp and ref_resp.request_id)
            )
            if completed:
                h.reflection = critique[:1200]
                h.evidence_alignment = alignment
                result.reflection_calls_completed += 1
            else:
                h.reflection = "MedGemma REFLECT did not return a valid final judgment."
                h.evidence_alignment = "unknown"
            result.reflection_receipts.append({
                "hypothesis_id": h.hypothesis_id,
                "model_request_id": ref_resp.request_id if ref_resp else None,
                "model_response_sha256": (
                    hashlib.sha256(ref_resp.text.encode()).hexdigest()
                    if ref_resp else None
                ),
                "model_response_chars": len(ref_resp.text) if ref_resp else 0,
                "evidence_alignment": alignment,
                "completed": completed,
            })

        # Enforce honesty gate — drop hypothesis evidence urls not in seen_urls.
        from oncology_arbiter.tools.honesty import filter_evidence_by_seen_urls
        gated_evidence = filter_evidence_by_seen_urls(
            [dict(e) for e in evidence], seen_urls
        )
        result.evidence_kept = len(gated_evidence)
        result.evidence_dropped = len(evidence) - len(gated_evidence)
        evidence = gated_evidence

        # Drop URLs from hypotheses that got filtered out.
        surviving_urls = {e["url"] for e in evidence if "url" in e}
        for h in hypotheses:
            h.evidence_urls = [u for u in h.evidence_urls if u in surviving_urls]

        if result.reflection_calls_completed != result.reflection_calls_expected:
            result.model_state = "llm_reflection_incomplete"
            result.notes = (
                "MedGemma REFLECT incomplete: "
                f"{result.reflection_calls_completed}/"
                f"{result.reflection_calls_expected} judgments carried valid receipts"
            )
        else:
            # ── TOURNAMENT ─────────────────────────────────────────────
            _run_tournament(hypotheses, llm, result)

            # ── META_REVIEW ────────────────────────────────────────────
            if result.tournament_pairs_completed != result.tournament_pairs_expected:
                result.model_state = "llm_tournament_incomplete"
                result.notes = (
                    "Pairwise MedGemma tournament incomplete: "
                    f"{result.tournament_pairs_completed}/"
                    f"{result.tournament_pairs_expected} judgments carried valid receipts"
                )
            else:
                try:
                    ranked = sorted(hypotheses, key=lambda h: h.elo, reverse=True)
                    meta_prompt = _build_meta_review_prompt(stage, ranked, evidence)
                    meta_resp = llm.chat(
                        [
                            {"role": "system", "content": _SYSTEM_META_REVIEW},
                            {"role": "user", "content": meta_prompt},
                        ],
                        max_tokens=1200,
                        temperature=0.4,
                    )
                    _record_llm(result, meta_resp)
                    result.meta_review_request_id = meta_resp.request_id
                    result.meta_review_response_sha256 = hashlib.sha256(
                        meta_resp.text.encode()
                    ).hexdigest()
                    result.meta_review = _final_model_text(meta_resp.text)
                    if not result.meta_review or not result.meta_review_request_id:
                        result.model_state = "llm_parse_failed"
                        result.notes = "MedGemma META_REVIEW had no final text or receipt"
                except LlmUnavailable:
                    result.model_state = "llm_unavailable"
                    result.notes = "MedGemma unavailable during META_REVIEW"

    except LlmUnavailable as e:
        result.model_state = "llm_unavailable"
        result.notes = f"LLM ladder exhausted: {e}"

    # Sort final hypotheses by Elo desc.
    result.hypotheses = sorted(hypotheses, key=lambda h: h.elo, reverse=True) if 'hypotheses' in locals() else []
    result.evidence = evidence
    result.seen_urls = sorted(seen_urls)
    result.seen_urls_count = len(seen_urls)
    result.latency_s = time.time() - t0
    return result


# ─── Prompt builders ───────────────────────────────────────────────────────

def _build_generate_prompt(stage: str, context: Mapping[str, Any], n: int) -> str:
    """Compose the GENERATE user prompt from patient context."""
    del n  # count lives in the system prompt; user carries context only
    ctx_lines = [f"Stage: {stage}"]
    for k, v in context.items():
        if v is None or v == "":
            continue
        if isinstance(v, (list, dict)):
            v = json.dumps(v, default=str)[:500]
        else:
            v = str(v)[:500]
        ctx_lines.append(f"- {k}: {v}")
    ctx_block = "\n".join(ctx_lines)
    return (
        "PATIENT CONTEXT:\n"
        + ctx_block
        + "\n\nReturn ONLY one JSON object with key 'hypotheses'."
    )


def _build_meta_review_prompt(
    stage: str,
    ranked: Sequence[Hypothesis],
    evidence: Sequence[Mapping[str, Any]],
) -> str:
    lines = [f"Stage: {stage}", "", "Top hypotheses (Elo-ranked):"]
    for i, h in enumerate(ranked[:5], 1):
        lines.append(
            f"{i}. [Elo {h.elo:.0f}, align={h.evidence_alignment}] {h.claim}"
        )
        if h.reflection:
            lines.append(f"   Reflection: {h.reflection[:300]}")
        if h.evidence_urls:
            lines.append(f"   URLs: {', '.join(h.evidence_urls[:3])}")
    if evidence:
        lines.append("")
        lines.append("Evidence URLs actually fetched:")
        for e in evidence[:8]:
            lines.append(f"- {e.get('url','?')}: {str(e.get('title',''))[:80]}")
    return "\n".join(lines)


# ─── Parsing helpers ───────────────────────────────────────────────────────

def _final_model_text(text: str) -> str:
    """Remove MedGemma reasoning wrappers and repeated turn terminators."""
    final_marker = "<unused95>final"
    thought_marker = "<unused94>thought"
    if final_marker in text:
        text = text.rsplit(final_marker, 1)[1]
    elif thought_marker in text:
        # A thought-only response exhausted its token budget before producing
        # the required final answer and must not be treated as completion.
        return ""
    for marker in ("<end_of_turn>", "<eos>", "</s>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return (
        text.replace("<end_of_turn>", "")
        .replace("<eos>", "")
        .replace("</s>", "")
        .strip()
    )


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Return the last valid JSON object from plain or reasoning-wrapped output."""
    t = _final_model_text(text)
    if not t and "<unused94>thought" in text:
        t = text.split("<unused94>thought", 1)[1]
        for marker in ("<unused95>final", "<end_of_turn>", "<eos>", "</s>"):
            if marker in t:
                t = t.split(marker, 1)[0]
        t = t.strip()
    if not t:
        return None
    # Strip common code-fence patterns
    if t.startswith("```"):
        m = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", t, re.DOTALL)
        if m:
            t = m.group(1)
    # MedGemma's causal chat template can emit a ``thought`` block containing
    # pseudo-JSON before a valid ``final`` object. Prefer the valid object that
    # ends latest, then its outermost start, so the final top-level object wins.
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, Dict[str, Any]]] = []
    for start in [match.start() for match in re.finditer(r"\{", t)]:
        try:
            value, consumed = decoder.raw_decode(t[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((start + consumed, -start, value))
    return max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None


def _chat_json(
    llm: GemmaClient,
    result: StageResult,
    system: str,
    user: str,
    *,
    max_tokens: int,
    temperature: float,
) -> Tuple[LlmResponse | None, Dict[str, Any] | None]:
    """Ask MedGemma for one JSON object. Retry once if the model only thought."""
    resp: LlmResponse | None = None
    try:
        resp = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        _record_llm(result, resp)
        parsed = _extract_json(resp.text)
        thought_only = parsed is None and (
            "<unused94>thought" in resp.text or _final_model_text(resp.text) == ""
        )
        # Retry only thought-budget exhaustion — not arbitrary malformed JSON —
        # so incomplete REFLECT/TOURNAMENT stay fail-closed on the first receipt.
        if thought_only:
            resp = llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": (
                        user
                        + "\n\nReturn ONLY one JSON object. No thought block. "
                        "No markdown. No prose."
                    )},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            _record_llm(result, resp)
            parsed = _extract_json(resp.text)
        return resp, parsed
    except LlmUnavailable:
        return resp, None


def _parse_hypotheses(
    text: str,
    expected_n: int,
    parsed: Optional[Dict[str, Any]] = None,
) -> List[Hypothesis]:
    """Parse GENERATE output into hypotheses. Tolerates minor JSON drift."""
    payload = parsed if isinstance(parsed, dict) else _extract_json(text)
    if not payload:
        return []
    raw_list = payload.get("hypotheses", [])
    if not isinstance(raw_list, list):
        return []
    result: List[Hypothesis] = []
    for raw in raw_list[: max(expected_n, 12)]:
        if not isinstance(raw, dict):
            continue
        claim = str(
            raw.get("claim")
            or raw.get("statement")
            or raw.get("hypothesis")
            or ""
        ).strip()
        if not claim:
            continue
        conf_raw = raw.get("initial_confidence", raw.get("confidence", 0.5))
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        result.append(Hypothesis(
            hypothesis_id=uuid.uuid4().hex,
            claim=claim[:400],
            rationale=str(raw.get("rationale", raw.get("reason", "")))[:1200].strip(),
            source_scope=str(raw.get("source_scope", "clinical_history")),
            initial_confidence=conf,
        ))
    return result


# ─── Evidence retrieval ───────────────────────────────────────────────────

_SEARCH_TOOL_NAMES = frozenset({
    "pubmed_search",
    "europe_pmc_search",
    "arxiv_search",
})


def _run_coro(factory):
    """Run an async tool from sync ``execute_stage`` even inside a live loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(factory())).result(timeout=60)


def _retrieve_evidence(
    stage: str,
    hypothesis: Hypothesis,
    seen_urls: set[str],
    *,
    context: Mapping[str, Any] | None = None,
    errors: List[str] | None = None,
    max_hits: int = 3,
) -> List[Mapping[str, Any]]:
    """Fetch evidence for one hypothesis using the stage's EVIDENCE tools.

    Returns list of evidence dicts. Each has at minimum ``url``, ``title``,
    ``snippet``. Also mutates ``hypothesis.evidence_urls``. Never raises.
    Search failures are recorded on ``errors`` instead of being swallowed.
    """
    tool_modules = _STAGE_PLANS[stage].get(AgentPhase.EVIDENCE, [])
    primary = _hypothesis_to_query(hypothesis, context=context)
    tokens = primary.split()
    queries = [primary]
    if len(tokens) > 4:
        queries.append(" ".join(tokens[:4]))
    collected: List[Mapping[str, Any]] = []
    error_sink = errors if errors is not None else []
    hit_cap = max(1, min(int(max_hits or 3), 6))

    for query in queries:
        if len(collected) >= hit_cap:
            break
        for mod_path in tool_modules:
            if len(collected) >= hit_cap:
                break
            try:
                mod = importlib.import_module(mod_path)
            except ImportError as exc:
                error_sink.append(f"{mod_path}: ImportError {exc}")
                continue
            tool_cls = _find_tool_class(mod)
            if tool_cls is None:
                error_sink.append(f"{mod_path}: no tool class")
                continue
            tool_name = getattr(tool_cls, "name", "")
            if tool_name not in _SEARCH_TOOL_NAMES:
                continue
            try:
                tool = tool_cls()
                ctx = _dummy_tool_ctx()
                result = _run_coro(
                    lambda t=tool, q=query, c=ctx: t.call(
                        {"query": q, "max_results": 3}, c
                    )
                )
            except Exception as exc:
                error_sink.append(f"{mod_path}: {type(exc).__name__}: {exc}")
                continue
            if result.is_error or not result.content:
                error_sink.append(
                    f"{mod_path}: {result.error_message or 'empty content'} query={query!r}"
                )
                continue
            recs = result.content.get("results", []) if isinstance(result.content, dict) else []
            if not recs:
                error_sink.append(f"{mod_path}: 0 hits query={query!r}")
                continue
            for rec in recs:
                url = (
                    rec.get("url")
                    or rec.get("pubmed_url")
                    or rec.get("doi_url")
                    or rec.get("pmid_url")
                )
                if not url:
                    continue
                seen_urls.add(url)
                hypothesis.evidence_urls.append(url)
                collected.append({
                    "url": url,
                    "title": rec.get("title", ""),
                    "snippet": (rec.get("abstract") or rec.get("summary") or "")[:400],
                    "source": mod_path.split(".")[-1],
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "query": query,
                })
            time.sleep(0.35)
    return collected


def _hypothesis_to_query(
    h: Hypothesis,
    *,
    context: Mapping[str, Any] | None = None,
) -> str:
    """Convert a hypothesis and high-signal clinical context into a query.

    The context terms prevent generic markers such as ``Ki-67`` or
    ``recurrence`` from retrieving evidence for an unrelated cancer.
    """
    stop = {"the","a","an","of","and","or","in","to","for","with","by","on",
            "is","are","was","were","be","been","being","this","that","these",
            "those","which","from","as","at","it","its","has","have","had",
            "patient","patients","de-identified","woman","man","high","higher",
            "despite","suggest","suggests","indicate","indicates","need",
            "needs","necessitating","risk","reported"}
    claim_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+\-]{1,}", h.claim)
    context_text = " ".join(
        str(context[key])
        for key in (
            "cancer",
            "tumor_type",
            "diagnosis",
            "pathology",
            "receptor_panel",
            "clinical_question",
        )
        if context and context.get(key)
    )
    context_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+\-]{1,}", context_text)
    keep: list[str] = []
    seen: set[str] = set()
    # Lead with three pathology/diagnosis anchors, then add the discriminating
    # hypothesis terms. Long PubMed queries overconstrain to zero results.
    for token in [*context_tokens[:3], *claim_tokens]:
        lowered = token.lower()
        if lowered in stop or lowered in seen:
            continue
        seen.add(lowered)
        keep.append(token)
        if len(keep) >= 6:
            break
    return " ".join(keep) if keep else h.claim[:80]


def _find_tool_class(module) -> Optional[type]:
    """Find the first attribute in module that has a ``call`` coroutine."""
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and hasattr(obj, "call") and hasattr(obj, "name"):
            return obj
    return None


def _dummy_tool_ctx():
    from oncology_arbiter.tools.base import ToolCtx
    return ToolCtx(
        artifacts_dir=Path("/tmp/oa-tool-runs"),
        session_id=uuid.uuid4().hex,
        task_id=None,
        run_id=uuid.uuid4().hex,
    )


def _summarize_evidence_for_hypothesis(
    evidence: Sequence[Mapping[str, Any]],
    h: Hypothesis,
) -> str:
    matches = [e for e in evidence if e.get("hypothesis_id") == h.hypothesis_id]
    if not matches:
        return "(no evidence retrieved)"
    lines = []
    for e in matches[:3]:
        lines.append(f"- [{e.get('source','?')}] {e.get('title','')[:140]}")
        snip = e.get("snippet", "").strip()
        if snip:
            lines.append(f"  {snip[:300]}")
        lines.append(f"  URL: {e.get('url','?')}")
    return "\n".join(lines)


# ─── Elo tournament ───────────────────────────────────────────────────────

def _run_tournament(
    hypotheses: List[Hypothesis],
    llm: GemmaClient,
    result: StageResult,
) -> None:
    """Deterministic pairwise Elo tournament.

    Pairs are sorted by (hypothesis_id_a, hypothesis_id_b) so the outcome
    order is reproducible across runs (given the LLM produces the same
    responses).
    """
    K = 32.0
    n = len(hypotheses)
    if n < 2:
        return

    # Deterministic pair ordering
    pairs: List[Tuple[Hypothesis, Hypothesis]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = hypotheses[i], hypotheses[j]
            if a.hypothesis_id > b.hypothesis_id:
                a, b = b, a
            pairs.append((a, b))
    pairs.sort(key=lambda p: (p[0].hypothesis_id, p[1].hypothesis_id))

    # Cap number of pairwise calls to bound cost (N=6 → 15 pairs, N=8 → 28).
    MAX_PAIRS = 30
    if len(pairs) > MAX_PAIRS:
        pairs = pairs[:MAX_PAIRS]
    result.tournament_pairs_expected = len(pairs)

    for a, b in pairs:
        pre_a = a.elo
        pre_b = b.elo
        pair_user = (
            f"Hypothesis A:\n{a.claim}\n\nReflection A: {a.reflection[:300]}\n"
            f"Alignment A: {a.evidence_alignment}\n\n"
            f"Hypothesis B:\n{b.claim}\n\nReflection B: {b.reflection[:300]}\n"
            f"Alignment B: {b.evidence_alignment}"
        )
        resp, j = _chat_json(
            llm,
            result,
            _SYSTEM_TOURNAMENT,
            pair_user,
            max_tokens=1600,
            temperature=0.2,
        )
        winner_raw = str((j or {}).get("winner", "tie")).strip().upper()
        winner = {
            "HYPOTHESIS A": "A",
            "HYPOTHESIS B": "B",
            "DRAW": "TIE",
        }.get(winner_raw, winner_raw)
        margin = str((j or {}).get("margin", "")).strip().lower()
        reason = str((j or {}).get("reason", "")).strip()
        response_sha256 = (
            hashlib.sha256(resp.text.encode()).hexdigest() if resp else None
        )
        valid_judgment = (
            bool(j)
            and winner in {"A", "B", "TIE"}
            and margin in {"clear", "moderate", "marginal", "tie"}
            and bool(reason)
            and bool(resp and resp.request_id)
        )
        if not valid_judgment:
            result.tournament_matches.append({
                "hypothesis_a_id": a.hypothesis_id,
                "hypothesis_b_id": b.hypothesis_id,
                "hypothesis_a_claim": a.claim,
                "hypothesis_b_claim": b.claim,
                "winner": "tie",
                "margin": "unavailable",
                "reason": "missing valid MedGemma pairwise judgment or request receipt",
                "model_request_id": resp.request_id if resp else None,
                "model_response_sha256": response_sha256,
                "model_response_chars": len(resp.text) if resp else 0,
                "completed": False,
                "elo_a_before": round(pre_a, 3),
                "elo_b_before": round(pre_b, 3),
                "elo_a_after": round(a.elo, 3),
                "elo_b_after": round(b.elo, 3),
            })
            continue

        if winner == "A":
            s_a, s_b = 1.0, 0.0
            a.tournament_wins += 1
            b.tournament_losses += 1
        elif winner == "B":
            s_a, s_b = 0.0, 1.0
            b.tournament_wins += 1
            a.tournament_losses += 1
        else:
            s_a, s_b = 0.5, 0.5
            a.tournament_ties += 1
            b.tournament_ties += 1

        expected_a = 1.0 / (1.0 + 10.0 ** ((b.elo - a.elo) / 400.0))
        expected_b = 1.0 - expected_a
        a.elo += K * (s_a - expected_a)
        b.elo += K * (s_b - expected_b)
        result.tournament_pairs_completed += 1
        result.tournament_matches.append({
            "hypothesis_a_id": a.hypothesis_id,
            "hypothesis_b_id": b.hypothesis_id,
            "hypothesis_a_claim": a.claim,
            "hypothesis_b_claim": b.claim,
            "winner": winner,
            "margin": margin,
            "reason": reason,
            "model_request_id": resp.request_id,
            "model_response_sha256": response_sha256,
            "model_response_chars": len(resp.text),
            "completed": True,
            "elo_a_before": round(pre_a, 3),
            "elo_b_before": round(pre_b, 3),
            "elo_a_after": round(a.elo, 3),
            "elo_b_after": round(b.elo, 3),
        })


def _record_llm(result: StageResult, resp: LlmResponse) -> None:
    result.llm_calls += 1
    result.llm_prompt_tokens += resp.prompt_tokens
    result.llm_completion_tokens += resp.completion_tokens
    result.llm_total_tokens += resp.prompt_tokens + resp.completion_tokens
    result.llm_cost_usd += resp.est_cost_usd
    result.llm_latency_s += resp.latency_s
    result.llm_provider = resp.route if result.llm_provider in {None, resp.route} else "mixed"
    result.llm_model = resp.model if result.llm_model in {None, resp.model} else "mixed"
    app_version = resp.app_version or None
    result.llm_app_version = (
        app_version
        if result.llm_app_version in {None, app_version}
        else "mixed"
    )
    revision = resp.revision_sha256 or None
    result.llm_revision_sha256 = (
        revision
        if result.llm_revision_sha256 in {None, revision}
        else "mixed"
    )
    if resp.request_id and resp.request_id not in result.llm_request_ids:
        result.llm_request_ids.append(resp.request_id)
    if resp.honesty_warning and resp.honesty_warning not in result.llm_honesty_warnings:
        result.llm_honesty_warnings.append(resp.honesty_warning)
