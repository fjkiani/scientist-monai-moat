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
import json
import importlib
import importlib.util
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER
from oncology_arbiter.models.llm_client import GemmaClient, LlmUnavailable, LlmResponse

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

        - ``"executed"``           — real LLM loop ran end-to-end
        - ``"llm_unavailable"``    — LLM ladder exhausted, degraded to placeholder
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
    llm_cost_usd: float = 0.0
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
            "seen_urls_sample": self.seen_urls[:10],
            "evidence_kept": self.evidence_kept,
            "evidence_dropped": self.evidence_dropped,
            "llm_calls": self.llm_calls,
            "llm_total_tokens": self.llm_total_tokens,
            "llm_cost_usd": round(self.llm_cost_usd, 6),
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
                              "oncology_arbiter.tools.arxiv_search",
                              "oncology_arbiter.tools.web_fetch"],
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
    result = StageResult(stage=stage, model_state="executed")

    try:
        # ── GENERATE ────────────────────────────────────────────────────
        gen_prompt = _build_generate_prompt(stage, context, n_hypotheses)
        gen_resp = llm.chat(
            [{"role": "user", "content": gen_prompt}],
            max_tokens=1500,
            temperature=0.7,
        )
        _record_llm(result, gen_resp)
        hypotheses = _parse_hypotheses(gen_resp.text, expected_n=n_hypotheses)

        if not hypotheses:
            result.notes = "GENERATE returned no parseable hypotheses"
            result.model_state = "llm_parse_failed"
            result.latency_s = time.time() - t0
            return result

        # ── EVIDENCE ────────────────────────────────────────────────────
        # Rank by initial_confidence and fetch evidence for the top K.
        by_conf = sorted(hypotheses, key=lambda h: h.initial_confidence, reverse=True)
        for h in by_conf[:n_evidence_top_k]:
            ev = _retrieve_evidence(stage, h, seen_urls)
            evidence.extend(ev)

        # ── REFLECT ────────────────────────────────────────────────────
        for h in hypotheses:
            if not h.evidence_urls:
                h.reflection = "No evidence retrieved for this hypothesis in EVIDENCE phase."
                h.evidence_alignment = "unknown"
                continue
            ev_text = _summarize_evidence_for_hypothesis(evidence, h)
            try:
                ref_resp = llm.chat(
                    [
                        {"role": "system", "content": _SYSTEM_REFLECT},
                        {"role": "user", "content": (
                            f"Hypothesis:\n{h.claim}\n\nRationale:\n{h.rationale}\n\n"
                            f"Retrieved evidence:\n{ev_text}"
                        )},
                    ],
                    max_tokens=400,
                    temperature=0.3,
                )
                _record_llm(result, ref_resp)
                r_json = _extract_json(ref_resp.text)
                if r_json:
                    h.reflection = str(r_json.get("critique", ""))[:1200]
                    h.evidence_alignment = str(r_json.get("evidence_alignment", "unknown"))
            except LlmUnavailable:
                h.reflection = "LLM unavailable during REFLECT"
                h.evidence_alignment = "unknown"

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

        # ── TOURNAMENT ─────────────────────────────────────────────────
        _run_tournament(hypotheses, llm, result)

        # ── META_REVIEW ────────────────────────────────────────────────
        try:
            ranked = sorted(hypotheses, key=lambda h: h.elo, reverse=True)
            meta_prompt = _build_meta_review_prompt(stage, ranked, evidence)
            meta_resp = llm.chat(
                [
                    {"role": "system", "content": _SYSTEM_META_REVIEW},
                    {"role": "user", "content": meta_prompt},
                ],
                max_tokens=600,
                temperature=0.4,
            )
            _record_llm(result, meta_resp)
            result.meta_review = meta_resp.text.strip()
        except LlmUnavailable:
            result.meta_review = "LLM unavailable during META_REVIEW"

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
        _SYSTEM_GENERATE.format(n=n)
        + "\n\nPATIENT CONTEXT:\n" + ctx_block
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

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Strip code fences and try to parse JSON. Return None on failure."""
    t = text.strip()
    # Strip common code-fence patterns
    if t.startswith("```"):
        m = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", t, re.DOTALL)
        if m:
            t = m.group(1)
    # Grab from first { to last }
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _parse_hypotheses(text: str, expected_n: int) -> List[Hypothesis]:
    """Parse GENERATE output into hypotheses. Tolerates minor JSON drift."""
    parsed = _extract_json(text)
    if not parsed:
        return []
    raw_list = parsed.get("hypotheses", [])
    if not isinstance(raw_list, list):
        return []
    result: List[Hypothesis] = []
    for raw in raw_list[: max(expected_n, 12)]:
        if not isinstance(raw, dict):
            continue
        claim = str(raw.get("claim", "")).strip()
        if not claim:
            continue
        try:
            conf = float(raw.get("initial_confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        result.append(Hypothesis(
            hypothesis_id=uuid.uuid4().hex,
            claim=claim[:400],
            rationale=str(raw.get("rationale", ""))[:1200].strip(),
            source_scope=str(raw.get("source_scope", "clinical_history")),
            initial_confidence=conf,
        ))
    return result


# ─── Evidence retrieval ───────────────────────────────────────────────────

def _retrieve_evidence(
    stage: str,
    hypothesis: Hypothesis,
    seen_urls: set[str],
) -> List[Mapping[str, Any]]:
    """Fetch evidence for one hypothesis using the stage's EVIDENCE tools.

    Returns list of evidence dicts. Each has at minimum ``url``, ``title``,
    ``snippet``. Also mutates ``hypothesis.evidence_urls``. Never raises.
    """
    tool_modules = _STAGE_PLANS[stage].get(AgentPhase.EVIDENCE, [])
    query = _hypothesis_to_query(hypothesis)
    collected: List[Mapping[str, Any]] = []

    for mod_path in tool_modules:
        if len(collected) >= 3:  # cap per hypothesis
            break
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        tool_cls = _find_tool_class(mod)
        if tool_cls is None:
            continue
        try:
            tool = tool_cls()
            ctx = _dummy_tool_ctx()
            result = asyncio.run(tool.call({"query": query, "max_results": 3}, ctx))
        except Exception:
            continue
        if result.is_error or not result.content:
            continue
        # Extract records
        recs = result.content.get("results", []) if isinstance(result.content, dict) else []
        for rec in recs:
            url = rec.get("url") or rec.get("doi_url") or rec.get("pmid_url")
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
            })
    return collected


def _hypothesis_to_query(h: Hypothesis) -> str:
    """Convert a hypothesis claim into a search query.

    Strips filler words, keeps top ~8 content tokens.
    """
    stop = {"the","a","an","of","and","or","in","to","for","with","by","on",
            "is","are","was","were","be","been","being","this","that","these",
            "those","which","from","as","at","it","its","has","have","had"}
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", h.claim)
    keep = [t for t in tokens if t.lower() not in stop][:8]
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

    for a, b in pairs:
        try:
            resp = llm.chat(
                [
                    {"role": "system", "content": _SYSTEM_TOURNAMENT},
                    {"role": "user", "content": (
                        f"Hypothesis A:\n{a.claim}\n\nReflection A: {a.reflection[:300]}\n"
                        f"Alignment A: {a.evidence_alignment}\n\n"
                        f"Hypothesis B:\n{b.claim}\n\nReflection B: {b.reflection[:300]}\n"
                        f"Alignment B: {b.evidence_alignment}"
                    )},
                ],
                max_tokens=200,
                temperature=0.2,
            )
            _record_llm(result, resp)
            j = _extract_json(resp.text)
        except LlmUnavailable:
            j = None
        if not j:
            # No signal → tie
            expected_a = 1.0 / (1.0 + 10.0 ** ((b.elo - a.elo) / 400.0))
            a.elo += K * (0.5 - expected_a)
            b.elo += K * (0.5 - (1.0 - expected_a))
            continue

        winner = str(j.get("winner", "tie")).upper()
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

        expected_a = 1.0 / (1.0 + 10.0 ** ((b.elo - a.elo) / 400.0))
        expected_b = 1.0 - expected_a
        a.elo += K * (s_a - expected_a)
        b.elo += K * (s_b - expected_b)


def _record_llm(result: StageResult, resp: LlmResponse) -> None:
    result.llm_calls += 1
    result.llm_total_tokens += resp.prompt_tokens + resp.completion_tokens
    result.llm_cost_usd += resp.est_cost_usd
