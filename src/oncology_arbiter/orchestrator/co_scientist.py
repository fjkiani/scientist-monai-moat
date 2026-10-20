"""L5 Co-Scientist 4-phase loop.

Ports the *shape* of the Co-Scientist tumor-board loop:

    generate → reflect → rank (Elo tournament) → evolve

**Live mode (default when ``MEDGEMMA_MODAL_URL`` is set):**

  * Pairwise matches are judged by identity-locked MedGemma 1.5 4B
    (LLM-as-a-judge). Elo updates from the judge's winner, not a scalar hack.
  * ``evolve()`` asks MedGemma for clinical regimen / biomarker variants.
  * Variants still pass :mod:`.reflection` so cited URLs must be in ``seen_urls``.

**Offline mode (``mode=\"offline\"`` only — unit tests / ``/v1/offline_ranker/run``):**

  * Deterministic scalar ``_score_hypothesis`` + string-swap evolve. Never
    the product path for clinical reasoning.

Evidence entries attached to a hypothesis come from the caller or from
MedGemma-generated variants that only keep URLs already in ``seen_urls``.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from .reflection import LoopResult, reflect_and_filter

_JSON_RE = re.compile(r"\{[\s\S]*\}")


# --------------------------------------------------------------------------- #
# Data model


@dataclass
class Hypothesis:
    """One tumor-board hypothesis considered by the loop.

    Fields:
      hyp_id           — stable string id (e.g. 'biopsy:IDC:grade=2')
      stage            — 'screening' | 'biopsy' | 'therapy'
      statement        — short human-readable claim
      confidence       — [0,1] calibrated confidence carried from the stage
      evidence         — list of {url, quoted_text, source} dicts
      honesty_markers  — dict of flags read from stage envelope
                         (e.g. {'proxy': True, 'gated': False, 'weights_loaded': False})
      derived_from     — hyp_id of parent (for evolved variants); None otherwise
    """
    hyp_id: str
    stage: str
    statement: str
    confidence: float = 0.5
    evidence: list[dict[str, Any]] = field(default_factory=list)
    honesty_markers: dict[str, bool] = field(default_factory=dict)
    derived_from: str | None = None

    def as_evidence_record(self) -> dict[str, Any]:
        """Return the fields reflect_and_filter expects (dict shape)."""
        return {
            "hyp_id": self.hyp_id,
            "statement": self.statement,
            "evidence": list(self.evidence),
        }


@dataclass
class EloEntry:
    hypothesis: Hypothesis
    rating: float = 1500.0
    wins: int = 0
    losses: int = 0
    draws: int = 0


# --------------------------------------------------------------------------- #
# Phase 1 — generate


def generate_hypotheses(
    *,
    screening: dict[str, Any] | None,
    biopsy: dict[str, Any] | None,
    therapy: dict[str, Any] | None,
) -> list[Hypothesis]:
    """Enumerate hypotheses from stage envelopes.

    Each stage contributes at most 4 hypotheses so the tournament stays
    bounded. We use the stage envelope's own fields as ground truth — this
    module does NOT invent findings, subtypes, or regimens that the stage
    didn't already emit.
    """
    hyps: list[Hypothesis] = []

    if screening:
        prov = screening.get("provenance") or {}
        markers = {
            "proxy": (prov.get("model_state") or "").startswith("proxy_"),
            "gated": prov.get("model_state") == "gated",
            "loaded": (prov.get("model_state") or "").startswith("loaded"),
        }
        for f in (screening.get("findings") or [])[:4]:
            hyps.append(Hypothesis(
                hyp_id=f"screening:{f.get('label', 'unknown')}",
                stage="screening",
                statement=f"Screening finding '{f.get('label')}' scored {float(f.get('score', 0.0)):.4f}",
                confidence=float(f.get("score", 0.0)),
                evidence=list(screening.get("evidence") or []),
                honesty_markers=markers,
            ))

    if biopsy:
        prov = biopsy.get("provenance") or {}
        markers = {
            "proxy": (prov.get("model_state") or "").startswith("proxy_"),
            "gated": prov.get("model_state") == "gated",
            "loaded": (prov.get("model_state") or "").startswith("loaded"),
        }
        subtype = biopsy.get("subtype_prediction")
        conf = biopsy.get("confidence")
        if subtype is not None:
            hyps.append(Hypothesis(
                hyp_id=f"biopsy:{subtype}",
                stage="biopsy",
                statement=f"Biopsy subtype prediction: {subtype} at {(conf or 0.0):.2f} confidence",
                confidence=float(conf or 0.0),
                evidence=list(biopsy.get("evidence") or []),
                honesty_markers=markers,
            ))
        # Add a "wrong-subtype" alternate hypothesis so the tournament has
        # something to argue about. Its confidence is 1-p (residual mass).
        alternate_map = {
            "benign_or_normal": "invasive_carcinoma",
            "in_situ_carcinoma": "invasive_carcinoma",
            "invasive_carcinoma": "in_situ_carcinoma",
            # Preserve legacy caller-provided pathology subtypes without
            # equating the broader BACH labels above to IDC/DCIS.
            "IDC": "DCIS",
            "DCIS": "IDC",
            "benign": "IDC",
        }
        if subtype in alternate_map and conf is not None:
            alternates = alternate_map[subtype]
            hyps.append(Hypothesis(
                hyp_id=f"biopsy:{alternates}:alternate",
                stage="biopsy",
                statement=f"Alternate biopsy hypothesis: {alternates} instead of {subtype}",
                confidence=max(0.0, 1.0 - float(conf)),
                evidence=list(biopsy.get("evidence") or []),
                honesty_markers=markers,
            ))

    if therapy:
        prov = therapy.get("provenance") or {}
        markers = {
            "proxy": (prov.get("model_state") or "").startswith("proxy_"),
            "gated": prov.get("model_state") == "gated",
            "loaded": (prov.get("model_state") or "").startswith("loaded"),
        }
        for opt in (therapy.get("recommended_options") or [])[:3]:
            hyps.append(Hypothesis(
                hyp_id=f"therapy:{opt.get('regimen', 'unknown')}:line{opt.get('line_of_therapy', 1)}",
                stage="therapy",
                statement=f"Therapy: {opt.get('regimen')} (line {opt.get('line_of_therapy')})",
                confidence=0.5,  # therapy engine doesn't emit a probability
                evidence=list(opt.get("evidence") or []),
                honesty_markers=markers,
            ))
    return hyps


# --------------------------------------------------------------------------- #
# Phase 2 — reflect (honesty gate)


def reflect_hypotheses(
    hyps: Iterable[Hypothesis],
    seen_urls: set[str],
) -> tuple[list[Hypothesis], list[str]]:
    """Route every hypothesis through the honesty gate.

    Drops evidence entries whose URLs were not registered in `seen_urls`.
    Hypotheses whose evidence list becomes empty are STILL KEPT — dropping
    them entirely would silently narrow the tournament. Instead they get a
    `no_evidence_after_reflect:<hyp_id>` warning surfaced to the caller.
    """
    loop = LoopResult(seen_urls=set(seen_urls))
    kept: list[Hypothesis] = []
    warnings: list[str] = []
    for h in hyps:
        record, stage_warnings = reflect_and_filter(
            h.as_evidence_record(), loop, require_evidence=False,
        )
        h.evidence = list(record.get("evidence") or [])
        kept.append(h)
        for w in stage_warnings:
            warnings.append(f"reflect:{h.hyp_id}:{w}")
        if not h.evidence:
            warnings.append(f"no_evidence_after_reflect:{h.hyp_id}")
    return kept, warnings


# --------------------------------------------------------------------------- #
# Phase 3 — rank (Elo tournament)


def _score_hypothesis(h: Hypothesis) -> float:
    """OFFLINE-ONLY scalar. Forbidden on the live clinical path.

    Kept solely so ``mode=\"offline\"`` unit tests / ``/v1/offline_ranker/run``
    stay deterministic. Live ranking must call :func:`_judge_match_medgemma`.
    """
    ev = min(1.0, len(h.evidence) / 10.0)
    marker = 0.0
    if h.honesty_markers.get("loaded"):
        marker += 0.5
    if h.honesty_markers.get("proxy"):
        marker -= 0.3
    if h.honesty_markers.get("gated"):
        marker -= 0.5
    return h.confidence + ev + marker


def _expected(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected-score formula."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Prefer the supervisor MedGemma parser (thought/final + last valid object)."""
    try:
        from oncology_arbiter.agents.supervisor import _extract_json as _sup_extract
        parsed = _sup_extract(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    text = (text or "").strip()
    text = re.sub(
        r"<unused\d+>thought[\s\S]*?(?:</?unused\d+>|</thought>|$)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _chat_json_with_retry(
    llm: Any,
    *,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> tuple[Any, dict[str, Any]]:
    """Call MedGemma; retry once if thought-budget ate the JSON."""
    resp = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    parsed = _extract_json_object(resp.text) or {}
    usable = bool(parsed.get("winner")) or isinstance(parsed.get("variants"), list)
    if not usable:
        # Second shot: force final JSON when thought-budget ate the object.
        resp = llm.chat(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        user
                        + "\n\nReturn ONLY one JSON object. No thought block. "
                        "No markdown. No prose. First character must be '{'."
                    ),
                },
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        parsed = _extract_json_object(resp.text) or {}
    return resp, parsed


_JUDGE_SYSTEM = (
    "RUO oncology judge. Compare Hypothesis A vs B on efficacy, toxicity, "
    "and resistance. Prefer decisive A or B when biomarkers favor one option. "
    "Emit ONE JSON object (schema exact):\n"
    '{"winner":"A"|"B"|"DRAW","clinical_rationale":"<=40 words","margin":'
    '"clear"|"moderate"|"marginal"|"tie"}\n'
    "Start with '{'. RESEARCH USE ONLY."
)


def _judge_match_medgemma(
    hyp_a: Hypothesis,
    hyp_b: Hypothesis,
    *,
    patient_context: dict[str, Any] | None,
    llm: Any,
) -> dict[str, Any]:
    """LLM-as-a-judge pairwise clinical evaluation via MedGemma."""
    user = (
        "PATIENT_CONTEXT_JSON:\n"
        + json.dumps(patient_context or {}, ensure_ascii=False)[:2500]
        + "\n\nHypothesis A (id="
        + hyp_a.hyp_id
        + "):\n"
        + hyp_a.statement
        + "\nEvidence A: "
        + json.dumps(hyp_a.evidence[:2], ensure_ascii=False)[:800]
        + "\n\nHypothesis B (id="
        + hyp_b.hyp_id
        + "):\n"
        + hyp_b.statement
        + "\nEvidence B: "
        + json.dumps(hyp_b.evidence[:2], ensure_ascii=False)[:800]
        + "\n\nJSON:"
    )
    resp, parsed = _chat_json_with_retry(
        llm,
        system=_JUDGE_SYSTEM,
        user=user,
        # MedGemma 4B burns ~1k tokens on thought before emitting JSON.
        max_tokens=1400,
        temperature=0.2,
    )
    winner_raw = str(parsed.get("winner") or "DRAW").strip().upper()
    winner = {
        "A": "A",
        "B": "B",
        "HYPOTHESIS A": "A",
        "HYPOTHESIS B": "B",
        "DRAW": "DRAW",
        "TIE": "DRAW",
    }.get(winner_raw, "DRAW")
    rationale = str(parsed.get("clinical_rationale") or parsed.get("reason") or "").strip()
    margin = str(parsed.get("margin") or ("tie" if winner == "DRAW" else "moderate")).strip().lower()
    if margin not in {"clear", "moderate", "marginal", "tie"}:
        margin = "moderate"
    if not rationale:
        rationale = "MedGemma judge returned no clinical_rationale; treated as DRAW."
        winner = "DRAW"
        margin = "tie"
    return {
        "winner": winner,
        "clinical_rationale": rationale[:1200],
        "margin": margin,
        "model_request_id": getattr(resp, "request_id", None),
        "model": getattr(resp, "model", None),
        "route": getattr(resp, "route", None),
    }


def rank_hypotheses(
    hyps: list[Hypothesis],
    *,
    k_factor: int = 16,
    seed: int = 20260703,
    mode: Literal["live", "offline"] = "offline",
    patient_context: dict[str, Any] | None = None,
    llm: Any | None = None,
    transcript: list[dict[str, Any]] | None = None,
) -> list[EloEntry]:
    """Round-robin Elo tournament.

    ``mode=\"live\"``: each pair is judged by MedGemma (required ``llm``).
    ``mode=\"offline\"``: legacy scalar ``_score_hypothesis`` (tests / offline_ranker only).
    """
    rng = random.Random(seed)
    entries = {h.hyp_id: EloEntry(hypothesis=h) for h in hyps}
    ids = sorted(entries.keys())
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    rng.shuffle(pairs)
    # Bound live LLM cost (Modal thought tokens are expensive).
    if mode == "live" and len(pairs) > 3:
        pairs = pairs[:3]

    for pair_i, (a, b) in enumerate(pairs, start=1):
        if mode == "live":
            print(
                f"[co_scientist] live match {pair_i}/{len(pairs)}: {a} vs {b}",
                flush=True,
            )
        ea = entries[a]
        eb = entries[b]
        match_rec: dict[str, Any] = {
            "hypothesis_a_id": a,
            "hypothesis_b_id": b,
            "hypothesis_a_statement": ea.hypothesis.statement,
            "hypothesis_b_statement": eb.hypothesis.statement,
            "elo_a_before": round(ea.rating, 3),
            "elo_b_before": round(eb.rating, 3),
            "mode": mode,
        }
        if mode == "live":
            if llm is None:
                raise RuntimeError("live rank requires a MedGemma GemmaClient")
            judgment = _judge_match_medgemma(
                ea.hypothesis,
                eb.hypothesis,
                patient_context=patient_context,
                llm=llm,
            )
            winner = judgment["winner"]
            match_rec.update(judgment)
            if winner == "A":
                score_a, score_b = 1.0, 0.0
                ea.wins += 1
                eb.losses += 1
            elif winner == "B":
                score_a, score_b = 0.0, 1.0
                ea.losses += 1
                eb.wins += 1
            else:
                score_a, score_b = 0.5, 0.5
                ea.draws += 1
                eb.draws += 1
        else:
            sa = _score_hypothesis(ea.hypothesis)
            sb = _score_hypothesis(eb.hypothesis)
            if sa > sb:
                score_a, score_b = 1.0, 0.0
                ea.wins += 1
                eb.losses += 1
                match_rec["winner"] = "A"
            elif sb > sa:
                score_a, score_b = 0.0, 1.0
                ea.losses += 1
                eb.wins += 1
                match_rec["winner"] = "B"
            else:
                if a < b:
                    score_a, score_b = 1.0, 0.0
                    ea.wins += 1
                    eb.losses += 1
                    match_rec["winner"] = "A"
                else:
                    score_a, score_b = 0.0, 1.0
                    ea.losses += 1
                    eb.wins += 1
                    match_rec["winner"] = "B"
                ea.draws += 1
                eb.draws += 1
            match_rec["clinical_rationale"] = "offline_scalar_score"
            match_rec["margin"] = "n/a"
        exp_a = _expected(ea.rating, eb.rating)
        exp_b = 1.0 - exp_a
        ea.rating += k_factor * (score_a - exp_a)
        eb.rating += k_factor * (score_b - exp_b)
        match_rec["elo_a_after"] = round(ea.rating, 3)
        match_rec["elo_b_after"] = round(eb.rating, 3)
        if transcript is not None:
            transcript.append(match_rec)

    ordered = sorted(
        entries.values(),
        key=lambda e: (-e.rating, -e.wins, e.losses, e.hypothesis.hyp_id),
    )
    return ordered


# --------------------------------------------------------------------------- #
# Phase 4 — evolve


_EVOLVE_SYSTEM = (
    "You are a research-use-only oncology co-scientist. Given a winning "
    "clinical hypothesis and the patient biomarker profile, propose exactly "
    "N clinical variations (dose escalation/de-escalation, adding a CDK4/6 "
    "partner, pre-emptive resistance interception, alternate line, etc.). "
    "Return ONLY JSON:\n"
    '{"variants":[{"statement":"...","rationale":"...","confidence":0.0}]}\n'
    "confidence in [0,1]. Cite only URLs listed under ALLOWED_URLS. "
    "No markdown. First character '{'. RESEARCH USE ONLY."
)


def _evolve_offline(
    ranked: list[EloEntry],
    *,
    top_n: int,
    n_variants: int,
) -> list[Hypothesis]:
    """Legacy string-swap / conf-perturb evolve — offline tests only."""
    variants: list[Hypothesis] = []
    for entry in ranked[:top_n]:
        parent = entry.hypothesis
        if parent.stage == "screening":
            for delta in (0.1, -0.1)[:n_variants]:
                new_conf = max(0.0, min(1.0, parent.confidence + delta))
                variants.append(Hypothesis(
                    hyp_id=f"{parent.hyp_id}:conf{new_conf:.2f}",
                    stage=parent.stage,
                    statement=f"{parent.statement} (perturb conf {delta:+.2f})",
                    confidence=new_conf,
                    evidence=list(parent.evidence),
                    honesty_markers=dict(parent.honesty_markers),
                    derived_from=parent.hyp_id,
                ))
        elif parent.stage == "biopsy":
            if "invasive_carcinoma" in parent.hyp_id:
                new_subtype = "in_situ_carcinoma"
            elif "in_situ_carcinoma" in parent.hyp_id:
                new_subtype = "invasive_carcinoma"
            elif "benign_or_normal" in parent.hyp_id:
                new_subtype = "invasive_carcinoma"
            elif "IDC" in parent.hyp_id:
                new_subtype = "DCIS"
            elif "DCIS" in parent.hyp_id:
                new_subtype = "IDC"
            elif "benign" in parent.hyp_id:
                new_subtype = "IDC"
            else:
                continue
            variants.append(Hypothesis(
                hyp_id=f"{parent.hyp_id}:swap:{new_subtype}",
                stage=parent.stage,
                statement=f"{parent.statement} → swap subtype to {new_subtype}",
                confidence=parent.confidence,
                evidence=list(parent.evidence),
                honesty_markers=dict(parent.honesty_markers),
                derived_from=parent.hyp_id,
            ))
        elif parent.stage == "therapy":
            for delta in (1, -1)[:n_variants]:
                variants.append(Hypothesis(
                    hyp_id=f"{parent.hyp_id}:line{delta:+d}",
                    stage=parent.stage,
                    statement=f"{parent.statement} → line_of_therapy {delta:+d}",
                    confidence=parent.confidence,
                    evidence=list(parent.evidence),
                    honesty_markers=dict(parent.honesty_markers),
                    derived_from=parent.hyp_id,
                ))
    return variants


def _evolve_medgemma(
    ranked: list[EloEntry],
    *,
    top_n: int,
    n_variants: int,
    patient_context: dict[str, Any] | None,
    llm: Any,
    seen_urls: set[str],
) -> list[Hypothesis]:
    """Generative clinical variants from MedGemma; evidence clipped to seen_urls."""
    variants: list[Hypothesis] = []
    allowed = sorted(seen_urls)[:20]
    for entry in ranked[:top_n]:
        parent = entry.hypothesis
        user = (
            f"N={n_variants}\n"
            f"WINNING_HYPOTHESIS:\n{parent.statement}\n"
            f"STAGE={parent.stage} HYP_ID={parent.hyp_id}\n"
            f"PATIENT_CONTEXT_JSON:\n{json.dumps(patient_context or {}, ensure_ascii=False)[:3500]}\n"
            f"ALLOWED_URLS:\n{json.dumps(allowed, ensure_ascii=False)}\n"
            "JSON:"
        )
        _resp, parsed = _chat_json_with_retry(
            llm,
            system=_EVOLVE_SYSTEM.replace("N clinical", f"{n_variants} clinical"),
            user=user,
            max_tokens=1400,
            temperature=0.4,
        )
        rows = parsed.get("variants") if isinstance(parsed.get("variants"), list) else []
        for idx, row in enumerate(rows[:n_variants]):
            if not isinstance(row, dict):
                continue
            statement = str(row.get("statement") or "").strip()
            if not statement:
                continue
            try:
                conf = float(row.get("confidence") if row.get("confidence") is not None else parent.confidence)
            except (TypeError, ValueError):
                conf = parent.confidence
            conf = max(0.0, min(1.0, conf))
            rationale = str(row.get("rationale") or "").strip()
            # Only inherit parent evidence URLs that remain in seen_urls —
            # never invent new citations outside the honesty gate.
            evidence = [
                e for e in parent.evidence
                if isinstance(e, dict) and e.get("url") in seen_urls
            ]
            variants.append(Hypothesis(
                hyp_id=f"{parent.hyp_id}:gen{idx}:{abs(hash(statement)) % 10_000_000:07d}",
                stage=parent.stage,
                statement=statement if not rationale else f"{statement} — {rationale[:240]}",
                confidence=conf,
                evidence=evidence,
                honesty_markers={
                    **dict(parent.honesty_markers),
                    "loaded": True,
                    "generative_evolve": True,
                },
                derived_from=parent.hyp_id,
            ))
    return variants


def evolve_hypotheses(
    ranked: list[EloEntry],
    *,
    top_n: int = 3,
    n_variants: int = 2,
    mode: Literal["live", "offline"] = "offline",
    patient_context: dict[str, Any] | None = None,
    llm: Any | None = None,
    seen_urls: set[str] | None = None,
) -> list[Hypothesis]:
    """Spawn variants of the top hypotheses.

    Live mode asks MedGemma for clinical variations. Offline mode keeps the
    legacy string-swap catalog for deterministic unit tests.
    """
    if mode == "live":
        if llm is None:
            raise RuntimeError("live evolve requires a MedGemma GemmaClient")
        return _evolve_medgemma(
            ranked,
            top_n=top_n,
            n_variants=n_variants,
            patient_context=patient_context,
            llm=llm,
            seen_urls=set(seen_urls or []),
        )
    return _evolve_offline(ranked, top_n=top_n, n_variants=n_variants)


# --------------------------------------------------------------------------- #
# Top-level driver


def _resolve_mode(mode: Literal["live", "offline", "auto"]) -> Literal["live", "offline"]:
    if mode == "live":
        return "live"
    if mode == "offline":
        return "offline"
    return "live" if (os.environ.get("MEDGEMMA_MODAL_URL") or "").strip() else "offline"


def run_co_scientist(
    *,
    screening: dict[str, Any] | None,
    biopsy: dict[str, Any] | None,
    therapy: dict[str, Any] | None,
    seen_urls: Iterable[str] | None = None,
    top_n_evolve: int = 3,
    n_variants: int = 2,
    return_top: int = 8,
    patient_context: dict[str, Any] | None = None,
    mode: Literal["live", "offline", "auto"] = "auto",
    llm: Any | None = None,
) -> dict[str, Any]:
    """Run the four-phase loop end-to-end.

    Live mode (default when MedGemma URL is configured) uses MedGemma
    pairwise judging + generative evolve. Offline mode is the deterministic
    scalar path used only by ``/v1/offline_ranker/run`` and unit tests.
    """
    resolved = _resolve_mode(mode)
    seen_urls_set: set[str] = set(seen_urls or [])
    warnings: list[str] = []
    transcript: list[dict[str, Any]] = []

    live_llm = llm
    if resolved == "live" and live_llm is None:
        from oncology_arbiter.models.llm_client import GemmaClient
        from oncology_arbiter.models.medgemma_1_5_4b_wiring import assert_pinned_chat_url

        assert_pinned_chat_url((os.environ.get("MEDGEMMA_MODAL_URL") or "").strip())
        live_llm = GemmaClient()

    ctx = patient_context or {
        "screening_model_state": (screening or {}).get("provenance", {}).get("model_state"),
        "biopsy_subtype": (biopsy or {}).get("subtype_prediction"),
        "biopsy_receptors": (biopsy or {}).get("receptor_panel"),
        "therapy_options": [
            o.get("regimen") for o in ((therapy or {}).get("recommended_options") or [])
        ],
    }

    initial = generate_hypotheses(
        screening=screening, biopsy=biopsy, therapy=therapy,
    )
    initial_count = len(initial)

    reflected, ref_warnings = reflect_hypotheses(initial, seen_urls_set)
    warnings.extend(ref_warnings)
    after_reflect = len(reflected)

    ranked_pre = rank_hypotheses(
        reflected,
        mode=resolved,
        patient_context=ctx,
        llm=live_llm,
        transcript=transcript,
    )

    variants = evolve_hypotheses(
        ranked_pre,
        top_n=top_n_evolve,
        n_variants=n_variants,
        mode=resolved,
        patient_context=ctx,
        llm=live_llm,
        seen_urls=seen_urls_set,
    )
    variants, var_warnings = reflect_hypotheses(variants, seen_urls_set)
    warnings.extend(var_warnings)

    all_hyps = [entry.hypothesis for entry in ranked_pre] + variants
    after_evolve = len(all_hyps)

    ranked_final = rank_hypotheses(
        all_hyps,
        mode=resolved,
        patient_context=ctx,
        llm=live_llm,
        transcript=transcript,
    )

    top = []
    for entry in ranked_final[:return_top]:
        h = entry.hypothesis
        top.append({
            "hyp_id": h.hyp_id,
            "stage": h.stage,
            "statement": h.statement,
            "confidence": h.confidence,
            "evidence": list(h.evidence),
            "honesty_markers": dict(h.honesty_markers),
            "derived_from": h.derived_from,
            "rating": round(entry.rating, 4),
            "wins": entry.wins,
            "losses": entry.losses,
            "draws": entry.draws,
        })

    return {
        "phases": ["generate", "reflect", "rank", "evolve", "rank"],
        "engine": (
            "medgemma_pairwise_judge"
            if resolved == "live"
            else "offline_scalar_ranker"
        ),
        "mode": resolved,
        "warnings": warnings,
        "initial_count": initial_count,
        "after_reflect": after_reflect,
        "after_evolve": after_evolve,
        "hypotheses": top,
        "tournament_transcript": transcript,
        "patient_context": ctx,
    }
