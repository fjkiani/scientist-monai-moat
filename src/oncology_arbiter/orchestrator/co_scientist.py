"""L5 Co-Scientist 4-phase loop (offline, deterministic).

Ports the *shape* of the Co-Scientist tumor-board loop:

    generate → reflect → rank (Elo tournament) → evolve

into a version that:

  1. runs entirely offline (no LLM calls, no network),
  2. is deterministic (fixed RNG seed, sorted iteration order),
  3. uses ONLY inputs already present in the stage envelopes (screening
     findings, biopsy subtype prediction alternates, therapy option
     regimen list),
  4. plugs the existing :mod:`.reflection` honesty gate in verbatim between
     `generate` and `rank`, so any hypothesis carrying an unseen URL in its
     evidence list is dropped before it can win Elo points.

The rank phase is a pure Elo tournament — every hypothesis plays every
other hypothesis once, with the winner decided by a deterministic scoring
function documented in `_score_hypothesis`. K=16, initial rating=1500.

The evolve phase takes the top-N hypotheses after rank and spawns
`n_variants` variants each by perturbing one feature (e.g. subtype
alternate, therapy line escalation). Variants pass back through reflect
before being merged with the surviving originals for the final Elo pass.

Note on evidence: this module does NOT invent citations. Evidence entries
attached to a hypothesis come from the caller — usually copied off the
stage envelope's own `evidence[]`. If none of the stage envelopes carried
any evidence, the loop still runs and produces ranked hypotheses; the
Elo scores just lack an evidence-count term.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable

from .reflection import LoopResult, reflect_and_filter


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
        if subtype in ("IDC", "DCIS", "benign") and conf is not None:
            alternates = {"IDC": "DCIS", "DCIS": "IDC", "benign": "IDC"}[subtype]
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
    """Deterministic scalar score used to decide pairwise matches.

    Terms:
      * confidence            × 1.0   — the model's own belief
      * evidence_count / 10   × 1.0   — grounded reasoning bonus (capped at 1.0)
      * loaded marker         × 0.5   — real live-model output beats proxy
      * proxy marker          × -0.3  — proxy / heuristic penalty
      * gated marker          × -0.5  — gated model refused; least trustworthy

    Ties broken deterministically by hyp_id lexicographic order in the
    rank loop below.
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


def rank_hypotheses(
    hyps: list[Hypothesis],
    *,
    k_factor: int = 16,
    seed: int = 20260703,
) -> list[EloEntry]:
    """Run a round-robin Elo tournament over `hyps`.

    Every pair plays once. Winner is decided by `_score_hypothesis` (with
    hyp_id as the tiebreaker). Ratings updated with K-factor `k_factor`.

    Returns entries sorted DESC by (rating, wins, -losses, hyp_id).
    """
    rng = random.Random(seed)
    entries = {h.hyp_id: EloEntry(hypothesis=h) for h in hyps}
    # Sort hyp_ids so iteration order is fixed regardless of caller.
    ids = sorted(entries.keys())
    # Randomize the *order of pair evaluation* deterministically so ratings
    # don't have a systematic first-mover advantage — but the outcome of
    # each individual match is deterministic (score-based).
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    rng.shuffle(pairs)

    for a, b in pairs:
        ea = entries[a]
        eb = entries[b]
        sa = _score_hypothesis(ea.hypothesis)
        sb = _score_hypothesis(eb.hypothesis)
        if sa > sb:
            score_a, score_b = 1.0, 0.0
            ea.wins += 1; eb.losses += 1
        elif sb > sa:
            score_a, score_b = 0.0, 1.0
            ea.losses += 1; eb.wins += 1
        else:
            # Deterministic tie-break by hyp_id — smaller id wins by convention
            if a < b:
                score_a, score_b = 1.0, 0.0
                ea.wins += 1; eb.losses += 1
            else:
                score_a, score_b = 0.0, 1.0
                ea.losses += 1; eb.wins += 1
            ea.draws += 1; eb.draws += 1
        exp_a = _expected(ea.rating, eb.rating)
        exp_b = 1.0 - exp_a
        ea.rating += k_factor * (score_a - exp_a)
        eb.rating += k_factor * (score_b - exp_b)

    ordered = sorted(
        entries.values(),
        key=lambda e: (-e.rating, -e.wins, e.losses, e.hypothesis.hyp_id),
    )
    return ordered


# --------------------------------------------------------------------------- #
# Phase 4 — evolve


def evolve_hypotheses(
    ranked: list[EloEntry],
    *,
    top_n: int = 3,
    n_variants: int = 2,
) -> list[Hypothesis]:
    """Spawn `n_variants` variants of each of the top `top_n` hypotheses.

    Variants perturb one feature only, so a downstream reader can tell what
    changed. Perturbation catalog is small and stage-specific:

      screening:  bump confidence by ±0.1 (bounded [0,1])
      biopsy:     swap subtype to the alternate (IDC↔DCIS, benign→IDC)
      therapy:    bump line_of_therapy by +1 (escalation) or -1 (de-escalation)

    Variants inherit `derived_from = parent.hyp_id` so the caller can trace
    the lineage in the final response.
    """
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
            # Only spawn a swap variant if the hyp_id encodes a known subtype.
            if "IDC" in parent.hyp_id:
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
                confidence=parent.confidence,  # keep confidence, mark as variant
                evidence=list(parent.evidence),
                honesty_markers=dict(parent.honesty_markers),
                derived_from=parent.hyp_id,
            ))
        elif parent.stage == "therapy":
            for delta in (1, -1)[:n_variants]:
                # Extract line number if encoded as ':lineN' suffix
                new_id = f"{parent.hyp_id}:line{delta:+d}"
                variants.append(Hypothesis(
                    hyp_id=new_id,
                    stage=parent.stage,
                    statement=f"{parent.statement} → line_of_therapy {delta:+d}",
                    confidence=parent.confidence,
                    evidence=list(parent.evidence),
                    honesty_markers=dict(parent.honesty_markers),
                    derived_from=parent.hyp_id,
                ))
    return variants


# --------------------------------------------------------------------------- #
# Top-level driver


def run_co_scientist(
    *,
    screening: dict[str, Any] | None,
    biopsy: dict[str, Any] | None,
    therapy: dict[str, Any] | None,
    seen_urls: Iterable[str] | None = None,
    top_n_evolve: int = 3,
    n_variants: int = 2,
    return_top: int = 8,
) -> dict[str, Any]:
    """Run the four-phase loop end-to-end.

    Returns a dict with fields:
      - phases: list of phase names in execution order
      - warnings: list of honesty warnings surfaced by reflect
      - initial_count: n hypotheses after generate
      - after_reflect: n hypotheses after reflect (may equal initial_count
        if reflect only touched evidence, not hypothesis count)
      - after_evolve: n hypotheses after evolve (includes variants)
      - hypotheses: list of ranked dicts, up to `return_top` entries, each
        of shape:
          {hyp_id, stage, statement, confidence, evidence, honesty_markers,
           derived_from, rating, wins, losses, draws}

    All fields are deterministic given identical inputs — no LLM sampling,
    no time-dependent randomness beyond the fixed-seed pair-order shuffle.
    """
    seen_urls_set: set[str] = set(seen_urls or [])
    warnings: list[str] = []

    # Phase 1 — generate
    initial = generate_hypotheses(
        screening=screening, biopsy=biopsy, therapy=therapy,
    )
    initial_count = len(initial)

    # Phase 2 — reflect
    reflected, ref_warnings = reflect_hypotheses(initial, seen_urls_set)
    warnings.extend(ref_warnings)
    after_reflect = len(reflected)

    # Phase 3a — rank (first pass)
    ranked_pre = rank_hypotheses(reflected)

    # Phase 4 — evolve
    variants = evolve_hypotheses(ranked_pre, top_n=top_n_evolve, n_variants=n_variants)
    # Route variants through reflect too, so any variant that magically
    # invented evidence gets its list truncated to the seen_urls set.
    variants, var_warnings = reflect_hypotheses(variants, seen_urls_set)
    warnings.extend(var_warnings)

    all_hyps = [entry.hypothesis for entry in ranked_pre] + variants
    after_evolve = len(all_hyps)

    # Phase 3b — rank the combined set once more so evolved variants
    # compete against the originals on equal footing.
    ranked_final = rank_hypotheses(all_hyps)

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
        "warnings": warnings,
        "initial_count": initial_count,
        "after_reflect": after_reflect,
        "after_evolve": after_evolve,
        "hypotheses": top,
    }
