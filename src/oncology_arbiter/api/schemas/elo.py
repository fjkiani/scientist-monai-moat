"""Schemas for /v1/elo/rank."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import ApiEnvelope, EvidenceRecord



# --------------------------------------------------------------------------- #
# /v1/elo/rank  (v0.3.0-alpha)


class EloDrugCandidate(BaseModel):
    """One therapy candidate submitted to the Elo tournament.

    Deliberately small: the endpoint does NOT try to be a therapy engine.
    It only ranks the candidates the caller already assembled (e.g. from
    /v1/therapy/reason recommended_options, an NCCN pocket-guide branch,
    or a manually curated list). The tournament decides the *order*.

    Fields:
      - drug_id:      stable id used as the hyp_id (e.g. "olaparib_maintenance")
      - regimen:      display name (e.g. "Olaparib 300 mg BID maintenance")
      - line:         line of therapy (1 = first-line, 2 = second-line, …)
      - confidence:   caller-supplied prior in [0,1]. If unknown, use 0.5.
      - evidence:     list of {url, quoted_text, source} — same schema as
                      EvidenceRecord. Contributes to the honesty score.
      - honesty_markers: {proxy, gated, loaded} flags carried from the
                      caller's stage envelope; used verbatim by the scorer.
    """
    model_config = ConfigDict(extra="forbid")
    drug_id: str = Field(..., min_length=1, max_length=128,
                         description="Stable id used as Elo hypothesis id.")
    regimen: str = Field(..., min_length=1, max_length=256)
    line: int = Field(..., ge=1, le=5, description="Line of therapy (1..5)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    honesty_markers: dict[str, bool] = Field(
        default_factory=dict,
        description="{'proxy': bool, 'gated': bool, 'loaded': bool} — carried "
                    "from the caller's stage envelope. Missing keys default to false.",
    )



class EloRankRequest(BaseModel):
    """POST body for /v1/elo/rank.

    Runs a Co-Scientist Elo tournament twice:
      1. baseline_ranking  — the plain tournament, scores from the candidate
                             fields alone (this is what /v1/case/full does
                             for hypotheses derived from stage envelopes).
      2. enriched_ranking  — same candidates but with `modifiers` applied
                             to bump/discount specific drugs based on
                             disease context.

    `modifiers` is a free-form dict keyed by drug_id whose values are
    scalar deltas added to that drug's Elo `_score_hypothesis` before the
    tournament runs. Deltas are unclamped so callers can either boost a
    biomarker-matched drug or explicitly stress-test a low-confidence
    one. Undocumented drug_ids in `modifiers` are surfaced as warnings
    but do not fail the request.

    `disease_context` is echoed back verbatim in the response and used to
    stamp provenance — no server-side lookup, no hidden bumps.
    """
    model_config = ConfigDict(extra="forbid")
    drugs: list[EloDrugCandidate] = Field(
        ...,
        min_length=2, max_length=32,
        description="Between 2 and 32 candidates. Fewer than 2 makes no "
                    "tournament sense; 32 caps CPU cost at 496 pairs.",
    )
    modifiers: dict[str, float] = Field(
        default_factory=dict,
        description="{drug_id: score_delta}. Deltas added to the Elo score "
                    "for the enriched tournament. Not clamped.",
    )
    disease_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Echoed verbatim on the response (e.g. cancer, stage, "
                    "HRD status, PD-L1 CPS, prior_lines). Purely provenance.",
    )
    k_factor: int = Field(default=16, ge=1, le=64,
                          description="Elo K-factor. 16 mirrors run_co_scientist.")
    seed: int = Field(default=20260703, ge=0,
                      description="RNG seed. Same seed → same ranking.")



class EloMatchRecord(BaseModel):
    """One row of the ranking-diff table.

    For each drug, we report where it sat under the baseline tournament,
    where it moved to under the enriched tournament, and the reason
    (which modifier deltas applied). This is what the SPA's
    EloRankingPanel renders as the ‘why did the ranking change’ table.
    """
    model_config = ConfigDict(extra="forbid")
    drug_id: str
    regimen: str
    line: int
    baseline_rank: int = Field(..., ge=1,
                               description="1-based rank in the baseline tournament.")
    enriched_rank: int = Field(..., ge=1,
                               description="1-based rank in the enriched tournament.")
    baseline_rating: float
    enriched_rating: float
    rank_delta: int = Field(...,
        description="baseline_rank − enriched_rank. Positive → moved up (improved).")
    rating_delta: float
    applied_modifier: float = Field(default=0.0,
        description="Score delta applied to this drug from the modifiers map "
                    "(0.0 if no modifier for this drug_id).")
    reason: str = Field(default="",
        description="Human-readable reason: 'HRD+PARP boost', 'no modifier', "
                    "'unknown drug_id in modifiers', etc.")



class EloRankedEntry(BaseModel):
    """One row of a ranked list.

    Same shape as the dict rows emitted by run_co_scientist(hypotheses), but
    with the drug metadata (regimen, line) surfaced so the SPA doesn't have
    to cross-reference by hyp_id.
    """
    model_config = ConfigDict(extra="forbid")
    rank: int = Field(..., ge=1)
    drug_id: str
    regimen: str
    line: int
    rating: float
    wins: int = Field(..., ge=0)
    losses: int = Field(..., ge=0)
    draws: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    honesty_markers: dict[str, bool] = Field(default_factory=dict)
    n_evidence: int = Field(..., ge=0)



class EloRankResponse(ApiEnvelope):
    """Response for /v1/elo/rank.

    Carries the standard ApiEnvelope fields (disclaimer/caveat/provenance/
    honesty_gate/evidence/warnings) plus the two rankings and the diff.

    The disease_context and applied_modifiers fields are echoed verbatim
    so a downstream reader can reproduce the tournament byte-for-byte.
    """
    contract_version: Literal["v0.3.0-alpha"] = "v0.3.0-alpha"
    baseline_ranking: list[EloRankedEntry] = Field(default_factory=list)
    enriched_ranking: list[EloRankedEntry] = Field(default_factory=list)
    matches: list[EloMatchRecord] = Field(
        default_factory=list,
        description="Per-drug baseline↔enriched diff, ordered by enriched_rank.",
    )
    disease_context: dict[str, Any] = Field(default_factory=dict,
        description="Echo of the request's disease_context, verbatim.")
    applied_modifiers: dict[str, float] = Field(default_factory=dict,
        description="Echo of the request's modifiers, verbatim.")
    n_candidates: int = Field(..., ge=2, le=32,
        description="Count of drugs in the tournament (echoes request len).")
