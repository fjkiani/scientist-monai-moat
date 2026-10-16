"""Schemas for /v1/offline_ranker/run (deterministic evidence ranker)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from .common import ApiEnvelope, EvidenceRecord



# --------------------------------------------------------------------------- #
# POST /v1/offline_ranker/run — deterministic evidence ranker, not Co-Scientist.
#
# The honesty contract is the load-bearing bit: any evidence URL not in
# `seed_urls` is dropped by the REFLECT phase. A hostile caller who
# hallucinates URLs will see them stripped from the response and reported
# in `warnings[]` as `dropped N hallucinated citation(s): [...]`.
# Real MedGemma GENERATE→EVIDENCE→REFLECT→Elo lives at POST /v1/co-scientist/run.


class CoScientistRunRequest(BaseModel):
    """Input for POST /v1/offline_ranker/run.

    The wired endpoint is an OFFLINE DETERMINISTIC RANKER: a fixed Elo-style
    ordering over caller-supplied stage envelopes. It performs no LLM call and
    its output must never be labelled as model-generated reasoning. Callers
    pass whatever stage envelopes they already have (screening, biopsy,
    therapy) plus the union of URLs their own tool loop fetched; any citation
    to an unseen URL is dropped.
    """

    screening: dict[str, Any] | None = Field(
        default=None,
        description=(
            "ScreeningResponse-shaped envelope (dict) or None. Used by "
            "generate_hypotheses to seed screening-track hypotheses."
        ),
    )
    biopsy: dict[str, Any] | None = Field(
        default=None,
        description=(
            "BiopsyResponse-shaped envelope (dict) or None. Used by "
            "generate_hypotheses to seed biopsy-track hypotheses."
        ),
    )
    therapy: dict[str, Any] | None = Field(
        default=None,
        description=(
            "TherapyResponse-shaped envelope (dict) or None. Used by "
            "generate_hypotheses to seed therapy-track hypotheses."
        ),
    )
    seed_urls: list[str] = Field(
        default_factory=list,
        description=(
            "URLs the tool-loop actually fetched. Any evidence URL NOT "
            "in this set is stripped by REFLECT. This is the ONLY "
            "authority on what was actually seen — plumb your fetch "
            "list here verbatim, no client-side filtering."
        ),
    )
    top_n_evolve: int = Field(
        default=3, ge=1, le=8,
        description="Top N ranked hypotheses to feed EVOLVE.",
    )
    n_variants: int = Field(
        default=2, ge=1, le=4,
        description="EVOLVE variants per seed hypothesis.",
    )
    return_top: int = Field(
        default=8, ge=1, le=32,
        description="Cap on hypotheses returned in `hypotheses[]`.",
    )



class CoScientistRunResponse(ApiEnvelope):
    """Output for POST /v1/offline_ranker/run.

    This is a deterministic evidence/Elo ranker, not the MedGemma
    Co-Scientist synthesis stage. Standard ApiEnvelope fields + per-phase
    counts + the ranked hypotheses are returned.
    The two drop counts are what a reviewer looks at to confirm the
    honesty gate did its job on a hostile input:

    - `urls_dropped_hallucinated`: total URL references stripped across
      all hypotheses because they weren't in `seed_urls`.
    - `hypotheses_dropped`: count of hypotheses whose evidence list was
      emptied by REFLECT (they're kept in `hypotheses[]` but flagged in
      `warnings[]` as `no_evidence_after_reflect:<hyp_id>`).
    """

    phases: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of phases executed, e.g. "
            "['generate','reflect','rank','evolve','rank']."
        ),
    )
    hypotheses: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Ranked hypotheses, up to `request.return_top`. Each entry has "
            "hyp_id, stage, statement, confidence, evidence, "
            "honesty_markers, derived_from, rating, wins, losses, draws."
        ),
    )
    initial_count: int = Field(
        default=0, ge=0,
        description="Hypothesis count after GENERATE.",
    )
    after_reflect: int = Field(
        default=0, ge=0,
        description="Hypothesis count after REFLECT (may equal initial_count).",
    )
    after_evolve: int = Field(
        default=0, ge=0,
        description="Hypothesis count after EVOLVE (includes variants).",
    )
    urls_dropped_hallucinated: int = Field(
        default=0, ge=0,
        description=(
            "Total URL references stripped by REFLECT because they weren't "
            "in `seed_urls`. This is the primary honesty metric — a "
            "hostile input with N fake URLs should see this ≥ N."
        ),
    )
    hypotheses_dropped: int = Field(
        default=0, ge=0,
        description=(
            "Hypotheses whose evidence list was emptied by REFLECT. Kept "
            "in `hypotheses[]` but a caller SHOULD treat them as untrusted."
        ),
    )


class RealCoScientistRequest(BaseModel):
    """Clinical context for the identity-locked MedGemma Co-Scientist loop."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal["screening", "biopsy", "therapy", "case_full"]
    context: dict[str, Any] = Field(
        ...,
        min_length=1,
        description="De-identified clinical context supplied to GENERATE.",
    )
    seed_urls: list[HttpUrl] = Field(
        default_factory=list,
        max_length=10,
        description="Optional URLs already fetched by the authenticated caller.",
    )
    n_hypotheses: int = Field(default=3, ge=2, le=8)
    n_evidence_top_k: int = Field(
        default=3, ge=1, le=6,
        description="Max literature hits retrieved per generated hypothesis.",
    )


class CoScientistHypothesis(BaseModel):
    """One reflected hypothesis after pairwise Elo ranking."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    claim: str
    rationale: str
    source_scope: str
    initial_confidence: float
    reflection: str
    evidence_alignment: str
    evidence_urls: list[str] = Field(default_factory=list)
    elo: float
    tournament_wins: int
    tournament_losses: int
    tournament_ties: int


class ReflectionReceipt(BaseModel):
    """Auditable receipt for one evidence-grounded MedGemma reflection."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    model_request_id: str | None = None
    model_response_sha256: str | None = None
    model_response_chars: int
    evidence_alignment: str
    completed: bool


class EloTournamentMatch(BaseModel):
    """Auditable receipt for one MedGemma pairwise judgment."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_a_id: str
    hypothesis_b_id: str
    hypothesis_a_claim: str
    hypothesis_b_claim: str
    winner: Literal["A", "B", "TIE", "tie"]
    margin: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    model_request_id: str | None = None
    model_response_sha256: str | None = None
    model_response_chars: int
    completed: bool
    elo_a_before: float
    elo_b_before: float
    elo_a_after: float
    elo_b_after: float


class RealCoScientistResponse(ApiEnvelope):
    """Evidence-grounded MedGemma Co-Scientist result with Elo receipts."""

    stage: Literal["screening", "biopsy", "therapy", "case_full"]
    supervisor_version: str
    hypotheses: list[CoScientistHypothesis] = Field(default_factory=list)
    meta_review: str
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    seen_urls: list[str] = Field(default_factory=list)
    reflection_calls_expected: int
    reflection_calls_completed: int
    reflection_receipts: list[ReflectionReceipt] = Field(default_factory=list)
    tournament_pairs_expected: int
    tournament_pairs_completed: int
    tournament_matches: list[EloTournamentMatch] = Field(default_factory=list)
    llm_calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    provider: str
    provider_chain: list[str] = Field(default_factory=list)
    model_id: str
    app_name: str
    model_revision: str
    model_request_ids: list[str] = Field(default_factory=list)
    meta_review_request_id: str
    meta_review_response_sha256: str
    latency_s: float
