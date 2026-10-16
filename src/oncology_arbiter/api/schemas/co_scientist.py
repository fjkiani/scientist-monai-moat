"""Schemas for /v1/offline_ranker/run (deterministic evidence ranker)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .common import ApiEnvelope



# --------------------------------------------------------------------------- #
# v0.4.0-alpha — POST /v1/co_scientist/run
#
# Surfaces the 4-phase Co-Scientist loop (generate → reflect → rank → evolve →
# rank) as a first-class endpoint, so callers can run the honesty tournament
# without going through /v1/case/full. Deterministic, no live LLM.
#
# The honesty contract is the load-bearing bit: any evidence URL not in
# `seed_urls` is dropped by the REFLECT phase. A hostile caller who
# hallucinates URLs will see them stripped from the response and reported
# in `warnings[]` as `dropped N hallucinated citation(s): [...]`.


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
