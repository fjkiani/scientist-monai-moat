"""Identity-locked L3 logistic arbiters for the oncology stage funnel.

The screening and biopsy v1 artifacts are trained on patient-disjoint public
CBIS-DDSM training cases. The therapy v1 artifact models observed METABRIC
chemotherapy receipt and does not claim treatment efficacy or causal benefit.
Production factories verify each artifact SHA and reject ``n_training == 0``.
Every score retains a research-use-only disclaimer, retrospective-validation
caveat, coefficient-level contributions, and the sum-of-terms invariant.
"""
from __future__ import annotations

from .logistic import (
    ArbiterResult,
    L2LogisticArbiter,
    RISK_BUCKETS,
    load_arbiter,
    screening_arbiter,
    biopsy_arbiter,
    therapy_arbiter,
)

__all__ = [
    "ArbiterResult",
    "L2LogisticArbiter",
    "RISK_BUCKETS",
    "load_arbiter",
    "screening_arbiter",
    "biopsy_arbiter",
    "therapy_arbiter",
]
