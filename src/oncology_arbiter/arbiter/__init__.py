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
from oncology_arbiter.immunotherapy.cart_ici_v3 import (
    load_arbiter_v3_contract,
    score_immunotherapy as immunotherapy_v3_arbiter,
)
from oncology_arbiter.ovarian.two_door_arbiter import (
    score_ovarian_case as ovarian_two_door_arbiter,
)


def mfap4_arbiter() -> L2LogisticArbiter:
    """Return the HGSOC MFAP4 platinum-resistance arbiter (GSE63885)."""
    return load_arbiter("mfap4")


__all__ = [
    "ArbiterResult",
    "L2LogisticArbiter",
    "RISK_BUCKETS",
    "load_arbiter",
    "screening_arbiter",
    "biopsy_arbiter",
    "therapy_arbiter",
    "mfap4_arbiter",
    "ovarian_two_door_arbiter",
    "immunotherapy_v3_arbiter",
    "load_arbiter_v3_contract",
]
