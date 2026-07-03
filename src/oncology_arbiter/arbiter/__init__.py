"""L3 calibrated logistic arbiters for mammography-driven oncology.

Mirrors the ``progression_arbiter`` pattern from
``fjkiani/org.backend/capabilities/progression_arbiter`` (L2-regularised
logistic regression scoring frozen JSON coefficients) — adapted for the
three-stage screening → biopsy → therapy decision funnel in
:mod:`oncology_arbiter`.

Three arbiters ship in this package:

* ``screening_arbiter``  — recall from mammogram vs. routine 1-year follow-up
* ``biopsy_arbiter``     — proceed to core-needle biopsy vs. 6-month short-interval follow-up
* ``therapy_arbiter``    — treatment intensity given biopsy result

Every arbiter loads a frozen JSON coefficient file (see ``models/``) and
returns a dict with:

    - ``p_positive``                — calibrated probability of the positive class
    - ``logit``                     — raw log-odds
    - ``risk_bucket``               — LOW / MID / HIGH
    - ``recommendation``            — arbiter-specific verb (e.g. RECALL_FOR_DIAGNOSTIC_WORKUP)
    - ``term_contributions``        — feature → contribution to logit
    - ``driving_feature``           — argmax |contribution|, excluding intercept
    - ``driving_feature_contribution`` — signed contribution of the driver
    - ``disclaimer``                — RUO_DISCLAIMER
    - ``caveat``                    — AUROC_CAVEAT

All three template models ship with ``n_training=0`` and the AUROC_CAVEAT
explicitly stating "TEMPLATE — coefficients illustrative, not fit on real
mammography outcomes". This is deliberate: the wiring, schema, honesty
gates, and sum-of-terms determinism are all lockable now, while the actual
frozen weights will be replaced once the Phase 3 EMBED/CBIS-DDSM training
runs land.

RESEARCH USE ONLY — see :data:`oncology_arbiter.RUO_DISCLAIMER`.
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
