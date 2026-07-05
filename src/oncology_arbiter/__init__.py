"""oncology-arbiter — RESEARCH USE ONLY.

Multi-stage breast oncology reasoning platform:
    L4a screening → L4b biopsy → L4c therapy
each guarded by an L3 calibrated arbiter, orchestrated by an L5 Co-Scientist agent loop,
with an L2 evidence layer that enforces seen-URLs citation honesty and an L1 DICOM/WSI
data layer.

Not FDA-cleared. Not CE-marked. Not intended for clinical use.
"""
from __future__ import annotations

__version__ = "0.2.0-alpha"

RUO_DISCLAIMER = (
    "RESEARCH USE ONLY — not validated for clinical decision-making. "
    "Not FDA-cleared. Not CE-marked. Investigational / IRB context only."
)

# AUROC_CAVEAT copied verbatim from progression_arbiter's frozen model JSON,
# preserved to keep the honest-performance pattern intact across the platform.
AUROC_CAVEAT = (
    "AUROC reflects the model's ability to discriminate within literature-derived "
    "events whose labels and features co-originate from the same published narratives. "
    "This circularity inflates apparent performance. Expected prospective AUROC: "
    "0.70–0.85 based on independent validation."
)
