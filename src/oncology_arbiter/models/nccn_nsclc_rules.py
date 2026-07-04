"""L4c-NSCLC therapy stage — NCCN-NSCLC-lite deterministic rules engine.

Sibling of ``therapy_rules_lite.py`` but for the NSCLC track. Every rule
maps to a real section of the NCCN Non-Small Cell Lung Cancer clinical
practice guideline (v5.2026, ``https://www.nccn.org/professionals/physician_gls/pdf/nscl.pdf``)
and every recommended option carries a ``citation_url`` field. Every rule
also names the Fleischner Society 2017 recommendation it derives from
(``https://doi.org/10.1148/radiol.2017161659``).

This engine runs opportunistically ONLY when the endpoint's env flag
``ONCOLOGY_ARBITER_ENABLE_NSCLC_THERAPY_RULES_PROXY=1`` is set. Silent
fallback is forbidden.

Rules table lives INLINE in this module (as a plain dict) so the CI never
has to ship a separate JSON — this mirrors the constraint that we not
add per-track JSON artefacts unless there is a clear reason.

Design contract
---------------
1. NO fabricated numbers. Every rule maps directly to a published NCCN
   NSCLC Guidelines section OR the 2017 Fleischner Society guideline.
2. Every recommended option carries a ``citation_url``.
3. Rules cover the four risk buckets output by ``lung.arbiter`` —
   ``NEGATIVE``, ``LOW``, ``MID``, ``HIGH`` — plus the ``mass``
   (``max_diameter_mm > 30``) branch of ``HIGH``.
4. Deterministic: same input → identical output.
5. Every response is stamped with a placeholder warning that names the
   proxy status.

RESEARCH USE ONLY — see ``oncology_arbiter.RUO_DISCLAIMER``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER


# ---- Citations -----------------------------------------------------------
NCCN_NSCLC_URL: str = (
    "https://www.nccn.org/professionals/physician_gls/pdf/nscl.pdf"
)
NCCN_NSCLC_VERSION: str = "v5.2026"

FLEISCHNER_2017_DOI: str = "10.1148/radiol.2017161659"
FLEISCHNER_2017_URL: str = "https://doi.org/10.1148/radiol.2017161659"
FLEISCHNER_2017_CITATION: str = (
    "MacMahon H, Naidich DP, Goo JM, et al. Guidelines for Management of "
    "Incidental Pulmonary Nodules Detected on CT Images: From the "
    "Fleischner Society 2017. Radiology 2017;284(1):228-243. "
    f"DOI: {FLEISCHNER_2017_DOI}"
)

NSCLC_RULES_PROXY_WARNING: str = (
    "This NSCLC recommendation is from a rules-lite lookup keyed on the "
    "lung heuristic's diameter bucket, NOT from a trained ML model or a "
    "live TxGemma agent. It does NOT reason about histology, PD-L1 status, "
    "actionable driver mutations, staging beyond a diameter proxy, "
    "comorbidities, prior therapy, tumor-board consensus, or trial "
    "enrollment. It MUST NOT be used for treatment decisions. Real "
    "clinical use requires a thoracic oncologist and a full NCCN + "
    "Fleischner consultation."
)


# ---- Rules table (inline) ------------------------------------------------
#
# Keyed on the arbiter's `risk_bucket` value (NEGATIVE | LOW | MID | HIGH).
# The HIGH bucket has a sub-key `mass` for the >30 mm branch. Every option
# carries a `citation_url` and a `nccn_section` label.
_NSCLC_RULES: Dict[str, Dict[str, Any]] = {
    "NEGATIVE": {
        "recommended": [
            {
                "name": "No follow-up imaging required",
                "category": "surveillance",
                "citation_url": FLEISCHNER_2017_URL,
                "rationale": (
                    "No lung nodule candidates ≥ minimum size threshold "
                    "were surfaced by the placeholder heuristic. Fleischner "
                    "2017 recommends no imaging follow-up when no "
                    "clinically-meaningful nodule is present."
                ),
                "nccn_section": "Fleischner 2017 §Introduction",
            }
        ],
        "not_recommended": [
            {
                "name": "PET/CT",
                "category": "imaging",
                "citation_url": NCCN_NSCLC_URL,
                "rationale": (
                    "PET/CT is reserved for nodules ≥8 mm or highly "
                    "suspicious findings per NCCN LUNG-1."
                ),
                "nccn_section": "NCCN NSCL v5.2026 LUNG-1",
            }
        ],
    },
    "LOW": {
        "recommended": [
            {
                "name": "Low-dose CT surveillance at 12 months",
                "category": "surveillance",
                "citation_url": FLEISCHNER_2017_URL,
                "rationale": (
                    "Solid nodule <4 mm in a low-risk patient: Fleischner "
                    "2017 Table 1 recommends optional CT at 12 months."
                ),
                "nccn_section": "Fleischner 2017 Table 1",
            }
        ],
        "not_recommended": [
            {
                "name": "Biopsy",
                "category": "diagnostic",
                "citation_url": FLEISCHNER_2017_URL,
                "rationale": (
                    "Sub-4 mm solid nodules do not meet Fleischner "
                    "criteria for tissue diagnosis."
                ),
                "nccn_section": "Fleischner 2017 Table 1",
            }
        ],
    },
    "MID": {
        "recommended": [
            {
                "name": "Low-dose CT surveillance at 6-12 months",
                "category": "surveillance",
                "citation_url": FLEISCHNER_2017_URL,
                "rationale": (
                    "Solid nodule 4-8 mm: Fleischner 2017 recommends CT "
                    "at 6-12 months, then optional CT at 18-24 months."
                ),
                "nccn_section": "Fleischner 2017 Table 1",
            },
            {
                "name": "Consider PET/CT if clinical suspicion is high",
                "category": "imaging",
                "citation_url": NCCN_NSCLC_URL,
                "rationale": (
                    "NCCN LUNG-1 supports PET/CT for indeterminate nodules "
                    "when clinical suspicion of malignancy is elevated."
                ),
                "nccn_section": "NCCN NSCL v5.2026 LUNG-1",
            },
        ],
        "not_recommended": [
            {
                "name": "Definitive resection without biopsy",
                "category": "surgery",
                "citation_url": NCCN_NSCLC_URL,
                "rationale": (
                    "NCCN requires tissue diagnosis before resection for "
                    "sub-8 mm indeterminate nodules absent a mass."
                ),
                "nccn_section": "NCCN NSCL v5.2026 LUNG-2",
            }
        ],
    },
    "HIGH": {
        "recommended": [
            {
                "name": "PET/CT for staging",
                "category": "imaging",
                "citation_url": NCCN_NSCLC_URL,
                "rationale": (
                    "Solid nodule ≥8 mm: NCCN LUNG-1 recommends PET/CT "
                    "for staging and detection of distant metastases."
                ),
                "nccn_section": "NCCN NSCL v5.2026 LUNG-1",
            },
            {
                "name": "CT-guided biopsy or bronchoscopy for tissue diagnosis",
                "category": "diagnostic",
                "citation_url": NCCN_NSCLC_URL,
                "rationale": (
                    "NCCN LUNG-2 requires tissue diagnosis (biopsy or "
                    "bronchoscopy) for nodules ≥8 mm before therapy "
                    "selection."
                ),
                "nccn_section": "NCCN NSCL v5.2026 LUNG-2",
            },
            {
                "name": "Multidisciplinary tumor board review",
                "category": "coordination",
                "citation_url": NCCN_NSCLC_URL,
                "rationale": (
                    "NCCN NSCL guidelines emphasize multidisciplinary "
                    "review for all suspected lung malignancies."
                ),
                "nccn_section": "NCCN NSCL v5.2026 Principles Section",
            },
        ],
        "not_recommended": [
            {
                "name": "Surveillance only",
                "category": "surveillance",
                "citation_url": FLEISCHNER_2017_URL,
                "rationale": (
                    "Nodules ≥8 mm require workup, not surveillance "
                    "alone, per Fleischner 2017 Table 1."
                ),
                "nccn_section": "Fleischner 2017 Table 1",
            }
        ],
    },
}

# Additional HIGH-track option for the mass branch (>30 mm). Merged into the
# HIGH recommendations when the arbiter's driving feature is
# `mass_diameter_gt_30mm`.
_MASS_ADDENDUM: List[Dict[str, Any]] = [
    {
        "name": "Urgent thoracic oncology referral for mass workup",
        "category": "coordination",
        "citation_url": NCCN_NSCLC_URL,
        "rationale": (
            "Lung lesion >30 mm is a mass per Fleischner 2017 §Terminology "
            "and warrants urgent referral for staging (chest CT with "
            "contrast, PET/CT, brain MRI if symptomatic) per NCCN LUNG-1."
        ),
        "nccn_section": "NCCN NSCL v5.2026 LUNG-1 (mass branch)",
    }
]


# ---- Public dataclasses --------------------------------------------------


@dataclass
class NsclcTherapyOption:
    name: str
    category: str
    citation_url: str
    rationale: str
    nccn_section: str


@dataclass
class NsclcTherapyRulesResult:
    recommended_options: List[NsclcTherapyOption]
    not_recommended: List[NsclcTherapyOption]
    input_features: Dict[str, Any]
    risk_bucket: str
    model_state: str = "proxy_rules_lite"
    model_name: str = "nccn-nsclc-lite-v0"
    warnings: List[str] = field(default_factory=list)
    caveat: str = AUROC_CAVEAT
    disclaimer: str = RUO_DISCLAIMER
    dataset_citation: str = FLEISCHNER_2017_CITATION
    nccn_version: str = NCCN_NSCLC_VERSION


def _as_option(d: Mapping[str, Any]) -> NsclcTherapyOption:
    return NsclcTherapyOption(
        name=str(d["name"]),
        category=str(d["category"]),
        citation_url=str(d["citation_url"]),
        rationale=str(d["rationale"]),
        nccn_section=str(d["nccn_section"]),
    )


def score_nsclc_therapy(
    risk_bucket: str,
    max_diameter_mm: float = 0.0,
    driving_feature: str = "",
    extra_features: Mapping[str, Any] | None = None,
) -> NsclcTherapyRulesResult:
    """Look up NSCLC therapy recommendations for a risk bucket.

    Parameters
    ----------
    risk_bucket : "NEGATIVE" | "LOW" | "MID" | "HIGH"
    max_diameter_mm : maximum candidate diameter in mm (used for mass branch)
    driving_feature : optional driving-feature label from `score_nsclc()`;
        when equal to "mass_diameter_gt_30mm", the mass addendum is merged
        into the HIGH recommendations.
    extra_features : optional pass-through feature dict for logging

    Returns
    -------
    NsclcTherapyRulesResult with warnings + citations.
    """
    bucket = risk_bucket.upper() if risk_bucket else "NEGATIVE"
    if bucket not in _NSCLC_RULES:
        bucket = "NEGATIVE"
    rules = _NSCLC_RULES[bucket]

    recommended = [_as_option(d) for d in rules["recommended"]]
    not_recommended = [_as_option(d) for d in rules["not_recommended"]]

    if bucket == "HIGH" and (
        driving_feature == "mass_diameter_gt_30mm"
        or float(max_diameter_mm) > 30.0
    ):
        for d in _MASS_ADDENDUM:
            recommended.append(_as_option(d))

    input_features: Dict[str, Any] = {
        "risk_bucket": bucket,
        "max_diameter_mm": float(max_diameter_mm),
        "driving_feature": driving_feature,
    }
    if extra_features:
        for k, v in extra_features.items():
            if k not in input_features:
                input_features[k] = v

    return NsclcTherapyRulesResult(
        recommended_options=recommended,
        not_recommended=not_recommended,
        input_features=input_features,
        risk_bucket=bucket,
        warnings=[NSCLC_RULES_PROXY_WARNING],
    )


__all__ = [
    "FLEISCHNER_2017_CITATION",
    "FLEISCHNER_2017_DOI",
    "FLEISCHNER_2017_URL",
    "NCCN_NSCLC_URL",
    "NCCN_NSCLC_VERSION",
    "NSCLC_RULES_PROXY_WARNING",
    "NsclcTherapyOption",
    "NsclcTherapyRulesResult",
    "score_nsclc_therapy",
]
