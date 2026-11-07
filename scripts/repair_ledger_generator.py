#!/usr/bin/env python
"""Retire proxy subsystems in the PROGRESS_LEDGER *generator*.

docs/PROGRESS_LEDGER.json is generated output: the source of truth is the
``SUBSYSTEMS`` literal inside scripts/update_progress_ledger.py.  Editing the
JSON alone is silently reverted on the next regeneration, so the honesty repair
has to land here.

Three entries claimed status LIVE while their backing behaviour was retired:

* ``L4a-screening-siglip-proxy`` -- a general-domain google/siglip-base-patch16-224
  encoder pointed at mammograms, with the now-deleted proxy test as evidence.
* ``L3-biopsy-arbiter``         -- an n_training=0 hand-drafted template,
  superseded by the METABRIC-fitted Breast DSS v3.
* ``L3-screening-arbiter`` / ``L3-therapy-arbiter`` -- roles that did not state
  the missingness policy or the explicit-inputs-only suppression rule.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "scripts" / "update_progress_ledger.py"
src = GEN.read_text()
applied: list[str] = []


def swap(old: str, new: str, label: str) -> None:
    global src
    if new in src:
        applied.append(f"  = {label} (already applied)")
        return
    if old not in src:
        raise SystemExit(f"ANCHOR MISSING for {label}:\n{old[:200]}")
    src = src.replace(old, new, 1)
    applied.append(f"  + {label}")


# --- L4a-screening-siglip-proxy -> RETIRED --------------------------------- #
swap(
    '''        "id": "L4a-screening-siglip-proxy",
        "layer": "L4a",
        "role": "Apache-2.0 ungated general-domain SigLIP used ONLY as a development proxy when HAI-DEF is denied. NEVER labeled as MedSigLIP.",
        "status": "LIVE",
        "not_wired_reason": None,
        "current_backend": "google/siglip-base-patch16-224",
        "hai_def_gate_state": None,
        "wired_files": [
            "src/oncology_arbiter/models/siglip_baseline.py",
            "src/oncology_arbiter/api/app.py",
        ],
        "evidence": [
            "tests/models/test_siglip_baseline.py",
            "tests/unit/test_screening_medsiglip_wiring.py::test_medsiglip_disabled_proxy_still_works",
            "/mnt/results/screening_response_medsiglip_gated_with_proxy_fallback.json",
        ],
    },''',
    '''        "id": "L4a-screening-siglip-proxy",
        "layer": "L4a",
        "role": (
            "RETIRED. General-domain google/siglip-base-patch16-224 was used as "
            "a development proxy on mammograms when HAI-DEF was denied. The "
            "screening route is now strict MedSigLIP-448 only: a denied or "
            "failing required stage returns pipeline_status="
            "'failed_required_stage' with findings=[] and overall_score=None "
            "rather than degrading to a non-medical image encoder."
        ),
        "status": "RETIRED",
        "not_wired_reason": (
            "Retired by production-integrity policy. A general-domain image "
            "encoder is not an acceptable substitute for MedSigLIP-448 on "
            "mammography, and its zero-shot scores were never calibrated for "
            "malignancy."
        ),
        "current_backend": None,
        "hai_def_gate_state": None,
        "wired_files": [],
        "evidence": [
            "tests/unit/test_screening_production_contract.py",
        ],
    },''',
    "L4a-screening-siglip-proxy -> RETIRED",
)

# --- L3-biopsy-arbiter -> RETIRED (superseded by Breast DSS v3) ------------- #
swap(
    '''        "evidence": [
            "tests/unit/test_api_arbiter_wiring.py::test_biopsy_endpoint_returns_arbiter_score",
        ],''',
    '''        "superseded_by": "breast_dss_v3",
        "evidence": [
            "tests/unit/test_breast_dss_api.py",
            "tests/unit/test_api_arbiter_wiring.py::test_biopsy_endpoint_emits_no_template_arbiter_score",
        ],''',
    "L3-biopsy-arbiter evidence -> Breast DSS v3",
)
swap(
    '''        "id": "L3-biopsy-arbiter",
        "layer": "L3",
        "role": "Logistic arbiter for biopsy stage classification.",
        "status": "LIVE",
        "not_wired_reason": None,''',
    '''        "id": "L3-biopsy-arbiter",
        "layer": "L3",
        "role": (
            "RETIRED. Hand-drafted template logistic arbiter for the biopsy "
            "stage. The biopsy route now returns arbiter_score=None and emits "
            "breast_dss_prognosis only when the full explicit seven-feature "
            "vector is supplied (no parser imputation)."
        ),
        "status": "RETIRED",
        "not_wired_reason": (
            "Superseded by breast_dss_arbiter_v3_metabric (n=1375, events=601, "
            "OOF AUROC 0.7074686031463495, incremental vs NPI+age delta "
            "0.019632112929677783, p=0.006). The template carried n_training=0 "
            "and illustrative coefficients."
        ),''',
    "L3-biopsy-arbiter -> RETIRED",
)

# --- L3-therapy-arbiter: state the suppression rule ------------------------ #
swap(
    '''        "id": "L3-therapy-arbiter",
        "layer": "L3",
        "role": "Logistic arbiter for therapy recommendation.",''',
    '''        "id": "L3-therapy-arbiter",
        "layer": "L3",
        "role": (
            "Deterministic logistic TRIAGE over explicit patient inputs only "
            "(illustrative coefficients, n_training=0). node_status_positive = "
            "lymph_nodes_pos > 0; zero nodes contributes exactly 0.0; missing "
            "nodes suppress scoring entirely and are never encoded as 0.5. Not "
            "treatment-benefit evidence and not a therapy recommendation -- "
            "recommendations come only from the authenticated SL bridge."
        ),''',
    "L3-therapy-arbiter role -> explicit-inputs-only triage",
)
swap(
    '''        "evidence": [
            "tests/unit/test_api_arbiter_wiring.py::test_therapy_endpoint_returns_arbiter_score",
        ],''',
    '''        "evidence": [
            "tests/unit/test_api_arbiter_wiring.py::test_therapy_arbiter_scores_only_from_explicit_inputs",
            "tests/unit/test_api_arbiter_wiring.py::test_therapy_arbiter_suppressed_when_nodes_missing",
        ],''',
    "L3-therapy-arbiter evidence -> explicit-input tests",
)

# --- L3-screening-arbiter: state the missingness policy -------------------- #
swap(
    '''        "id": "L3-screening-arbiter",
        "layer": "L3",
        "role": "L2-regularised logistic arbiter mapping BI-RADS-like features \\u2192 p_positive \\u2192 risk_bucket for screening triage.",'''
    if '\\u2192' in src
    else '''        "id": "L3-screening-arbiter",
        "layer": "L3",
        "role": "L2-regularised logistic arbiter mapping BI-RADS-like features \u2192 p_positive \u2192 risk_bucket for screening triage.",''',
    '''        "id": "L3-screening-arbiter",
        "layer": "L3",
        "role": (
            "L2-regularised logistic arbiter mapping BI-RADS-like features to "
            "p_positive and a risk bucket for screening triage (illustrative "
            "template coefficients, n_training=0). Unobserved booleans encode "
            "to the reference level 0.0 and contribute exactly zero log-odds; "
            "the prior 0.5 midpoint injected +1.15 log-odds and inflated p "
            "from 0.11920292202211755 to 0.29943286 (+151.20%) from absent "
            "data alone."
        ),''',
    "L3-screening-arbiter role -> missingness policy stated",
)

GEN.write_text(src)
print("GENERATOR REPAIRS")
for a in applied:
    print(a)
