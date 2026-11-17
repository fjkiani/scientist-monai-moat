"""Full offline pipeline export (CLI entry)."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .cohorts import load_all_responder_cohorts, load_published_subgroup_contrasts
from .constants import CONTRACT_STATUS, MOA_CLASSES_15, MOA_CLASSES_24, RUO_DISCLAIMER, SIGNED_AXES_8D
from .decoder_io import bilinear_contract_path, load_bilinear_contract
from .fit import compose_w_moa_v2_from_v1_matrix, fit_bilinear_w_moa
from .loto import (
    decode_all_trials_with_w_moa,
    run_loto_and_heldout_validation,
    run_loto_and_heldout_validation_v2,
)
from .paths import _CASPRO_ROOT, _CRISPR_MAIN_ROOT, _MOAT_ROOT
from .routing import route_485_tcga_coadread_mss_patients


def run_full_8d_bilinear_pipeline_and_export() -> dict[str, Any]:
    """Fit W_MoA, decode all 42 trials, run LOTO validation, route 485 CRC patients, and export JSONs."""
    cohorts = load_all_responder_cohorts()
    contrasts = load_published_subgroup_contrasts()

    W_MoA = fit_bilinear_w_moa(cohorts, contrasts)
    decoded_bundle = decode_all_trials_with_w_moa(W_MoA)
    loto_validation = run_loto_and_heldout_validation(cohorts, contrasts, W_MoA)
    routing_485 = route_485_tcga_coadread_mss_patients(W_MoA, decoded_bundle)

    # Format W_MoA as both matrix and named dictionary
    w_moa_named: dict[str, dict[str, float]] = {}
    for i, ax in enumerate(SIGNED_AXES_8D):
        w_moa_named[ax] = {
            MOA_CLASSES_15[j]: round(float(W_MoA[i, j]), 6)
            for j in range(len(MOA_CLASSES_15))
        }

    model_payload = {
        "schema_version": "1.0.0",
        "model_id": "crispro_8d_bilinear_decoder_v1",
        "ruo": True,
        "ruo_disclaimer": RUO_DISCLAIMER,
        "contract_status": CONTRACT_STATUS,
        "signed_axes_8d": list(SIGNED_AXES_8D),
        "moa_classes_15": list(MOA_CLASSES_15),
        "W_MoA_shape": [8, 15],
        "W_MoA_matrix": [[round(float(val), 6) for val in row] for row in W_MoA],
        "W_MoA_by_axis_and_moa": w_moa_named,
        "loto_validation_summary": loto_validation["summary"],
        "loto_patient_cohorts": loto_validation["loto_patient_cohorts"],
        "gdsc2_mbd4_pharmacology_contrasts": loto_validation["gdsc2_mbd4_pharmacology_contrasts"],
        "loto_subgroup_hr_contrasts": loto_validation["loto_subgroup_hr_contrasts"],
    }

    vectors_v3_payload = {
        "metadata": {
            "version": "v3_learned_signed_bilinear_8d",
            "source_registry": decoded_bundle["source_registry"],
            "source_registry_sha256": decoded_bundle["source_registry_sha256"],
            "total_trials": 42,
            "signed_axes_8d": list(SIGNED_AXES_8D),
            "moa_classes_15": list(MOA_CLASSES_15),
            "ruo": True,
            "ruo_disclaimer": RUO_DISCLAIMER,
            "contract_status": CONTRACT_STATUS,
            "unquarantined_programs": ["ATR_DDR (5/5)", "CEACAM5 (5/5)", "BREAK_CRC_001 (STC-1010)", "IO_APPENDIX (15/15)", "HISTORICAL_BENCHMARK (6/6)", "GBM_ESCAPEMAP (3/3)"],
        },
        "W_MoA_by_axis_and_moa": w_moa_named,
        "trials_42_learned_8d": decoded_bundle["decoded_42_trials"],
        "comparator_trials_learned_8d": decoded_bundle["decoded_comparator_trials"],
        "loto_responder_validation": loto_validation,
        "tcga_coadread_485_mss_routing": routing_485,
    }

    # Export to requested paths across scientist-monai-moat, caspro, and crispr-assistant-main
    out_paths_vectors = [
        _CASPRO_ROOT / "src/data/brenus/learned_42_trial_8d_vectors_v3.json",
        _CRISPR_MAIN_ROOT / "src/data/brenus/learned_42_trial_8d_vectors_v3.json",
        _MOAT_ROOT / "src/data/brenus/learned_42_trial_8d_vectors_v3.json",
    ]
    out_paths_model = [
        _MOAT_ROOT / "models/crispro_8d_bilinear_decoder_v1.json",
        _MOAT_ROOT / "src/oncology_arbiter/arbiter/models/crispro_8d_bilinear_decoder_v1.json",
        _CRISPR_MAIN_ROOT / "models/crispro_8d_bilinear_decoder_v1.json",
        _CASPRO_ROOT / "src/data/brenus/crispro_8d_bilinear_decoder_v1.json",
    ]

    vec_json_str = json.dumps(vectors_v3_payload, indent=2) + "\n"
    mod_json_str = json.dumps(model_payload, indent=2) + "\n"

    for p in out_paths_vectors:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(vec_json_str, encoding="utf-8")

    for p in out_paths_model:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(mod_json_str, encoding="utf-8")

    return vectors_v3_payload


def export_bilinear_decoder_v2_contract() -> dict[str, Any]:
    """Compose 8×24 W from frozen v1 weights + receipt-backed extension; run LOTO v2; write JSONs."""
    v1 = load_bilinear_contract(bilinear_contract_path("v1"))
    W_v1 = np.asarray(v1["W_MoA_matrix"], dtype=float)
    W_v2 = compose_w_moa_v2_from_v1_matrix(W_v1)

    cohorts = load_all_responder_cohorts()
    contrasts = load_published_subgroup_contrasts()
    loto_validation = run_loto_and_heldout_validation_v2(cohorts, contrasts, W_v1)

    w_moa_named: dict[str, dict[str, float]] = {}
    for i, ax in enumerate(SIGNED_AXES_8D):
        w_moa_named[ax] = {
            MOA_CLASSES_24[j]: round(float(W_v2[i, j]), 6)
            for j in range(len(MOA_CLASSES_24))
        }

    model_payload = {
        "schema_version": "2.0.0",
        "model_id": "crispro_8d_bilinear_decoder_v2",
        "ruo": True,
        "ruo_disclaimer": RUO_DISCLAIMER,
        "contract_status": CONTRACT_STATUS,
        "signed_axes_8d": list(SIGNED_AXES_8D),
        "moa_classes_15": list(MOA_CLASSES_15),
        "moa_classes_24": list(MOA_CLASSES_24),
        "W_MoA_shape": [8, 24],
        "W_MoA_matrix": [[round(float(val), 6) for val in row] for row in W_v2],
        "W_MoA_by_axis_and_moa": w_moa_named,
        "W_MoA_v1_frozen_block_sha256": __import__("hashlib").sha256(
            json.dumps(v1["W_MoA_matrix"], sort_keys=True).encode()
        ).hexdigest(),
        "W_MoA_extension_source": "build_biological_anchor_prior_w_moa_extension",
        "loto_validation_summary": loto_validation["summary"],
        "loto_patient_cohorts": loto_validation["loto_patient_cohorts"],
        "gdsc2_mbd4_pharmacology_contrasts": loto_validation["gdsc2_mbd4_pharmacology_contrasts"],
        "loto_subgroup_hr_contrasts": loto_validation["loto_subgroup_hr_contrasts"],
        "provenance": {
            "composed_from": "crispro_8d_bilinear_decoder_v1.json",
            "prompt_2_human_failure_contrasts": "ABSENT_ON_DISK",
            "trial_609_audit": "docs/proofs/trial_609_shell_audit.json (2 RECRUITING, 607 UNKNOWN/MISSING)",
        },
    }

    loto_receipt = {
        "decoder_version": "v2",
        "W_MoA_shape": [8, 24],
        **loto_validation,
    }

    out_paths_model = [
        _MOAT_ROOT / "models/crispro_8d_bilinear_decoder_v2.json",
        _MOAT_ROOT / "src/oncology_arbiter/arbiter/models/crispro_8d_bilinear_decoder_v2.json",
        _CRISPR_MAIN_ROOT / "models/crispro_8d_bilinear_decoder_v2.json",
        _CASPRO_ROOT / "src/data/brenus/crispro_8d_bilinear_decoder_v2.json",
    ]
    out_paths_loto = [
        _MOAT_ROOT / "src/oncology_arbiter/arbiter/models/crispro_8d_bilinear_decoder_v2_loto_receipt.json",
        _CRISPR_MAIN_ROOT / "docs/proofs/crispro_8d_bilinear_decoder_v2_loto_receipt.json",
    ]

    mod_json_str = json.dumps(model_payload, indent=2) + "\n"
    loto_json_str = json.dumps(loto_receipt, indent=2) + "\n"

    for p in out_paths_model:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(mod_json_str, encoding="utf-8")

    for p in out_paths_loto:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(loto_json_str, encoding="utf-8")

    return model_payload
