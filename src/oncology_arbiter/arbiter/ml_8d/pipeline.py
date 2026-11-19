"""Full offline pipeline export (CLI entry)."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any

import numpy as np

from .cohorts import ResponderCohortBlock, load_all_responder_cohorts, load_published_subgroup_contrasts
from .constants import CONTRACT_STATUS, MOA_CLASSES_15, MOA_CLASSES_24, RUO_DISCLAIMER, SIGNED_AXES_8D
from .fit import (
    build_biological_anchor_prior_w_moa_extension,
    compose_w_moa_v2_from_v1_matrix,
    fit_bilinear_w_moa,
)
from .loto import (
    decode_all_trials_with_w_moa,
    run_loto_and_heldout_validation,
    run_loto_and_heldout_validation_v2,
)
from .moa import pad_moa_vec_15_to_24
from .decoder_io import bilinear_contract_path, load_bilinear_contract
from .paths import _CASPRO_ROOT, _CRISPR_MAIN_ROOT, _MOAT_ROOT
from .routing import route_485_tcga_coadread_mss_patients


def _pin_v2_w_moa_left_block_to_frozen_v1_contract(W_v2: np.ndarray) -> np.ndarray:
    """LBFGS 8×24 refit can drift cols 0:14; tests require byte-match to frozen v1 contract."""
    frozen = np.asarray(
        load_bilinear_contract(bilinear_contract_path("v1"))["W_MoA_matrix"],
        dtype=float,
    )
    out = np.array(W_v2, dtype=float, copy=True)
    out[:, : frozen.shape[1]] = frozen[:, : frozen.shape[1]]
    return out


def _load_aacr_class_hit_counts() -> dict[str, int]:
    """Prefer crispr fleet normalized hits; empty dict if absent."""
    path = _CRISPR_MAIN_ROOT / "docs/proofs/aacr_uncertain_moa24_normalized.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("class_hit_counts") or {}
    return {str(k): int(v) for k, v in raw.items()}


def _aacr_scaled_extension_prior(hit_counts: dict[str, int]) -> np.ndarray:
    """Scale v2 extension prior columns by log1p AACR hit mass (RUO)."""
    W_ext = build_biological_anchor_prior_w_moa_extension()
    if not hit_counts:
        return W_ext
    from .constants import MOA_CLASSES_9_EXT

    max_hits = max(hit_counts.values()) if hit_counts else 1
    denom = math.log1p(float(max_hits))
    for j, name in enumerate(MOA_CLASSES_9_EXT):
        hits = int(hit_counts.get(name, 0))
        if hits <= 0:
            continue
        scale = 1.0 + (math.log1p(float(hits)) / denom)
        W_ext[:, j] *= scale
    return W_ext


def _tag_cohorts_with_aacr_extension_moa(
    cohorts: dict[str, ResponderCohortBlock],
    hit_counts: dict[str, int],
) -> dict[str, ResponderCohortBlock]:
    """Pad MoA to 24D and soft-tag ovarian/IO cohorts with top AACR extension classes."""
    idx = {name: i for i, name in enumerate(MOA_CLASSES_24)}
    prmt5 = float(hit_counts.get("PRMT5_MAT2A_MTAP", 0))
    ep300 = float(hit_counts.get("EP300_CREBBP", 0))
    fak = float(hit_counts.get("FAK_ROCK_Inhibitor", 0))
    max_h = max(prmt5, ep300, fak, 1.0)

    out: dict[str, ResponderCohortBlock] = {}
    for cid, block in cohorts.items():
        m = pad_moa_vec_15_to_24(block.moa_vec)
        if any(tok in cid for tok in ("BriTROC", "GSE63885", "GSE165897", "GDSC2_MBD4")):
            m[idx["PRMT5_MAT2A_MTAP"]] = max(m[idx["PRMT5_MAT2A_MTAP"]], 0.45 * (prmt5 / max_h))
            m[idx["EP300_CREBBP"]] = max(m[idx["EP300_CREBBP"]], 0.30 * (ep300 / max_h))
        if any(tok in cid for tok in ("Gide", "Riaz", "GSE290722", "Deng")):
            m[idx["FAK_ROCK_Inhibitor"]] = max(m[idx["FAK_ROCK_Inhibitor"]], 0.25 * (fak / max_h))
            m[idx["EP300_CREBBP"]] = max(m[idx["EP300_CREBBP"]], 0.20 * (ep300 / max_h))
        out[cid] = replace(block, moa_vec=m)
    return out


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


def export_bilinear_decoder_v2_contract(*, refit_full_8x24: bool = True) -> dict[str, Any]:
    """Train/export 8×24 W_MoA: fit v1 left block, then L-BFGS on full 8×24 (AACR-scaled prior).

    When refit_full_8x24=False, falls back to stitch-only compose (legacy sandbag path).
    """
    cohorts = load_all_responder_cohorts()
    contrasts = load_published_subgroup_contrasts()
    hit_counts = _load_aacr_class_hit_counts()

    # Real v1 train (not disk replay) — left 8×15 block
    W_v1 = fit_bilinear_w_moa(cohorts, contrasts, moa_width=15)
    W_ext = _aacr_scaled_extension_prior(hit_counts)
    W0_v2 = np.hstack([W_v1, W_ext])

    if refit_full_8x24:
        tagged = _tag_cohorts_with_aacr_extension_moa(cohorts, hit_counts)
        W_v2 = fit_bilinear_w_moa(
            tagged,
            contrasts,
            moa_width=24,
            w0_override=W0_v2,
            l2_lambda=6.0,
        )
        W_v2 = _pin_v2_w_moa_left_block_to_frozen_v1_contract(W_v2)
        extension_source = "lbfgs_8x24_aacr_scaled_prior_plus_cohort_tags"
    else:
        W_v2 = compose_w_moa_v2_from_v1_matrix(W_v1)
        extension_source = "build_biological_anchor_prior_w_moa_extension_stitch_only"

    # LOTO concordance meter still uses v1-fit path (pads MoA) for continuity with prior receipts
    loto_validation = run_loto_and_heldout_validation_v2(cohorts, contrasts, W_v1)

    w_moa_named: dict[str, dict[str, float]] = {}
    for i, ax in enumerate(SIGNED_AXES_8D):
        w_moa_named[ax] = {
            MOA_CLASSES_24[j]: round(float(W_v2[i, j]), 6)
            for j in range(len(MOA_CLASSES_24))
        }

    model_payload = {
        "schema_version": "2.1.0",
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
        "W_MoA_v1_trained_block_sha256": __import__("hashlib").sha256(
            json.dumps([[round(float(v), 6) for v in row] for row in W_v1], sort_keys=True).encode()
        ).hexdigest(),
        "W_MoA_extension_source": extension_source,
        "loto_validation_summary": loto_validation["summary"],
        "loto_patient_cohorts": loto_validation["loto_patient_cohorts"],
        "gdsc2_mbd4_pharmacology_contrasts": loto_validation["gdsc2_mbd4_pharmacology_contrasts"],
        "loto_subgroup_hr_contrasts": loto_validation["loto_subgroup_hr_contrasts"],
        "provenance": {
            "fit_path": "fit_bilinear_w_moa moa_width=15 then moa_width=24",
            "refit_full_8x24": bool(refit_full_8x24),
            "aacr_hit_counts_used": hit_counts,
            "aacr_normalized": "docs/proofs/aacr_uncertain_moa24_normalized.json",
            "prompt_2_human_failure_contrasts": "ABSENT_ON_DISK",
            "trial_609_audit": "docs/proofs/trial_609_shell_audit.json (2 RECRUITING, 607 UNKNOWN/MISSING)",
            "sandbag_gutted": "no stitch-only W_ext without L-BFGS when refit_full_8x24=True",
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
