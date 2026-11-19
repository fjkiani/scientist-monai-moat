"""Single-trial scoring and 42-trial ranking (frozen W_MoA)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .constants import SIGNED_AXES_8D
from .decoder_io import contract_moa_classes, load_bilinear_contract
from .exceptions import BilinearDecoderInputError
from .moa import moa_vec
from .patient import build_signed_8d_patient_vector
from .registry import (
    COMPARATOR_MOA_SPECS,
    RECOGNIZED_CLINICAL_FEATURE_KEYS,
    REGISTRY_42_MOA_SPECS,
)
from .scoring import signed_cosine_similarity, signed_fit_score

def score_bilinear_ml_loto(
    payload: Mapping[str, Any],
    *,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Score one signed 8D patient against one MoA mixture using frozen JSON weights.

    Query path. Does not call fit_bilinear_w_moa, load_all_responder_cohorts,
    build_signed_8d_patient_vector, or any RNG. PATH A patient_vector is rejected
    by omission: this function reads only p_signed_8d.
    """
    if payload.get("p_signed_8d") is None:
        raise BilinearDecoderInputError(
            "bilinear_ml_loto requires p_signed_8d (length 8, values in [-1, 1]). "
            "PATH A patient_vector is a different coordinate system and is not accepted.",
            code="missing_features",
            missing=["p_signed_8d"],
        )
    raw_p = payload["p_signed_8d"]
    if not isinstance(raw_p, (list, tuple)) or len(raw_p) != 8:
        raise BilinearDecoderInputError(
            "p_signed_8d must be a length-8 list",
            code="shape_mismatch",
            missing=["p_signed_8d"],
        )
    try:
        p = np.asarray([float(x) for x in raw_p], dtype=float)
    except (TypeError, ValueError) as exc:
        raise BilinearDecoderInputError(
            "p_signed_8d contains non-numeric values",
            code="shape_mismatch",
        ) from exc
    if not bool(np.all(np.isfinite(p))):
        raise BilinearDecoderInputError(
            "p_signed_8d contains non-finite values",
            code="shape_mismatch",
        )
    if bool(np.any(np.abs(p) > 1.0 + 1e-6)):
        raise BilinearDecoderInputError(
            "p_signed_8d values must lie in [-1, 1]",
            code="shape_mismatch",
        )

    moa = payload.get("moa_weights")
    if not isinstance(moa, Mapping) or len(moa) == 0:
        raise BilinearDecoderInputError(
            "bilinear_ml_loto requires moa_weights over contract MoA columns",
            code="missing_features",
            missing=["moa_weights"],
        )
    try:
        weights = {str(k): float(v) for k, v in moa.items()}
    except (TypeError, ValueError) as exc:
        raise BilinearDecoderInputError(
            "moa_weights values must be numeric",
            code="shape_mismatch",
        ) from exc
    if not any(abs(v) > 0.0 for v in weights.values()):
        raise BilinearDecoderInputError(
            "moa_weights are all zero",
            code="shape_mismatch",
        )

    contract = load_bilinear_contract(contract_path)
    moa_classes = contract_moa_classes(contract)
    w_moa = np.asarray(contract["W_MoA_matrix"], dtype=float)
    if w_moa.shape[0] != 8:
        raise BilinearDecoderInputError(
            f"W_MoA_matrix shape {w_moa.shape} is not (8, K)",
            code="contract_shape_mismatch",
        )
    unknown = [str(key) for key in weights if key not in moa_classes]
    if unknown:
        raise BilinearDecoderInputError(
            f"unknown moa columns: {unknown}",
            code="shape_mismatch",
        )
    m_vec = moa_vec(moa_classes, **weights)
    trial_vec = w_moa @ m_vec
    norm_t = float(np.linalg.norm(trial_vec))
    if norm_t <= 1e-12:
        raise BilinearDecoderInputError(
            "frozen W_MoA @ moa_weights has zero norm",
            code="shape_mismatch",
        )
    score = signed_fit_score(p, trial_vec)
    return {
        "engine": "bilinear_ml_loto",
        "model_id": contract.get("model_id"),
        "coordinate_system": "signed_bilinear_8d",
        "signed_axes_8d": list(contract["signed_axes_8d"]),
        "distinct_from_path_a_axes": True,
        "refit_on_query": False,
        "weights_source": "crispro_8d_bilinear_decoder_v1.json",
        "fallback_to_path_a": False,
        "ruo": True,
        "ruo_disclaimer": contract.get("ruo_disclaimer"),
        "contract_status": contract.get("contract_status"),
        "score": score,
        "trial_vector_raw": [float(x) for x in trial_vec],
        "l2_norm_trial": norm_t,
        "p_signed_8d": [float(x) for x in p],
        "moa_weights": weights,
        "loto_validation_summary_loaded": contract.get("loto_validation_summary"),
    }


def rank_42_trials_bilinear_ml(
    payload: Mapping[str, Any],
    *,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Rank all 42 decoded trials (and adjudicated comparators) for a patient using frozen W_MoA.

    Accepts either:
      - p_signed_8d: list of 8 floats in [-1, 1]
      - clinical_features: non-empty dict of recognized biomarker/genomic features
    Fail-closed on empty, malformed, or PATH A patient_vector inputs.
    """
    if not isinstance(payload, Mapping) or len(payload) == 0:
        raise BilinearDecoderInputError(
            "POST /v1/ranker/8d/bilinear requires p_signed_8d or clinical_features",
            code="missing_features",
            missing=["p_signed_8d", "clinical_features"],
        )

    if (
        payload.get("patient_vector") is not None
        and payload.get("p_signed_8d") is None
        and payload.get("clinical_features") is None
    ):
        raise BilinearDecoderInputError(
            "PATH A patient_vector is a different coordinate system and is rejected; "
            "provide p_signed_8d or clinical_features.",
            code="coordinate_system_mismatch",
            missing=["p_signed_8d"],
        )

    raw_p = payload.get("p_signed_8d")
    raw_cf = payload.get("clinical_features")

    if raw_p is not None:
        if not isinstance(raw_p, (list, tuple)) or len(raw_p) != 8:
            raise BilinearDecoderInputError(
                "p_signed_8d must be a length-8 list",
                code="shape_mismatch",
                missing=["p_signed_8d"],
            )
        try:
            p = np.asarray([float(x) for x in raw_p], dtype=float)
        except (TypeError, ValueError) as exc:
            raise BilinearDecoderInputError(
                "p_signed_8d contains non-numeric values",
                code="shape_mismatch",
            ) from exc
        if not bool(np.all(np.isfinite(p))):
            raise BilinearDecoderInputError(
                "p_signed_8d contains non-finite values",
                code="shape_mismatch",
            )
        if bool(np.any(np.abs(p) > 1.0 + 1e-6)):
            raise BilinearDecoderInputError(
                "p_signed_8d values must lie in [-1, 1]",
                code="shape_mismatch",
            )
        input_mode = "p_signed_8d"
    elif raw_cf is not None:
        if not isinstance(raw_cf, Mapping) or len(raw_cf) == 0:
            raise BilinearDecoderInputError(
                "clinical_features must be a non-empty object",
                code="missing_features",
                missing=["clinical_features"],
            )
        recognized = set(raw_cf.keys()) & RECOGNIZED_CLINICAL_FEATURE_KEYS
        if not recognized:
            raise BilinearDecoderInputError(
                "clinical_features contains no recognized 8D biomarker keys",
                code="missing_features",
                missing=["clinical_features"],
            )
        try:
            p = build_signed_8d_patient_vector(raw_cf)
        except (TypeError, ValueError) as exc:
            raise BilinearDecoderInputError(
                f"clinical_features contains malformed values: {exc}",
                code="shape_mismatch",
            ) from exc
        if not bool(np.all(np.isfinite(p))):
            raise BilinearDecoderInputError(
                "constructed p_signed_8d contains non-finite values",
                code="shape_mismatch",
            )
        input_mode = "clinical_features"
    else:
        raise BilinearDecoderInputError(
            "POST /v1/ranker/8d/bilinear requires either p_signed_8d or clinical_features",
            code="missing_features",
            missing=["p_signed_8d", "clinical_features"],
        )

    contract = load_bilinear_contract(contract_path)
    moa_classes = contract_moa_classes(contract)
    w_moa = np.asarray(contract["W_MoA_matrix"], dtype=float)

    def _score_spec_entry(spec: Mapping[str, Any]) -> dict[str, Any]:
        moa_map = {str(k): float(v) for k, v in spec["moa"].items()}
        m_j = moa_vec(moa_classes, **moa_map)
        t_raw = w_moa @ m_j
        norm_t = float(np.linalg.norm(t_raw))
        max_abs = float(np.max(np.abs(t_raw)))
        t_signed = (t_raw / max_abs) if max_abs > 1e-12 else np.zeros(8, dtype=float)
        fit_score = signed_fit_score(p, t_raw)
        cos_sim = signed_cosine_similarity(p, t_raw)
        calibrated_prob = float(1.0 / (1.0 + math.exp(-2.2 * fit_score)))
        axis_contribs = {
            SIGNED_AXES_8D[i]: round(float(p[i] * t_signed[i]), 6) for i in range(8)
        }
        reasons: list[str] = []
        if p[2] <= -0.50 and (moa_map.get("Anti_EGFR", 0.0) >= 0.50 or t_signed[2] >= 0.35):
            reasons.append("RAS_RAF_MUTANT_ANTI_EGFR_REPULSION")
        if p[1] >= 0.45 and (
            moa_map.get("PARP_Inhibitor", 0.0) >= 0.50
            or moa_map.get("ATR_WEE1_Inhibitor", 0.0) >= 0.50
        ):
            reasons.append("DOOR2_HIBERNATION_REPELS_DIVIDING_DDR")
        if (
            p[5] >= 0.55
            and p[4] <= -0.25
            and moa_map.get("Anti_PD1_PDL1", 0.0) >= 0.65
            and moa_map.get("Vaccine_STC1010", 0.0) == 0.0
            and moa_map.get("Anti_VEGF_Bev", 0.0) == 0.0
        ):
            reasons.append("STROMAL_MYELOID_BARRIER_IO_MONOTHERAPY_EXCLUSION")
        if p[7] <= -0.50:
            reasons.append("GATE0_SAFETY_OR_REGULATORY_HOLD")

        contraindicated = len(reasons) > 0 or fit_score < -0.25
        return {
            "registry_index": spec.get("idx"),
            "id": str(spec["id"]),
            "trial_name": str(spec["name"]),
            "program": str(spec["program"]),
            "moa_design_weights": moa_map,
            "learned_8d_vector_raw": [round(float(x), 6) for x in t_raw],
            "learned_8d_vector_signed": [round(float(x), 6) for x in t_signed],
            "learned_8d_by_axis": {
                SIGNED_AXES_8D[i]: round(float(t_signed[i]), 6) for i in range(8)
            },
            "bilinear_score": round(float(fit_score), 6),
            "cosine_similarity": round(float(cos_sim), 6),
            "calibrated_loto_probability": round(float(calibrated_prob), 6),
            "axis_contributions_8d": axis_contribs,
            "contraindicated": bool(contraindicated),
            "contraindication_reasons": reasons,
        }

    scored_42 = [_score_spec_entry(spec) for spec in REGISTRY_42_MOA_SPECS]
    scored_42.sort(key=lambda r: (not r["contraindicated"], r["bilinear_score"]), reverse=True)
    for rank_idx, row in enumerate(scored_42, start=1):
        row["rank"] = rank_idx

    scored_comparators = [_score_spec_entry(cspec) for cspec in COMPARATOR_MOA_SPECS]
    scored_comparators.sort(key=lambda r: (not r["contraindicated"], r["bilinear_score"]), reverse=True)
    for rank_idx, row in enumerate(scored_comparators, start=1):
        row["rank"] = rank_idx

    blacklist = [
        {
            "id": r["id"],
            "trial_name": r["trial_name"],
            "program": r["program"],
            "bilinear_score": r["bilinear_score"],
            "contraindication_reasons": r["contraindication_reasons"],
        }
        for r in (scored_42 + scored_comparators)
        if r["contraindicated"]
    ]

    custom_moa_eval = None
    if payload.get("moa_weights") is not None:
        single_res = score_bilinear_ml_loto(
            {"p_signed_8d": [float(x) for x in p], "moa_weights": payload["moa_weights"]},
            contract_path=contract_path,
        )
        custom_moa_eval = {
            "score": single_res["score"],
            "calibrated_loto_probability": round(
                float(1.0 / (1.0 + math.exp(-2.2 * float(single_res["score"])))), 6
            ),
            "trial_vector_raw": single_res["trial_vector_raw"],
            "l2_norm_trial": single_res["l2_norm_trial"],
            "moa_weights": single_res["moa_weights"],
        }

    top_trial = scored_42[0]
    return {
        "engine": "bilinear_ml_loto",
        "model_id": contract.get("model_id"),
        "coordinate_system": "signed_bilinear_8d",
        "input_mode": input_mode,
        "signed_axes_8d": list(SIGNED_AXES_8D),
        "distinct_from_path_a_axes": True,
        "refit_on_query": False,
        "weights_source": "crispro_8d_bilinear_decoder_v1.json",
        "fallback_to_path_a": False,
        "ruo": True,
        "ruo_disclaimer": contract.get("ruo_disclaimer"),
        "contract_status": contract.get("contract_status"),
        "p_signed_8d": [round(float(x), 6) for x in p],
        "p_signed_8d_named": {
            SIGNED_AXES_8D[i]: round(float(p[i]), 6) for i in range(8)
        },
        "top_trial": top_trial,
        "calibrated_loto_probability": top_trial["calibrated_loto_probability"],
        "top_bilinear_score": top_trial["bilinear_score"],
        "ranked_trials_42": scored_42,
        "ranked_comparator_trials": scored_comparators,
        "contraindication_gate_status": {
            "gate_active": True,
            "n_contraindicated_trials": len(blacklist),
            "repelled_trial_ids": [b["id"] for b in blacklist],
        },
        "contraindicated_blacklist": blacklist,
        "custom_moa_eval": custom_moa_eval,
        "loto_validation_summary": contract.get("loto_validation_summary"),
    }
