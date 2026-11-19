"""Identity-locked wiring for ovarian_resistance_multivariate_v2.joblib.

The multivariate joblib + ``gse63885_multivariate_metrics.json`` are the
official ablation proof (permutation / ElasticNet isolate MFAP4).

Production scoring for ``POST /v1/case/full?cancer=hgsoc`` uses the
**best-OOF** model. When ``beats_univariate_loocv`` is false (HGB CV5 0.5007 /
Brier 0.3075 loses to univariate MFAP4 CV5 0.6801 / Brier 0.2377), live
predictions come from ``ovarian_mfap4_arbiter_v1.json`` with the multivariate
ablation comparison attached in provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

CAPABILITY = "hgsoc-platinum-resistance-multivariate-v2"
ARTIFACT_FILENAME = "ovarian_resistance_multivariate_v2.joblib"
ARTIFACT_SHA256 = (
    "82febea7003675941afeaffe268ec841281d17146188339900fba8f70f941e26"
)
GEO = "GSE63885"
HONEST_N = 75
# Locked from docs/proofs/gse63885_multivariate_metrics.json top_features[0]
MFAP4_PERMUTATION_IMPORTANCE = 0.13768531802965087
UNI_CV5_AUROC = 0.6800573888091822
UNI_CV5_BRIER = 0.23165745157982215
# Proof table LOOCV Brier for uni; Alpha cites CV Brier 0.2377 (LOOCV)
UNI_LOOCV_BRIER = 0.23774185025360312
HGB_CV5_AUROC = 0.5007173601147776
HGB_CV5_BRIER = 0.378260959380372


def artifact_path() -> Path:
    return Path(__file__).parent / "models" / ARTIFACT_FILENAME


def proof_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "proofs"
        / "gse63885_multivariate_metrics.json"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_sha256() -> str:
    return ARTIFACT_SHA256


def load_proof_receipt() -> dict[str, Any]:
    path = proof_path()
    if not path.is_file():
        raise FileNotFoundError(f"{CAPABILITY} proof missing: {path}")
    return json.loads(path.read_text())


def load_multivariate_bundle() -> dict[str, Any]:
    path = artifact_path()
    if not path.is_file():
        raise FileNotFoundError(f"{CAPABILITY} artifact missing: {path}")
    actual = _sha256(path)
    expected = expected_sha256()
    if actual != expected:
        raise RuntimeError(
            f"{CAPABILITY} artifact identity mismatch: "
            f"expected {expected}, got {actual}"
        )
    bundle = joblib.load(path)
    if int(bundle.get("n_training", 0)) != HONEST_N:
        raise RuntimeError(
            f"{CAPABILITY} refuses n_training={bundle.get('n_training')}"
        )
    loo = float(bundle.get("loocv_auroc", 0.0))
    if loo >= 0.75:
        raise RuntimeError(
            f"{CAPABILITY} refuses loocv_auroc={loo} — "
            "retracted n=101 0.7625 must never ship"
        )
    return bundle


def vectorize_features(
    features: dict[str, Any],
    feature_names: list[str],
) -> np.ndarray:
    """Build a 1×F row from a feature dict; missing → NaN (imputer handles)."""
    row = []
    for name in feature_names:
        if name not in features or features[name] is None:
            row.append(np.nan)
        else:
            row.append(float(features[name]))
    return np.asarray([row], dtype=float)


def _mfap4_expr_from_features(features: dict[str, Any]) -> float | None:
    for key in ("EMT_MFAP4", "mfap4_expr", "MFAP4"):
        if key in features and features[key] is not None:
            return float(features[key])
    return None


def _ablation_provenance(bundle: dict[str, Any], proof: dict[str, Any]) -> dict[str, Any]:
    perm = MFAP4_PERMUTATION_IMPORTANCE
    top = proof.get("top_features") or bundle.get("importances") or []
    if top and top[0].get("feature") == "EMT_MFAP4":
        perm = float(top[0].get("weight", perm))
    return {
        "beats_univariate_loocv": bool(
            proof.get("beats_univariate_loocv", bundle.get("beats_univariate_loocv"))
        ),
        "mfap4_permutation_importance": perm,
        "ablation_artifact": ARTIFACT_FILENAME,
        "ablation_artifact_sha256": ARTIFACT_SHA256,
        "ablation_winner_algorithm": bundle.get("winner"),
        "ablation_loocv_auroc": bundle.get("loocv_auroc"),
        "ablation_cv5_auroc": bundle.get("cv5_auroc"),
        "univariate_cv5_auroc": UNI_CV5_AUROC,
        "univariate_loocv_brier": UNI_LOOCV_BRIER,
        "ablation_hgb_cv5_auroc": HGB_CV5_AUROC,
        "ablation_hgb_cv5_brier": HGB_CV5_BRIER,
        "selection_rule": (
            "best_oof_wins: if beats_univariate_loocv is false, serve "
            "ovarian_mfap4_arbiter_v1; multivariate joblib retained as ablation proof"
        ),
    }


def _bucket(proba: float) -> tuple[str, str]:
    if proba < 0.3:
        return "LOW", "PLATINUM_SENSITIVE_LEAN"
    if proba < 0.7:
        return "MID", "INDETERMINATE"
    return "HIGH", "PLATINUM_RESISTANCE_LEAN"


def score_resistance(features: dict[str, Any]) -> dict[str, Any]:
    """Score platinum-resistance using best-OOF model + multivariate ablation.

    Multivariate ``ovarian_resistance_multivariate_v2.joblib`` stays on disk as
    the ablation proof. Live probability comes from univariate MFAP4 when the
    multivariate OOF does not beat it (clinical calibration safety).
    """
    from oncology_arbiter.arbiter.stage_mfap4_wiring import (
        ARTIFACT_SHA256 as MFAP4_SHA,
        load_stage_mfap4_arbiter,
    )

    bundle = load_multivariate_bundle()
    proof = load_proof_receipt()
    provenance = _ablation_provenance(bundle, proof)
    beats = provenance["beats_univariate_loocv"]

    if beats:
        # Multivariate won OOF — serve HGB pipeline (not the current GSE63885 state).
        names: list[str] = list(bundle["feature_names"])
        X = vectorize_features(features, names)
        proba = float(bundle["pipeline"].predict_proba(X)[0, 1])
        bucket, rec = _bucket(proba)
        return {
            "model_name": bundle["model_name"],
            "model_state": "loaded",
            "production_scorer": bundle["winner"],
            "winner_algorithm": bundle["winner"],
            "p_positive": round(proba, 6),
            "positive_class": bundle["positive_class"],
            "risk_bucket": bucket,
            "recommendation": rec,
            "n_training": bundle["n_training"],
            "loocv_auroc": bundle["loocv_auroc"],
            "cv5_auroc": bundle["cv5_auroc"],
            "beats_univariate_loocv": True,
            "mfap4_permutation_importance": provenance[
                "mfap4_permutation_importance"
            ],
            "univariate_comparator_auroc": bundle.get("univariate_comparator_auroc"),
            "feature_names": names,
            "features_used": {k: features.get(k) for k in names},
            "top_features": (bundle.get("importances") or [])[:10],
            "geo": GEO,
            "disclaimer": bundle.get("disclaimer"),
            "artifact_sha256": expected_sha256(),
            "provenance": provenance,
        }

    # Best-OOF: univariate MFAP4 (CV5 0.6801, Brier ~0.2377)
    mfap4_expr = _mfap4_expr_from_features(features)
    if mfap4_expr is None:
        raise ValueError(
            "best-OOF scorer requires EMT_MFAP4 (or mfap4_expr); "
            "multivariate ablation retained in provenance only"
        )
    arb = load_stage_mfap4_arbiter()
    result = arb.score({"mfap4_expr": mfap4_expr})
    return {
        "model_name": arb.model_name,
        "model_state": "loaded",
        "production_scorer": "ovarian_mfap4_arbiter_v1",
        "winner_algorithm": "univariate_mfap4_best_oof",
        "p_positive": result.p_positive,
        "positive_class": arb.positive_class,
        "risk_bucket": result.risk_bucket,
        "recommendation": result.recommendation,
        "n_training": arb.n_training,
        "loocv_auroc": None,
        "cv5_auroc": UNI_CV5_AUROC,
        "cv5_brier": UNI_CV5_BRIER,
        "loocv_brier": UNI_LOOCV_BRIER,
        "held_out_test_auroc": float(
            arb.performance.get("held_out_test_auroc", 0.6836)
        ),
        "beats_univariate_loocv": False,
        "mfap4_permutation_importance": provenance["mfap4_permutation_importance"],
        "univariate_comparator_auroc": bundle.get("univariate_comparator_auroc"),
        "feature_names": ["mfap4_expr"],
        "features_used": {"mfap4_expr": mfap4_expr, "EMT_MFAP4": mfap4_expr},
        "top_features": (proof.get("top_features") or [])[:10],
        "geo": GEO,
        "disclaimer": arb.disclaimer,
        "artifact_sha256": MFAP4_SHA,
        "ablation_artifact_sha256": ARTIFACT_SHA256,
        "provenance": provenance,
    }


def identity_block() -> dict[str, Any]:
    """Model identity without a patient probability (no features supplied)."""
    from oncology_arbiter.arbiter.stage_mfap4_wiring import (
        ARTIFACT_SHA256 as MFAP4_SHA,
        load_stage_mfap4_arbiter,
    )

    bundle = load_multivariate_bundle()
    proof = load_proof_receipt()
    provenance = _ablation_provenance(bundle, proof)
    arb = load_stage_mfap4_arbiter()
    return {
        "model_name": arb.model_name,
        "model_state": "loaded",
        "production_scorer": "ovarian_mfap4_arbiter_v1",
        "winner_algorithm": "univariate_mfap4_best_oof",
        "p_positive": None,
        "n_training": arb.n_training,
        "cv5_auroc": UNI_CV5_AUROC,
        "loocv_brier": UNI_LOOCV_BRIER,
        "beats_univariate_loocv": False,
        "mfap4_permutation_importance": provenance["mfap4_permutation_importance"],
        "univariate_comparator_auroc": bundle.get("univariate_comparator_auroc"),
        "ablation_loocv_auroc": bundle.get("loocv_auroc"),
        "ablation_cv5_auroc": bundle.get("cv5_auroc"),
        "top_features": (proof.get("top_features") or [])[:10],
        "artifact_sha256": MFAP4_SHA,
        "ablation_artifact_sha256": ARTIFACT_SHA256,
        "warning": (
            "No resistance_features supplied — best-OOF MFAP4 identity loaded; "
            "patient probability not emitted. Multivariate ablation proof retained."
        ),
        "disclaimer": arb.disclaimer,
        "provenance": provenance,
    }
