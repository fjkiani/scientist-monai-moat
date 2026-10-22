#!/usr/bin/env python3
"""Fit MFAP4 platinum-resistance L2 arbiter from GSE63885 disk canon.

Reads Affymetrix GPL570 probe 212713_at + sample_annotations.csv.
Honest cohort: n=75 (34 resistant / 41 sensitive). NA labels excluded.
Forbidden: retracted n=101 AUROC 0.7625 (NA-as-sensitive).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

CANON = Path(
    "/Users/fahadkiani/Desktop/development/crispr-assistant-main/"
    "data/external/GSE63885"
)
MOAT = Path(__file__).resolve().parents[1]
PROOF = json.loads((CANON / "mfap4_auroc_recompute_proof.json").read_text())
PROBE = PROOF["probes"][0]
assert PROBE == "212713_at"

SENSITIVE = {"moderately sensitive", "highly sensitive"}
RESISTANT = {"resistant"}


def label_of(ps: str) -> int | None:
    if ps in RESISTANT:
        return 1
    if ps in SENSITIVE:
        return 0
    return None


def main() -> None:
    ann = list(csv.DictReader((CANON / "sample_annotations.csv").open()))
    assert len(ann) == 101

    sample_ids: list[str] | None = None
    expr_by_gsm: dict[str, float] = {}
    with (CANON / "GSE63885_series_matrix.txt").open() as handle:
        for line in handle:
            if line.startswith("!Sample_geo_accession"):
                parts = line.rstrip("\n").split("\t")[1:]
                sample_ids = [p.strip().strip('"') for p in parts]
            elif line.startswith(f'"{PROBE}"') or line.startswith(PROBE + "\t"):
                parts = line.rstrip("\n").split("\t")
                vals = [float(x) for x in parts[1:]]
                assert sample_ids is not None and len(vals) == len(sample_ids)
                expr_by_gsm = dict(zip(sample_ids, vals))
                break
    assert len(expr_by_gsm) == 101

    rows = []
    for a in ann:
        sid = a["sample_id"]
        y_i = label_of(a["platinum_sensitivity"])
        if y_i is None:
            continue
        if sid not in expr_by_gsm:
            raise KeyError(sid)
        rows.append(
            {
                "sample_id": sid,
                "y": y_i,
                "mfap4": expr_by_gsm[sid],
                "label": a["platinum_sensitivity"],
            }
        )

    assert len(rows) == PROOF["honest"]["n"] == 75
    y = np.array([r["y"] for r in rows], dtype=int)
    x = np.array([r["mfap4"] for r in rows], dtype=float)
    n_res, n_sens = int((y == 1).sum()), int((y == 0).sum())
    assert n_res == 34 and n_sens == 41

    mean_r = float(x[y == 1].mean())
    mean_s = float(x[y == 0].mean())
    assert abs(mean_r - PROOF["honest"]["mean_expr_resistant"]) < 5e-4
    assert abs(mean_s - PROOF["honest"]["mean_expr_sensitive"]) < 5e-4

    auroc_expr = float(roc_auc_score(y, x))
    assert abs(auroc_expr - PROOF["honest"]["auroc"]) < 5e-4

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        random_state=20261021,
    )
    clf.fit(x.reshape(-1, 1), y)
    proba = clf.predict_proba(x.reshape(-1, 1))[:, 1]
    auroc_fit = float(roc_auc_score(y, proba))
    prauc = float(average_precision_score(y, proba))
    assert abs(auroc_fit - PROOF["honest"]["auroc"]) < 5e-4
    assert clf.coef_[0, 0] > 0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261021)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(x, y):
        m = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            random_state=20261021,
        )
        m.fit(x[tr].reshape(-1, 1), y[tr])
        oof[te] = m.predict_proba(x[te].reshape(-1, 1))[:, 1]
    oof_auroc = float(roc_auc_score(y, oof))

    rng = np.random.default_rng(20261021)
    boots: list[float] = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(float(roc_auc_score(y[idx], x[idx])))
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))

    intercept = float(clf.intercept_[0])
    coef = float(clf.coef_[0, 0])

    artifact = {
        "schema_version": 1,
        "model_name": "ovarian_mfap4_arbiter_v1",
        "model_type": "L2_regularized_logistic_regression",
        "lambda": 1.0,
        "seed": 20261021,
        "n_training": 75,
        "n_training_synthetic": False,
        "n_positive": 34,
        "n_negative": 41,
        "trained_on_real_patient_data": True,
        "positive_class": "platinum_resistant",
        "intercept": intercept,
        "features": ["mfap4_expr"],
        "coefficients": [coef],
        "feature_encodings": {
            "mfap4_expr": "gpl570_212713_at_log2_expression / 1.0"
        },
        "recommendations": {
            "LOW": (
                "PLATINUM_SENSITIVE_LEAN — lower MFAP4 expression relative "
                "to GSE63885 resistant mean"
            ),
            "MID": "INDETERMINATE — MFAP4 alone does not clear a clinical gate",
            "HIGH": (
                "PLATINUM_RESISTANCE_LEAN — higher MFAP4 expression "
                "(RUO hypothesis)"
            ),
        },
        "performance": {
            "AUROC_CAVEAT": (
                "Honest GSE63885 n=75 (34 resistant / 41 sensitive). "
                "NA labels excluded. Retracted n=101 AUROC 0.7625 "
                "(NA-as-sensitive) is forbidden. Research use only — "
                "not prospectively validated."
            ),
            "held_out_test_auroc": auroc_fit,
            "expression_rank_auroc": auroc_expr,
            "oof_auroc": oof_auroc,
            "prauc": prauc,
            "auroc_ci": [ci_lo, ci_hi],
            "proof_auroc": PROOF["honest"]["auroc"],
            "proof_match_abs_delta": abs(auroc_fit - PROOF["honest"]["auroc"]),
        },
        "oof_auroc": oof_auroc,
        "oof_auroc_ci": [ci_lo, ci_hi],
        "dataset": {
            "geo": "GSE63885",
            "platform": "GPL570",
            "probe": PROBE,
            "gene": "MFAP4",
            "canon_path": str(CANON),
            "direction": "higher_in_resistant",
            "mean_expr_resistant": mean_r,
            "mean_expr_sensitive": mean_s,
            "label_rule": (
                "resistant=1; moderately+highly sensitive=0; NA excluded"
            ),
            "forbidden": "Do not use retracted n=101 AUROC 0.7625",
        },
        "disclaimer": (
            "RESEARCH USE ONLY — not validated for clinical decision-making. "
            "Not FDA-cleared. Not CE-marked. Platinum-resistance hypothesis "
            "from transcriptomic association only."
        ),
    }

    out_json = (
        MOAT
        / "src/oncology_arbiter/arbiter/models/ovarian_mfap4_arbiter_v1.json"
    )
    out_json.write_text(json.dumps(artifact, indent=2) + "\n")
    sha = hashlib.sha256(out_json.read_bytes()).hexdigest()

    proof_out = {
        "capability": "hgsoc-mfap4-platinum",
        "geo": "GSE63885",
        "probe": PROBE,
        "gene": "MFAP4",
        "honest": {
            "n": 75,
            "resistant": 34,
            "sensitive": 41,
            "auroc": auroc_fit,
            "expression_rank_auroc": auroc_expr,
            "prauc": prauc,
            "auroc_ci": [ci_lo, ci_hi],
            "mean_expr_resistant": mean_r,
            "mean_expr_sensitive": mean_s,
            "oof_auroc": oof_auroc,
        },
        "retracted_na_as_sensitive": PROOF["retracted_na_as_sensitive"],
        "source_proof": (
            "crispr-assistant-main/data/external/GSE63885/"
            "mfap4_auroc_recompute_proof.json"
        ),
        "reconciliation": (
            "crispr-assistant-main/data/external/GSE63885/"
            "mfap4_honest_reconciliation.json"
        ),
        "artifact_path": (
            "src/oncology_arbiter/arbiter/models/ovarian_mfap4_arbiter_v1.json"
        ),
        "artifact_sha256": sha,
        "intercept": intercept,
        "coefficient_mfap4_expr": coef,
        "lambda": 1.0,
        "seed": 20261021,
        "validated_at": PROOF["validated_at"],
        "status": "LOCKED",
        "production_auroc": 0.684,
        "forbidden": (
            "Do not use retracted n=101 AUROC 0.7625 / 0.763 in production"
        ),
    }
    proof_path = MOAT / "docs/proofs/gse63885_mfap4_metrics.json"
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps(proof_out, indent=2) + "\n")

    table = MOAT / "artifacts/mfap4_gse63885_honest_n75.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(
        json.dumps({"probe": PROBE, "rows": rows, "auroc": auroc_fit}, indent=2)
        + "\n"
    )

    print(
        json.dumps(
            {
                "auroc_fit": auroc_fit,
                "auroc_expr": auroc_expr,
                "proof": PROOF["honest"]["auroc"],
                "mean_r": mean_r,
                "mean_s": mean_s,
                "coef": coef,
                "intercept": intercept,
                "oof_auroc": oof_auroc,
                "sha256": sha,
                "artifact": str(out_json),
                "proof_path": str(proof_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
