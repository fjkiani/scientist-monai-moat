#!/usr/bin/env python3
"""Multivariate platinum-resistance ML on GSE63885 (honest n=75).

Builds EMT + DDR + clinical + biological ratio features, runs stratified
5-fold + LOOCV OOF benchmarks, exports the winning model as joblib.
Forbidden: retracted n=101 AUROC 0.7625; in-sample train metrics as proof.
"""
from __future__ import annotations

import csv
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SelectCols(BaseEstimator, TransformerMixin):
    """Column subsetter for univariate / EMT-panel pipelines (picklable)."""

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.cols]

warnings.filterwarnings("ignore", category=FutureWarning)

CANON = Path(
    "/Users/fahadkiani/Desktop/development/crispr-assistant-main/"
    "data/external/GSE63885"
)
MOAT = Path(__file__).resolve().parents[1]
SEED = 20261022
PROOF_UNI = 0.6836
DDR_GENES = ["BRCA1", "BRCA2", "RAD51", "PARP1", "ATM", "ATR"]

FIGO_ORD = {
    "I": 1,
    "IA": 1,
    "IB": 1,
    "IC": 1,
    "II": 2,
    "IIA": 2,
    "IIB": 2,
    "IIC": 2,
    "III": 3,
    "IIIA": 3,
    "IIIB": 3,
    "IIIC": 3,
    "IV": 4,
}
RESIDUAL_ORD = {"R0": 0, "R1": 1, "R2": 2}
GRADE_ORD = {"G1": 1, "G2": 2, "G3": 3, "G4": 4}


def _strip(v: str) -> str:
    return v.strip().strip('"')


def load_emt_mapping() -> dict[str, list[str]]:
    return json.loads((CANON / "emt_probe_mapping.json").read_text())


def load_ddr_probes() -> dict[str, list[str]]:
    """Map DDR gene symbols → GPL570 probe IDs via Gene symbol column."""
    out: dict[str, list[str]] = {g: [] for g in DDR_GENES}
    header = None
    gs_idx = None
    with (CANON / "GPL570.annot.txt").open() as handle:
        for line in handle:
            if header is None and line.startswith("ID\t"):
                header = line.rstrip("\n").split("\t")
                gs_idx = header.index("Gene symbol")
                continue
            if header is None or gs_idx is None:
                continue
            if line.startswith("!") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if gs_idx >= len(parts):
                continue
            gene = _strip(parts[gs_idx])
            if gene in out:
                out[gene].append(_strip(parts[0]))
    for gene, probes in out.items():
        if not probes:
            raise RuntimeError(f"No GPL570 probes for DDR gene {gene}")
    return out


def parse_series_clinical_and_expression(
    needed_probes: set[str],
) -> tuple[list[str], dict[str, dict], dict[str, np.ndarray]]:
    """Parse accession order, clinical characteristics, and probe expression."""
    sample_ids: list[str] | None = None
    clinical_rows: dict[str, list[str]] = {}
    expr: dict[str, np.ndarray] = {}

    with (CANON / "GSE63885_series_matrix.txt").open() as handle:
        for line in handle:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [_strip(x) for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [_strip(x) for x in line.rstrip("\n").split("\t")[1:]]
                # key = text before first ':'
                key = vals[0].split(":", 1)[0].strip().lower() if vals else "unknown"
                # normalize known keys
                if "histophatological" in key or "histopathological" in key:
                    key = "histology"
                elif key.startswith("tumor grade"):
                    key = "grade"
                elif key.startswith("figo"):
                    key = "figo"
                elif key.startswith("residual"):
                    key = "residual"
                elif key.startswith("brca1"):
                    key = "brca1"
                elif "platinium sensitivity" in key or "platinum sensitivity" in key:
                    key = "platinum"
                elif key.startswith("adjuwant") or key.startswith("adjuvant"):
                    key = "chemo"
                elif "tp53 accumulation" in key:
                    key = "tp53_ihc"
                elif "somatic tp53" in key:
                    key = "tp53_mut"
                clinical_rows[key] = vals
            elif line.startswith("!") or line.startswith("^"):
                continue
            else:
                # expression table rows: "PROBE"\tvals...
                parts = line.rstrip("\n").split("\t")
                if not parts:
                    continue
                probe = _strip(parts[0])
                if probe in needed_probes:
                    expr[probe] = np.array([float(x) for x in parts[1:]], dtype=float)

    if sample_ids is None:
        raise RuntimeError("missing Sample_geo_accession")
    missing = needed_probes - set(expr)
    if missing:
        raise RuntimeError(f"probes missing from series matrix: {sorted(missing)[:20]}")
    clinical = {sid: {} for sid in sample_ids}
    for key, vals in clinical_rows.items():
        if len(vals) != len(sample_ids):
            continue
        for sid, raw in zip(sample_ids, vals):
            if ":" in raw:
                clinical[sid][key] = raw.split(":", 1)[1].strip()
            else:
                clinical[sid][key] = raw
    return sample_ids, clinical, expr


def gene_mean(expr: dict[str, np.ndarray], probes: list[str], idx: int) -> float:
    return float(np.mean([expr[p][idx] for p in probes]))


def encode_clinical(c: dict) -> dict[str, float]:
    grade_raw = c.get("grade", "NA")
    figo_raw = c.get("figo", "NA")
    residual_raw = c.get("residual", "NA")
    brca1_raw = c.get("brca1", "NA")
    hist_raw = c.get("histology", "NA").lower()

    grade = GRADE_ORD.get(grade_raw.upper().replace(" ", ""), np.nan)
    figo = FIGO_ORD.get(figo_raw.upper().replace(" ", ""), np.nan)
    residual = RESIDUAL_ORD.get(residual_raw.upper().replace(" ", ""), np.nan)
    if brca1_raw.upper() in {"NA", ""}:
        brca1 = np.nan
    elif "no mutation" in brca1_raw.lower():
        brca1 = 0.0
    else:
        brca1 = 1.0
    serous = 1.0 if "serous" in hist_raw else 0.0
    undiff = 1.0 if "undifferentiated" in hist_raw else 0.0
    return {
        "grade_ord": float(grade) if grade == grade else np.nan,
        "figo_ord": float(figo) if figo == figo else np.nan,
        "residual_ord": float(residual) if residual == residual else np.nan,
        "brca1_mut": brca1,
        "hist_serous": serous,
        "hist_undiff": undiff,
    }


def build_matrix() -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict]:
    emt_map = load_emt_mapping()
    ddr_map = load_ddr_probes()
    needed: set[str] = set()
    for probes in emt_map.values():
        needed.update(probes)
    for probes in ddr_map.values():
        needed.update(probes)

    sample_ids, clinical, expr = parse_series_clinical_and_expression(needed)
    ann = {r["sample_id"]: r for r in csv.DictReader((CANON / "sample_annotations.csv").open())}

    sensitive = {"moderately sensitive", "highly sensitive"}
    resistant = {"resistant"}

    rows_x: list[list[float]] = []
    rows_y: list[int] = []
    kept_ids: list[str] = []
    feature_names = (
        [f"EMT_{g}" for g in ["MFAP4", "EFEMP1", "VIM", "CDH1", "SNAI1"]]
        + [f"DDR_{g}" for g in DDR_GENES]
        + ["ratio_mes_epi", "grade_ord", "figo_ord", "residual_ord", "brca1_mut", "hist_serous", "hist_undiff"]
    )

    for i, sid in enumerate(sample_ids):
        a = ann.get(sid)
        if a is None:
            continue
        ps = a["platinum_sensitivity"]
        if ps in resistant:
            y = 1
        elif ps in sensitive:
            y = 0
        else:
            continue

        emt_vals = {
            g: gene_mean(expr, emt_map[g], i) for g in ["MFAP4", "EFEMP1", "VIM", "CDH1", "SNAI1"]
        }
        ddr_vals = {g: gene_mean(expr, ddr_map[g], i) for g in DDR_GENES}
        ratio = (
            emt_vals["MFAP4"] + emt_vals["VIM"] + emt_vals["EFEMP1"]
        ) - emt_vals["CDH1"]
        clin = encode_clinical(clinical[sid])
        row = (
            [emt_vals[g] for g in ["MFAP4", "EFEMP1", "VIM", "CDH1", "SNAI1"]]
            + [ddr_vals[g] for g in DDR_GENES]
            + [
                ratio,
                clin["grade_ord"],
                clin["figo_ord"],
                clin["residual_ord"],
                clin["brca1_mut"],
                clin["hist_serous"],
                clin["hist_undiff"],
            ]
        )
        rows_x.append(row)
        rows_y.append(y)
        kept_ids.append(sid)

    X = np.asarray(rows_x, dtype=float)
    y = np.asarray(rows_y, dtype=int)
    assert X.shape[0] == 75, X.shape
    assert int((y == 1).sum()) == 34 and int((y == 0).sum()) == 41
    meta = {
        "emt_map": emt_map,
        "ddr_map": ddr_map,
        "sample_ids": kept_ids,
        "n": 75,
        "clinical_source": (
            "GSE63885_series_matrix.txt !Sample_characteristics_ch1 "
            "(sample_annotations.csv has only platinum_sensitivity + clinical_response)"
        ),
    }
    return X, y, feature_names, kept_ids, meta


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], scores[idx]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def oof_predict(model, X: np.ndarray, y: np.ndarray, cv) -> np.ndarray:
    oof = np.full(len(y), np.nan, dtype=float)
    for tr, te in cv.split(X, y):
        clf = clone(model)
        clf.fit(X[tr], y[tr])
        if hasattr(clf, "predict_proba"):
            oof[te] = clf.predict_proba(X[te])[:, 1]
        else:
            oof[te] = clf.decision_function(X[te])
    return oof


def metrics_block(y: np.ndarray, oof: np.ndarray) -> dict:
    auroc = float(roc_auc_score(y, oof))
    prauc = float(average_precision_score(y, oof))
    brier = float(brier_score_loss(y, oof))
    ci = bootstrap_ci(y, oof)
    return {
        "auroc": auroc,
        "prauc": prauc,
        "brier": brier,
        "auroc_ci95": list(ci),
    }


def make_models(feature_names: list[str]) -> dict:
    emt_idx = [
        i
        for i, n in enumerate(feature_names)
        if n.startswith("EMT_") or n == "ratio_mes_epi"
    ]
    uni_idx = [feature_names.index("EMT_MFAP4")]

    models = {}
    models["univariate_mfap4"] = Pipeline(
        [
            ("sel", SelectCols(uni_idx)),
            ("imp", SimpleImputer(strategy="median")),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    models["l2_emt_panel"] = Pipeline(
        [
            ("sel", SelectCols(emt_idx)),
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=np.logspace(-2, 2, 12),
                    cv=5,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    scoring="roc_auc",
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )
    models["elasticnet_emt_ddr_clinical"] = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=np.logspace(-2, 2, 10),
                    l1_ratios=[0.15, 0.5, 0.85],
                    cv=5,
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=8000,
                    scoring="roc_auc",
                    random_state=SEED,
                    n_jobs=1,
                    tol=1e-3,
                ),
            ),
        ]
    )
    models["hist_gradient_boosting"] = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=3,
                    learning_rate=0.05,
                    max_iter=200,
                    min_samples_leaf=8,
                    random_state=SEED,
                ),
            ),
        ]
    )
    models["random_forest"] = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=4,
                    min_samples_leaf=4,
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return models


def extract_importances(name: str, model, feature_names: list[str]) -> list[dict]:
    clf = model.named_steps["clf"]
    if name == "univariate_mfap4":
        coef = float(clf.coef_.ravel()[0])
        return [{"feature": "EMT_MFAP4", "weight": coef, "kind": "coef"}]
    if name == "l2_emt_panel":
        sel = model.named_steps["sel"]
        cols = [feature_names[i] for i in sel.cols]
        coefs = clf.coef_.ravel()
        return sorted(
            [{"feature": f, "weight": float(c), "kind": "coef"} for f, c in zip(cols, coefs)],
            key=lambda d: abs(d["weight"]),
            reverse=True,
        )
    if name == "elasticnet_emt_ddr_clinical":
        coefs = clf.coef_.ravel()
        return sorted(
            [
                {"feature": f, "weight": float(c), "kind": "coef"}
                for f, c in zip(feature_names, coefs)
            ],
            key=lambda d: abs(d["weight"]),
            reverse=True,
        )
    if name in ("hist_gradient_boosting", "random_forest"):
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            return sorted(
                [
                    {"feature": f, "weight": float(w), "kind": "importance"}
                    for f, w in zip(feature_names, imp)
                ],
                key=lambda d: abs(d["weight"]),
                reverse=True,
            )
    return []


def main() -> None:
    X, y, feature_names, kept_ids, meta = build_matrix()
    models = make_models(feature_names)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    loo = LeaveOneOut()

    comparison = []
    oof_store = {}
    for name, model in models.items():
        print(f"CV {name} 5-fold...", flush=True)
        oof5 = oof_predict(model, X, y, skf)
        m5 = metrics_block(y, oof5)
        print(f"  5fold AUROC={m5['auroc']:.4f}", flush=True)
        print(f"CV {name} LOOCV...", flush=True)
        oof_loo = oof_predict(model, X, y, loo)
        mloo = metrics_block(y, oof_loo)
        print(f"  LOO AUROC={mloo['auroc']:.4f}", flush=True)
        comparison.append(
            {
                "model": name,
                "n_features": int(
                    X.shape[1]
                    if "emt_ddr" in name or name in ("hist_gradient_boosting", "random_forest")
                    else (
                        1
                        if name == "univariate_mfap4"
                        else sum(
                            1
                            for n in feature_names
                            if n.startswith("EMT_") or n == "ratio_mes_epi"
                        )
                    )
                ),
                "cv5": m5,
                "loocv": mloo,
            }
        )
        oof_store[name] = {"cv5": oof5.tolist(), "loocv": oof_loo.tolist()}

    # Winner = highest LOOCV AUROC among multivariate models; report vs univariate
    uni = next(c for c in comparison if c["model"] == "univariate_mfap4")
    multi = [c for c in comparison if c["model"] != "univariate_mfap4"]
    winner = max(multi, key=lambda c: c["loocv"]["auroc"])
    beats_uni = winner["loocv"]["auroc"] > uni["loocv"]["auroc"]

    # Fit winner on full data for export
    win_name = winner["model"]
    win_model = models[win_name]
    win_model.fit(X, y)
    importances = extract_importances(win_name, win_model, feature_names)

    # Calibrate on full (isotonic) for deployed probabilities — optional wrapper
    # Keep raw pipeline; OOF metrics already honest.

    artifact_path = (
        MOAT
        / "src/oncology_arbiter/arbiter/models/ovarian_resistance_multivariate_v2.joblib"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": "ovarian_resistance_multivariate_v2",
        "winner": win_name,
        "pipeline": win_model,
        "feature_names": feature_names,
        "n_training": 75,
        "n_positive": 34,
        "n_negative": 41,
        "positive_class": "platinum_resistant",
        "seed": SEED,
        "geo": "GSE63885",
        "probe_maps": {"emt": meta["emt_map"], "ddr": meta["ddr_map"]},
        "clinical_source": meta["clinical_source"],
        "univariate_comparator_auroc": PROOF_UNI,
        "loocv_auroc": winner["loocv"]["auroc"],
        "cv5_auroc": winner["cv5"]["auroc"],
        "beats_univariate_loocv": beats_uni,
        "importances": importances,
        "disclaimer": (
            "RESEARCH USE ONLY — multivariate GSE63885 platinum-resistance model. "
            "Not FDA-cleared. Not CE-marked. OOF metrics only."
        ),
    }
    joblib.dump(payload, artifact_path)
    sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    proof = {
        "capability": "hgsoc-platinum-resistance-multivariate-v2",
        "geo": "GSE63885",
        "honest_n": 75,
        "resistant": 34,
        "sensitive": 41,
        "feature_names": feature_names,
        "clinical_source": meta["clinical_source"],
        "ddr_probe_map": meta["ddr_map"],
        "emt_probe_map": meta["emt_map"],
        "comparison_oof": comparison,
        "winner": win_name,
        "beats_univariate_loocv": beats_uni,
        "univariate_proof_auroc": PROOF_UNI,
        "top_features": importances[:15],
        "artifact_path": "src/oncology_arbiter/arbiter/models/ovarian_resistance_multivariate_v2.joblib",
        "artifact_sha256": sha,
        "forbidden": "Do not use retracted n=101 AUROC 0.7625",
        "status": "LOCKED",
        "seed": SEED,
    }
    proof_path = MOAT / "docs/proofs/gse63885_multivariate_metrics.json"
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps(proof, indent=2) + "\n")

    # Print comparison table
    print("\n=== OOF MODEL COMPARISON (honest n=75) ===")
    print(f"{'model':35s} {'CV5 AUROC':>10s} {'LOO AUROC':>10s} {'PR-AUC':>8s} {'Brier':>8s}")
    for c in comparison:
        print(
            f"{c['model']:35s} {c['cv5']['auroc']:10.4f} {c['loocv']['auroc']:10.4f} "
            f"{c['loocv']['prauc']:8.4f} {c['loocv']['brier']:8.4f}"
        )
    print(f"\nWINNER: {win_name} LOOCV={winner['loocv']['auroc']:.4f} "
          f"(uni LOOCV={uni['loocv']['auroc']:.4f}, beats={beats_uni})")
    print("TOP FEATURES:")
    for row in importances[:12]:
        print(f"  {row['feature']:24s} {row['weight']:+.6f} ({row['kind']})")
    print(f"artifact sha256={sha}")
    print(f"proof={proof_path}")


if __name__ == "__main__":
    main()
