"""Train a scikit-learn LogisticRegression probe on CBIS-DDSM MedSigLIP embeddings.

Reads::

    <artifacts>/embeddings.npy   float32 (N, 1152)
    <artifacts>/labels.npy       int8    (N,)  1=cancer, 0=not_cancer
    <artifacts>/splits.npy       int8    (N,)  1=train, 0=test
    <artifacts>/paths.npy        object  (N,)  absolute PNG paths

Fits a StandardScaler + LogisticRegression (L2, saga, max_iter=5000) with
5-fold stratified CV on train, then evaluates once on the held-out test set.
Saves the trained pipeline to ``<out>/cbis_ddsm_logreg_v1.joblib`` and a
metrics JSON so downstream code (probe endpoint, model card) can consume it.

Reproducibility: RNG seed=42 for CV split & LR solver.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

DEFAULT_ART = Path("/workspace/cbis_ddsm_1024/artifacts")
DEFAULT_OUT = Path("/workspace/cbis_ddsm_1024/artifacts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=DEFAULT_ART)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    art: Path = args.artifacts
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] artifacts from {art}")
    X = np.load(art / "embeddings.npy")
    y = np.load(art / "labels.npy")
    splits = np.load(art / "splits.npy")
    paths = np.load(art / "paths.npy", allow_pickle=True)
    n, dim = X.shape
    print(f"[load] X={X.shape}  y_pos={int(y.sum())}  y_neg={int((y==0).sum())}")
    train_mask = splits == 1
    test_mask = splits == 0
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    print(f"[load] train n={len(ytr)}  test n={len(yte)}")

    # Deferred imports so the module loads even without sklearn available.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def make_pipe() -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=1.0,
                        penalty="l2",
                        solver="saga",
                        max_iter=5000,
                        n_jobs=-1,
                        random_state=args.seed,
                    ),
                ),
            ]
        )

    # 5-fold CV on the training set for a leakage-free ranking metric.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_aucs: list[float] = []
    cv_pr_aucs: list[float] = []
    t0 = time.time()
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(Xtr, ytr)):
        pipe = make_pipe()
        pipe.fit(Xtr[tr_idx], ytr[tr_idx])
        proba = pipe.predict_proba(Xtr[va_idx])[:, 1]
        auc = float(roc_auc_score(ytr[va_idx], proba))
        prauc = float(average_precision_score(ytr[va_idx], proba))
        cv_aucs.append(auc)
        cv_pr_aucs.append(prauc)
        print(f"[cv fold {fold_i}]  n_tr={len(tr_idx)}  n_va={len(va_idx)}  AUC={auc:.4f}  PR-AUC={prauc:.4f}")
    cv_seconds = time.time() - t0
    print(
        f"[cv] mean AUC={float(np.mean(cv_aucs)):.4f} ± {float(np.std(cv_aucs)):.4f}  "
        f"mean PR-AUC={float(np.mean(cv_pr_aucs)):.4f}  ({cv_seconds:.1f}s)"
    )

    # Refit on FULL train, evaluate on held-out test once.
    t0 = time.time()
    final = make_pipe()
    final.fit(Xtr, ytr)
    fit_seconds = time.time() - t0

    proba_te = final.predict_proba(Xte)[:, 1]
    pred_te = (proba_te >= 0.5).astype(int)
    test_auc = float(roc_auc_score(yte, proba_te))
    test_prauc = float(average_precision_score(yte, proba_te))
    test_acc = float(accuracy_score(yte, pred_te))
    test_f1 = float(f1_score(yte, pred_te))
    test_prec = float(precision_score(yte, pred_te, zero_division=0))
    test_rec = float(recall_score(yte, pred_te))
    test_brier = float(brier_score_loss(yte, proba_te))
    cm = confusion_matrix(yte, pred_te).tolist()

    print(
        f"[test] AUC={test_auc:.4f}  PR-AUC={test_prauc:.4f}  acc={test_acc:.4f}  "
        f"F1={test_f1:.4f}  prec={test_prec:.4f}  recall={test_rec:.4f}  "
        f"brier={test_brier:.4f}  cm={cm}"
    )

    # Persist the trained pipeline for use at inference time.
    from joblib import dump as joblib_dump

    model_path = out_dir / "cbis_ddsm_logreg_v1.joblib"
    joblib_dump(final, model_path)
    print(f"[save] {model_path}")

    # Metrics dossier — consumed by the model card and probe endpoint.
    metrics = {
        "dataset": {
            "name": "CBIS-DDSM_1024",
            "source": "https://huggingface.co/datasets/dbaek111/CBIS-DDSM_1024",
            "n_total": int(n),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_train_positive": int((y[train_mask] == 1).sum()),
            "n_train_negative": int((y[train_mask] == 0).sum()),
            "n_test_positive": int((y[test_mask] == 1).sum()),
            "n_test_negative": int((y[test_mask] == 0).sum()),
        },
        "features": {
            "source_model": "google/medsiglip-448",
            "modal_endpoint": "https://crispro-test--medsiglip-embed-batch.modal.run",
            "embedding_dim": int(dim),
            "input_resolution": 448,
            "input_format": "pixels",  # PNGs, not DICOMs
        },
        "classifier": {
            "class": "sklearn.pipeline.Pipeline(StandardScaler,LogisticRegression)",
            "penalty": "l2",
            "C": 1.0,
            "solver": "saga",
            "max_iter": 5000,
            "seed": int(args.seed),
        },
        "cv_5fold": {
            "auc_per_fold": [round(x, 6) for x in cv_aucs],
            "pr_auc_per_fold": [round(x, 6) for x in cv_pr_aucs],
            "auc_mean": round(float(np.mean(cv_aucs)), 6),
            "auc_std": round(float(np.std(cv_aucs)), 6),
            "pr_auc_mean": round(float(np.mean(cv_pr_aucs)), 6),
            "seconds": round(cv_seconds, 3),
        },
        "test": {
            "auc": round(test_auc, 6),
            "pr_auc": round(test_prauc, 6),
            "accuracy": round(test_acc, 6),
            "f1": round(test_f1, 6),
            "precision": round(test_prec, 6),
            "recall": round(test_rec, 6),
            "brier": round(test_brier, 6),
            "confusion_matrix": cm,
            "threshold": 0.5,
        },
        "fit_seconds_on_full_train": round(fit_seconds, 3),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    (out_dir / "cbis_ddsm_logreg_v1_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(f"[save] {out_dir / 'cbis_ddsm_logreg_v1_metrics.json'}")


if __name__ == "__main__":
    main()
