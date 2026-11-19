#!/usr/bin/env python3
"""Train StandardScaler+LogisticRegression on RSNA MedSigLIP embeddings (RUO).

Expects ``embeddings.npy`` / ``labels.npy`` / ``splits.npy`` from
``scripts/embed_mammo_png_tree.py`` (splits: 1=train, 0=test).

Writes joblib + ``evaluation_v2.json`` + ``sha256.txt`` under ``--out-dir``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dataset-id", type=str, default="rsna_medsiglip")
    ap.add_argument("--joblib-name", type=str, default="rsna_medsiglip_logreg.joblib")
    args = ap.parse_args()

    emb = np.load(args.emb_dir / "embeddings.npy")
    labels = np.load(args.emb_dir / "labels.npy")
    splits = np.load(args.emb_dir / "splits.npy")
    if emb.ndim != 2 or emb.shape[1] != 1152:
        raise SystemExit(f"bad embeddings shape {emb.shape}")
    if len(labels) != len(emb) or len(splits) != len(emb):
        raise SystemExit("labels/splits length mismatch")

    train_m = splits == 1
    test_m = splits == 0
    X_tr, y_tr = emb[train_m], labels[train_m]
    X_te, y_te = emb[test_m], labels[test_m]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        raise SystemExit(
            f"need both classes in train/test; "
            f"train={np.bincount(y_tr.astype(int))} test={np.bincount(y_te.astype(int))}"
        )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                ),
            ),
        ]
    )
    t0 = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_auc = float(roc_auc_score(y_te, proba))
    test_prauc = float(average_precision_score(y_te, proba))
    test_acc = float(accuracy_score(y_te, pred))
    test_f1 = float(f1_score(y_te, pred, zero_division=0))
    cm = confusion_matrix(y_te, pred).tolist()
    wall = round(time.time() - t0, 2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = args.out_dir / args.joblib_name
    dump(pipe, joblib_path)
    sha = hashlib.sha256(joblib_path.read_bytes()).hexdigest()
    (args.out_dir / "sha256.txt").write_text(sha + "\n")

    receipt = {
        "status": "ok",
        "dataset_id": args.dataset_id,
        "n_total": int(len(emb)),
        "n_train": int(train_m.sum()),
        "n_test": int(test_m.sum()),
        "n_pos": int(labels.sum()),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "test_auc": test_auc,
        "test_prauc": test_prauc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "confusion_matrix": cm,
        "joblib": str(joblib_path),
        "joblib_sha256": sha,
        "backbone": "MedSigLIP-1152",
        "split": "embed_tree_train_test_folders",
        "class_weight": "balanced",
        "wall_s": wall,
        "band": "24-48",
        "meets_10k_floor": bool(len(emb) >= 10000),
        "emb_dir": str(args.emb_dir),
    }
    out_json = args.out_dir / "evaluation_v2.json"
    out_json.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
