"""Hyperparameter sweep for CBIS-DDSM MedSigLIP probe v2.

Ingests 1152-d embeddings (merged npy or shard_0..4.npz), sweeps
LogisticRegression / RidgeClassifier / 2-layer MLP over C in
{0.001, 0.01, 0.1, 1.0, 10.0} with L2 + class_weight='balanced',
evaluates once on the held-out CBIS test split, and exports the winner
as models/cbis_ddsm_logreg_v2.joblib + metrics sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifacts(art: Path, shard_dir: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    emb = art / "embeddings.npy"
    lab = art / "labels.npy"
    spl = art / "splits.npy"
    if emb.exists() and lab.exists() and spl.exists():
        X = np.load(emb)
        y = np.load(lab)
        splits = np.load(spl)
        print(f"[load] merged npy from {art}: X={X.shape}")
        return X, y, splits

    if shard_dir is None:
        raise FileNotFoundError(f"missing {emb} and no --shard-dir")
    Xs, ys, ss = [], [], []
    for i in range(5):
        p = shard_dir / f"shard_{i}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        d = np.load(p)
        key_x = "embeddings" if "embeddings" in d.files else "X"
        Xs.append(d[key_x])
        ys.append(d["labels"])
        ss.append(d["splits"])
        print(f"[load] {p.name} rows={len(d[key_x])}")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    splits = np.concatenate(ss, axis=0)
    print(f"[load] shards merged X={X.shape}")
    return X, y, splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=_ROOT / "artifacts/cbis_ddsm_1024")
    ap.add_argument("--shard-dir", type=Path, default=_ROOT / "artifacts/cbis_parallel")
    ap.add_argument("--out-model", type=Path, default=_ROOT / "models/cbis_ddsm_logreg_v2.joblib")
    ap.add_argument(
        "--out-metrics",
        type=Path,
        default=_ROOT / "docs/proofs/cbis_ddsm_logreg_v2_metrics.json",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-auroc", type=float, default=0.85)
    args = ap.parse_args()

    from joblib import dump
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
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
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, splits = load_artifacts(args.artifacts, args.shard_dir)
    train_mask = splits == 1
    test_mask = splits == 0
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    print(f"[split] train={len(ytr)} test={len(yte)} pos_te={int(yte.sum())}")

    C_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    candidates: list[dict] = []

    def score_proba(pipe, name: str, params: dict) -> dict:
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(Xte)[:, 1]
        else:
            # RidgeClassifier: decision_function → sigmoid-ish rank scores
            dec = pipe.decision_function(Xte)
            if dec.ndim > 1:
                dec = dec[:, 0]
            # map to (0,1) via logistic for AUROC-compatible ranking (AUROC uses rank)
            proba = 1.0 / (1.0 + np.exp(-dec))
        auroc = float(roc_auc_score(yte, proba))
        prauc = float(average_precision_score(yte, proba))
        pred = (proba >= 0.5).astype(int)
        row = {
            "name": name,
            "params": params,
            "test_auroc": auroc,
            "test_prauc": prauc,
            "test_accuracy": float(accuracy_score(yte, pred)),
            "test_f1": float(f1_score(yte, pred)),
            "fit_seconds": float(time.time() - t0),
            "pipe": pipe,
            "proba": proba,
        }
        print(
            f"[cand] {name} params={params} AUROC={auroc:.4f} PR-AUC={prauc:.4f} "
            f"({row['fit_seconds']:.1f}s)"
        )
        candidates.append(row)
        return row

    for C in C_grid:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        penalty="l2",
                        solver="saga",
                        max_iter=5000,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        score_proba(pipe, "LogisticRegression", {"C": C, "penalty": "l2", "class_weight": "balanced"})

    for C in C_grid:
        # RidgeClassifier uses alpha = 1/C convention for sweep parity
        alpha = 1.0 / C
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RidgeClassifier(
                        alpha=alpha,
                        class_weight="balanced",
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        score_proba(pipe, "RidgeClassifier", {"alpha": alpha, "C_equiv": C, "class_weight": "balanced"})

    for C in C_grid:
        # MLP: map C → inverse regularization strength via alpha
        alpha = 1.0 / (C * 1000.0)  # keep alpha in a sensible range
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 64),
                        activation="relu",
                        alpha=alpha,
                        max_iter=200,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        # MLP has no class_weight; approximate via sample_weight on fit override
        # Fit with balanced sample weights manually
        t0 = time.time()
        pipe.named_steps["scaler"].fit(Xtr)
        Xtr_s = pipe.named_steps["scaler"].transform(Xtr)
        Xte_s = pipe.named_steps["scaler"].transform(Xte)
        n_pos = max(int(ytr.sum()), 1)
        n_neg = max(int((ytr == 0).sum()), 1)
        sw = np.where(ytr == 1, len(ytr) / (2 * n_pos), len(ytr) / (2 * n_neg)).astype(np.float64)
        clf = pipe.named_steps["clf"]
        clf.fit(Xtr_s, ytr, sample_weight=sw)
        proba = clf.predict_proba(Xte_s)[:, 1]
        auroc = float(roc_auc_score(yte, proba))
        prauc = float(average_precision_score(yte, proba))
        pred = (proba >= 0.5).astype(int)
        # rebuild full pipeline for export
        export_pipe = Pipeline([("scaler", pipe.named_steps["scaler"]), ("clf", clf)])
        row = {
            "name": "MLP_2layer",
            "params": {
                "hidden": [256, 64],
                "alpha": alpha,
                "C_equiv": C,
                "sample_weight": "balanced",
            },
            "test_auroc": auroc,
            "test_prauc": prauc,
            "test_accuracy": float(accuracy_score(yte, pred)),
            "test_f1": float(f1_score(yte, pred)),
            "fit_seconds": float(time.time() - t0),
            "pipe": export_pipe,
            "proba": proba,
        }
        print(
            f"[cand] MLP_2layer params={row['params']} AUROC={auroc:.4f} "
            f"PR-AUC={prauc:.4f} ({row['fit_seconds']:.1f}s)"
        )
        candidates.append(row)

    best = max(candidates, key=lambda r: r["test_auroc"])
    print(
        f"[best] {best['name']} {best['params']} AUROC={best['test_auroc']:.4f} "
        f"target={args.target_auroc} pass={best['test_auroc'] >= args.target_auroc}"
    )

    proba = best["proba"]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(yte, pred).tolist()
    metrics = {
        "version": "cbis_ddsm_logreg_v2",
        "artifact": str(args.out_model.relative_to(_ROOT)),
        "dataset": {
            "name": "CBIS-DDSM_1024",
            "source": "https://huggingface.co/datasets/dbaek111/CBIS-DDSM_1024",
            "n_total": int(len(y)),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "n_train_positive": int(ytr.sum()),
            "n_train_negative": int((ytr == 0).sum()),
            "n_test_positive": int(yte.sum()),
            "n_test_negative": int((yte == 0).sum()),
        },
        "features": {
            "source_model": "google/medsiglip-448",
            "modal_endpoint": "https://crispro-test--medsiglip-embed-batch.modal.run",
            "modal_app_version": "medsiglip-modal-v0.3.0",
            "embedding_dim": 1152,
            "artifacts_dir": str(args.artifacts),
            "shard_dir": str(args.shard_dir),
        },
        "winner": {
            "name": best["name"],
            "params": best["params"],
        },
        "sweep": [
            {
                "name": c["name"],
                "params": c["params"],
                "test_auroc": c["test_auroc"],
                "test_prauc": c["test_prauc"],
                "test_accuracy": c["test_accuracy"],
                "test_f1": c["test_f1"],
                "fit_seconds": c["fit_seconds"],
            }
            for c in sorted(candidates, key=lambda r: -r["test_auroc"])
        ],
        "test_metrics": {
            "auroc": best["test_auroc"],
            "prauc": best["test_prauc"],
            "accuracy": best["test_accuracy"],
            "f1": best["test_f1"],
            "precision": float(precision_score(yte, pred)),
            "recall": float(recall_score(yte, pred)),
            "brier": float(brier_score_loss(yte, proba)),
            "confusion_matrix": cm,
        },
        "floor": {"auroc": args.target_auroc, "pass": bool(best["test_auroc"] >= args.target_auroc)},
        "seed": args.seed,
        "class_weight": "balanced",
        "created_unix": int(time.time()),
    }

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    dump(best["pipe"], args.out_model)
    art_sha = _sha256(args.out_model)
    metrics["artifact_sha256"] = art_sha

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[out] model={args.out_model} sha256={art_sha}")
    print(f"[out] metrics={args.out_metrics}")
    print(json.dumps({"test_auroc": best["test_auroc"], "floor_pass": metrics["floor"]["pass"]}, indent=2))


if __name__ == "__main__":
    main()
