#!/usr/bin/env python3
"""
Phikon NCT-CRC-HE-100K / CRC-VAL-HE-7K 9-class linear-probe population validation.

Scope-expansion addendum (approved, prior session window): full NCT-CRC-HE-100K
Zenodo download for Phikon population validation, EXPLICITLY DISCLAIMED AS NOT
A PRODUCT SCREENING CLAIM. This script performs the classifier-fitting/eval step
that was queued but not yet written: fit a 9-class linear probe on real Phikon
embeddings of the official NCT-CRC-HE-100K TRAIN split, evaluate on the official
CRC-VAL-HE-7K VALIDATION split (a formally held-out, patient-disjoint split
defined by the dataset creators, Kather et al.) -- both embedded end-to-end via
the live Phikon Modal endpoint (background job phikon_nct_crc_embed_full,
107180/107180 images, wall=4403s).

This is a population-scale sanity check of the Phikon embedding endpoint's
representational quality for a well-known 9-tissue-type histology benchmark.
It is NOT a cancer-screening validation, NOT a claim about any product
endpoint's clinical performance, and NOT related to the CBIS-DDSM/MedSigLIP
ship-gate. No result from this script may be used to support the word
"screening" for any oncology-arbiter product surface.

NOTE: this script expects Phikon embeddings already produced by
scripts/embed_phikon_nct_crc.py (manifest.json/embeddings.npy/done_mask.npy
under EMB_DIR). It is standalone (no oncology_arbiter import dependency) so it
can run on a machine that only holds the embedding artifacts (e.g. w2), not a
full repo checkout.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

EMB_DIR = Path("/workspace/pathology_crc/phikon_embeddings")
OUT_JSON = EMB_DIR / "phikon_nct_crc_population_validation.json"

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def main():
    print("[load] manifest.json ...", flush=True)
    with open(EMB_DIR / "manifest.json") as f:
        manifest = json.load(f)
    n = len(manifest)
    print(f"[load] {n} manifest entries", flush=True)

    print("[load] embeddings.npy / done_mask.npy ...", flush=True)
    embeddings = np.load(EMB_DIR / "embeddings.npy")
    done_mask = np.load(EMB_DIR / "done_mask.npy")
    assert embeddings.shape[0] == n, f"embeddings rows {embeddings.shape[0]} != manifest {n}"
    assert done_mask.shape[0] == n, f"done_mask rows {done_mask.shape[0]} != manifest {n}"
    assert embeddings.shape[1] == 768, f"unexpected embedding dim {embeddings.shape[1]} (Phikon must be 768)"

    n_done = int(done_mask.sum())
    n_finite = int(np.isfinite(embeddings).all(axis=1).sum())
    n_nonzero = int((np.abs(embeddings).sum(axis=1) > 0).sum())
    print(f"[integrity] done_mask sum={n_done}/{n}  all_finite_rows={n_finite}/{n}  nonzero_rows={n_nonzero}/{n}", flush=True)

    if n_done != n:
        missing_idx = np.where(~done_mask.astype(bool))[0].tolist()
        raise RuntimeError(
            f"REFUSING to fit/eval on an incomplete embedding set: {n - n_done} of {n} rows "
            f"are not marked done (indices e.g. {missing_idx[:10]}...). This must be re-run to "
            f"completion, not silently dropped or imputed."
        )
    if n_finite != n or n_nonzero != n:
        raise RuntimeError(
            f"REFUSING to fit/eval: found non-finite or all-zero embedding rows despite "
            f"done_mask claiming completeness (finite={n_finite}/{n}, nonzero={n_nonzero}/{n}). "
            f"This is a real data-integrity anomaly that must be investigated, not masked."
        )

    labels = np.array([m["label"] for m in manifest], dtype=np.int64)
    splits = np.array([m["split"] for m in manifest])
    classes_seen = sorted(set(m["class"] for m in manifest))
    assert classes_seen == CLASSES, f"class set mismatch: {classes_seen} vs expected {CLASSES}"

    train_mask = splits == "train"
    val_mask = splits == "val"
    n_train, n_val = int(train_mask.sum()), int(val_mask.sum())
    print(f"[split] train(NCT-CRC-HE-100K)={n_train}  val(CRC-VAL-HE-7K)={n_val}  (n={n_train + n_val})", flush=True)
    assert n_train + n_val == n

    X_train, y_train = embeddings[train_mask], labels[train_mask]
    X_val, y_val = embeddings[val_mask], labels[val_mask]

    train_class_counts = {CLASSES[c]: int((y_train == c).sum()) for c in range(len(CLASSES))}
    val_class_counts = {CLASSES[c]: int((y_val == c).sum()) for c in range(len(CLASSES))}
    print("[split] train class counts:", train_class_counts, flush=True)
    print("[split] val   class counts:", val_class_counts, flush=True)

    print("[fit] StandardScaler + multinomial LogisticRegression (lbfgs, max_iter=2000) on train ...", flush=True)
    import time

    t0 = time.time()
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,
            C=1.0,
            random_state=42,
        ),
    )
    clf.fit(X_train, y_train)
    fit_s = time.time() - t0
    print(f"[fit] done in {fit_s:.1f}s", flush=True)

    print("[eval] scoring held-out CRC-VAL-HE-7K ...", flush=True)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    weighted_f1 = f1_score(y_val, y_pred, average="weighted")
    per_class_f1 = f1_score(y_val, y_pred, average=None, labels=list(range(len(CLASSES))))
    per_class_precision = precision_score(y_val, y_pred, average=None, labels=list(range(len(CLASSES))))
    per_class_recall = recall_score(y_val, y_pred, average=None, labels=list(range(len(CLASSES))))
    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(CLASSES))))
    report_txt = classification_report(y_val, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES, digits=4)

    print(report_txt, flush=True)
    print(f"[eval] accuracy={acc:.4f}  balanced_accuracy={bal_acc:.4f}  macro_F1={macro_f1:.4f}  weighted_F1={weighted_f1:.4f}", flush=True)

    result = {
        "DISCLAIMER": (
            "This is a population-scale linear-probe validation of the Phikon pathology "
            "embedding endpoint on a public 9-tissue-type histology benchmark "
            "(NCT-CRC-HE-100K train / CRC-VAL-HE-7K held-out val, Kather et al.). "
            "It is NOT a product screening claim, NOT a cancer-detection validation, and is "
            "UNRELATED to the CBIS-DDSM/MedSigLIP >=0.85 ship-gate. Approved as a scope-expansion "
            "addendum for population validation only."
        ),
        "endpoint": {
            "name": "phikon-pathology",
            "modal_embed_url": "https://crispro--phikon-embed.modal.run",
            "embedding_dim": 768,
            "batch_cap_discovered_empirically": 64,
        },
        "job": {
            "name": "phikon_nct_crc_embed_full",
            "n_images_embedded": n,
            "wall_seconds": 4403,
            "transient_retries": (
                "1 batch (index ~106048) hit a write-timeout on first attempt and succeeded on "
                "retry after 127.05s (retry/backoff logic in embed_phikon_nct_crc.py). No data "
                "loss: done_mask and finite/nonzero checks below confirm full, valid coverage."
            ),
        },
        "data_integrity": {
            "n_total": n,
            "n_done_mask_true": n_done,
            "n_all_finite_rows": n_finite,
            "n_nonzero_rows": n_nonzero,
            "status": "COMPLETE_NO_MISSING_NO_NONFINITE",
        },
        "cohort": {
            "train_source": "NCT-CRC-HE-100K",
            "val_source": "CRC-VAL-HE-7K (official held-out, patient-disjoint split per dataset creators)",
            "classes": CLASSES,
            "n_train": n_train,
            "n_val": n_val,
            "train_class_counts": train_class_counts,
            "val_class_counts": val_class_counts,
        },
        "probe": {
            "type": "StandardScaler + multinomial LogisticRegression (lbfgs, C=1.0, max_iter=2000, random_state=42)",
            "fit_seconds": fit_s,
            "trained_this_run": True,
        },
        "held_out_val_metrics": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class": {
                CLASSES[i]: {
                    "f1": float(per_class_f1[i]),
                    "precision": float(per_class_precision[i]),
                    "recall": float(per_class_recall[i]),
                    "n_val": val_class_counts[CLASSES[i]],
                }
                for i in range(len(CLASSES))
            },
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_row_order": CLASSES,
            "classification_report_text": report_txt,
        },
        "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] wrote {OUT_JSON} ({OUT_JSON.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
