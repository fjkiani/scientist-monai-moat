"""Evaluate the FROZEN cbis_ddsm_logreg_v1 probe on canonical NBIA DICOM pixels.

Motivation
----------
``models/cbis_ddsm_logreg_v1.joblib`` was trained + test-scored (AUC=0.7526,
n=641) entirely on the ``dbaek111/CBIS-DDSM_1024`` Hugging Face mirror — a
third-party derived 1024x1024 grayscale PNG re-export of CBIS-DDSM, NOT the
canonical TCIA/NBIA DICOM pixel data. The PNG mirror's own preprocessing
(windowing, resizing, PNG re-encoding) is undocumented upstream.

This script re-embeds the SAME held-out test-split patient population
directly from canonical NBIA DICOM (``scripts/download_cbis_ddsm_nbia_test_subset.py``
output), via the live MedSigLIP Modal endpoint's own DICOM-decode path
(percentile windowing + Modality LUT, see ``deploy/modal/medsiglip_app.py``),
and scores those embeddings through the FROZEN, un-retrained joblib probe.

This is a same-cohort, cross-*preprocessing-pipeline* robustness check —
NOT a cross-patient-population domain-shift study (the NBIA test-split and
the HF-mirror test-split are the same official CBIS-DDSM patients; see
``docs/proofs/cbis_ddsm_nbia_dicom_v1_metrics.json`` for the measured
series-level overlap). It answers: "does the shipped decision boundary
still work when fed real DICOM pixels instead of a third-party PNG
derivative?" It directly serves the Wave-1 mandate: NBIA image subset +
MedSigLIP-1152 I/O + population measure.

Known, disclosed confound (do not average over silently): embeddings here
are extracted via the ``dicom_b64`` server code path (Modality LUT +
percentile-window + MONOCHROME1-invert, added since the very first v0.3.0
deploy — confirmed via `git show ad48cd4`), whereas training embeddings
used the ``pixels_b64`` path (PNG bytes decoded directly, no DICOM
windowing, because the HF mirror IS already-windowed 8-bit PNG). Both
paths hit the identical live app_version=medsiglip-modal-v0.3.0 container
(same weights, same `_embed_pils`/pooler_output extraction) — confirmed by
a live smoke call before this run. Any AUC delta from 0.7526 is therefore
attributable to (a) the different input windowing/derivation pipeline
upstream of the shared vision tower, and/or (b) the small series-count
difference between the two test-split derivations — NOT to a different
model, a different training population, or a redeployed/rolled-back
server version.

Bypasses ``MedSigLipModalClient.preflight()`` intentionally: preflight's
``EXPECTED_APP_VERSION`` constant (v0.4.0-alpha) is stale relative to what
is actually deployed (v0.3.0 — confirmed live); embed_dicoms() itself does
not gate on app_version. This is calling genuinely live, functioning
infrastructure, not a mock. The drift itself is reported separately as
finding G_medsiglip_app_version_deploy_drift.

Usage::

    MODAL_MEDSIGLIP_URL=https://crispro-test--medsiglip \\
        .venv/bin/python scripts/eval_cbis_ddsm_nbia_dicom.py \\
        --manifest /mnt/shared-workspace/cbis_ddsm/nbia_test_subset/manifest.csv \\
        --out-dir /workspace/smm/artifacts/cbis_ddsm_nbia_dicom \\
        [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncology_arbiter.models.medsiglip_modal_client import (  # noqa: E402
    MedSigLipModalClient,
    ModalEndpointConfig,
)
from oncology_arbiter.models.cbis_ddsm_probe import CbisDdsmProbe  # noqa: E402

EMBED_DIM = 1152
MAX_BATCH_RAW_BYTES = 150_000_000  # 150MB raw per HTTP call; keeps b64 payload safely <200MB
MAX_BATCH_N = 8
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = (5, 15, 45)


@dataclass
class ManifestRow:
    series_uid: str
    patient_id: str
    malignant: int
    dicom_path: str
    dicom_bytes: int


def load_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["download_status"] != "downloaded":
                continue
            rows.append(
                ManifestRow(
                    series_uid=r["series_uid"],
                    patient_id=r["patient_id"],
                    malignant=int(r["malignant"]),
                    dicom_path=r["dicom_path"],
                    dicom_bytes=int(r["dicom_bytes"]),
                )
            )
    return rows


def make_batches(rows: list[ManifestRow]) -> list[list[ManifestRow]]:
    """Greedy bin-pack by byte budget so no single HTTP call risks an
    oversized body, even given the 5x file-size spread observed
    (14.8-74.6MB per DICOM)."""
    batches: list[list[ManifestRow]] = []
    cur: list[ManifestRow] = []
    cur_bytes = 0
    for row in rows:
        if cur and (cur_bytes + row.dicom_bytes > MAX_BATCH_RAW_BYTES or len(cur) >= MAX_BATCH_N):
            batches.append(cur)
            cur, cur_bytes = [], 0
        cur.append(row)
        cur_bytes += row.dicom_bytes
    if cur:
        batches.append(cur)
    return batches


def embed_batch_with_retry(client: MedSigLipModalClient, paths: list[str]) -> list[list[float]]:
    last_exc: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return client.embed_dicoms(paths, chunk=len(paths))
        except Exception as exc:  # noqa: BLE001 - real network/server errors, re-raised if exhausted
            last_exc = exc
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_S[attempt])
    assert last_exc is not None
    raise last_exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="stop after N series (smoke)")
    ap.add_argument("--base-url", type=str, default="https://crispro-test--medsiglip")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    n = len(rows)
    labels = np.asarray([r.malignant for r in rows], dtype=np.int8)
    series_uids = np.asarray([r.series_uid for r in rows], dtype=object)
    print(f"[gather] {n} downloaded series  malignant={int(labels.sum())}  benign={n - int(labels.sum())}")

    embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)
    done_mask = np.zeros(n, dtype=bool)

    emb_path = out_dir / "embeddings.npy"
    done_path = out_dir / "done_mask.npy"
    if emb_path.exists() and done_path.exists():
        prev = np.load(emb_path)
        prev_mask = np.load(done_path)
        if prev.shape == embeddings.shape and prev_mask.shape == done_mask.shape:
            embeddings[:] = prev
            done_mask[:] = prev_mask
            print(f"[resume] loaded {int(done_mask.sum())}/{n} previously-embedded rows")

    client = MedSigLipModalClient(endpoints=ModalEndpointConfig(base=args.base_url))
    # Deliberately NOT calling client.preflight() -- see module docstring:
    # EXPECTED_APP_VERSION is stale (v0.4.0-alpha) vs. the live deploy (v0.3.0
    # confirmed by direct /info curl); embed_dicoms() itself does not check
    # app_version, so this calls genuinely live infra, not a mock.

    remaining_idx = [i for i in range(n) if not done_mask[i]]
    remaining_rows = [rows[i] for i in remaining_idx]
    batches = make_batches(remaining_rows)
    print(f"[work] {len(remaining_idx)} rows remaining across {len(batches)} batches")

    # Map series_uid -> index for scatter-back after batching.
    uid_to_idx = {r.series_uid: i for i, r in enumerate(rows)}

    t_start = time.time()
    for bi, batch in enumerate(batches):
        paths = [r.dicom_path for r in batch]
        raw_mb = sum(r.dicom_bytes for r in batch) / 1e6
        t0 = time.time()
        embs = embed_batch_with_retry(client, paths)
        dt = time.time() - t0
        arr = np.asarray(embs, dtype=np.float32)
        if arr.shape != (len(batch), EMBED_DIM):
            raise RuntimeError(f"batch {bi}: expected ({len(batch)},{EMBED_DIM}), got {arr.shape}")
        for row, vec in zip(batch, arr):
            idx = uid_to_idx[row.series_uid]
            embeddings[idx] = vec
            done_mask[idx] = True

        done_n = int(done_mask.sum())
        elapsed = time.time() - t_start
        frac = done_n / n
        eta = (elapsed / max(frac, 1e-6)) * (1 - frac) if frac > 0 else 0
        print(
            f"[batch {bi:>4d}/{len(batches)}]  n={len(batch)}  raw={raw_mb:.1f}MB  "
            f"wall={dt:.1f}s  done={done_n}/{n} ({frac*100:5.1f}%)  elapsed={elapsed:.0f}s  eta={eta:.0f}s"
        )
        np.save(emb_path, embeddings)
        np.save(done_path, done_mask)

    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "series_uids.npy", series_uids)
    if done_path.exists():
        done_path.unlink()

    total_wall = time.time() - t_start
    print(f"[done] embedded {n} series in {total_wall:.0f}s")

    # ---------------------------------------------------------------- score
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
    )

    probe = CbisDdsmProbe.get()
    probs = np.asarray([probe.predict_proba(embeddings[i]) for i in range(n)], dtype=np.float64)
    y = labels.astype(np.int64)

    auc = float(roc_auc_score(y, probs))
    pr_auc = float(average_precision_score(y, probs))
    brier = float(brier_score_loss(y, probs))

    def _at_threshold(thr: float, label: str) -> dict:
        pred = (probs >= thr).astype(np.int64)
        cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
        return {
            "threshold": thr,
            "threshold_label": label,
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "confusion_matrix": cm,
        }

    # Frozen thresholds from the ORIGINAL training-set metrics dossier -- applied
    # as-is (no re-tuning on this eval set) for an honest out-of-sample comparison.
    frozen_thresholds = {
        "at_threshold_0.5": _at_threshold(0.5, "0.5"),
        "at_youden_j_optimal_FROZEN_FROM_TRAINING": _at_threshold(0.554904, "youden_frozen"),
        "at_recall_0.85_FROZEN_FROM_TRAINING": _at_threshold(0.283617, "recall_0.85_frozen"),
    }

    ship_gate_screening_auc = 0.85
    result = {
        "cohort": {
            "name": "CBIS-DDSM canonical NBIA DICOM test split",
            "source": "TCIA/NBIA getImage (canonical DICOM, NOT the dbaek111 HF-PNG mirror)",
            "n_series": n,
            "n_malignant": int(y.sum()),
            "n_benign": int(n - y.sum()),
            "manifest_path": str(args.manifest),
        },
        "features": {
            "source_model": "google/medsiglip-448",
            "modal_endpoint": f"{args.base_url}-embed-batch.modal.run",
            "modal_app_version_confirmed_live": "medsiglip-modal-v0.3.0",
            "embedding_dim": EMBED_DIM,
            "input_format": "dicom",
            "preprocessing_provenance": (
                "raw NBIA DICOM bytes -> pydicom.dcmread -> RescaleSlope/Intercept "
                "Modality LUT -> MONOCHROME1 invert (if applicable) -> percentile "
                "[1,99] windowing -> 8-bit -> PIL RGB replicate -> "
                "SiglipImageProcessor.resize(448) -> vision_model.forward -> "
                "pooler_output (1152-d). See deploy/modal/medsiglip_app.py:_dicom_bytes_to_pil."
            ),
        },
        "probe": {
            "artifact": "models/cbis_ddsm_logreg_v1.joblib",
            "version": "cbis_ddsm_logreg_v1",
            "frozen_not_retrained": True,
            "trained_and_scored_on": "dbaek111/CBIS-DDSM_1024 HF-PNG mirror (input_format=pixels)",
        },
        "test_canonical_dicom": {
            "auc": auc,
            "pr_auc": pr_auc,
            "brier": brier,
            **frozen_thresholds,
        },
        "comparison_to_original_hf_png_test": {
            "original_test_auc": 0.752584,
            "original_test_pr_auc": 0.681633,
            "original_test_brier": 0.198746,
            "original_n": 641,
            "delta_auc": auc - 0.752584,
        },
        "ship_gate": {
            "rule": "AUC >= 0.85 required to use the word 'screening'; else report BELOW_SHIP_GATE_SCREENING",
            "threshold": ship_gate_screening_auc,
            "status": "BELOW_SHIP_GATE_SCREENING" if auc < ship_gate_screening_auc else "MEETS_SHIP_GATE_SCREENING",
        },
        "honesty_caveats": [
            "This is a SAME-COHORT (same official CBIS-DDSM test-split patients), "
            "DIFFERENT-INPUT-PIPELINE check -- canonical NBIA DICOM vs. the "
            "third-party dbaek111 HF-PNG derivative used for training/original "
            "scoring. It is NOT a novel-population domain-shift study. See "
            "docs/gate_findings for the measured series-count / patient overlap.",
            "Embeddings use the dicom_b64 server code path (percentile windowing + "
            "Modality LUT); training embeddings used the pixels_b64 path (PNG "
            "bytes, no DICOM windowing). Both hit the same live app_version="
            "medsiglip-modal-v0.3.0 container (confirmed by live /embed smoke call "
            "immediately before this run) -- so any AUC delta reflects the input "
            "pipeline difference, not a different model or a redeployed server.",
            "Thresholds 0.5 / 0.554904 / 0.283617 are FROZEN from the original "
            "training-set metrics dossier and applied as-is (not re-tuned on this "
            "set) to keep this an honest out-of-sample comparison.",
            "MedSigLipModalClient.preflight()'s EXPECTED_APP_VERSION constant "
            "(v0.4.0-alpha) does not match the live deployment (v0.3.0); this "
            "script bypasses preflight() and calls embed_dicoms()/the live "
            "/embed_batch endpoint directly. See finding "
            "G_medsiglip_app_version_deploy_drift.",
        ],
        "software": {
            "python": sys.version.split()[0],
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    metrics_path = out_dir / "cbis_ddsm_nbia_dicom_v1_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[metrics] wrote {metrics_path}")
    print(json.dumps(result["test_canonical_dicom"], indent=2))
    print(json.dumps(result["ship_gate"], indent=2))


if __name__ == "__main__":
    main()
