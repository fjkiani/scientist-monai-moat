#!/usr/bin/env python3
"""Prepare RSNA Screening Mammography into CBIS-compatible PNG layout (RUO).

Canonical competition: ``rsna-breast-cancer-detection`` (Kaggle).
Target layout::

    data/rsna_mammo/png_tree/{train,test}/{cancer,not_cancer}/*.png

This script FAIL-CLOSES without Kaggle credentials. It does not invent downloads.

Usage::

    # 1) Place ~/.kaggle/kaggle.json (competition rules accepted on Kaggle UI)
    # 2) pip install kaggle pillow pandas
    # 3) python3 scripts/prepare_rsna_mammo_layout.py --download --limit-patients 500
    # 4) MODAL_MEDSIGLIP_URL=... python3 scripts/embed_mammo_png_tree.py \\
    #       --root data/rsna_mammo/png_tree --out-dir artifacts/rsna_mammo/embeddings_v1 \\
    #       --dataset-id rsna_breast_cancer_detection
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "rsna-breast-cancer-detection"
RAW_DIR = ROOT / "data" / "rsna_mammo" / "raw"
PNG_TREE = ROOT / "data" / "rsna_mammo" / "png_tree"
RECEIPT = ROOT / "artifacts" / "rsna_mammo" / "prepare_receipt.json"


def _kaggle_ready() -> tuple[bool, str]:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.is_file():
        return False, f"missing {kaggle_json}"
    try:
        subprocess.run(
            [sys.executable, "-c", "import kaggle"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return False, "python package 'kaggle' not installed (pip install kaggle)"
    return True, "ok"


def _ensure_layout() -> None:
    for split in ("train", "test"):
        for cls in ("cancer", "not_cancer"):
            (PNG_TREE / split / cls).mkdir(parents=True, exist_ok=True)


def download_competition() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(RAW_DIR),
    ]
    print("[download]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return RAW_DIR


def materialize_from_csv(
    train_csv: Path,
    *,
    limit_patients: int | None,
    image_root: Path,
) -> dict:
    """Map RSNA train.csv cancer labels into png_tree using existing PNGs/DICOMs.

    Expects columns including patient_id, image_id, cancer (0/1). Images may be
    DICOM under ``image_root``; when only CSVs exist, this records a dry plan.
    """
    import pandas as pd

    df = pd.read_csv(train_csv)
    required = {"patient_id", "image_id", "cancer"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{train_csv} missing columns: {sorted(missing)}")

    patients = sorted(df["patient_id"].astype(str).unique())
    if limit_patients:
        patients = patients[:limit_patients]
    keep = df[df["patient_id"].astype(str).isin(patients)].copy()

    # Patient-level holdout: last 20% of sorted patients → test
    cut = max(1, int(len(patients) * 0.8))
    train_patients = set(patients[:cut])
    copied = {"train": {"cancer": 0, "not_cancer": 0}, "test": {"cancer": 0, "not_cancer": 0}}
    skipped_missing_image = 0

    for _, row in keep.iterrows():
        pid = str(row["patient_id"])
        iid = str(row["image_id"])
        split = "train" if pid in train_patients else "test"
        cls = "cancer" if int(row["cancer"]) == 1 else "not_cancer"
        # RSNA ships DICOM; accept already-converted PNG beside raw if present.
        candidates = [
            image_root / pid / f"{iid}.png",
            image_root / f"{pid}_{iid}.png",
            RAW_DIR / "png" / pid / f"{iid}.png",
        ]
        src = next((p for p in candidates if p.is_file()), None)
        if src is None:
            skipped_missing_image += 1
            continue
        dest = PNG_TREE / split / cls / f"{pid}_{iid}.png"
        if not dest.exists():
            shutil.copy2(src, dest)
        copied[split][cls] += 1

    return {
        "n_patients": len(patients),
        "n_rows": int(len(keep)),
        "copied": copied,
        "skipped_missing_image": skipped_missing_image,
        "train_csv": str(train_csv),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="kaggle competitions download")
    ap.add_argument("--limit-patients", type=int, default=None)
    ap.add_argument(
        "--train-csv",
        type=Path,
        default=RAW_DIR / "train.csv",
        help="Path to RSNA train.csv after unzip",
    )
    ap.add_argument(
        "--image-root",
        type=Path,
        default=RAW_DIR / "train_images",
        help="Root of patient/image files (DICOM or PNG)",
    )
    ap.add_argument(
        "--materialize",
        action="store_true",
        help="Copy/convert available PNGs into png_tree from train.csv",
    )
    args = ap.parse_args()

    _ensure_layout()
    ready, reason = _kaggle_ready()
    receipt: dict = {
        "competition": COMPETITION,
        "png_tree": str(PNG_TREE),
        "raw_dir": str(RAW_DIR),
        "kaggle_ready": ready,
        "kaggle_reason": reason,
        "status": "scaffolded",
    }

    if args.download:
        if not ready:
            receipt["status"] = "fail_closed_no_kaggle"
            RECEIPT.parent.mkdir(parents=True, exist_ok=True)
            RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt, indent=2))
            raise SystemExit(
                "FAIL-CLOSED: Kaggle not configured. "
                "Accept competition rules on Kaggle, place ~/.kaggle/kaggle.json, "
                "pip install kaggle, then re-run with --download."
            )
        download_competition()
        receipt["status"] = "downloaded"

    if args.materialize:
        if not args.train_csv.is_file():
            receipt["status"] = "fail_closed_missing_train_csv"
            RECEIPT.parent.mkdir(parents=True, exist_ok=True)
            RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt, indent=2))
            raise SystemExit(f"missing {args.train_csv} — download+unzip first")
        stats = materialize_from_csv(
            args.train_csv,
            limit_patients=args.limit_patients,
            image_root=args.image_root,
        )
        receipt["materialize"] = stats
        receipt["status"] = "materialized" if sum(
            stats["copied"][s][c] for s in stats["copied"] for c in stats["copied"][s]
        ) else "materialize_zero_images"

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
