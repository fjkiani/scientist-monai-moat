#!/usr/bin/env python3
"""Layout theoviel RSNA 256 PNGs into cancer/not_cancer tree using train.csv labels."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-src", type=Path, default=ROOT / "data/rsna_mammo/png_src")
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data/rsna_mammo/png_tree")
    ap.add_argument("--limit", type=int, default=12000)
    ap.add_argument("--receipt", type=Path, default=ROOT / "artifacts/rsna_mammo/layout_v1.json")
    args = ap.parse_args()

    label_by_key: dict[str, int] = {}
    with args.train_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = f"{row['patient_id']}_{row['image_id']}"
            label_by_key[key] = int(row.get("cancer") or 0)

    for split in ("train", "test"):
        for cls in ("cancer", "not_cancer"):
            (args.out / split / cls).mkdir(parents=True, exist_ok=True)

    n_pos = n_neg = n_skip = 0
    for png in sorted(args.png_src.glob("*.png")):
        if n_pos + n_neg >= args.limit:
            break
        key = png.stem
        if key not in label_by_key:
            n_skip += 1
            continue
        cls = "cancer" if label_by_key[key] == 1 else "not_cancer"
        # keep all in train for embedding pass; later scripts can split
        dest = args.out / "train" / cls / png.name
        if not dest.exists():
            shutil.copy2(png, dest)
        if cls == "cancer":
            n_pos += 1
        else:
            n_neg += 1

    receipt = {
        "status": "ok",
        "png_src": str(args.png_src),
        "train_csv": str(args.train_csv),
        "out": str(args.out),
        "limit": args.limit,
        "n_cancer": n_pos,
        "n_not_cancer": n_neg,
        "n_skip_unlabeled": n_skip,
        "source_dataset": "theoviel/rsna-breast-cancer-256-pngs",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
