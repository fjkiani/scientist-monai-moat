#!/usr/bin/env python3
"""Generic MedSigLIP-1152 embedder for mammo PNG trees (CBIS / RSNA / VinDr).

Expects the CBIS-compatible layout::

    {root}/{train,test}/{cancer,not_cancer}/*.png

Produces ``embeddings.npy`` (N×1152), ``labels.npy``, ``splits.npy``, ``paths.npy``
under ``--out-dir``. Resumes via ``done_mask.npy``.

Requires ``MODAL_MEDSIGLIP_URL``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient  # noqa: E402


def gather_paths(root: Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    paths: list[Path] = []
    labels: list[int] = []
    splits: list[int] = []
    layout = [
        ("train", "cancer", 1, 1),
        ("train", "not_cancer", 0, 1),
        ("test", "cancer", 1, 0),
        ("test", "not_cancer", 0, 0),
    ]
    for split_name, cls_name, label, split_flag in layout:
        folder = root / split_name / cls_name
        if not folder.is_dir():
            raise FileNotFoundError(f"missing folder: {folder}")
        files = sorted(folder.glob("*.png"))
        print(f"  {split_name}/{cls_name}: {len(files)}")
        paths.extend(files)
        labels.extend([label] * len(files))
        splits.extend([split_flag] * len(files))
    return paths, np.asarray(labels, dtype=np.int8), np.asarray(splits, dtype=np.int8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="PNG tree root")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dataset-id", type=str, default="mammo_png_tree")
    args = ap.parse_args()

    if not os.environ.get("MODAL_MEDSIGLIP_URL"):
        raise SystemExit("set MODAL_MEDSIGLIP_URL to the Modal MedSigLIP base URL")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gather] dataset_id={args.dataset_id} root={args.root}")
    paths, labels, splits = gather_paths(args.root)
    if args.limit:
        paths = paths[: args.limit]
        labels = labels[: args.limit]
        splits = splits[: args.limit]
    n = len(paths)
    if n == 0:
        raise SystemExit(f"no PNGs under {args.root}")
    print(f"[gather] {n} images  train={int(splits.sum())} test={n - int(splits.sum())}")

    dim = 1152
    embeddings = np.zeros((n, dim), dtype=np.float32)
    done_mask = np.zeros(n, dtype=bool)
    emb_path = out_dir / "embeddings.npy"
    done_path = out_dir / "done_mask.npy"
    if emb_path.exists() and done_path.exists():
        prev = np.load(emb_path)
        prev_mask = np.load(done_path)
        if prev.shape == embeddings.shape and prev_mask.shape == done_mask.shape:
            embeddings[:] = prev
            done_mask[:] = prev_mask
            print(f"[resume] {int(done_mask.sum())}/{n}")

    client = MedSigLipModalClient(batch_chunk=args.chunk)
    gate = client.preflight()
    if gate.access_level.value != "allowed":
        raise SystemExit(f"preflight not allowed: {gate.reason}")

    remaining = np.where(~done_mask)[0].tolist()
    t_start = time.time()
    for chunk_i, i in enumerate(range(0, len(remaining), args.chunk)):
        idxs = remaining[i : i + args.chunk]
        batch_paths = [str(paths[j]) for j in idxs]
        embs = client.embed_dicoms(batch_paths, chunk=len(batch_paths))
        arr = np.asarray(embs, dtype=np.float32)
        if arr.shape != (len(idxs), dim):
            raise RuntimeError(f"bad batch shape {arr.shape}")
        embeddings[idxs] = arr
        done_mask[idxs] = True
        if chunk_i % 5 == 0 or i + args.chunk >= len(remaining):
            np.save(emb_path, embeddings)
            np.save(done_path, done_mask)
            print(f"[batch {chunk_i}] {int(done_mask.sum())}/{n}")

    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "splits.npy", splits)
    np.save(out_dir / "paths.npy", np.asarray([str(p) for p in paths], dtype=object))
    (out_dir / "dataset_id.txt").write_text(args.dataset_id + "\n")
    if done_path.exists():
        done_path.unlink()
    print(f"[done] {n}×{dim} wall={time.time() - t_start:.1f}s → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
