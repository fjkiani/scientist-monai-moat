"""Embed every CBIS-DDSM_1024 PNG via the Modal MedSigLIP endpoint.

Walks::

    /workspace/cbis_ddsm_1024/{train,test}/{cancer,not_cancer}/*.png

and produces four artifacts in ``--out-dir``:

* ``embeddings.npy``  — float32, shape ``(N, 1152)``
* ``labels.npy``      — int8, 1=cancer, 0=not_cancer
* ``splits.npy``      — int8, 1=train, 0=test
* ``paths.npy``       — object array of absolute PNG paths (same order)

Batches are chunked to 16 (Modal server cap is 32; keeping a margin for the
HTTP body size). Progress is printed every 5 batches. On failure the script
resumes from ``embeddings.npy`` on-disk if present.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the src package is importable when running the script directly.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient  # noqa: E402


DEFAULT_ROOT = Path("/workspace/cbis_ddsm_1024")
DEFAULT_OUT = Path("/workspace/cbis_ddsm_1024/artifacts")


def gather_paths(root: Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Walk the four class/split folders. Returns (paths, labels, splits)."""
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
        # Deterministic order → downstream reproducibility.
        files = sorted(folder.glob("*.png"))
        print(f"  {split_name}/{cls_name}: {len(files)}")
        paths.extend(files)
        labels.extend([label] * len(files))
        splits.extend([split_flag] * len(files))
    return paths, np.asarray(labels, dtype=np.int8), np.asarray(splits, dtype=np.int8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N images (for smoke)")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("MODAL_MEDSIGLIP_URL"):
        raise SystemExit("set MODAL_MEDSIGLIP_URL to the Modal base URL")

    print(f"[gather] scanning {args.root}")
    paths, labels, splits = gather_paths(args.root)
    n = len(paths)
    if args.limit:
        paths = paths[: args.limit]
        labels = labels[: args.limit]
        splits = splits[: args.limit]
        n = len(paths)
    print(f"[gather] {n} images total  (train={int(splits.sum())}, test={n - int(splits.sum())})")
    print(f"[gather] positives={int(labels.sum())}  negatives={n - int(labels.sum())}")

    # Pre-allocate embeddings matrix; fill in place so we can save partial progress.
    dim = 1152
    embeddings = np.zeros((n, dim), dtype=np.float32)
    done_mask = np.zeros(n, dtype=bool)

    # Resume-from-disk if partial artifacts exist.
    emb_path = out_dir / "embeddings.npy"
    done_path = out_dir / "done_mask.npy"
    if emb_path.exists() and done_path.exists():
        prev = np.load(emb_path)
        prev_mask = np.load(done_path)
        if prev.shape == embeddings.shape and prev_mask.shape == done_mask.shape:
            embeddings[:] = prev
            done_mask[:] = prev_mask
            print(f"[resume] loaded {int(done_mask.sum())}/{n} previously-embedded rows")

    client = MedSigLipModalClient(batch_chunk=args.chunk)
    gate = client.preflight()
    print(f"[gate] access={gate.access_level.value}  reason={gate.reason}")
    if gate.access_level.value != "allowed":
        raise SystemExit(f"preflight not allowed: {gate.reason}")

    remaining = np.where(~done_mask)[0].tolist()
    print(f"[work] {len(remaining)} rows to embed  chunk={args.chunk}")

    t_start = time.time()
    total_seconds_server = 0.0
    for chunk_i, i in enumerate(range(0, len(remaining), args.chunk)):
        idxs = remaining[i : i + args.chunk]
        batch_paths = [str(paths[j]) for j in idxs]
        t0 = time.time()
        # embed_dicoms auto-detects PNG vs DICOM and picks pixels_b64.
        embs = client.embed_dicoms(batch_paths, chunk=len(batch_paths))
        dt = time.time() - t0
        total_seconds_server += dt
        arr = np.asarray(embs, dtype=np.float32)
        if arr.shape != (len(idxs), dim):
            raise RuntimeError(f"batch {chunk_i}: expected ({len(idxs)},{dim}), got {arr.shape}")
        embeddings[idxs] = arr
        done_mask[idxs] = True

        if chunk_i % 5 == 0 or i + args.chunk >= len(remaining):
            done_n = int(done_mask.sum())
            frac = done_n / n
            elapsed = time.time() - t_start
            eta = (elapsed / max(frac, 1e-6)) * (1 - frac) if frac > 0 else 0
            print(
                f"[batch {chunk_i:>4d}]  {done_n:>4d}/{n}  ({frac*100:5.1f}%)  "
                f"batch_wall={dt:.2f}s  elapsed={elapsed:.0f}s  eta={eta:.0f}s"
            )
            # Persist progress every 5 batches.
            np.save(emb_path, embeddings)
            np.save(done_path, done_mask)

    # Final save
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "splits.npy", splits)
    np.save(out_dir / "paths.npy", np.asarray([str(p) for p in paths], dtype=object))
    if done_path.exists():
        done_path.unlink()  # done_mask no longer needed once everything is embedded

    total_wall = time.time() - t_start
    print(
        f"[done] {n} embeddings, dim={dim}  wall={total_wall:.1f}s  "
        f"server_sum={total_seconds_server:.1f}s  throughput={n/max(total_wall,1e-6):.1f} img/s"
    )
    print(f"[out] wrote to {out_dir}/")


if __name__ == "__main__":
    main()
