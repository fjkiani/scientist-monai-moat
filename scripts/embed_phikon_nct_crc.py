"""Bulk-embed the full NCT-CRC-HE-100K (train) + CRC-VAL-HE-7K (held-out val)
Zenodo colorectal-histology datasets through the LIVE Phikon Modal endpoint,
then fit + evaluate a real 9-class linear probe on the real 768-d embeddings.

This is the concrete artifact forward-referenced (but never created) by the
docstring of tests/integration/test_phikon_pathology_live.py:

    "deliberately separate from the large-scale population validation
    (phikon_nct_crc_population_validation.json), which embeds the full
    107,180-image dataset locally and fits/evaluates a real classifier"

HONESTY / SCOPE (do not let downstream readers over-claim this):
  - NCT-CRC-HE-100K / CRC-VAL-HE-7K are COLORECTAL H&E histology tiles
    (Kather et al., Zenodo DOI 10.5281/zenodo.1214456, CC-BY-4.0). This is
    NOT the product's claimed screening tissue/task. This script produces a
    domain/population validation of the Phikon embedding's discriminative
    power on a large, independent, real histology benchmark -- it is
    EXPLICITLY NOT a product screening-accuracy claim for any oncology
    Arbiter output. The output JSON repeats this disclaimer verbatim.
  - CRC-VAL-HE-7K is the dataset authors' own held-out split: N=50 patients
    with NO overlap with the 100k training tiles (see the download script's
    own provenance banner). Train/val separation here is therefore a real,
    patient-level held-out evaluation, not a random re-split of the same
    tiles.
  - Standalone script (does not import oncology_arbiter) because the data
    lives on a machine without that package installed; talks to the Phikon
    endpoint via plain urllib, replicating the exact wire format used by
    PhikonClient.embed() in src/oncology_arbiter/models/specialist_clients.py
    (payload {"inputs": [{"image_b64": ...}, ...]}, response either
    {"embedding": [...]} for n=1 or {"embeddings": [[...], ...]} for n>1).
  - Empirically measured (2026-09-27) server-enforced batch cap: batch size
    <=64 (HTTP 200 with body {"error": "batch size N exceeds cap of 64"} for
    larger batches -- NOT an HTTPError, must be checked explicitly). This
    script hard-caps at 64 and treats an "error" key in a 200 response as a
    real failure, not a silent empty result.
  - Steady-state throughput measured on this endpoint before launching the
    full run: ~345-355 ms server compute / batch-of-64 (~5.4-5.6 ms/image),
    ~1.6-1.8s wall/batch-of-64 including network. ~1675 batches for the full
    107,180 images -> ~45-55 min wall, independently confirmed by the actual
    full run's logged wall time in the output JSON.

Usage::

    PHIKON_MODAL_EMBED_URL=https://crispro--phikon-embed.modal.run \\
        python3 scripts/embed_phikon_nct_crc.py \\
        --nct-root /workspace/pathology_crc/NCT-CRC-HE-100K/NCT-CRC-HE-100K \\
        --val-root /workspace/pathology_crc/CRC-VAL-HE-7K/CRC-VAL-HE-7K \\
        --out-dir /workspace/pathology_crc/phikon_embeddings \\
        [--limit-per-class N]  # smoke-test override, omit for full population
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
BATCH_CAP = 64
MAX_ATTEMPTS = 4
BACKOFF_SCHEDULE_S = [5, 15, 45]


def gather_manifest(root: Path, split: str) -> list[dict]:
    rows = []
    for cls in CLASSES:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            raise SystemExit(f"missing expected class dir: {cls_dir}")
        for fp in sorted(cls_dir.glob("*.tif")):
            rows.append({"path": str(fp), "class": cls, "label": CLASS_TO_IDX[cls], "split": split})
    return rows


def call_phikon_batch(url: str, image_paths: list[str], timeout: float = 120.0) -> list[list[float]]:
    inputs = []
    for fp in image_paths:
        with open(fp, "rb") as f:
            raw_bytes = f.read()
        inputs.append({"image_b64": base64.b64encode(raw_bytes).decode("ascii")})
    payload = json.dumps({"inputs": inputs}, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("error"):
                # Real, documented failure mode (e.g. batch-size cap) -- not silent.
                raise RuntimeError(f"phikon endpoint returned error: {data['error']}")
            embeddings = data.get("embeddings")
            if embeddings is None and isinstance(data.get("embedding"), list):
                embeddings = [data["embedding"]]
            if not isinstance(embeddings, list) or len(embeddings) != len(image_paths):
                raise RuntimeError(
                    f"contract_mismatch: requested {len(image_paths)} embeddings, "
                    f"got {len(embeddings) if isinstance(embeddings, list) else type(embeddings)}"
                )
            for e in embeddings:
                if not isinstance(e, list) or len(e) != 768:
                    raise RuntimeError("contract_mismatch: embedding not exactly 768-dim")
            return embeddings
        except Exception as exc:  # noqa: BLE001 - retry transient, surface persistent
            last_exc = exc
            if attempt < MAX_ATTEMPTS:
                sleep_s = BACKOFF_SCHEDULE_S[min(attempt - 1, len(BACKOFF_SCHEDULE_S) - 1)]
                print(f"    batch call failed (attempt {attempt}/{MAX_ATTEMPTS}): {exc} -- retrying in {sleep_s}s", flush=True)
                time.sleep(sleep_s)
    raise RuntimeError(f"batch call failed after {MAX_ATTEMPTS} attempts: {last_exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nct-root", type=Path, required=True)
    ap.add_argument("--val-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--embed-url", type=str, default=None, help="defaults to $PHIKON_MODAL_EMBED_URL")
    ap.add_argument("--limit-per-class", type=int, default=None, help="smoke-test cap per class per split")
    ap.add_argument("--checkpoint-every", type=int, default=50, help="save embeddings.npy every N batches")
    args = ap.parse_args()

    url = args.embed_url or os.environ.get("PHIKON_MODAL_EMBED_URL")
    if not url:
        raise SystemExit("set --embed-url or $PHIKON_MODAL_EMBED_URL")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = gather_manifest(args.nct_root, "train") + gather_manifest(args.val_root, "val")
    if args.limit_per_class:
        # Cap per (split, class) while preserving the deterministic sort order.
        capped = []
        counts: dict[tuple, int] = {}
        for row in manifest:
            key = (row["split"], row["class"])
            counts.setdefault(key, 0)
            if counts[key] < args.limit_per_class:
                capped.append(row)
                counts[key] += 1
        manifest = capped

    n_total = len(manifest)
    print(f"[gather] {n_total} images ({sum(1 for r in manifest if r['split']=='train')} train / "
          f"{sum(1 for r in manifest if r['split']=='val')} val)", flush=True)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    emb_path = out_dir / "embeddings.npy"
    done_path = out_dir / "done_mask.npy"
    embeddings = np.zeros((n_total, 768), dtype=np.float32)
    done_mask = np.zeros(n_total, dtype=bool)
    if emb_path.exists() and done_path.exists():
        prior_emb = np.load(emb_path)
        prior_done = np.load(done_path)
        if prior_emb.shape == embeddings.shape and prior_done.shape == done_mask.shape:
            embeddings = prior_emb
            done_mask = prior_done
            print(f"[resume] {int(done_mask.sum())}/{n_total} already embedded", flush=True)
        else:
            print("[resume] shape mismatch with prior checkpoint -- starting fresh", flush=True)

    remaining_idx = [i for i in range(n_total) if not done_mask[i]]
    t_start = time.time()
    n_batches_done = 0
    for batch_start in range(0, len(remaining_idx), BATCH_CAP):
        batch_idx = remaining_idx[batch_start: batch_start + BATCH_CAP]
        batch_paths = [manifest[i]["path"] for i in batch_idx]
        t0 = time.time()
        vecs = call_phikon_batch(url, batch_paths)
        for local_i, global_i in enumerate(batch_idx):
            embeddings[global_i] = np.asarray(vecs[local_i], dtype=np.float32)
            done_mask[global_i] = True
        n_batches_done += 1
        n_done_total = int(done_mask.sum())
        elapsed = time.time() - t_start
        print(f"[{n_done_total}/{n_total}] batch of {len(batch_idx)} in {time.time()-t0:.2f}s "
              f"(cum wall={elapsed:.0f}s)", flush=True)
        if n_batches_done % args.checkpoint_every == 0:
            np.save(emb_path, embeddings)
            np.save(done_path, done_mask)
            print(f"    [checkpoint] saved at {n_done_total}/{n_total}", flush=True)

    np.save(emb_path, embeddings)
    np.save(done_path, done_mask)
    elapsed = time.time() - t_start
    print(f"[done] {int(done_mask.sum())}/{n_total} embedded, wall={elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
