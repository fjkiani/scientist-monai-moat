"""Upload LUNA16 raw data + MONAI baseline bundle to Modal Volumes.

Volumes populated
-----------------
- ``luna16-data``            → subset*.zip, LUNA16_datasplit/, seg-lungs, annotations, candidates, evaluationScript.
- ``luna16-baseline-weights``→ MONAI ``lung_nodule_ct_detection`` bundle (config/models/scripts).

Usage
-----
    modal run scripts/upload_luna16_to_modal.py::upload \
        --luna16-dir /workspace/data/luna16 \
        --bundle-dir /workspace/monai_bundles/lung_nodule_ct_detection

This runs INSIDE a Modal function so it can call ``vol.commit()`` and
stream large files without a local CLI upload session.

Idempotent — skips files that already exist in the Volume.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import modal

app = modal.App("luna16-upload")

LUNA16_VOL = modal.Volume.from_name("luna16-data", create_if_missing=True)
BASELINE_VOL = modal.Volume.from_name(
    "luna16-baseline-weights", create_if_missing=True
)

UPLOAD_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "tqdm==4.66.5",
)


def _mount_kwargs(luna16_dir: Path, bundle_dir: Path) -> dict:
    """Modal mounts for local dirs → container paths."""
    return {
        "mounts": [
            modal.Mount.from_local_dir(str(luna16_dir), remote_path="/local/luna16"),
            modal.Mount.from_local_dir(str(bundle_dir), remote_path="/local/bundle"),
        ]
    }


@app.function(
    image=UPLOAD_IMAGE,
    volumes={
        "/vol/luna16": LUNA16_VOL,
        "/vol/baseline": BASELINE_VOL,
    },
    timeout=4 * 60 * 60,
    memory=8 * 1024,
    cpu=4.0,
)
def _upload_from_mounts() -> dict:
    """Runs inside Modal. Copies /local/luna16/* → /vol/luna16 and bundle → /vol/baseline."""
    import shutil
    from pathlib import Path

    def _copy_tree(src: Path, dst: Path) -> tuple[int, int]:
        n_copied = 0
        n_skipped = 0
        for p in src.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(src)
            target = dst / rel
            if target.exists() and target.stat().st_size == p.stat().st_size:
                n_skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            n_copied += 1
        return n_copied, n_skipped

    luna_dst = Path("/vol/luna16")
    bundle_dst = Path("/vol/baseline/lung_nodule_ct_detection")

    luna_new, luna_skip = _copy_tree(Path("/local/luna16"), luna_dst)
    bundle_new, bundle_skip = _copy_tree(Path("/local/bundle"), bundle_dst)

    LUNA16_VOL.commit()
    BASELINE_VOL.commit()
    return {
        "luna16": {"copied": luna_new, "skipped": luna_skip},
        "bundle": {"copied": bundle_new, "skipped": bundle_skip},
    }


@app.local_entrypoint()
def upload(luna16_dir: str, bundle_dir: str) -> None:
    """Copy local LUNA16 + bundle dirs to Modal Volumes.

    Args
    ----
    luna16_dir : str
        Local dir containing subset*.zip, LUNA16_datasplit/, etc.
    bundle_dir : str
        Local dir containing the ``lung_nodule_ct_detection`` MONAI bundle.
    """
    luna16 = Path(luna16_dir).resolve()
    bundle = Path(bundle_dir).resolve()
    if not luna16.is_dir():
        raise SystemExit(f"luna16-dir not a directory: {luna16}")
    if not bundle.is_dir():
        raise SystemExit(f"bundle-dir not a directory: {bundle}")

    # Attach mounts dynamically then run
    fn = _upload_from_mounts.with_options(**_mount_kwargs(luna16, bundle))
    result = fn.remote()
    import json
    print(json.dumps(result, indent=2))
