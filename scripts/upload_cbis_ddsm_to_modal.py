"""Upload CBIS-DDSM DICOM corpus + series manifest to Modal Volume.

Volume populated
----------------
- ``cbis-ddsm-data`` → ``CBIS-DDSM_full/`` DICOM tree + ``series_manifest.json``.

Also uploads the current ``src/oncology_arbiter/`` tree to ``oa-repo-code``
Volume so the training container can import the scaffold helpers.

Usage
-----
    modal run scripts/upload_cbis_ddsm_to_modal.py::upload \
        --cbis-dir /workspace/data/CBIS-DDSM_full \
        --manifest /workspace/data/CBIS-DDSM_full/series_manifest.json \
        --oa-src /workspace/oa-repo/src
"""
from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("cbis-ddsm-upload")

CBIS_VOL = modal.Volume.from_name("cbis-ddsm-data", create_if_missing=True)
CODE_VOL = modal.Volume.from_name("oa-repo-code", create_if_missing=True)

UPLOAD_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "tqdm==4.66.5"
)


def _mount_kwargs(cbis_dir: Path, oa_src: Path) -> dict:
    return {
        "mounts": [
            modal.Mount.from_local_dir(str(cbis_dir), remote_path="/local/cbis"),
            modal.Mount.from_local_dir(str(oa_src), remote_path="/local/oa_src"),
        ]
    }


@app.function(
    image=UPLOAD_IMAGE,
    volumes={
        "/vol/cbis": CBIS_VOL,
        "/vol/code": CODE_VOL,
    },
    timeout=6 * 60 * 60,
    memory=16 * 1024,
    cpu=8.0,
)
def _upload_from_mounts() -> dict:
    import shutil
    from pathlib import Path

    def _copy_tree(src: Path, dst: Path, only_ext: tuple[str, ...] | None = None) -> tuple[int, int]:
        n_copied = 0
        n_skipped = 0
        for p in src.rglob("*"):
            if not p.is_file():
                continue
            if only_ext and p.suffix.lower() not in only_ext:
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

    # 1. CBIS DICOM corpus (large — 163 GB target)
    cbis_dst = Path("/vol/cbis/CBIS-DDSM_full")
    cbis_new, cbis_skip = _copy_tree(Path("/local/cbis"), cbis_dst)

    # 2. Series manifest JSON (small)
    for f in Path("/local/cbis").glob("series_manifest.json*"):
        shutil.copy2(f, Path("/vol/cbis") / f.name)

    # 3. Code (small)
    code_dst = Path("/vol/code/src")
    code_new, code_skip = _copy_tree(Path("/local/oa_src"), code_dst)

    CBIS_VOL.commit()
    CODE_VOL.commit()
    return {
        "cbis": {"copied": cbis_new, "skipped": cbis_skip},
        "code": {"copied": code_new, "skipped": code_skip},
    }


@app.local_entrypoint()
def upload(cbis_dir: str, oa_src: str) -> None:
    cbis = Path(cbis_dir).resolve()
    src = Path(oa_src).resolve()
    if not cbis.is_dir():
        raise SystemExit(f"cbis-dir not a directory: {cbis}")
    if not src.is_dir():
        raise SystemExit(f"oa-src not a directory: {src}")

    fn = _upload_from_mounts.with_options(**_mount_kwargs(cbis, src))
    result = fn.remote()
    import json
    print(json.dumps(result, indent=2))
