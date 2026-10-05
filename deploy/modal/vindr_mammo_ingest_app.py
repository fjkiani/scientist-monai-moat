"""Download VinDr-Mammo onto Modal volume (local disk too small — 54GB zip).

    MODAL_PROFILE=fjkiani modal run deploy/modal/vindr_mammo_ingest_app.py::download_zip
"""
from __future__ import annotations

import modal

app = modal.App("vindr-mammo-ingest")
VOL = modal.Volume.from_name("rsna-mammo-png", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==0.30.2", "hf_transfer==0.1.9")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=IMAGE,
    volumes={"/vol/mammo": VOL},
    timeout=6 * 60 * 60,
    memory=16384,
    # Modal requires ephemeral_disk ∈ [512 GiB, 3072 GiB]
    ephemeral_disk=512 * 1024,
)
def download_zip() -> dict:
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    import os
    import shutil

    out = Path("/vol/mammo/vindr_raw")
    out.mkdir(parents=True, exist_ok=True)
    # Prefer volume for HF cache so commit keeps progress across retries
    cache = Path("/vol/mammo/hf_cache")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    path = hf_hub_download(
        repo_id="hassanzo/VinDr-Mammo",
        repo_type="dataset",
        filename=(
            "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-"
            "and-diagnosis-in-full-field-digital-mammography-1.0.0.zip"
        ),
        local_dir=str(out),
    )
    p = Path(path)
    VOL.commit()
    free = shutil.disk_usage("/").free
    return {
        "status": "ok",
        "path": str(p),
        "bytes": p.stat().st_size,
        "root_free_bytes": free,
    }


@app.local_entrypoint()
def main() -> None:
    print(download_zip.remote())
