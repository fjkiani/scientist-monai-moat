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


@app.function(
    image=IMAGE.apt_install("unzip"),
    volumes={"/vol/mammo": VOL},
    timeout=6 * 60 * 60,
    memory=32768,
    ephemeral_disk=512 * 1024,
    cpu=4.0,
)
def unzip_vindr() -> dict:
    """Unzip VinDr archive already on volume → /vol/mammo/vindr_extracted."""
    from pathlib import Path
    import subprocess
    import shutil

    zip_path = Path(
        "/vol/mammo/vindr_raw/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-"
        "and-diagnosis-in-full-field-digital-mammography-1.0.0.zip"
    )
    out = Path("/vol/mammo/vindr_extracted")
    out.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        return {"status": "missing_zip", "expected": str(zip_path)}
    # Skip if already extracted with content
    existing = list(out.rglob("*.dicom"))[:1] or list(out.rglob("*.dcm"))[:1] or list(out.iterdir())
    if len(list(out.iterdir())) > 2:
        n_files = sum(1 for _ in out.rglob("*") if _.is_file())
        return {"status": "already_extracted", "out": str(out), "n_files": n_files, "bytes": zip_path.stat().st_size}
    subprocess.check_call(["unzip", "-q", "-o", str(zip_path), "-d", str(out)])
    VOL.commit()
    n_files = sum(1 for _ in out.rglob("*") if _.is_file())
    return {
        "status": "ok",
        "out": str(out),
        "n_files": n_files,
        "zip_bytes": zip_path.stat().st_size,
        "root_free_bytes": shutil.disk_usage("/").free,
    }


@app.local_entrypoint()
def main() -> None:
    print(download_zip.remote())
