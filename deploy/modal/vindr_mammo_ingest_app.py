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


@app.function(
    image=IMAGE.pip_install("pydicom==3.0.1", "pillow==11.1.0", "numpy==2.2.3"),
    volumes={"/vol/mammo": VOL},
    timeout=2 * 60 * 60,
    memory=32768,
    ephemeral_disk=512 * 1024,
    cpu=4.0,
)
def convert_png_sample(limit: int = 512) -> dict:
    """Convert a labeled VinDr DICOM sample → PNG tree on the volume for MedSigLIP."""
    import csv
    from pathlib import Path
    import numpy as np
    from PIL import Image
    import pydicom

    root = Path(
        "/vol/mammo/vindr_extracted/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-"
        "and-diagnosis-in-full-field-digital-mammography-1.0.0"
    )
    ann = root / "breast-level_annotations.csv"
    images = root / "images"
    out = Path("/vol/mammo/vindr_png_tree")
    for split in ("train", "test"):
        for cls in ("cancer", "not_cancer"):
            (out / split / cls).mkdir(parents=True, exist_ok=True)
    if not ann.is_file():
        return {"status": "missing_annotations", "ann": str(ann)}

    # BI-RADS 4/5 → cancer; 1/2 → not_cancer; skip 3/0
    rows = []
    with ann.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            birads = (row.get("breast_birads") or "").upper()
            if "BI-RADS 5" in birads or "BI-RADS 4" in birads:
                label = "cancer"
            elif "BI-RADS 1" in birads or "BI-RADS 2" in birads:
                label = "not_cancer"
            else:
                continue
            rows.append((row, label))

    n_cancer = n_neg = n_skip = 0
    for row, label in rows:
        if n_cancer + n_neg >= limit:
            break
        study = row["study_id"]
        image_id = row["image_id"]
        split = "train" if (row.get("split") or "training").startswith("train") else "test"
        dicom = images / study / f"{image_id}.dicom"
        if not dicom.is_file():
            # some releases nest differently
            alt = list((images / study).glob(f"*{image_id}*")) if (images / study).is_dir() else []
            if not alt:
                n_skip += 1
                continue
            dicom = alt[0]
        try:
            ds = pydicom.dcmread(str(dicom))
            arr = ds.pixel_array.astype(np.float32)
            arr = arr - arr.min()
            denom = arr.max() - arr.min()
            if denom <= 0:
                n_skip += 1
                continue
            arr = (arr / denom * 255.0).astype(np.uint8)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            img = Image.fromarray(arr).convert("L").resize((256, 256))
            dest = out / split / label / f"{study}_{image_id}.png"
            img.save(dest)
            if label == "cancer":
                n_cancer += 1
            else:
                n_neg += 1
        except Exception:  # noqa: BLE001
            n_skip += 1
            continue

    VOL.commit()
    return {
        "status": "ok",
        "out": str(out),
        "limit": limit,
        "n_cancer": n_cancer,
        "n_not_cancer": n_neg,
        "n_skip": n_skip,
        "n_candidate_rows": len(rows),
    }


@app.local_entrypoint()
def main(mode: str = "download", limit: int = 512) -> None:
    if mode == "download":
        print(download_zip.remote())
    elif mode == "unzip":
        print(unzip_vindr.remote())
    elif mode == "convert":
        print(convert_png_sample.remote(limit=limit))
    else:
        raise SystemExit(f"unknown mode {mode!r}; use download|unzip|convert")
