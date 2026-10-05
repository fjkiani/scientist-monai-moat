"""RSNA screening mammo → Modal volume PNG tree (RUO).

Requires KAGGLE_API_TOKEN in the environment (or Modal secret ``kaggle-api``).

    export KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)"
    MODAL_PROFILE=fjkiani modal run deploy/modal/rsna_kaggle_ingest_app.py \\
        --max-patients 2500
"""
from __future__ import annotations

import modal

app = modal.App("rsna-kaggle-ingest")
VOL = modal.Volume.from_name("rsna-mammo-png", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "kaggle==1.7.4.5",
        "pydicom==3.0.1",
        "pillow==11.1.0",
        "numpy==2.2.3",
        "pandas==2.2.3",
    )
)


@app.function(
    image=IMAGE,
    volumes={"/vol/mammo": VOL},
    timeout=6 * 60 * 60,
    memory=16384,
    ephemeral_disk=512 * 1024,
    secrets=[modal.Secret.from_name("kaggle-api", required=False)],
)
def ingest(max_patients: int = 2500) -> dict:
    import csv
    import os
    import subprocess
    import zipfile
    from pathlib import Path

    import numpy as np
    import pydicom
    from PIL import Image

    token = os.environ.get("KAGGLE_API_TOKEN") or ""
    if not token:
        return {"status": "failed", "error": "KAGGLE_API_TOKEN missing on Modal"}

    raw = Path("/vol/mammo/rsna_raw")
    png = Path("/vol/mammo/png_tree")
    raw.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for cls in ("cancer", "not_cancer"):
            (png / split / cls).mkdir(parents=True, exist_ok=True)

    # train.csv
    csv_path = raw / "train.csv"
    if not csv_path.is_file():
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "rsna-breast-cancer-detection",
                "-f",
                "train.csv",
                "-p",
                str(raw),
            ],
            check=True,
            env={**os.environ, "KAGGLE_API_TOKEN": token},
        )
        z = raw / "train.csv.zip"
        if z.is_file():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(raw)

    rows = list(csv.DictReader(csv_path.open()))
    # one image per patient (prefer CC/MLO any), stratified-ish by cancer
    by_patient: dict[str, list[dict]] = {}
    for r in rows:
        by_patient.setdefault(r["patient_id"], []).append(r)

    cancer_pids = [p for p, rs in by_patient.items() if any(int(x.get("cancer") or 0) == 1 for x in rs)]
    neg_pids = [p for p in by_patient if p not in set(cancer_pids)]
    # half/half up to max_patients
    n_pos = min(len(cancer_pids), max_patients // 2)
    n_neg = min(len(neg_pids), max_patients - n_pos)
    selected = cancer_pids[:n_pos] + neg_pids[:n_neg]

    written = 0
    errors = 0
    for pid in selected:
        cand = by_patient[pid][0]
        image_id = cand["image_id"]
        label = "cancer" if int(cand.get("cancer") or 0) == 1 else "not_cancer"
        out_png = png / "train" / label / f"{pid}_{image_id}.png"
        if out_png.is_file():
            written += 1
            continue
        rel = f"train_images/{pid}/{image_id}.dcm"
        dest = raw / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not dest.is_file():
                subprocess.run(
                    [
                        "kaggle",
                        "competitions",
                        "download",
                        "-c",
                        "rsna-breast-cancer-detection",
                        "-f",
                        rel,
                        "-p",
                        str(dest.parent),
                    ],
                    check=True,
                    env={**os.environ, "KAGGLE_API_TOKEN": token},
                    capture_output=True,
                )
                # kaggle may write basename into -p dir
                maybe = dest.parent / f"{image_id}.dcm"
                if maybe.is_file() and not dest.is_file():
                    maybe.rename(dest)
            ds = pydicom.dcmread(str(dest))
            arr = ds.pixel_array.astype(np.float32)
            arr = arr - arr.min()
            if arr.max() > 0:
                arr = arr / arr.max()
            img = Image.fromarray((arr * 255.0).astype(np.uint8)).convert("L")
            img = img.resize((512, 512))
            img.save(out_png)
            written += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if errors <= 5:
                print(f"err {pid}: {type(exc).__name__}: {exc}", flush=True)
        if written % 50 == 0:
            VOL.commit()
            print(f"progress written={written} errors={errors}", flush=True)

    VOL.commit()
    return {
        "status": "ok",
        "max_patients": max_patients,
        "selected": len(selected),
        "written": written,
        "errors": errors,
        "png_root": str(png),
    }


@app.local_entrypoint()
def main(max_patients: int = 2500) -> None:
    import json
    import os
    from pathlib import Path

    token = os.environ.get("KAGGLE_API_TOKEN") or ""
    if not token and Path.home().joinpath(".kaggle/api_token").is_file():
        os.environ["KAGGLE_API_TOKEN"] = Path.home().joinpath(".kaggle/api_token").read_text().strip()
    # Ensure Modal secret exists for durable runs
    try:
        modal.Secret.from_name("kaggle-api")
    except Exception:
        if os.environ.get("KAGGLE_API_TOKEN"):
            modal.Secret.create_deployed(
                "kaggle-api",
                {"KAGGLE_API_TOKEN": os.environ["KAGGLE_API_TOKEN"]},
            )
            print("created Modal secret kaggle-api")
    print(json.dumps(ingest.remote(max_patients=max_patients), indent=2))
