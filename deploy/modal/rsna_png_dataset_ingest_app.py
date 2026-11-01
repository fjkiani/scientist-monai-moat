"""Pull preprocessed RSNA PNG dataset onto Modal volume (RUO).

Uses Kaggle dataset ``theoviel/rsna-breast-cancer-256-pngs`` (~1GB) as the
fast path past CBIS-only. Larger 512/1024 variants can swap dataset_id.

    export KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)"
    MODAL_PROFILE=fjkiani modal run deploy/modal/rsna_png_dataset_ingest_app.py
"""
from __future__ import annotations

import modal

app = modal.App("rsna-png-dataset-ingest")
VOL = modal.Volume.from_name("rsna-mammo-png", create_if_missing=True)
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("kaggle==1.7.4.5")
)


@app.function(
    image=IMAGE,
    volumes={"/vol/mammo": VOL},
    timeout=3 * 60 * 60,
    memory=8192,
    ephemeral_disk=512 * 1024,
    secrets=[modal.Secret.from_name("kaggle-api")],
)
def download_dataset(
    dataset_id: str = "theoviel/rsna-breast-cancer-256-pngs",
) -> dict:
    import os
    import subprocess
    import zipfile
    from pathlib import Path

    out = Path("/vol/mammo/rsna_png_256")
    out.mkdir(parents=True, exist_ok=True)
    marker = out / ".download_complete"
    if marker.is_file():
        n = sum(1 for _ in out.rglob("*.png"))
        return {"status": "already_present", "png_count": n, "root": str(out)}

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_id,
            "-p",
            str(out),
            "--unzip",
        ],
        check=True,
        env=os.environ.copy(),
    )
    # some versions leave a zip
    for z in out.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        z.unlink(missing_ok=True)
    n = sum(1 for _ in out.rglob("*.png"))
    marker.write_text(f"png_count={n}\n")
    VOL.commit()
    return {"status": "ok", "dataset_id": dataset_id, "png_count": n, "root": str(out)}


@app.local_entrypoint()
def main(dataset_id: str = "theoviel/rsna-breast-cancer-256-pngs") -> None:
    import json

    print(json.dumps(download_dataset.remote(dataset_id=dataset_id), indent=2))
