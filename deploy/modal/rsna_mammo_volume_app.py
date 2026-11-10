"""Modal volumes for RSNA mammo expansion (RUO).

Creates durable volumes (empty until upload):
- ``rsna-mammo-png`` — CBIS-compatible PNG tree
- ``rsna-mammo-embeddings`` — MedSigLIP 1152-d artifacts

    MODAL_PROFILE=fjkiani modal run deploy/modal/rsna_mammo_volume_app.py::ensure_volumes
"""
from __future__ import annotations

import modal

app = modal.App("rsna-mammo-volumes")

PNG_VOL = modal.Volume.from_name("rsna-mammo-png", create_if_missing=True)
EMB_VOL = modal.Volume.from_name("rsna-mammo-embeddings", create_if_missing=True)


@app.function(
    volumes={
        "/vol/rsna_png": PNG_VOL,
        "/vol/rsna_emb": EMB_VOL,
    },
)
def ensure_volumes() -> dict:
    """Touch volumes so they exist in the active Modal workspace."""
    from pathlib import Path

    Path("/vol/rsna_png/.keep").write_text("rsna-mammo-png\n")
    Path("/vol/rsna_emb/.keep").write_text("rsna-mammo-embeddings\n")
    PNG_VOL.commit()
    EMB_VOL.commit()
    return {
        "status": "ok",
        "volumes": ["rsna-mammo-png", "rsna-mammo-embeddings"],
        "note": "Empty until prepare_rsna_mammo_layout + upload. RESEARCH USE ONLY.",
    }


@app.local_entrypoint()
def main() -> None:
    print(ensure_volumes.remote())
