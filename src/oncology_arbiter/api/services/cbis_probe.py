"""CBIS-DDSM supervised probe runner (embedding-only)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .medsiglip import get_medsiglip

def run_cbis_ddsm_probe_on_bytes(
    dicom_bytes: bytes,
    preprocess_result: Any = None,
    *,
    embedding: list[float] | None = None,
    dicom_path: str | Path | None = None,
    modal_client: Any | None = None,
) -> Any | None:
    """Run the trained CBIS-DDSM supervised probe on a mammogram.

    Scores the **embedding-only** CBIS probe (currently
    ``models/cbis_ddsm_logreg_v2.joblib``) over a 1152-d MedSigLIP-448 embedding.
    Embedding sources (first hit wins):

      1. explicit ``embedding`` list (preferred — reuse the screening receipt)
      2. ``modal_client.embed_dicom(dicom_path)`` when both are supplied
      3. a fresh Modal embed of ``dicom_bytes`` via ``MedSigLipModalClient``

    Returns ``None`` when no Modal embedding path is available so callers can
    keep the zero-shot findings without poisoning the response.
    """
    from oncology_arbiter.models.cbis_ddsm_probe import CbisDdsmProbe
    from oncology_arbiter.models.medsiglip_modal_client import MedSigLipModalClient

    emb = embedding
    if emb is None and dicom_path is not None:
        client = modal_client if isinstance(modal_client, MedSigLipModalClient) else MedSigLipModalClient()
        emb = client.embed_dicom(dicom_path)
    if emb is None:
        client = modal_client if isinstance(modal_client, MedSigLipModalClient) else None
        if client is None:
            ms = get_medsiglip()
            if isinstance(ms, MedSigLipModalClient):
                client = ms
            else:
                # Prefer an explicit Modal client over a local encoder that
                # may not expose 1152-d embeddings on this process.
                try:
                    client = MedSigLipModalClient()
                except Exception:
                    return None
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tf:
            tf.write(dicom_bytes)
            tmp_path = tf.name
        try:
            emb = client.embed_dicom(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return CbisDdsmProbe.get().predict(emb)



class CbisProbeRuntime:
    """Class facade for the CBIS-DDSM probe."""

    run_on_bytes = staticmethod(run_cbis_ddsm_probe_on_bytes)
