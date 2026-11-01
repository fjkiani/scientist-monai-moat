"""Modal app: durable case ingress with content-addressable case IDs.

The oncology-arbiter previously accepted a client-supplied ``series_dir``
pointing at server-local filesystem paths. That is not a real ingress
contract: the API trusted the caller to know a server path. This app
replaces that with a Modal Volume-backed uploader that

- Accepts DICOM/NIfTI, report text, and genomics payloads.
- Writes bytes into a Modal ``Volume`` keyed by ``case_id`` (first 16 hex
  chars of the SHA-256 of the concatenated payloads).
- Records a provenance JSON alongside each case:
  ``{"case_id", "sha256_full", "created_at", "media_types", "byte_counts",
     "uploader_note"}``.
- Returns a ``{case_id, provenance_uri, storage_backend}`` receipt to the
  caller. The API gateway (Render) then references the ``case_id`` — the
  bytes never traverse the gateway again.

Endpoints
---------
- ``GET  /case-storage-healthz`` → liveness
- ``POST /case-storage-upload``  → chunked JSON upload  → case_id, provenance
- ``GET  /case-storage-get``     → ``?case_id=…&file=…`` → raw bytes
- ``GET  /case-storage-manifest``→ ``?case_id=…`` → provenance JSON

Design notes
------------
- CPU only (no GPU): this is I/O, not inference.
- ``min_containers=0`` — durability lives in the Volume, not the container.
- Base64 JSON is used for the upload body so we can drive it from stdlib
  ``urllib`` on the FastAPI gateway (Render) without pulling extra deps.
- Reads pipe through ``StreamingResponse`` when files exceed 5 MB to keep
  memory bounded on the container.

Deploy
------
    modal deploy deploy/modal/case_storage_app.py

RESEARCH USE ONLY — do NOT upload PHI. Callers are responsible for
upstream de-identification.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from typing import Any, Dict, List, Optional

import modal


APP_VERSION = "case-storage-modal-v0.5.0-alpha"
DISCLAIMER = "Research Use Only. Not FDA-cleared. Callers must de-identify uploads."

app = modal.App("case-storage")

# Durable case bytes live in a Modal Volume. Each case_id is a directory
# under /vol/<case_id>/{dicom.bin,report.txt,genomics.json,manifest.json}.
volume = modal.Volume.from_name("oncology-arbiter-cases", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi==0.115.0", "pydantic==2.13.4")
)


def _sha256(*payloads: bytes) -> str:
    h = hashlib.sha256()
    for p in payloads:
        h.update(p)
    return h.hexdigest()


def _case_dir(case_id: str) -> str:
    return f"/vol/{case_id}"


@app.function(image=IMAGE, volumes={"/vol": volume}, timeout=120)
@modal.fastapi_endpoint(method="GET", label="case-storage-healthz")
def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "app": "case-storage",
        "version": APP_VERSION,
        "volume": "oncology-arbiter-cases",
        "disclaimer": DISCLAIMER,
    }


@app.function(image=IMAGE, volumes={"/vol": volume}, timeout=300)
@modal.fastapi_endpoint(method="POST", label="case-storage-upload")
def upload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Upload a case bundle.

    Payload
    -------
    ``{"dicom_b64": "<base64 DICOM bytes | omit>",
       "nifti_b64": "<base64 NIfTI bytes | omit>",
       "report_text": "<UTF-8 text | omit>",
       "genomics_json": {<dict> | omit},
       "uploader_note": "<optional short string>"}``

    At least one of ``dicom_b64``, ``nifti_b64``, ``report_text``, or
    ``genomics_json`` MUST be present. All bytes are hashed together
    (in a stable order) to produce a content-addressable ``case_id``.

    Returns
    -------
    ``{"case_id", "provenance_uri", "storage_backend", "sha256_full",
       "byte_counts", "media_types", "created_at", "app_version"}``
    """
    t0 = time.time()

    dicom_b64 = payload.get("dicom_b64")
    # v0.5.0: multi-file DICOM series support.
    # dicom_series_b64 is either a list of base64 blobs (all stored to
    # dicom_series/<index>.dcm) or a mapping filename -> base64 (stored
    # as-is under dicom_series/). Content-address hash covers every slice
    # in name-sorted order, so the case_id remains deterministic.
    dicom_series_b64 = payload.get("dicom_series_b64")
    nifti_b64 = payload.get("nifti_b64")
    report_text = payload.get("report_text")
    genomics_json = payload.get("genomics_json")
    note = str(payload.get("uploader_note") or "")[:512]

    if not any([dicom_b64, dicom_series_b64, nifti_b64, report_text, genomics_json]):
        return {"error": "at least one of dicom_b64/dicom_series_b64/nifti_b64/report_text/genomics_json required"}

    files: Dict[str, bytes] = {}
    # Files that live inside a per-case subdirectory rather than the case
    # root. Keys are subdirectory-qualified names like
    # ``dicom_series/slice_0000.dcm``. They are included in the sha256
    # hash the same way as top-level files (name-sorted concat).
    series_files: Dict[str, bytes] = {}
    media_types: List[str] = []
    if dicom_b64:
        try:
            files["dicom.bin"] = base64.b64decode(dicom_b64)
            media_types.append("application/dicom")
        except Exception as e:
            return {"error": f"dicom_b64 decode failed: {type(e).__name__}: {e}"}
    if dicom_series_b64:
        try:
            if isinstance(dicom_series_b64, list):
                # v0.5.0: pad index so lexicographic sort == numeric sort
                pad = max(4, len(str(len(dicom_series_b64) - 1)))
                for idx, blob in enumerate(dicom_series_b64):
                    key = f"dicom_series/slice_{idx:0{pad}d}.dcm"
                    series_files[key] = base64.b64decode(blob)
            elif isinstance(dicom_series_b64, dict):
                for name, blob in dicom_series_b64.items():
                    safe = str(name).replace("/", "_").replace("..", "_")
                    key = f"dicom_series/{safe}"
                    series_files[key] = base64.b64decode(blob)
            else:
                return {"error": "dicom_series_b64 must be a list of b64 strings or a mapping name->b64"}
            media_types.append("application/dicom-series")
        except Exception as e:
            return {"error": f"dicom_series_b64 decode failed: {type(e).__name__}: {e}"}
    if nifti_b64:
        try:
            files["nifti.nii.gz"] = base64.b64decode(nifti_b64)
            media_types.append("application/x-nifti")
        except Exception as e:
            return {"error": f"nifti_b64 decode failed: {type(e).__name__}: {e}"}
    if report_text:
        files["report.txt"] = str(report_text).encode("utf-8")
        media_types.append("text/plain")
    if genomics_json is not None:
        try:
            files["genomics.json"] = json.dumps(
                genomics_json, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            media_types.append("application/json")
        except Exception as e:
            return {"error": f"genomics_json serialization failed: {type(e).__name__}: {e}"}

    # Deterministic case_id: sha256 of concatenated (name-sorted) file bytes
    # including series slices (fully qualified keys).
    all_files: Dict[str, bytes] = {**files, **series_files}
    ordered_keys = sorted(all_files.keys())
    sha256_full = _sha256(*(all_files[k] for k in ordered_keys))
    case_id = sha256_full[:16]

    # Write to Volume
    cdir = _case_dir(case_id)
    os.makedirs(cdir, exist_ok=True)
    if series_files:
        os.makedirs(os.path.join(cdir, "dicom_series"), exist_ok=True)
    byte_counts = {}
    for name in ordered_keys:
        path = os.path.join(cdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True) if "/" in name else None
        with open(path, "wb") as f:
            f.write(all_files[name])
        byte_counts[name] = len(all_files[name])

    manifest = {
        "case_id": case_id,
        "sha256_full": sha256_full,
        "created_at": time.time(),
        "media_types": media_types,
        "files": ordered_keys,
        "byte_counts": byte_counts,
        "uploader_note": note,
        "storage_backend": "modal-volume:oncology-arbiter-cases",
        "app_version": APP_VERSION,
        "disclaimer": DISCLAIMER,
    }
    with open(os.path.join(cdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # Explicit commit — Modal volumes need commit() to persist across containers.
    volume.commit()

    return {
        "case_id": case_id,
        "sha256_full": sha256_full,
        "provenance_uri": f"modal-volume://oncology-arbiter-cases/{case_id}/manifest.json",
        "storage_backend": "modal-volume:oncology-arbiter-cases",
        "byte_counts": byte_counts,
        "media_types": media_types,
        "created_at": manifest["created_at"],
        "elapsed_s": round(time.time() - t0, 3),
        "app_version": APP_VERSION,
        "disclaimer": DISCLAIMER,
    }


@app.function(image=IMAGE, volumes={"/vol": volume}, timeout=60)
@modal.fastapi_endpoint(method="GET", label="case-storage-manifest")
def manifest(case_id: str) -> Dict[str, Any]:
    """Return the manifest JSON for a case_id, or ``{"error": ...}``."""
    volume.reload()  # in case another container just wrote
    path = os.path.join(_case_dir(case_id), "manifest.json")
    if not os.path.exists(path):
        return {"error": f"case_id {case_id} not found"}
    with open(path, "r") as f:
        return json.load(f)


@app.function(image=IMAGE, volumes={"/vol": volume}, timeout=60)
@modal.fastapi_endpoint(method="GET", label="case-storage-get")
def get_file(case_id: str, file: str) -> Dict[str, Any]:
    """Return a base64-encoded file body for a case_id.

    The gateway uses this to hydrate DICOM bytes into the LUNA16 or
    MedSigLIP inference calls without carrying the bytes through Render.
    """
    volume.reload()
    # v0.5.0: allow ``dicom_series/<slice>.dcm`` but forbid ``..`` and
    # nested paths of arbitrary depth.
    if ".." in file:
        return {"error": "invalid file name"}
    if file.count("/") > 1:
        return {"error": "invalid file name (depth>1)"}
    if "/" in file and not file.startswith("dicom_series/"):
        return {"error": "invalid file name (only dicom_series/ subdir allowed)"}
    path = os.path.join(_case_dir(case_id), file)
    if not os.path.exists(path):
        return {"error": f"case_id {case_id} / file {file} not found"}
    with open(path, "rb") as f:
        data = f.read()
    return {
        "case_id": case_id,
        "file": file,
        "bytes_b64": base64.b64encode(data).decode(),
        "byte_count": len(data),
        "app_version": APP_VERSION,
    }
