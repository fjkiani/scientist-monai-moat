#!/usr/bin/env python3
"""Live E2E for Strikes 1–4: MedSigLIP + Phikon + MedGemma + PubMed via /v1/case/full.

Auth: ONCOLOGY_ARBITER_AUTH_MODE=off (local). Writes a JSON receipt under artifacts/live_io/.
Converts fixture mammogram PNG → DICOM so screening accepts dicom_bytes_b64.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("ONCOLOGY_ARBITER_AUTH_MODE", "off")
os.environ.setdefault("MEDSIGLIP_BACKEND", "modal")
os.environ.setdefault("MODAL_PROFILE", "crispro")
os.environ.setdefault("MODAL_MEDSIGLIP_URL", "https://crispro--medsiglip")
os.environ.setdefault("ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP", "0")


def png_to_dicom_bytes(png_path: Path) -> bytes:
    img = Image.open(png_path).convert("L")
    arr = np.asarray(img, dtype=np.uint16)
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    ds = FileDataset("mammo.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "MG"
    ds.PatientName = "ANON^RUO"
    ds.PatientID = "patient_sample_01"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    buf = io.BytesIO()
    ds.save_as(buf)
    return buf.getvalue()


def main() -> int:
    from fastapi.testclient import TestClient

    from oncology_arbiter.api.app import create_app
    from oncology_arbiter.models.phikon_probe_wiring import predict_tissue_class

    fixture = ROOT / "fixtures" / "patient_sample_01"
    report = (fixture / "pathology_note.txt").read_text()
    dicom = png_to_dicom_bytes(fixture / "mammogram.png")

    # Real NCT-CRC histology row. Held-out tile phikon-val-TUM-TUM-TCGA-AAHIDTWA.
    sample_id = "phikon-val-TUM-TUM-TCGA-AAHIDTWA"
    samples = json.loads((ROOT / "artifacts/phikon_staging/samples.json").read_text())["samples"]
    row_index = next(i for i, row in enumerate(samples) if row["sample_id"] == sample_id)
    matrix = np.load(ROOT / "artifacts/phikon_staging/embeddings.npy", mmap_mode="r")
    emb = np.asarray(matrix[row_index], dtype=np.float32)
    if emb.shape != (768,) or not np.isfinite(emb).all():
        raise SystemExit(f"phikon embedding contract failed: {emb.shape}")
    emb_sha = hashlib.sha256(emb.tobytes()).hexdigest()
    probe_smoke = predict_tissue_class(emb.tolist())
    if probe_smoke.get("pred_label") != "TUM":
        raise SystemExit(f"real TUM tile did not score TUM: {probe_smoke}")

    client = TestClient(create_app())
    payload = {
        "screening_input": {
            "dicom_bytes_b64": base64.b64encode(dicom).decode("ascii"),
        },
        "biopsy_input": {
            "report_text": report[:6000],
            "phikon_embedding": emb.tolist(),
        },
        "therapy_context": {
            "age": 54,
            "menopausal_status": "post",
        },
        "run_co_scientist": False,
    }

    t0 = time.time()
    resp = client.post("/v1/case/full?cancer=breast", json=payload)
    elapsed = round(time.time() - t0, 3)
    body = (
        resp.json()
        if resp.headers.get("content-type", "").startswith("application/json")
        else {"raw": resp.text[:2000]}
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "artifacts" / "live_io"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped = out_dir / f"e2e_case_full_medgemma_phikon_pubmed_{ts}.json"
    latest = out_dir / "e2e_case_full_medgemma_phikon_pubmed_latest.json"

    receipt = {
        "ts_utc": ts,
        "elapsed_s": elapsed,
        "status_code": resp.status_code,
        "probe_smoke": probe_smoke,
        "inputs": {
            "fixture": str(fixture.relative_to(ROOT)),
            "dicom_bytes": len(dicom),
            "report_chars": len(report[:6000]),
            "phikon_source": "artifacts/phikon_staging/embeddings.npy",
            "phikon_sample_id": sample_id,
            "phikon_row_index": row_index,
            "phikon_embedding_sha256": emb_sha,
            "phikon_embedding_dim": 768,
            "phikon_generator": "real_histology_row",
        },
        "gates": {
            "screening_state": ((body.get("screening") or {}).get("provenance") or {}).get(
                "model_state"
            ),
            "biopsy_state": ((body.get("biopsy") or {}).get("provenance") or {}).get(
                "model_state"
            ),
            "therapy_state": ((body.get("therapy") or {}).get("provenance") or {}).get(
                "model_state"
            ),
            "phikon_probe": ((body.get("biopsy") or {}).get("phikon_embedding") or {}).get(
                "probe"
            ),
            "n_evidence": len(body.get("evidence") or []),
            "therapy_n_evidence": len((body.get("therapy") or {}).get("evidence") or []),
            "recommended": [
                o.get("regimen")
                for o in ((body.get("therapy") or {}).get("recommended_options") or [])
            ],
            "therapy_transport": next(
                (
                    s.get("input_reference")
                    for s in ((body.get("therapy") or {}).get("stage_receipts") or [])
                    if s.get("stage") == "medgemma_therapy_reason"
                ),
                None,
            ),
        },
        "screening_compact": {
            "provenance": (body.get("screening") or {}).get("provenance"),
            "overall_score": (body.get("screening") or {}).get("overall_score"),
            "findings": (body.get("screening") or {}).get("findings"),
            "warnings": (body.get("screening") or {}).get("warnings") or [],
            "stage_receipts": (body.get("screening") or {}).get("stage_receipts"),
        },
        "biopsy_compact": {
            "provenance": (body.get("biopsy") or {}).get("provenance"),
            "subtype": (body.get("biopsy") or {}).get("subtype_prediction"),
            "receptor_panel": (body.get("biopsy") or {}).get("receptor_panel"),
            "probe": ((body.get("biopsy") or {}).get("phikon_embedding") or {}).get("probe"),
        },
        "therapy_compact": {
            "provenance": (body.get("therapy") or {}).get("provenance"),
            "recommended_options": (body.get("therapy") or {}).get("recommended_options"),
            "evidence": (body.get("therapy") or {}).get("evidence"),
            "stage_receipts": (body.get("therapy") or {}).get("stage_receipts"),
        },
    }
    if resp.status_code != 200:
        receipt["error_body"] = body
    stamped.write_text(json.dumps(receipt, indent=2)[:2_000_000])
    latest.write_text(stamped.read_text())
    print(json.dumps({"receipt": str(stamped), "gates": receipt["gates"], "elapsed_s": elapsed, "status": resp.status_code}, indent=2))

    screen_blob = json.dumps(receipt["screening_compact"])
    therapy_blob = json.dumps(receipt["therapy_compact"]).lower()
    regimens = " ".join(
        f"{o.get('regimen','')} {o.get('rationale','')}"
        for o in (receipt["therapy_compact"].get("recommended_options") or [])
    ).lower()
    evidence_blob = json.dumps(receipt["therapy_compact"].get("evidence") or []).lower()
    cancer_score = next(
        (
            f.get("score")
            for f in (receipt["screening_compact"].get("findings") or [])
            if str(f.get("label") or "").startswith("cbis_ddsm_logreg_v2")
        ),
        None,
    )
    ok = (
        resp.status_code == 200
        and receipt["gates"]["biopsy_state"] == "loaded_phikon"
        and receipt["gates"]["therapy_state"] == "loaded_medgemma_1_5_4b_therapy"
        and receipt["gates"]["therapy_n_evidence"] >= 1
        and (receipt["gates"]["phikon_probe"] or {}).get("pred_label") == "TUM"
        and "standard_normal" not in json.dumps(receipt)
        and "gaussian" not in json.dumps(receipt).lower()
        and "StandardScaler" not in screen_blob
        and "1309" not in screen_blob
        and cancer_score is not None
        and ("trastuzumab" in regimens or "herceptin" in regimens)
        and ("trastuzumab" in evidence_blob or "herceptin" in evidence_blob)
        and "nab-paclitaxel breast cancer therapy" not in therapy_blob
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
