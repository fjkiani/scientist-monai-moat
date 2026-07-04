"""
API-level tests for the NSCLC branch of ``POST /v1/case/full?cancer=nsclc``.

These use ``fastapi.testclient.TestClient`` so we exercise the same middleware
stack (auth, CORS, rate-limit, prometheus) plus the schema envelope. Two
paths are covered:

1. Placeholder / shape-only branch — no ``nsclc_ct_input`` OR the env gate
   ``ONCOLOGY_ARBITER_ALLOW_SERIES_DIR`` is not truthy. Response MUST come
   back with ``provenance.model_state == "placeholder"``, an ``nsclc``
   envelope populated with a placeholder model_state, and at least one
   warning that explains what the caller needs to flip.
2. Real proxy pipeline branch — synthetic on-disk CT series (built at test
   time with pydicom) + ``ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1``. Response
   MUST come back with ``model_state == "proxy_lung_heuristic"``, a real
   ``risk_bucket``, at least one candidate blob, and at least one therapy
   recommendation.

We build the DICOM stack in a tmp dir; no LIDC-IDRI cohort is required so
CI can run this without a 128 GB download.
"""

from __future__ import annotations

import os

import numpy as np
import pydicom
import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


# --------------------------------------------------------------------------- #
# helpers


def _write_synthetic_ct_series(series_dir, n_slices: int = 8, shape=(64, 64)) -> None:
    """Write ``n_slices`` DICOM CT slices under ``series_dir``.

    The slices carry a solid nodule in the middle so the heuristic will
    actually find something. Values are picked so ``rescale_slope * pixel +
    rescale_intercept`` lands in the Hounsfield ranges the pipeline expects:
    body silhouette ~ -50 HU, lung ~ -800 HU, nodule ~ +50 HU, gantry air
    ~ -1200 HU.
    """
    series_dir.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    for i in range(n_slices):
        vol_slice = np.full(shape, -1200, dtype=np.int16)  # gantry air
        vol_slice[10:-10, 10:-10] = -50  # body silhouette
        vol_slice[15:-15, 15:-15] = -800  # lung
        # planted nodule slabbed through slices 3..5
        if 3 <= i <= 5:
            cx = shape[0] // 2
            cy = shape[1] // 2
            vol_slice[cx - 3:cx + 4, cy - 3:cy + 4] = 50

        ds = Dataset()
        ds.PatientID = "SYN-0001"
        ds.PatientName = "SYN^0001"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = generate_uid()
        ds.Modality = "CT"
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [0.0, 0.0, float(i) * 2.5]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 2.5
        ds.Rows, ds.Columns = shape
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope = 1
        ds.RescaleIntercept = 0
        ds.PixelData = vol_slice.tobytes()

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        pydicom.dcmwrite(str(series_dir / f"slice_{i:03d}.dcm"), ds, enforce_file_format=True)


@pytest.fixture(scope="module")
def synthetic_ct_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("case_full_nsclc")
    series_dir = root / "SYN-0001" / "STUDY" / "CT_SYN"
    _write_synthetic_ct_series(series_dir)
    return series_dir


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    from oncology_arbiter.api.app import create_app
    return TestClient(create_app())


# --------------------------------------------------------------------------- #
# tests


def test_case_full_nsclc_placeholder_when_no_ct_input(client):
    r = client.post("/v1/case/full?cancer=nsclc", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provenance"]["model_state"] == "placeholder"
    assert body["nsclc"] is not None
    assert body["nsclc"]["model_state"] == "placeholder"
    assert body["nsclc"]["risk_score"] is None
    assert any("shape only" in w for w in body["warnings"])


def test_case_full_nsclc_placeholder_when_gate_off(client, monkeypatch, synthetic_ct_dir):
    # Force gate off even though the caller provided a series_dir.
    monkeypatch.delenv("ONCOLOGY_ARBITER_ALLOW_SERIES_DIR", raising=False)
    r = client.post(
        "/v1/case/full?cancer=nsclc",
        json={"nsclc_ct_input": {"series_dir": str(synthetic_ct_dir), "top_n": 5}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["provenance"]["model_state"] == "placeholder"
    # Placeholder must ALSO warn about the ignored series_dir.
    assert any("ONCOLOGY_ARBITER_ALLOW_SERIES_DIR" in w for w in body["warnings"])


def test_case_full_nsclc_real_pipeline_on_synthetic_ct(client, monkeypatch, synthetic_ct_dir):
    monkeypatch.setenv("ONCOLOGY_ARBITER_ALLOW_SERIES_DIR", "1")
    r = client.post(
        "/v1/case/full?cancer=nsclc",
        json={"nsclc_ct_input": {"series_dir": str(synthetic_ct_dir), "top_n": 5}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provenance"]["model_state"] == "proxy_lung_heuristic"
    assert body["provenance"]["model_name"] == "nsclc_lung_heuristic_v0+nccn_nsclc_lite_v0"
    n = body["nsclc"]
    assert n is not None
    assert n["model_state"] == "proxy_lung_heuristic"
    # Pipeline stats must be present and non-trivial
    assert n["n_slices"] == 8
    assert n["read_seconds"] is not None and n["read_seconds"] >= 0
    assert n["heuristic_seconds"] is not None and n["heuristic_seconds"] >= 0
    assert n["lung_voxel_fraction"] is not None and n["lung_voxel_fraction"] > 0
    assert n["max_diameter_mm"] is not None and n["max_diameter_mm"] > 0
    assert n["risk_bucket"] in {"NEGATIVE", "LOW", "MID", "HIGH"}
    assert n["driving_feature"] is not None
    # NCCN-lite must return SOMETHING — even NEGATIVE has recommendations.
    assert len(n["therapy_recommended"]) > 0
    for opt in n["therapy_recommended"]:
        assert opt["citation_url"].startswith("https://")
        assert opt["nccn_section"]
    # Honesty warning must mention proxy + rules-lite.
    assert any("PROXY" in w for w in n["warnings"])


def test_case_full_nsclc_real_pipeline_400_on_missing_dir(client, monkeypatch):
    monkeypatch.setenv("ONCOLOGY_ARBITER_ALLOW_SERIES_DIR", "1")
    r = client.post(
        "/v1/case/full?cancer=nsclc",
        json={"nsclc_ct_input": {"series_dir": "/no/such/path", "top_n": 5}},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "series_dir" in detail or "failed to read" in detail
