"""API-level tests for L4a MONAI detector wiring into /v1/screening/analyze.

The MonaiDetector runs on the real (mocked here) preprocess-mammogram
output. To avoid needing a real DICOM fixture, we monkey-patch the
screening endpoint's preprocess call to return a controllable
PreprocessResult carrying a fixed image + breast_mask.
"""
from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.api.schemas import ModelState


# --------------------------------------------------------------------------- #
# Fake preprocess result
# --------------------------------------------------------------------------- #


def _make_fake_preprocess_result(seed: int = 42) -> Any:
    """Build a namespace matching PreprocessResult's duck-type surface."""
    rng = np.random.default_rng(seed=seed)
    H, W = 256, 256
    image = rng.random((H, W)).astype(np.float32) * 0.3
    image[100:130, 100:130] += 0.5  # inject bright square (lesion hint)
    breast_mask = np.zeros((H, W), dtype=bool)
    breast_mask[50:250, 30:200] = True

    metadata = SimpleNamespace(
        laterality=SimpleNamespace(value="L"),
        view=SimpleNamespace(value="CC"),
        orientation_flipped=False,
    )
    return SimpleNamespace(
        image=image,
        breast_mask=breast_mask,
        metadata=metadata,
    )


@pytest.fixture
def screening_client(monkeypatch) -> TestClient:
    """FastAPI TestClient with a monkey-patched preprocessor + flags reset."""
    for flag in (
        "ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP",
        "ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY",
        "ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR",
    ):
        monkeypatch.delenv(flag, raising=False)

    import oncology_arbiter.mammography as mm
    from oncology_arbiter.mammography import pipeline as pl

    def _fake_pipeline(*args, **kwargs) -> Any:
        return _make_fake_preprocess_result()

    monkeypatch.setattr(pl, "preprocess_mammogram", _fake_pipeline)
    monkeypatch.setattr(mm, "preprocess_mammogram", _fake_pipeline)
    return TestClient(create_app())


def _post_screening(client: TestClient) -> Any:
    return client.post(
        "/v1/screening/analyze",
        json={"dicom_bytes_b64": base64.b64encode(b"fake-dicom").decode()},
    )


# --------------------------------------------------------------------------- #
# 1. MONAI detector OFF by default → no monai_heuristic findings
# --------------------------------------------------------------------------- #


def test_monai_detector_off_by_default(screening_client: TestClient) -> None:
    resp = _post_screening(screening_client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    monai_findings = [f for f in body["findings"] if f["label"].startswith("monai_heuristic:")]
    assert monai_findings == [], "MONAI findings should be empty when flag off"


# --------------------------------------------------------------------------- #
# 2. MONAI detector ON → produces bbox findings
# --------------------------------------------------------------------------- #


def test_monai_detector_on_produces_bbox_findings(
    monkeypatch, screening_client: TestClient
) -> None:
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR", "1")
    resp = _post_screening(screening_client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    monai_findings = [f for f in body["findings"] if f["label"].startswith("monai_heuristic:")]
    assert len(monai_findings) >= 1, (
        f"expected at least one monai heuristic finding; got findings={body['findings']}"
    )
    for f in monai_findings:
        bbox = f["location_bbox_normalized"]
        assert bbox is not None
        assert len(bbox) == 4
        x0, y0, x1, y1 = bbox
        assert 0.0 <= x0 < x1 <= 1.0
        assert 0.0 <= y0 < y1 <= 1.0
        assert 0.0 <= f["score"] <= 1.0


# --------------------------------------------------------------------------- #
# 3. MONAI detector ON + no other backend → provenance shows proxy_monai_heuristic
# --------------------------------------------------------------------------- #


def test_monai_detector_promotes_placeholder_to_proxy(
    monkeypatch, screening_client: TestClient
) -> None:
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR", "1")
    # MedSigLIP + SigLIP-proxy both off → placeholder path, then MONAI upgrades
    resp = _post_screening(screening_client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["provenance"]["model_state"] == "proxy_monai_heuristic"


# --------------------------------------------------------------------------- #
# 4. MONAI honesty warning surfaced on the response envelope
# --------------------------------------------------------------------------- #


def test_monai_honesty_warning_surfaced(
    monkeypatch, screening_client: TestClient
) -> None:
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR", "1")
    resp = _post_screening(screening_client)
    body = resp.json()
    assert any(
        "heuristic" in w.lower() and "no trained weights" in w.lower()
        for w in body["warnings"]
    ), body["warnings"]
