"""Production screening accepts only strict MedSigLIP-448 inference."""
from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.api.schemas import ModelState


class _Preprocessed:
    image = np.zeros((8, 8), dtype=np.float32)
    breast_mask = np.ones((8, 8), dtype=bool)
    metadata = SimpleNamespace(
        laterality=SimpleNamespace(value="L"),
        view=SimpleNamespace(value="CC"),
        orientation_flipped=False,
    )


def _client(monkeypatch, modal_client) -> TestClient:
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    monkeypatch.setattr("oncology_arbiter.mammography.preprocess_mammogram", lambda *args, **kwargs: _Preprocessed())
    monkeypatch.setattr("oncology_arbiter.models.medsiglip_modal_client.MedSigLipModalClient", modal_client)
    return TestClient(create_app())


def _payload() -> dict[str, str]:
    return {"dicom_bytes_b64": base64.b64encode(b"deidentified-dicom").decode("ascii")}


def test_success_requires_1152_dimensional_production_receipt(monkeypatch) -> None:
    class ModalClient:
        def run(self, path):
            return SimpleNamespace(
                labels=["malignant", "without malignancy"],
                probs=[0.2, 0.8],
                warnings=["off-label"],
                model_repo="google/medsiglip-448",
                app_version="medsiglip-modal-v0.4.0-alpha",
                input_resolution=448,
                embedding_dim=1152,
                embedding_sha256="a" * 64,
                prompts=["malignant", "without malignancy"],
                inference_seconds=1.25,
                gate_report=None,
            )

    response = _client(monkeypatch, ModalClient).post("/v1/screening/analyze", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_status"] == "complete"
    assert body["provenance"]["model_state"] == ModelState.LOADED_MEDSIGLIP.value
    assert body["medsiglip"]["embedding_dim"] == 1152
    assert body["medsiglip"]["embedding_sha256"] == "a" * 64
    assert body["stage_receipts"][0]["status"] == "succeeded"
    assert body["arbiter_score"] is None


def test_failure_is_terminal_without_proxy_or_heuristic(monkeypatch) -> None:
    class ModalClient:
        def run(self, path):
            raise TimeoutError("remote inference timed out")

    response = _client(monkeypatch, ModalClient).post("/v1/screening/analyze", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_status"] == "failed_required_stage"
    assert body["stage_receipts"][0]["status"] == "failed_required"
    assert body["findings"] == []
    assert body["overall_score"] is None
    assert body["medsiglip"] is None
    assert body["arbiter_score"] is None
    assert body["provenance"]["model_state"] == ModelState.UNAVAILABLE.value


def test_url_ingestion_is_not_silently_substituted(monkeypatch) -> None:
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    response = TestClient(create_app()).post(
        "/v1/screening/analyze",
        json={"dicom_url": "https://example.org/study.dcm"},
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# Retired-substitute contract. Folded in from the deleted proxy-era file
# tests/unit/test_screening_medsiglip_wiring.py, inverted: instead of pinning
# that a proxy state exists, these pin that no substitute can be emitted.
# --------------------------------------------------------------------------- #


def test_no_retired_substitute_state_is_constructible() -> None:
    from oncology_arbiter.api.schemas import RETIRED_MODEL_STATE_VALUES

    live = {m.value for m in ModelState}
    assert live.isdisjoint(RETIRED_MODEL_STATE_VALUES), live & RETIRED_MODEL_STATE_VALUES
    assert ModelState.LOADED_MEDSIGLIP.value == "loaded_medsiglip"
    for other in (ModelState.PLACEHOLDER, ModelState.GATED, ModelState.UNAVAILABLE):
        assert ModelState.LOADED_MEDSIGLIP.value != other.value


def test_app_module_imports_no_substitute_backend() -> None:
    """app.py must not import any retired stand-in for a required specialist."""
    import ast
    import importlib
    from pathlib import Path

    app_src = Path(importlib.import_module("oncology_arbiter.api.app").__file__)
    tree = ast.parse(app_src.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    banned = {
        "oncology_arbiter.models.siglip_baseline",   # general-domain SigLIP proxy
        "oncology_arbiter.nlp.report_parser_v2",     # regex / regex+BERT fusion parser
        "oncology_arbiter.models.therapy_rules_lite",  # NCCN-lite static rules
        "oncology_arbiter.models.nccn_nsclc_rules",    # NCCN-lite static rules (NSCLC)
        "oncology_arbiter.models.txgemma_client",      # gated TxGemma therapy path
        "oncology_arbiter.lung.pipeline",              # HU-threshold nodule heuristic
        "oncology_arbiter.lung",
    }
    assert not (imported & banned), sorted(imported & banned)
    text = app_src.read_text(encoding="utf-8")
    for token in ("_get_siglip_proxy", "_run_siglip_proxy_on_preprocessed",
                  "SiglipBaseline", "NsclcCTInput"):
        assert token not in text, token


def test_zero_shot_probs_are_reported_as_independent_sigmoid_scores(monkeypatch) -> None:
    """SigLIP scores are pairwise-sigmoid, so they need not sum to 1.

    The live MedSigLIP receipt returned probs=[9.475e-06, 8.807e-06]
    (sum 1.83e-05). Presenting that as a normalised distribution -- or as a
    calibrated malignancy probability -- would be a false claim, so the receipt
    must carry the sum and the explicit non-normalisation flag.
    """
    class ModalClient:
        def run(self, path):
            return SimpleNamespace(
                labels=["malignant", "without malignancy"],
                probs=[9.475328624830581e-06, 8.806667210592423e-06],
                model_repo="google/medsiglip-448",
                app_version="medsiglip-modal-v0.4.0-alpha",
                input_resolution=448,
                embedding_dim=1152,
                embedding_sha256="c" * 64,
                prompts=["a mammogram showing malignancy", "a mammogram without malignancy"],
                inference_seconds=0.25,
                warnings=[],
                gate_report=None,
            )

    client = _client(monkeypatch, ModalClient)
    body = client.post("/v1/screening/analyze", json=_payload()).json()
    block = body["medsiglip"]
    assert block["score_semantics"] == "independent_uncalibrated_sigmoid_zero_shot"
    assert block["probs_are_normalised_distribution"] is False
    assert block["probs_sum"] == pytest.approx(1.8281995835423004e-05, rel=1e-9)
    assert block["probs_sum"] < 0.5  # emphatically not a distribution
    assert body["overall_score"] == pytest.approx(9.475328624830581e-06, rel=1e-12)
    assert body["arbiter_score"] is None
    assert any("uncalibrated" in w.lower() for w in body["warnings"]), body["warnings"]
