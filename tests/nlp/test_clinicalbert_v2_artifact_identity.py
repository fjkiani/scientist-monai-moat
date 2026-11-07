"""Unique identity and sliding-window contract tests for ClinicalBERT v2."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from oncology_arbiter.nlp import clinicalbert_modal_client as modal_client
from oncology_arbiter.nlp.clinicalbert_runtime_v2 import (
    ARTIFACT_FILENAME,
    BASE_REVISION,
    EXPECTED_ARTIFACT_SHA256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "clinicalbert" / ARTIFACT_FILENAME
CONFIG_PATH = REPO_ROOT / "artifacts" / "clinicalbert" / "clinicalbert_head_v2_config.json"
RUNTIME_PATH = REPO_ROOT / "src" / "oncology_arbiter" / "nlp" / "clinicalbert_runtime_v2.py"
MODAL_APP_PATH = REPO_ROOT / "deploy" / "modal" / "clinicalbert_app.py"
EXPECTED_SHA = "429f804d7f348d7c4eeb27821f766cc2de65c4072f20db0c3b3afaefa9068e50"
EXPECTED_BASE_REVISION = "d5892b39a4adaed74b92212a44081509db72f87b"


def test_clinicalbert_v2_artifact_is_exact_two_tensor_head() -> None:
    assert ARTIFACT_PATH.is_file()
    assert ARTIFACT_PATH.stat().st_size > 100_000
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == EXPECTED_SHA
    assert EXPECTED_ARTIFACT_SHA256 == EXPECTED_SHA
    with ARTIFACT_PATH.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    tensor_names = {name for name in header if name != "__metadata__"}
    assert tensor_names == {"classifier.weight", "classifier.bias"}
    assert header["classifier.weight"]["shape"] == [39, 768]
    assert header["classifier.bias"]["shape"] == [39]
    assert header["__metadata__"]["non_o_margin_threshold"] == "0.25"


def test_clinicalbert_v2_config_and_wiring_bind_exact_identity() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    assert config["artifact_sha256"] == EXPECTED_SHA
    assert config["base_revision"] == EXPECTED_BASE_REVISION == BASE_REVISION
    assert config["n_training_synthetic"] is False
    assert config["max_length"] == 512
    assert config["stride"] == 128
    assert config["non_o_margin_threshold"] == 0.25
    runtime_source = RUNTIME_PATH.read_text()
    assert EXPECTED_SHA in runtime_source
    assert EXPECTED_BASE_REVISION in runtime_source
    assert ARTIFACT_FILENAME in runtime_source
    assert "return_overflowing_tokens=True" in runtime_source
    assert "window_count" in runtime_source
    assert "report_sha256" in runtime_source
    modal_source = MODAL_APP_PATH.read_text()
    assert EXPECTED_SHA in modal_source
    assert EXPECTED_BASE_REVISION in modal_source
    assert ARTIFACT_FILENAME in modal_source
    assert "ClinicalBertV2Runtime" in modal_source


def test_modal_product_client_rejects_identity_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    report = "A " * 700
    good = {
        "artifact_sha256": EXPECTED_SHA,
        "base_revision": EXPECTED_BASE_REVISION,
        "n_training_synthetic": False,
        "report_sha256": hashlib.sha256(report.encode()).hexdigest(),
        "window_count": 3,
        "spans": [],
        "parsed": {},
    }
    monkeypatch.setattr(modal_client, "_post_json", lambda *args, **kwargs: dict(good))
    client = modal_client.ClinicalBertModalClient(
        endpoints=modal_client.ClinicalBertModalEndpointConfig("https://example--clinicalbert")
    )
    response = client.parse(report)
    assert response["artifact_sha256"] == EXPECTED_SHA
    assert response["window_count"] == 3

    drifted = dict(good, artifact_sha256="0" * 64)
    monkeypatch.setattr(modal_client, "_post_json", lambda *args, **kwargs: drifted)
    with pytest.raises(modal_client.ClinicalBertModalError, match="artifact identity"):
        client.parse(report)
