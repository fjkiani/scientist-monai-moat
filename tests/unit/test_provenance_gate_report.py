"""Structured `provenance.gate_report` field on API responses.

Contract:
  * Placeholder responses have `provenance.gate_report is None` (no preflight ran).
  * When a HAI-DEF preflight runs and gets denied, the endpoint returns
    `model_state=gated` AND `provenance.gate_report` is populated with the
    exact repo_id + access_level + status_code + reason + has_token that the
    runtime `hai_def.GateReport` carried — no field drop, no re-labeling.
  * When a preflight succeeds, `provenance.gate_report.access_level == "allowed"`
    AND `provenance.gate_report.allowed is True`.
  * The pydantic converter refuses invalid access_level literals so a bad
    upstream can't silently smuggle a fake access level through the schema.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import _to_schema_gate_report, create_app
from oncology_arbiter.api.schemas import GateReport as SchemaGateReport
from oncology_arbiter.models.hai_def import (
    AccessLevel,
    GatedAccessError,
    GateReport as RuntimeGateReport,
)


def _cbis_dicom_b64() -> str:
    """Load the shipped CBIS-DDSM test fixture as base64 for POSTing."""
    p = Path(__file__).resolve().parents[1] / "fixtures" / "cbis_ddsm" / "Calc-Test_P_00038_LEFT_CC.dcm"
    if not p.is_file():
        pytest.skip(f"CBIS-DDSM fixture missing at {p}")
    return base64.b64encode(p.read_bytes()).decode("ascii")


# --------------------------------------------------------------------------- #
# _to_schema_gate_report — the runtime→schema converter

def test_to_schema_gate_report_allowed():
    runtime = RuntimeGateReport(
        repo_id="google/medsiglip-448",
        access_level=AccessLevel.ALLOWED,
        status_code=200,
        reason="preflight ok",
        has_token=True,
    )
    schema = _to_schema_gate_report(runtime)
    assert isinstance(schema, SchemaGateReport)
    assert schema.repo_id == "google/medsiglip-448"
    assert schema.access_level == "allowed"
    assert schema.status_code == 200
    assert schema.reason == "preflight ok"
    assert schema.has_token is True
    assert schema.allowed is True


def test_to_schema_gate_report_forbidden():
    runtime = RuntimeGateReport(
        repo_id="google/medsiglip-448",
        access_level=AccessLevel.FORBIDDEN,
        status_code=403,
        reason="accept terms at huggingface.co/google/medsiglip-448",
        has_token=True,
    )
    schema = _to_schema_gate_report(runtime)
    assert schema.access_level == "forbidden"
    assert schema.status_code == 403
    assert schema.has_token is True
    assert schema.allowed is False  # not ALLOWED → allowed=False


def test_to_schema_gate_report_unauthenticated():
    runtime = RuntimeGateReport(
        repo_id="google/medsiglip-448",
        access_level=AccessLevel.UNAUTHENTICATED,
        status_code=401,
        reason="no HF token discovered",
        has_token=False,
    )
    schema = _to_schema_gate_report(runtime)
    assert schema.access_level == "unauthenticated"
    assert schema.status_code == 401
    assert schema.has_token is False
    assert schema.allowed is False


def test_to_schema_gate_report_none_returns_none():
    assert _to_schema_gate_report(None) is None


# --------------------------------------------------------------------------- #
# Schema-level validation — the pydantic literal guard

def test_schema_gate_report_rejects_unknown_access_level():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SchemaGateReport(
            repo_id="google/medsiglip-448",
            access_level="something-made-up",  # not in the Literal
            status_code=200,
            reason="bogus",
            has_token=True,
            allowed=True,
        )


# --------------------------------------------------------------------------- #
# End-to-end: placeholder response has gate_report=None

def test_placeholder_response_has_no_gate_report(monkeypatch):
    # No HAI-DEF preflight is triggered on the therapy placeholder path.
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP", raising=False)
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY", raising=False)
    with TestClient(create_app()) as client:
        # /v1/therapy/reason is a pure placeholder today — no preflight, so
        # provenance.gate_report MUST be None.
        resp = client.post("/v1/therapy/reason", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["provenance"]["model_state"] == "placeholder"
        assert body["provenance"]["gate_report"] is None


# --------------------------------------------------------------------------- #
# End-to-end: gated screening response carries structured gate_report

def test_gated_screening_response_has_structured_gate_report(monkeypatch):
    """Simulate a HAI-DEF preflight denial on /v1/screening/analyze and
    confirm the response envelope carries:
      * provenance.model_state == "gated"
      * provenance.gate_report != None
      * provenance.gate_report matches the raised GatedAccessError shape
    """
    # Enable the MedSigLIP arm so app.py takes the gated branch.
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP", "1")
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY", raising=False)

    def _raise_gate(_preprocess_result):
        raise GatedAccessError(
            repo_id="google/medsiglip-448",
            access_level=AccessLevel.FORBIDDEN,
            status_code=403,
            reason="accept terms at huggingface.co/google/medsiglip-448",
        )

    # Import the module and patch the runner in-place. The endpoint calls
    # this name-locally so a monkeypatch of the module attribute is enough.
    import oncology_arbiter.api.app as app_module
    monkeypatch.setattr(app_module, "_run_medsiglip_on_preprocessed", _raise_gate)

    dicom_b64 = _cbis_dicom_b64()
    with TestClient(create_app()) as client:
        resp = client.post("/v1/screening/analyze", json={"dicom_bytes_b64": dicom_b64})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"]["model_state"] == "gated"
    gr = body["provenance"]["gate_report"]
    assert gr is not None, "provenance.gate_report must be populated on gated responses"
    assert gr["repo_id"] == "google/medsiglip-448"
    assert gr["access_level"] == "forbidden"
    assert gr["status_code"] == 403
    assert "accept terms" in gr["reason"]
    # has_token depends on the test env — assert only that the field exists
    # and is a bool so nobody can smuggle a None or a string through.
    assert isinstance(gr["has_token"], bool)
    assert gr["allowed"] is False


def test_allowed_screening_response_has_allowed_gate_report(monkeypatch):
    """When _run_medsiglip_on_preprocessed returns a normal result with a
    populated runtime GateReport, the wire gate_report must carry
    access_level='allowed'."""
    from types import SimpleNamespace
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP", "1")
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY", raising=False)

    fake_result = SimpleNamespace(
        model_repo="google/medsiglip-448",
        warnings=[],
        gate_report=RuntimeGateReport(
            repo_id="google/medsiglip-448",
            access_level=AccessLevel.ALLOWED,
            status_code=200,
            reason="preflight ok",
            has_token=True,
        ),
        probs=[0.12, 0.88],
        labels=["malignant lesion", "no lesion"],
    )

    def _return_ok(_preprocess_result):
        return fake_result

    import oncology_arbiter.api.app as app_module
    monkeypatch.setattr(app_module, "_run_medsiglip_on_preprocessed", _return_ok)

    dicom_b64 = _cbis_dicom_b64()
    with TestClient(create_app()) as client:
        resp = client.post("/v1/screening/analyze", json={"dicom_bytes_b64": dicom_b64})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provenance"]["model_state"] == "loaded_medsiglip"
    gr = body["provenance"]["gate_report"]
    assert gr is not None
    assert gr["access_level"] == "allowed"
    assert gr["allowed"] is True
    assert gr["repo_id"] == "google/medsiglip-448"
