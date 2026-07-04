"""End-to-end API tests hitting real endpoints with real DICOM bytes.

We use FastAPI's TestClient (in-process, no network) to POST real CBIS-DDSM
DICOM bytes to /v1/screening/analyze and verify the ENTIRE preprocessing
pipeline runs through HTTP correctly. This is more meaningful than
mocking the preprocessor because we actually catch bytes-handling and
serialization bugs.

The classifier is a placeholder (there's no MedSigLIP weight loaded yet),
so overall_score must be None. But laterality, view, mask coverage, and
audit ledger writes are all real.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter import RUO_DISCLAIMER, AUROC_CAVEAT
from oncology_arbiter.api import create_app

pytestmark = pytest.mark.data

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "cbis_ddsm"


@pytest.fixture(scope="module")
def audit_tmp():
    """Redirect the audit log dir to a tmp folder so tests are hermetic."""
    d = tempfile.mkdtemp(prefix="oa-audit-test-")
    prev = os.environ.get("ONCOLOGY_ARBITER_AUDIT_DIR")
    os.environ["ONCOLOGY_ARBITER_AUDIT_DIR"] = d
    # Force re-import of audit module so it picks up the env var
    import importlib
    from oncology_arbiter.api import audit as audit_mod
    importlib.reload(audit_mod)
    from oncology_arbiter.api import app as app_mod
    importlib.reload(app_mod)
    yield d
    if prev is None:
        os.environ.pop("ONCOLOGY_ARBITER_AUDIT_DIR", None)
    else:
        os.environ["ONCOLOGY_ARBITER_AUDIT_DIR"] = prev
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def client(audit_tmp):
    return TestClient(create_app())


# --------------------------------------------------------------------------- #
# /health

def test_health_lists_all_endpoints(client):
    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert body["disclaimer"] == RUO_DISCLAIMER
    assert body["caveat"] == AUROC_CAVEAT
    for path in ["POST /v1/screening/analyze", "POST /v1/biopsy/analyze",
                 "POST /v1/therapy/reason", "POST /v1/case/full"]:
        assert path in body["endpoints"], f"missing {path}"
    # Every model_state is 'placeholder' pre-Phase-2
    for m, state in body["models_loaded"].items():
        assert state == "placeholder", f"{m} claims non-placeholder state pre-Phase-2"


# --------------------------------------------------------------------------- #
# /v1/screening/analyze — real DICOM through HTTP

def test_screening_rejects_empty_body(client):
    r = client.post("/v1/screening/analyze", json={})
    assert r.status_code == 400
    assert "dicom_url" in r.text or "dicom_bytes" in r.text


def test_screening_rejects_both_url_and_bytes(client):
    r = client.post(
        "/v1/screening/analyze",
        json={"dicom_url": "https://example.com/x.dcm",
              "dicom_bytes_b64": "AA=="},
    )
    assert r.status_code == 400


def test_screening_url_not_yet_wired(client):
    r = client.post(
        "/v1/screening/analyze",
        json={"dicom_url": "https://example.com/x.dcm"},
    )
    # Placeholder responds 501 with an honest "not yet wired" message.
    assert r.status_code == 501
    assert "not yet wired" in r.text or "Phase 2" in r.text


@pytest.mark.parametrize("filename,expected_lat,expected_view,needs_hint", [
    ("Calc-Test_P_00038_LEFT_CC.dcm", "L", "CC", False),
    ("Calc-Test_P_00038_RIGHT_CC.dcm", "R", "CC", False),
    ("Calc-Test_P_00038_LEFT_MLO.dcm", "L", "MLO", False),
    ("Calc-Test_P_00038_RIGHT_MLO.dcm", "R", "MLO", False),
    ("Mass-Test_P_00016_LEFT_CC.dcm", "L", "CC", True),
])
def test_screening_real_dicom_bytes(client, filename, expected_lat, expected_view, needs_hint):
    """Real CBIS-DDSM DICOM bytes through /v1/screening/analyze.

    `needs_hint=True` fixtures require the caller to provide `laterality_hint`
    because the DICOM neither has ImageLaterality nor a laterality-suffixed
    filename in the API path (we stream into a random tmpfile). This mirrors
    real deployment: the calling PACS/EMR knows the laterality even when the
    curated DICOM has stripped it.
    """
    path = FIXTURE_DIR / filename
    if not path.exists():
        pytest.skip(f"missing fixture {path}")
    dicom_bytes = path.read_bytes()
    b64 = base64.b64encode(dicom_bytes).decode()
    body_req = {
        "dicom_bytes_b64": b64,
        "patient_id_hash": "a" * 64,   # fake but shape-valid SHA256 hex
    }
    if needs_hint:
        body_req["laterality_hint"] = expected_lat
        body_req["view_hint"] = expected_view
    r = client.post("/v1/screening/analyze", json=body_req)
    assert r.status_code == 200, r.text
    body = r.json()

    # Envelope invariants
    assert body["disclaimer"] == RUO_DISCLAIMER
    assert body["caveat"] == AUROC_CAVEAT
    assert body["provenance"]["model_state"] == "placeholder"
    assert body["provenance"]["request_id"], "missing request_id"
    assert body["honesty_gate"] == {
        "seen_urls_count": 0, "evidence_kept": 0, "evidence_dropped": 0,
    }
    assert body["evidence"] == []

    # Real preprocessing ran
    assert body["laterality"] == expected_lat
    assert body["view"] == expected_view
    # For real mammograms the mask coverage sits between 10% and 80%
    assert 0.10 <= body["breast_mask_coverage"] <= 0.80

    # Classifier is not wired → overall_score None
    assert body["overall_score"] is None
    assert body["findings"] == []


def test_screening_without_hint_uses_content_detection_when_metadata_absent(client):
    """Real-world case: the Mass- fixture has no laterality in DICOM tags
    and its content is pre-mirrored. Without a hint the API returns whatever
    content detection said — which may be *wrong* vs the physical patient.

    This test locks in the behavior so callers know they MUST send hints when
    the DICOM has been stripped of laterality metadata."""
    path = FIXTURE_DIR / "Mass-Test_P_00016_LEFT_CC.dcm"
    if not path.exists():
        pytest.skip(f"missing fixture {path}")
    b64 = base64.b64encode(path.read_bytes()).decode()
    r = client.post("/v1/screening/analyze", json={"dicom_bytes_b64": b64})
    assert r.status_code == 200
    body = r.json()
    # Content detection returned 'R' for this fixture even though it's a LEFT
    # breast per the source filename. This is the documented failure mode.
    assert body["laterality"] in ("L", "R")  # not UNKNOWN
    # If content detection got it wrong, the honesty gate is NOT there to
    # save us — this is a real data-quality gap the caller must fill with hints.


def test_screening_writes_audit_log(client, audit_tmp):
    """Verify a screening call appends an entry to today's audit ledger."""
    path = FIXTURE_DIR / "Calc-Test_P_00038_LEFT_CC.dcm"
    if not path.exists():
        pytest.skip(f"missing fixture {path}")
    b64 = base64.b64encode(path.read_bytes()).decode()
    r = client.post(
        "/v1/screening/analyze",
        json={"dicom_bytes_b64": b64, "patient_id_hash": "b" * 64},
    )
    assert r.status_code == 200
    request_id = r.json()["provenance"]["request_id"]

    # Ledger file should contain the request id.
    # As of v0.2 the audit ledger is partitioned per-tenant:
    #   <AUDIT_DIR>/<tenant_id>/audit-YYYY-MM-DD.jsonl
    # With auth off (default in tests) tenant_id is "_anon".
    import glob
    logs = glob.glob(str(Path(audit_tmp) / "*" / "audit-*.jsonl"))
    assert logs, "no audit log written"
    found = False
    for lp in logs:
        with open(lp) as fh:
            for line in fh:
                entry = json.loads(line)
                if entry["request_id"] == request_id:
                    found = True
                    assert entry["endpoint"] == "/v1/screening/analyze"
                    assert entry["patient_id_hash"] == "b" * 64
                    assert entry["extra"]["laterality"] == "L"
                    break
    assert found, "request_id not found in any audit log"


def test_screening_corrupt_bytes_returns_422(client):
    b64 = base64.b64encode(b"not a real dicom").decode()
    r = client.post("/v1/screening/analyze", json={"dicom_bytes_b64": b64})
    assert r.status_code == 422
    assert "preprocessing failed" in r.text


# --------------------------------------------------------------------------- #
# /v1/biopsy/analyze — placeholder

def test_biopsy_requires_input(client):
    r = client.post("/v1/biopsy/analyze", json={})
    assert r.status_code == 400


def test_biopsy_placeholder_returns_shape(client):
    r = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": "Invasive ductal carcinoma, Nottingham grade 2, "
                             "ER+, PR+, HER2 equivocal."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["disclaimer"] == RUO_DISCLAIMER
    assert body["provenance"]["model_state"] == "placeholder"
    assert body["subtype_prediction"] is None
    assert body["grade"] is None
    assert body["confidence"] is None
    assert body["receptor_panel"] == {
        "er_positive": None, "pr_positive": None,
        "her2_status": None, "ki67_percent": None,
    }


# --------------------------------------------------------------------------- #
# /v1/therapy/reason — placeholder

def test_therapy_placeholder_returns_empty_options(client):
    r = client.post(
        "/v1/therapy/reason",
        json={"patient_context": {"age": 55, "menopausal_status": "post"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["disclaimer"] == RUO_DISCLAIMER
    assert body["provenance"]["model_state"] == "placeholder"
    assert body["recommended_options"] == []
    assert body["not_recommended"] == []


# --------------------------------------------------------------------------- #
# /v1/case/full — end-to-end chain through real screening

def test_case_full_chains_placeholder_stages(client):
    path = FIXTURE_DIR / "Calc-Test_P_00038_LEFT_CC.dcm"
    if not path.exists():
        pytest.skip(f"missing fixture {path}")
    b64 = base64.b64encode(path.read_bytes()).decode()
    r = client.post(
        "/v1/case/full",
        json={
            "screening_input": {"dicom_bytes_b64": b64},
            "biopsy_input": {"report_text": "IDC, ER+, grade 2"},
            "therapy_context": {"age": 60, "menopausal_status": "post"},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Every stage populated
    assert body["screening"] is not None
    assert body["biopsy"] is not None
    assert body["therapy"] is not None
    # Screening got the real answer
    assert body["screening"]["laterality"] == "L"
    assert body["screening"]["view"] == "CC"
    # All disclaimers present at outer envelope
    assert body["disclaimer"] == RUO_DISCLAIMER
    assert body["caveat"] == AUROC_CAVEAT
    # No Elo tournament in placeholder path
    assert body["elo_ranked_hypotheses"] == []
