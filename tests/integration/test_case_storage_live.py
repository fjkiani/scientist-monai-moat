"""Live integration test for the Modal-backed case-storage app.

Skipped unless the environment carries ``CASE_STORAGE_MODAL_URL``. This is
NOT a mocked unit test: every assertion here hits the real
``crispro--case-storage`` Modal deployment over the network with a real
DICOM fixture, and every expected value (case_id, sha256_full, byte counts)
is computed independently in this test process *before* looking at the
server's response, so a server-side bug cannot pass by construction (the
classic "fake pass/fail" failure mode this suite is repairing).

``CaseStorageClient`` (src/oncology_arbiter/models/specialist_clients.py)
only exposes ``healthz``/``manifest``/``get_file`` -- there is no
``upload()`` method in the production client, so the upload leg below
calls the raw endpoint via the same ``_http_json`` helper the client uses
internally. That asymmetry (read-capable client, no write path) is a real
observation about the current codebase, not a test artifact.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pytest

APP_VERSION = "case-storage-modal-v0.5.0-alpha"

LIVE = pytest.mark.skipif(
    not os.environ.get("CASE_STORAGE_MODAL_URL"),
    reason="CASE_STORAGE_MODAL_URL not set (skipping live Modal test)",
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "cbis_ddsm"
DICOM_FIXTURE = FIXTURE_ROOT / "Calc-Test_P_00038_LEFT_CC.dcm"

# A case uploaded in a prior session (2026-08-13, real 121-slice LUNA16
# chest CT). Used only to test cross-session Modal Volume durability via a
# read-only manifest call -- we do not re-upload or mutate it.
DURABLE_PRIOR_CASE_ID = "892531e5e4dde1e8"


def _skip_if_missing(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            pytest.skip(f"fixture missing: {p}")


def _upload_url() -> str:
    base = os.environ["CASE_STORAGE_MODAL_URL"].rstrip("/")
    return f"{base}-upload.modal.run"


def _upload_dicom_bytes(raw: bytes, *, note: str) -> dict:
    """Raw POST to the upload endpoint (no client-side wrapper exists)."""
    import base64

    from oncology_arbiter.models.specialist_clients import _http_json

    payload = {"dicom_b64": base64.b64encode(raw).decode("ascii"), "uploader_note": note}
    data, elapsed = _http_json(_upload_url(), payload=payload, timeout=60.0)
    data["_elapsed_s"] = elapsed
    return data


@LIVE
def test_upload_real_dicom_is_content_addressable() -> None:
    """Server-returned case_id/sha256 must match hashes computed locally.

    This is the crux real-I/O assertion: we do not trust the server's
    self-reported case_id, we recompute sha256(raw_bytes) ourselves and
    require an exact match. A server that mints random or non-deterministic
    case_ids would fail this test even if it returned HTTP 200.
    """
    _skip_if_missing(DICOM_FIXTURE)
    raw = DICOM_FIXTURE.read_bytes()
    expected_sha256_full = hashlib.sha256(raw).hexdigest()
    expected_case_id = expected_sha256_full[:16]

    resp = _upload_dicom_bytes(raw, note=f"pytest-content-address-{uuid.uuid4().hex[:8]}")

    assert "error" not in resp, resp
    assert resp["case_id"] == expected_case_id, (
        f"server case_id {resp.get('case_id')!r} != locally computed "
        f"sha256(raw)[:16] {expected_case_id!r} -- content-addressing is broken"
    )
    assert resp["sha256_full"] == expected_sha256_full
    assert resp["byte_counts"] == {"dicom.bin": len(raw)}
    assert resp["media_types"] == ["application/dicom"]
    assert resp["app_version"] == APP_VERSION
    assert resp["storage_backend"] == "modal-volume:oncology-arbiter-cases"


@LIVE
def test_reupload_same_bytes_is_idempotent() -> None:
    """Uploading identical bytes twice must return the identical case_id.

    Real test of the "content-addressable" claim in the module docstring:
    idempotency is a load-bearing property (the API gateway relies on this
    to avoid duplicate storage), not just a nice-to-have.
    """
    _skip_if_missing(DICOM_FIXTURE)
    raw = DICOM_FIXTURE.read_bytes()
    resp1 = _upload_dicom_bytes(raw, note="pytest-idempotency-1")
    resp2 = _upload_dicom_bytes(raw, note="pytest-idempotency-2")
    assert "error" not in resp1 and "error" not in resp2
    assert resp1["case_id"] == resp2["case_id"]
    assert resp1["sha256_full"] == resp2["sha256_full"]


@LIVE
def test_manifest_matches_uploaded_case() -> None:
    """CaseStorageClient.manifest() (production read path) must reflect
    exactly what was just uploaded via the raw endpoint."""
    from oncology_arbiter.models.specialist_clients import CaseStorageClient

    _skip_if_missing(DICOM_FIXTURE)
    raw = DICOM_FIXTURE.read_bytes()
    expected_case_id = hashlib.sha256(raw).hexdigest()[:16]
    upload_resp = _upload_dicom_bytes(raw, note="pytest-manifest-check")
    assert upload_resp["case_id"] == expected_case_id

    client = CaseStorageClient()
    call = client.manifest(expected_case_id, request_id=f"pytest-{uuid.uuid4().hex[:8]}")
    body = call.output
    assert body["case_id"] == expected_case_id
    assert body["files"] == ["dicom.bin"]
    assert body["byte_counts"] == {"dicom.bin": len(raw)}
    assert body["app_version"] == APP_VERSION
    assert call.receipt["status"] == "succeeded"
    assert call.receipt["service_name"] == "case-storage"


@LIVE
def test_get_file_returns_byte_identical_content() -> None:
    """Round trip: bytes read back from the Modal Volume must be byte-for-
    byte identical (verified by independent sha256, not just length) to
    the bytes we uploaded."""
    from oncology_arbiter.models.specialist_clients import CaseStorageClient

    _skip_if_missing(DICOM_FIXTURE)
    raw = DICOM_FIXTURE.read_bytes()
    expected_sha256 = hashlib.sha256(raw).hexdigest()
    expected_case_id = expected_sha256[:16]
    upload_resp = _upload_dicom_bytes(raw, note="pytest-get-file-check")
    assert upload_resp["case_id"] == expected_case_id

    client = CaseStorageClient()
    call = client.get_file(expected_case_id, "dicom.bin", request_id=f"pytest-{uuid.uuid4().hex[:8]}")
    body = call.output
    assert body["case_id"] == expected_case_id
    assert body["byte_count"] == len(raw)
    assert body["sha256"] == expected_sha256
    assert body["bytes"] == raw, "retrieved bytes differ from the original upload"


@LIVE
def test_durable_prior_session_case_still_resolves() -> None:
    """A case uploaded in a *different, prior* session must still resolve.

    This is a cross-session durability check, distinct from the round-trip
    tests above which only prove within-run consistency. If the Modal
    Volume were ephemeral or scoped per-deployment, this would 404.
    """
    from oncology_arbiter.models.specialist_clients import CaseStorageClient

    client = CaseStorageClient()
    call = client.manifest(DURABLE_PRIOR_CASE_ID, request_id=f"pytest-{uuid.uuid4().hex[:8]}")
    body = call.output
    assert body["case_id"] == DURABLE_PRIOR_CASE_ID
    assert body["sha256_full"].startswith(DURABLE_PRIOR_CASE_ID)
    assert len(body["files"]) > 0
    assert body["app_version"] == APP_VERSION


@LIVE
def test_upload_with_no_payload_fields_raises_upstream_error_not_crash() -> None:
    """The app's own input contract: at least one payload field is
    required. The Modal app returns HTTP 200 with a structured
    ``{"error": ...}`` body (not a 500) when no file field is present --
    but ``_http_json`` (the shared transport helper used by every
    specialist client) treats *any* ``{"error": ...}`` response body as
    a hard failure and raises ``SpecialistServiceError("upstream_error",
    ...)`` before it ever reaches caller code. That is the real,
    production behavior: verify the raise, not a returned error dict.
    """
    from oncology_arbiter.models.specialist_clients import SpecialistServiceError, _http_json

    with pytest.raises(SpecialistServiceError) as excinfo:
        _http_json(_upload_url(), payload={"uploader_note": "pytest-empty-payload"}, timeout=30.0)
    assert excinfo.value.code == "upstream_error"
    assert "at least one of" in str(excinfo.value)
