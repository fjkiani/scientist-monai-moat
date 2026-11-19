"""Live integration test for the ``crispro--luna16-infer`` Modal app.

Skipped unless ``LUNA16_MODAL_DETECT_URL`` is set to the FULL detect
endpoint URL, e.g. ``https://crispro--luna16-detect.modal.run``.
``Luna16Client`` (specialist_clients.py) uses this value as-is with no
suffix appended, unlike ``CASE_STORAGE_MODAL_URL``/``CLINICALBERT_MODAL_URL``
which are base prefixes -- do not set this to a base prefix by mistake.

Uses the durable real chest-CT DICOM series already proven to persist in
case-storage this session (case_id ``892531e5e4dde1e8``, uploader_note
"live fleet inference sweep: real LUNA16 chest CT", media_type
``application/dicom-series``, 121 real slices -- reconfirmed live via the
case-storage manifest endpoint immediately before this file was written).
No synthetic volume, no all-zeros tensor, no ``volume_hu_b64`` stub.

``Luna16Client`` only exposes ``detect()`` -- there is no ``healthz``/
``info`` wrapper in the production client (the same read-capable-but-
narrow-surface pattern already observed for ``CaseStorageClient``) -- so
those two endpoints are hit directly via the shared ``_http_json``
transport helper the client itself is built on.
"""
from __future__ import annotations

import os

import pytest

LIVE = pytest.mark.skipif(
    not os.environ.get("LUNA16_MODAL_DETECT_URL"),
    reason="LUNA16_MODAL_DETECT_URL not set (skipping live Modal test)",
)

# Real chest CT (121 slices), uploaded and durability-verified this
# session via test_case_storage_live.py::test_durable_prior_session_case_still_resolves
# and re-confirmed via a fresh manifest call while writing this file.
REAL_CHEST_CT_CASE_ID = "892531e5e4dde1e8"

BUNDLE_VERSION = "0.6.9"
APP_VERSION = "luna16-infer-v0.5.2-alpha-resample"

# Claimed operating envelope from deploy/modal/luna16_infer_app.py's own
# DISCLAIMER string ("Nodule size range 3-30 mm"). Used below to
# mathematically interrogate any detection outside this range rather than
# silently accepting whatever the model returns.
CLAIMED_MIN_DIAMETER_MM = 3.0
CLAIMED_MAX_DIAMETER_MM = 30.0


def _detect_url() -> str:
    return os.environ["LUNA16_MODAL_DETECT_URL"]


def _healthz_url() -> str:
    return _detect_url().replace("-detect.modal.run", "-healthz.modal.run")


def _info_url() -> str:
    return _detect_url().replace("-detect.modal.run", "-info.modal.run")


@LIVE
def test_healthz_reports_ok_and_bundle_version() -> None:
    from oncology_arbiter.models.specialist_clients import _http_json

    data, _elapsed_ms = _http_json(_healthz_url(), payload=None, timeout=30)
    assert data["status"] == "ok"
    assert data["app"] == "luna16-infer"
    assert data["version"] == APP_VERSION
    assert data["bundle_version"] == BUNDLE_VERSION


@LIVE
def test_info_reports_loaded_model_state_and_preprocessing() -> None:
    from oncology_arbiter.models.specialist_clients import _http_json

    data, _elapsed_ms = _http_json(_info_url(), payload=None, timeout=120)
    assert data["model_state"] == "loaded_luna16_retinanet"
    assert data["bundle_version"] == BUNDLE_VERSION
    assert f"@{BUNDLE_VERSION}" in data["model_name"]
    prep = data["preprocessing"]
    assert prep["hu_range"] == [-1024.0, 300.0]
    assert len(prep["target_spacing_mm"]) == 3
    assert "disclaimer" in data


@LIVE
def test_detect_rejects_malformed_case_id_client_side() -> None:
    """16 characters, but not lowercase hex -- must be rejected by the
    client's regex guard before any network call is made."""
    from oncology_arbiter.models.specialist_clients import (
        Luna16Client,
        SpecialistServiceError,
    )

    client = Luna16Client(detect_url=_detect_url())
    with pytest.raises(SpecialistServiceError) as exc_info:
        client.detect("NOTHEX-CASE-ID!!", request_id="pytest-luna16-badformat")
    assert exc_info.value.code == "invalid_case_id"


@LIVE
def test_detect_valid_format_but_nonexistent_case_id_is_honest_upstream_error() -> None:
    """Syntactically valid (16 lowercase hex) but not an uploaded case --
    the server's own FileNotFoundError must surface as a genuine
    SpecialistServiceError, not be swallowed or silently return an empty
    success payload."""
    from oncology_arbiter.models.specialist_clients import (
        Luna16Client,
        SpecialistServiceError,
    )

    client = Luna16Client(detect_url=_detect_url(), timeout=180.0)
    with pytest.raises(SpecialistServiceError) as exc_info:
        client.detect("0000000000000000", request_id="pytest-luna16-nonexistent")
    assert exc_info.value.code == "upstream_error"


@LIVE
def test_detect_real_chest_ct_returns_plausible_detections() -> None:
    """The real-I/O centerpiece of this file: run the actual production
    Luna16Client.detect() against a real, 121-slice chest CT DICOM
    series already durable in case-storage. Cold start (bundle fetch +
    GPU container spin-up) can take well over a minute; the client's own
    default timeout is 600s and we do not shorten it."""
    from oncology_arbiter.models.specialist_clients import Luna16Client

    client = Luna16Client(detect_url=_detect_url(), timeout=600.0)
    call = client.detect(REAL_CHEST_CT_CASE_ID, request_id="pytest-luna16-real-ct", top_n=20)

    out = call.output
    assert out["model_state"] == "loaded_luna16_retinanet"
    assert out["bundle_version"] == BUNDLE_VERSION
    assert out["case_id"] == REAL_CHEST_CT_CASE_ID
    assert out["ingest_source"] and REAL_CHEST_CT_CASE_ID in out["ingest_source"]
    assert isinstance(out["detections"], list)
    assert out["n_detections"] == len(out.get("detections", [])) or out["n_detections"] >= len(
        out["detections"]
    )  # detections list is truncated to top_n; n_detections is the pre-truncation count

    # Receipt sanity (SpecialistCall contract, mirrors case-storage/phikon).
    assert call.receipt["status"] == "succeeded"
    assert call.receipt["service_name"] == "luna16-infer"
    assert call.receipt["model_version"] == BUNDLE_VERSION
    assert f"case_id:{REAL_CHEST_CT_CASE_ID}" in call.receipt["input_reference"]

    detections = out["detections"]
    print(
        f"\n[luna16-infer live detect] case_id={REAL_CHEST_CT_CASE_ID} "
        f"n_detections_total={out['n_detections']} n_returned={len(detections)} "
        f"top_score={out.get('top_score')} "
        f"inference_seconds={out.get('inference_seconds')} "
        f"total_seconds={out.get('total_seconds')} device={out.get('device')} "
        f"input_shape={out.get('input_shape')}"
    )

    if not detections:
        # A genuine, honestly-reported possibility on a real chest CT that
        # may simply have no nodules scoring above threshold. Not silently
        # treated as success OR failure -- surfaced for human review.
        pytest.skip(
            "live detector returned zero detections on the real chest CT "
            "(case_id 892531e5e4dde1e8) -- report as-is, not a test bug; "
            "see printed diagnostics above."
        )

    # Detections must be sorted descending by score (server-side claim in
    # luna16_infer_app.py: `boxes.sort(key=lambda b: b['score'], reverse=True)`).
    scores = [d["score"] for d in detections]
    assert scores == sorted(scores, reverse=True), (
        f"detections not sorted descending by score: {scores}"
    )
    assert scores[0] == out["top_score"]

    for d in detections:
        for key in (
            "center_z_mm", "center_y_mm", "center_x_mm",
            "width_mm", "height_mm", "depth_mm", "diameter_mm", "score",
        ):
            assert key in d and isinstance(d[key], (int, float))
        assert 0.0 <= d["score"] <= 1.0

    # Mathematically interrogate the app's own claimed operating envelope
    # (DISCLAIMER: "Nodule size range 3-30 mm") rather than assuming it
    # holds. Report exactly how many/which detections violate it, if any.
    diameters = [d["diameter_mm"] for d in detections]
    out_of_range = [
        d for d in diameters
        if not (CLAIMED_MIN_DIAMETER_MM <= d <= CLAIMED_MAX_DIAMETER_MM)
    ]
    if out_of_range:
        print(
            f"[luna16-infer live detect] ANOMALY: {len(out_of_range)}/{len(diameters)} "
            f"detections fall outside the app's claimed "
            f"{CLAIMED_MIN_DIAMETER_MM}-{CLAIMED_MAX_DIAMETER_MM} mm envelope: "
            f"{sorted(out_of_range)} mm. This is a real measurement, not "
            "excluded as a confound; it indicates either the disclaimer's "
            "claimed range is inaccurate or size post-processing can exceed "
            "the trained/expected range."
        )

    # Inference timing plausibility: docstring claims ~30-90s for a
    # 200-slice CT on A10G. A near-zero or wildly excessive figure on a
    # real 121-slice volume would itself be a signal of a stub/fake path.
    inf_s = out.get("inference_seconds")
    if isinstance(inf_s, (int, float)):
        assert inf_s > 0.01, f"suspiciously instantaneous inference_seconds={inf_s}"


NEEDS_CASE_STORAGE = pytest.mark.skipif(
    not os.environ.get("CASE_STORAGE_MODAL_URL"),
    reason="CASE_STORAGE_MODAL_URL not set (needed to read raw DICOM headers "
    "independently of the detect() call for this test)",
)


@LIVE
def test_detect_resamples_native_spacing_to_declared_target() -> None:
    """Live contract: detect() must emit actual_spacing_mm at target dz=1.25.

    Finding F (gate_violation_bf5e54f8) documented missing Spacingd on the
    case_id path. v0.5.2 wires ``_resample_volume_to_luna16`` before clip.
    This test calls live detect() and asserts preprocessing_summary —
    not a native-DICOM vs info() comparison that never exercised detect().
    """
    from oncology_arbiter.models.specialist_clients import Luna16Client, _http_json

    info_data, _ = _http_json(_info_url(), payload=None, timeout=120)
    target = info_data["preprocessing"]["target_spacing_mm"]  # [dz, dy, dx]
    assert abs(float(target[0]) - 1.25) < 1e-6

    out = Luna16Client(detect_url=_detect_url()).detect(
        REAL_CHEST_CT_CASE_ID, request_id="pytest-resample-contract"
    ).output
    assert out.get("error") is None, out
    ps = out["preprocessing_summary"]
    actual = ps["actual_spacing_mm"]
    source = ps.get("source_spacing_mm")
    print(
        f"\n[luna16-infer resample contract] case_id={REAL_CHEST_CT_CASE_ID} "
        f"source={source} actual={actual} was_resampled={ps.get('was_resampled')} "
        f"app={out.get('app_version')}"
    )
    assert abs(float(actual[0]) - float(target[0])) < 0.01, (
        f"detect() actual dz={actual[0]} != target dz={target[0]} — resample missing"
    )
    if source is not None and abs(float(source[0]) - float(target[0])) > 0.05:
        assert ps.get("was_resampled") is True
