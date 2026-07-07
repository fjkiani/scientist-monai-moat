"""Live end-to-end smoke for /v1/screening/analyze with Modal + CBIS-DDSM probe.

Skipped unless three env vars are set:
  * ``MODAL_MEDSIGLIP_URL``       — Modal app is deployed and reachable
  * ``MEDSIGLIP_BACKEND=modal``   — factory returns the remote client
  * ``ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP=1`` — endpoint calls MedSigLIP

The demo DICOM fixture is CBIS-DDSM Mass-Test_P_00016_LEFT_CC (BI-RADS 5,
IRREGULAR/SPICULATED mass, pathology MALIGNANT — see
mass_case_description_test_set.csv on ACSG-64/CBIS-DDSM-description-corrected).

This test is intentionally NOT a unit test — it hits real Modal endpoints,
so it catches drift between the deploy artifact and the API wiring.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest


LIVE_MODAL = pytest.mark.skipif(
    not (
        os.environ.get("MODAL_MEDSIGLIP_URL")
        and os.environ.get("MEDSIGLIP_BACKEND", "").lower() == "modal"
        and os.environ.get("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP") in ("1", "true", "yes")
    ),
    reason=(
        "MODAL_MEDSIGLIP_URL + MEDSIGLIP_BACKEND=modal + "
        "ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP=1 not all set"
    ),
)


@pytest.fixture(scope="module")
def demo_dicom_b64() -> str:
    """Fetch the CBIS-DDSM demo DICOM (idempotent) and return base64 bytes."""
    from oncology_arbiter.api.demo_fixtures import prewarm_demo_case

    path: Path | None = prewarm_demo_case()
    if path is None or not path.exists():
        pytest.skip("demo DICOM fixture unavailable (HF unreachable)")
    return base64.b64encode(path.read_bytes()).decode("ascii")


@pytest.fixture(scope="module")
def test_client():
    # Auth off for the smoke; the wiring under test is the MedSigLIP+probe
    # chain, not the tenant middleware.
    prev_auth = os.environ.get("ONCOLOGY_ARBITER_AUTH_MODE")
    os.environ["ONCOLOGY_ARBITER_AUTH_MODE"] = "off"
    # Skip demo pre-warm during app boot; we fetched it above.
    prev_skip = os.environ.get("ONCOLOGY_ARBITER_SKIP_DEMO_PREWARM")
    os.environ["ONCOLOGY_ARBITER_SKIP_DEMO_PREWARM"] = "1"
    try:
        from fastapi.testclient import TestClient
        from oncology_arbiter.api import create_app

        app = create_app()
        with TestClient(app) as c:
            yield c
    finally:
        if prev_auth is None:
            os.environ.pop("ONCOLOGY_ARBITER_AUTH_MODE", None)
        else:
            os.environ["ONCOLOGY_ARBITER_AUTH_MODE"] = prev_auth
        if prev_skip is None:
            os.environ.pop("ONCOLOGY_ARBITER_SKIP_DEMO_PREWARM", None)
        else:
            os.environ["ONCOLOGY_ARBITER_SKIP_DEMO_PREWARM"] = prev_skip


@LIVE_MODAL
def test_screening_analyze_uses_modal_and_reports_loaded_medsiglip(
    test_client, demo_dicom_b64
) -> None:
    """POST DICOM → 200 → provenance reflects Modal-backed MedSigLIP."""
    resp = test_client.post(
        "/v1/screening/analyze",
        json={"dicom_bytes_b64": demo_dicom_b64},
    )
    assert resp.status_code == 200, resp.text
    obj = resp.json()
    prov = obj["provenance"]
    assert prov["model_state"] == "loaded_medsiglip", prov
    assert prov["model_name"].startswith("google/medsiglip-448"), prov
    assert prov["gate_report"]["access_level"] == "allowed", prov["gate_report"]
    # Modal-remote fence: reason must call out modal-remote and dim=1152.
    assert "modal-remote" in prov["gate_report"]["reason"], prov["gate_report"]
    assert "dim=1152" in prov["gate_report"]["reason"], prov["gate_report"]


@LIVE_MODAL
def test_screening_analyze_returns_two_zero_shot_findings(
    test_client, demo_dicom_b64
) -> None:
    """MedSigLIP zero-shot always emits DEFAULT_ZERO_SHOT_LABELS worth of findings."""
    resp = test_client.post(
        "/v1/screening/analyze",
        json={"dicom_bytes_b64": demo_dicom_b64},
    )
    obj = resp.json()
    zs = [f for f in obj["findings"] if not f["label"].startswith("cbis_ddsm")]
    assert len(zs) >= 2, obj["findings"]
    for f in zs:
        assert 0.0 <= f["score"] <= 1.0, f
        # Off-label mammography: probs should be tiny (< 0.1) but nonzero.
        assert f["score"] < 0.1, f


@LIVE_MODAL
def test_screening_analyze_wires_cbis_ddsm_probe_when_enabled(
    test_client, demo_dicom_b64
) -> None:
    """With ONCOLOGY_ARBITER_ENABLE_CBIS_DDSM_PROBE=1 the trained probe fires
    and OVERRIDES overall_score, adds a probe finding, and appends the
    honesty warning.
    """
    prev = os.environ.get("ONCOLOGY_ARBITER_ENABLE_CBIS_DDSM_PROBE")
    os.environ["ONCOLOGY_ARBITER_ENABLE_CBIS_DDSM_PROBE"] = "1"
    try:
        resp = test_client.post(
            "/v1/screening/analyze",
            json={"dicom_bytes_b64": demo_dicom_b64},
        )
    finally:
        if prev is None:
            os.environ.pop("ONCOLOGY_ARBITER_ENABLE_CBIS_DDSM_PROBE", None)
        else:
            os.environ["ONCOLOGY_ARBITER_ENABLE_CBIS_DDSM_PROBE"] = prev

    assert resp.status_code == 200, resp.text
    obj = resp.json()

    # Probe finding present.
    probe_findings = [
        f for f in obj["findings"] if f["label"].startswith("cbis_ddsm_logreg_v1:")
    ]
    assert len(probe_findings) == 1, obj["findings"]
    pf = probe_findings[0]
    assert pf["label"] in {
        "cbis_ddsm_logreg_v1:cancer",
        "cbis_ddsm_logreg_v1:not_cancer",
    }
    assert 0.0 <= pf["score"] <= 1.0

    # Overall score is the probe score (not the zero-shot argmax which is <0.01).
    assert obj["overall_score"] == pytest.approx(pf["score"], rel=1e-6)
    assert obj["overall_score"] > 0.05, obj["overall_score"]

    # Ground-truth CBIS-DDSM Mass-Test_P_00016_LEFT_CC is MALIGNANT (BI-RADS 5).
    # At the default recall_0.85 op point (0.2836), and at 0.5, this case
    # should be recalled as cancer. Guard against silent regression of the
    # probe weights.
    assert pf["label"] == "cbis_ddsm_logreg_v1:cancer", (
        f"Ground-truth MALIGNANT case labeled {pf['label']!r} at score {pf['score']}"
    )

    # Model name chain reflects the probe.
    prov = obj["provenance"]
    assert "cbis_ddsm_logreg_v1" in prov["model_name"], prov

    # Honesty warning shipped.
    assert any(
        "cbis" in w.lower() and "auc" in w.lower() for w in obj["warnings"]
    ), obj["warnings"]


@LIVE_MODAL
def test_screening_analyze_preprocessing_still_runs_on_real_dicom(
    test_client, demo_dicom_b64
) -> None:
    """The Modal path must not skip local mammography preprocessing (laterality,
    view, mask). Regression fence against a code change that would forward
    the raw DICOM to Modal without also computing local metadata.
    """
    resp = test_client.post(
        "/v1/screening/analyze",
        json={"dicom_bytes_b64": demo_dicom_b64},
    )
    obj = resp.json()
    # CBIS-DDSM Mass-Test_P_00016_LEFT_CC — the CSV says LEFT/CC, our
    # preprocessing detects it. Anything else means preprocessing is broken
    # (or the demo DICOM was swapped without updating this test).
    assert obj["view"] == "CC", obj
    # Laterality-from-image detection can flip LEFT/RIGHT depending on
    # orientation; accept either and only enforce that it's not "U".
    assert obj["laterality"] in ("L", "R"), obj
    # Real breast mask coverage should be non-trivial on this well-exposed
    # test image (~0.2 in the manual smoke).
    assert 0.05 < obj["breast_mask_coverage"] < 0.9, obj
