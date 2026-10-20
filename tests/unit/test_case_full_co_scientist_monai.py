"""Case/full honesty: screening hypotheses must not invent evidence URLs.

Uses the CBIS-DDSM test DICOM when the fixture is on disk.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app


FIXTURE_DCM = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cbis_ddsm"
    / "Calc-Test_P_00038_LEFT_CC.dcm"
)


def _dicom_b64() -> str | None:
    if not FIXTURE_DCM.is_file():
        return None
    return base64.b64encode(FIXTURE_DCM.read_bytes()).decode("ascii")


@pytest.mark.skipif(
    not FIXTURE_DCM.is_file(),
    reason=f"CBIS-DDSM fixture missing at {FIXTURE_DCM}",
)
def test_case_full_monai_findings_do_not_bypass_honesty_gate(monkeypatch):
    """A MONAI heuristic finding has no evidence URL by construction — so
    Co-Scientist's honesty gate should keep the hypothesis (with a
    no_evidence_after_reflect warning under the hood) but NOT invent a URL
    that isn't in the stage response."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR", "1")
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST", "1")

    dicom_b64 = _dicom_b64()
    with TestClient(create_app()) as client:
        r = client.post(
            "/v1/case/full",
            json={
                "screening_input": {
                    "dicom_bytes_b64": dicom_b64,
                    "laterality": "L",
                    "view": "CC",
                },
                "therapy_context": {"age": 55, "menopausal_status": "post"},
            },
        )
    assert r.status_code == 200
    body = r.json()

    # For every screening-derived hypothesis, evidence[] MUST be a subset
    # of URLs actually present in the screening evidence[] block. Since
    # MONAI heuristic emits no URL evidence, every screening hypothesis
    # should have evidence == [] (never fabricated).
    screening_evidence_urls = {
        e["url"]
        for e in (body["screening"].get("evidence") or [])
        if isinstance(e, dict) and "url" in e
    }
    for h in body["elo_ranked_hypotheses"]:
        if h.get("stage") != "screening":
            continue
        for e in h.get("evidence") or []:
            assert e.get("url") in screening_evidence_urls, (
                f"screening hypothesis {h['hyp_id']} carries URL {e.get('url')} "
                f"not present in screening.evidence — honesty gate breach"
            )
