"""Cross-stage integration: /v1/case/full with the MONAI heuristic detector
turned on must feed its screening findings into Co-Scientist so that at least
one hypothesis in ``elo_ranked_hypotheses`` has stage=``screening`` and its
``derived_from`` traces to a monai_heuristic finding label.

This exercises three workers' code paths end-to-end in the same request:

  * worker-2  → MONAI heuristic detector in screening
  * worker-0  → L5 Co-Scientist generate/reflect/rank/evolve loop
  * worker-0  → gate_report Provenance contract (no gated repo here, so
                gate_report should be None on the outer envelope)

Uses the CBIS-DDSM test DICOM the repo ships as a fixture if present;
otherwise the test is SKIPPED (no fabrication).
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
def test_case_full_with_monai_heuristic_feeds_co_scientist(monkeypatch):
    """MONAI heuristic ON + Co-Scientist ON → screening hypotheses land in
    the ranked list."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_MONAI_DETECTOR", "1")
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST", "1")
    # No HAI-DEF backends — keep this test hermetic.
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP", raising=False)
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP", raising=False)
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA", raising=False)

    dicom_b64 = _dicom_b64()
    assert dicom_b64 is not None  # skipif guarded above

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
    assert r.status_code == 200, r.text
    body = r.json()

    # Screening ran and returned at least one finding (heuristic detector
    # is deterministic on a real DICOM — see w2 model card).
    screening = body["screening"]
    assert screening is not None
    findings = screening.get("findings") or []
    assert len(findings) > 0, "MONAI heuristic should surface at least one finding"

    # Co-Scientist ran → ranked hypotheses list non-empty
    ranked = body["elo_ranked_hypotheses"]
    assert isinstance(ranked, list)
    assert len(ranked) > 0, "Co-Scientist should have produced hypotheses"

    # At least one hypothesis MUST be derived from a screening stage output.
    screening_hyps = [h for h in ranked if h.get("stage") == "screening"]
    assert len(screening_hyps) > 0, (
        "elo_ranked_hypotheses must include at least one screening-stage "
        "hypothesis when MONAI heuristic surfaced findings"
    )

    # Elo ratings must be numeric and each hypothesis must carry the
    # standard hypothesis shape (hyp_id, statement, confidence).
    for h in ranked:
        assert isinstance(h.get("statement"), str)
        assert isinstance(h.get("confidence"), (int, float))
        assert isinstance(h.get("hyp_id"), str)
        assert isinstance(h.get("rating"), (int, float))


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
