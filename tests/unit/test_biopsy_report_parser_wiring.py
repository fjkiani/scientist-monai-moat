"""API-level tests for the v0.2.1 biopsy endpoint wiring.

Focus: the /v1/biopsy/analyze endpoint MUST call report_parser and surface
the extracted ER/PR/HER2/grade in the response, including per-field parse
state, so the frontend can gate therapy on user confirmation.

Auth is off by default in create_app(), so these tests can hit the endpoint
directly with TestClient (mirroring test_biopsy_therapy_endpoints.py).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """Fresh app with the MedSigLIP biopsy flag OFF (placeholder path).

    We only care about the parser wiring; MedSigLIP is orthogonal.
    """
    for flag in (
        "ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP",
        "ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA",
        "ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY",
    ):
        monkeypatch.delenv(flag, raising=False)
    return TestClient(create_app())


LUMINAL_A_REPORT = (
    "Age: 58, postmenopausal\n"
    "Stage: T1N0M0\n"
    "Pathology:\n"
    "  Invasive ductal carcinoma of the right breast, 1.4 cm.\n"
    "  Estrogen Receptor: Positive (95%).\n"
    "  Progesterone Receptor: Positive (80%).\n"
    "  HER2/neu: Negative (IHC 1+).\n"
    "  Nottingham Grade: 2.\n"
    "  Ki-67 index: 12%.\n"
)


TNBC_REPORT = (
    "Invasive ductal carcinoma. ER negative. PR negative. "
    "HER2 0 by IHC. Grade 3."
)


HER2_POS_REPORT = "ER positive. PR negative. HER2 3+. Grade 3."


def test_biopsy_luminal_a_receptor_panel_and_grade_populated(client: TestClient) -> None:
    """The canned demo report MUST land ER=T, PR=T, HER2=negative, grade=2.

    This is the regression test for the exact silent-TNBC-default defect
    that motivated v0.2.1.
    """
    resp = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": LUMINAL_A_REPORT},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    panel = body["receptor_panel"]
    assert panel["er_positive"] is True
    assert panel["pr_positive"] is True
    assert panel["her2_status"] == "negative"
    assert body["grade"] == 2

    ps = panel["parse_state"]
    assert ps["er"] == "matched"
    assert ps["pr"] == "matched"
    assert ps["her2"] == "matched"
    assert ps["grade"] == "matched"

    # Warnings must announce the parser source so an auditor can trace it.
    assert any(
        w.startswith("receptor_panel_source:proxy_regex_v0")
        for w in body["warnings"]
    ), body["warnings"]


def test_biopsy_tnbc_receptor_panel_all_negative(client: TestClient) -> None:
    resp = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": TNBC_REPORT},
    )
    assert resp.status_code == 200, resp.text
    panel = resp.json()["receptor_panel"]
    assert panel["er_positive"] is False
    assert panel["pr_positive"] is False
    assert panel["her2_status"] == "negative"


def test_biopsy_her2_positive_case(client: TestClient) -> None:
    resp = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": HER2_POS_REPORT},
    )
    assert resp.status_code == 200
    panel = resp.json()["receptor_panel"]
    assert panel["er_positive"] is True
    assert panel["pr_positive"] is False
    assert panel["her2_status"] == "positive"


def test_biopsy_no_report_text_returns_empty_panel_but_valid_response(
    client: TestClient,
) -> None:
    """If report_text is absent (WSI-only path), the panel stays empty.

    We still need a valid response; wsi_bytes_b64 provides the input.
    """
    import base64

    resp = client.post(
        "/v1/biopsy/analyze",
        json={"wsi_bytes_b64": base64.b64encode(b"x").decode()},
    )
    assert resp.status_code == 200
    body = resp.json()
    panel = body["receptor_panel"]
    # Every field is None; parse_state absent because the parser did not run.
    assert panel["er_positive"] is None
    assert panel["pr_positive"] is None
    assert panel["her2_status"] is None
    assert panel.get("parse_state") is None
    assert body["grade"] is None


def test_biopsy_no_match_report_returns_all_no_match(client: TestClient) -> None:
    """A report with no receptor mentions returns all no_match parse states."""
    resp = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": "Fibroadenoma. Benign. No further workup needed."},
    )
    assert resp.status_code == 200
    body = resp.json()
    panel = body["receptor_panel"]
    assert panel["er_positive"] is None
    assert panel["pr_positive"] is None
    assert panel["her2_status"] is None
    assert body["grade"] is None
    ps = panel["parse_state"]
    assert ps["er"] == "no_match"
    assert ps["pr"] == "no_match"
    assert ps["her2"] == "no_match"
    assert ps["grade"] == "no_match"


def test_biopsy_her2_equivocal_marked_ambiguous(client: TestClient) -> None:
    """HER2 2+ (IHC equivocal) MUST land as ambiguous, not silently coerced.

    The whole point of the parse_state contract is that the UI can flag this
    for the pathologist to enter a definitive value.
    """
    resp = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": "ER positive. PR positive. HER2 2+. Grade 2."},
    )
    assert resp.status_code == 200
    body = resp.json()
    panel = body["receptor_panel"]
    assert panel["her2_status"] == "equivocal"
    assert panel["parse_state"]["her2"] == "ambiguous"
    assert panel["parse_state"]["er"] == "matched"
