"""API-level tests for the v0.2.1 therapy override + case_full confirm gate.

Two invariants:

1. /v1/therapy/reason: when `receptors_override` is set on the request, its
   values MUST replace whatever the biopsy_output receptor_panel says, for
   the purpose of branch selection. This is what makes the Confirm-gate
   contract honest — pathologist wins over parser.

2. /v1/case/full: when `receptors_confirmed` is set on the top-level request,
   the internal therapy_reason call MUST receive it as receptors_override.

We exercise the rules-lite proxy path (deterministic, no HF) with
ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY=1 so branch selection is
observable via `branch_id` in the response.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """App with rules-lite proxy on so branch_id lands in the response."""
    for flag in (
        "ONCOLOGY_ARBITER_ENABLE_BIOPSY_MEDSIGLIP",
        "ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA",
        "ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST",
    ):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY", "1")
    return TestClient(create_app())


def _envelope_biopsy(er: bool | None, pr: bool | None,
                     her2: str | None, grade: int | None = 2) -> dict[str, Any]:
    """Minimal BiopsyResponse dict shape for TherapyRequest.biopsy_output."""
    return {
        "disclaimer": "RESEARCH USE ONLY",
        "caveat": "AUROC caveat",
        "provenance": {"model_state": "placeholder", "request_id": "req-t"},
        "honesty_gate": {"seen_urls_count": 0, "evidence_kept": 0, "evidence_dropped": 0},
        "evidence": [],
        "warnings": [],
        "subtype_prediction": None,
        "receptor_panel": {
            "er_positive": er,
            "pr_positive": pr,
            "her2_status": her2,
            "ki67_percent": None,
        },
        "grade": grade,
        "confidence": None,
    }


# ---------------------------------------------------------------------------
# /v1/therapy/reason: receptors_override
# ---------------------------------------------------------------------------


def test_therapy_uses_receptors_override_over_biopsy_output(client):
    """Stale biopsy says TNBC; override says ER+/PR+/HER2− → HR+/HER2− branch."""
    payload = {
        "biopsy_output": _envelope_biopsy(er=False, pr=False, her2="negative"),
        "receptors_override": {
            "er_positive": True,
            "pr_positive": True,
            "her2_status": "negative",
        },
        "patient_context": {"age": 58, "menopausal_status": "post"},
    }
    resp = client.post("/v1/therapy/reason", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["branch_id"] == "hr_positive_her2_negative", body
    assert any("receptors_source:user_confirmed" in w for w in body["warnings"])


def test_therapy_override_absent_falls_back_to_biopsy_panel(client):
    """No override → biopsy_output.receptor_panel wins (existing behaviour)."""
    payload = {
        "biopsy_output": _envelope_biopsy(er=False, pr=False, her2="negative"),
        "patient_context": {"age": 58, "menopausal_status": "post"},
    }
    resp = client.post("/v1/therapy/reason", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["branch_id"] == "triple_negative", body
    assert not any(
        "receptors_source:user_confirmed" in w for w in body["warnings"]
    )


def test_therapy_override_her2_positive_wins(client):
    """Override HER2 positive should hit HER2+ branch even if biopsy says negative."""
    payload = {
        "biopsy_output": _envelope_biopsy(er=True, pr=True, her2="negative"),
        "receptors_override": {
            "er_positive": True,
            "pr_positive": True,
            "her2_status": "positive",
        },
        "patient_context": {"age": 55, "menopausal_status": "post"},
    }
    resp = client.post("/v1/therapy/reason", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["branch_id"] == "her2_positive", body


def test_therapy_override_none_fields_treated_as_negative(client):
    """A partial override with None fields (e.g. HER2 status unknown from parser)
    should coerce to False for the branch selector, not raise.
    """
    payload = {
        "biopsy_output": _envelope_biopsy(er=False, pr=False, her2="negative"),
        "receptors_override": {
            "er_positive": True,
            "pr_positive": None,   # unknown
            "her2_status": None,   # unknown
        },
        "patient_context": {"age": 60, "menopausal_status": "post"},
    }
    resp = client.post("/v1/therapy/reason", json=payload)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /v1/case/full: receptors_confirmed
# ---------------------------------------------------------------------------


def test_case_full_receptors_confirmed_drives_therapy_branch(client):
    """The exact tumor-board demo defect:
    biopsy_input.report_text says TNBC-looking text, but the user has
    corrected the panel via Confirm to ER+/PR+/HER2- luminal-A. The
    therapy branch MUST honor the correction.
    """
    payload = {
        "biopsy_input": {
            "report_text": "Invasive ductal carcinoma. ER negative. HER2 0.",
        },
        "receptors_confirmed": {
            "er_positive": True,
            "pr_positive": True,
            "her2_status": "negative",
            "parse_state": {
                "er": "user_supplied",
                "pr": "user_supplied",
                "her2": "user_supplied",
                "grade": "user_supplied",
            },
        },
        "therapy_context": {"age": 58, "menopausal_status": "post"},
    }
    resp = client.post("/v1/case/full?cancer=breast", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Biopsy sub-response reflects what the parser found (TNBC-looking)
    assert body["biopsy"]["receptor_panel"]["er_positive"] is False
    # Therapy branch reflects the confirmed override (luminal-A)
    assert body["therapy"]["branch_id"] == "hr_positive_her2_negative", body["therapy"]
    assert any(
        "receptors_source:user_confirmed" in w
        for w in body["therapy"]["warnings"]
    )


def test_case_full_no_confirmed_falls_back_to_biopsy(client):
    """Sanity: without receptors_confirmed, biopsy panel drives branch."""
    payload = {
        "biopsy_input": {
            "report_text": "Invasive ductal carcinoma. ER positive. PR positive. HER2 negative. Grade 2.",
        },
        "therapy_context": {"age": 58, "menopausal_status": "post"},
    }
    resp = client.post("/v1/case/full?cancer=breast", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["biopsy"]["receptor_panel"]["er_positive"] is True
    assert body["therapy"]["branch_id"] == "hr_positive_her2_negative"
