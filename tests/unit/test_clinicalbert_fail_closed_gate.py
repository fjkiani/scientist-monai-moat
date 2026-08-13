"""Directive 3: the deployed parser must be refused before clinical use.

These tests use the EXACT report text and the EXACT wrong values recorded in
``artifacts/audit/modal_fleet_live_inference.json`` from the live
``clinicalbert-parse`` call against ``clinicalbert-modal-v0.5.1``. The point is
not that the parser is imperfect -- it is that two specific fields were wrong in
a direction that changes treatment, and the API must therefore emit nothing
parser-derived at all.

Two errors under test, both quoted from the live receipt:

  HER2: report says ``HER2/neu by immunohistochemistry scores 1+ (NEGATIVE)``
        parser returned ``HER2_VALUE = "positive"``
  Ki-67: report says ``Ki-67 proliferation index is 20%``
        parser returned ``KI67_PCT = 67``  (digits lifted from the analyte name)
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.nlp.parser_acceptance_gate import (
    BELOW_FLOOR_CODE,
    PARSER_DERIVED_FIELDS,
    PARSER_MIN_MICRO_F1,
    PARSER_REQUIRED_APP_VERSION,
    UNVERIFIABLE_CODE,
    VERSION_MISMATCH_CODE,
    assert_no_parser_derived_values,
    evaluate_parser_acceptance,
)

# ---- the exact audited report (verbatim from the live-inference receipt) ----
AUDITED_REPORT = (
    "SURGICAL PATHOLOGY REPORT\n"
    "Specimen: Left breast, lumpectomy with sentinel lymph node biopsy.\n"
    "Gross: A 2.2 cm firm, tan-white mass is identified.\n"
    "Microscopic: Invasive ductal carcinoma, Nottingham grade 2 "
    "(tubule 3, nuclear 2, mitotic 1; total score 6 of 9).\n"
    "Lymphovascular invasion is absent.\n"
    "Margins are negative; closest margin 4 mm (inferior).\n"
    "Sentinel lymph nodes: 0 of 3 involved by metastatic carcinoma.\n"
    "IMMUNOHISTOCHEMISTRY:\n"
    "Estrogen receptor: POSITIVE, 95% of tumor nuclei, strong intensity.\n"
    "Progesterone receptor: POSITIVE, 80% of tumor nuclei, strong intensity.\n"
    "HER2/neu by immunohistochemistry scores 1+ (NEGATIVE).\n"
    "Ki-67 proliferation index is 20%.\n"
)

#: Verbatim wrong output from the live v0.5.1 deployment.
DEPLOYED_V051_ENTITIES = [
    {"entity_type": "GRADE", "value": "2", "surface": "grade 2"},
    {"entity_type": "TUMOR_SIZE_MM", "value": "2.2", "surface": "2.2 cm"},
    {"entity_type": "LVI", "value": "absent", "surface": "invasion is absent"},
    {"entity_type": "ER_VALUE", "value": "positive", "surface": "Estrogen receptor : POSITIVE"},
    {"entity_type": "PR_VALUE", "value": "positive", "surface": "Progesterone receptor : POSITIVE"},
    # THE TWO CONSEQUENTIAL ERRORS
    {"entity_type": "HER2_VALUE", "value": "positive",
     "surface": "HER 2 / neu by immunohistochemistry"},
    {"entity_type": "KI67_PCT", "value": "67",
     "surface": "Ki - 67 proliferation index is 20%"},
]

DEPLOYED_V051_INFO = {
    "app_version": "clinicalbert-modal-v0.5.1",
    "test_micro_f1": 0.08089260808926081,
    "num_labels": 43,
    "provenance": "REAL-v0.5.1-snorkel-openrouter-llm",
}


@pytest.fixture()
def client():
    return TestClient(create_app())


# --------------------------------------------------------------------------
# Gate unit behaviour
# --------------------------------------------------------------------------
def test_deployed_v051_is_refused_on_version_before_f1():
    """Version is the first discriminator: a mismatched version means the
    reported F1 describes a different artifact, so the floor test is moot."""
    acc = evaluate_parser_acceptance(DEPLOYED_V051_INFO)
    assert acc.accepted is False
    assert acc.error_code == VERSION_MISMATCH_CODE
    assert acc.observed_app_version == "clinicalbert-modal-v0.5.1"
    assert acc.observed_micro_f1 == pytest.approx(0.08089260808926081)


def test_correct_version_but_below_floor_is_refused_on_f1():
    info = dict(DEPLOYED_V051_INFO, app_version=PARSER_REQUIRED_APP_VERSION)
    acc = evaluate_parser_acceptance(info)
    assert acc.accepted is False
    assert acc.error_code == BELOW_FLOOR_CODE
    assert acc.required_micro_f1 == PARSER_MIN_MICRO_F1


def test_missing_info_is_refused_not_assumed_good():
    acc = evaluate_parser_acceptance(None)
    assert acc.accepted is False
    assert acc.error_code == UNVERIFIABLE_CODE


def test_missing_metrics_with_right_version_is_refused():
    acc = evaluate_parser_acceptance({"app_version": PARSER_REQUIRED_APP_VERSION})
    assert acc.accepted is False
    assert acc.error_code == UNVERIFIABLE_CODE


def test_a_conforming_parser_is_accepted():
    acc = evaluate_parser_acceptance(
        {"app_version": PARSER_REQUIRED_APP_VERSION, "test_micro_f1": 0.74}
    )
    assert acc.accepted is True
    assert acc.error_code is None


def test_floor_is_exactly_0_70_and_inclusive():
    at = evaluate_parser_acceptance(
        {"app_version": PARSER_REQUIRED_APP_VERSION, "test_micro_f1": 0.70})
    just_under = evaluate_parser_acceptance(
        {"app_version": PARSER_REQUIRED_APP_VERSION, "test_micro_f1": 0.6999})
    assert at.accepted is True
    assert just_under.accepted is False
    assert just_under.error_code == BELOW_FLOOR_CODE


def test_refusal_payload_declares_no_fallback_of_any_kind():
    p = evaluate_parser_acceptance(DEPLOYED_V051_INFO).payload()
    assert p["report_parse"] is None
    assert p["regex_fallback_used"] is False
    assert p["fusion_used"] is False
    assert p["inferred_value_substitution_used"] is False
    for f in ("her2_status", "ki67_pct", "grade", "dss_prognosis", "therapy_options"):
        assert f in p["parser_derived_fields_withheld"]


def test_gate_cannot_be_constructed_with_a_fallback_permitted():
    from oncology_arbiter.nlp.parser_acceptance_gate import ParserAcceptance

    with pytest.raises(ValueError, match="no fallback is permitted"):
        ParserAcceptance(
            accepted=False, error_code=BELOW_FLOOR_CODE, detail="x",
            observed_app_version="v", observed_micro_f1=0.1,
            fallback_permitted=True,
        )


# --------------------------------------------------------------------------
# The two consequential values must never survive into a response
# --------------------------------------------------------------------------
def test_leak_detector_catches_her2_positive_anywhere_in_payload():
    leaky = {"provenance": {}, "biopsy": {"receptor_panel": {"her2_status": "positive"}}}
    leaks = assert_no_parser_derived_values(leaky)
    assert any("her2_status" in x for x in leaks)
    assert any("receptor_panel" in x for x in leaks)


def test_leak_detector_catches_ki67_67_nested_in_a_list():
    leaky = {"therapy": {"options": [{"rationale": "high risk", "ki67_pct": 67}]}}
    leaks = assert_no_parser_derived_values(leaky)
    assert any("ki67_pct" in x for x in leaks)


def test_leak_detector_passes_a_fully_withheld_payload():
    clean = {
        "report_parse": None,
        "biopsy": {"receptor_panel": None, "grade": None},
        "therapy": {"options": []},
        "dss_prognosis": None,
    }
    assert assert_no_parser_derived_values(clean) == []


# --------------------------------------------------------------------------
# End-to-end: /v1/biopsy/analyze with the audited report
# --------------------------------------------------------------------------
def _post_biopsy(client, monkeypatch, *, entities, info):
    """Force the parse path to return the deployed v0.5.1 output verbatim."""
    import oncology_arbiter.api.app as appmod

    def fake_parse(report_text, *, request_id):
        raise AssertionError(
            "the gate must refuse before any parse result is consumed"
        )

    monkeypatch.setattr(appmod, "_clinicalbert_deployment_info", lambda: info,
                        raising=False)
    return client.post(
        "/v1/biopsy/analyze",
        json={"report_text": AUDITED_REPORT, "patient_id_hash": "b" * 64},
    )


def test_biopsy_with_audited_report_emits_null_report_parse(client):
    r = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": AUDITED_REPORT, "patient_id_hash": "b" * 64},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("report_parse") is None, (
        "a parser below the validation floor must yield report_parse=null"
    )


def test_biopsy_with_audited_report_has_failed_required_stage(client):
    body = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": AUDITED_REPORT, "patient_id_hash": "b" * 64},
    ).json()
    stages = body["stage_receipts"]
    parse_stages = [s for s in stages if "clinicalbert" in s["stage"]]
    assert parse_stages, f"no clinicalbert stage in {[s['stage'] for s in stages]}"
    assert parse_stages[0]["status"] == "failed_required"


def test_biopsy_error_code_is_one_of_the_two_declared_codes(client):
    body = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": AUDITED_REPORT, "patient_id_hash": "b" * 64},
    ).json()
    stages = body["stage_receipts"]
    codes = [
        (s.get("error") or {}).get("code")
        for s in stages
        if "clinicalbert" in s["stage"]
    ]
    assert codes and codes[0] in {
        BELOW_FLOOR_CODE, VERSION_MISMATCH_CODE, UNVERIFIABLE_CODE
    }, codes


def test_her2_1plus_negative_never_becomes_positive_end_to_end(client):
    """The single most consequential invariant in this file."""
    body = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": AUDITED_REPORT, "patient_id_hash": "b" * 64},
    ).json()
    blob = repr(body).lower()
    assert '"her2_status": "positive"' not in blob.replace("'", '"')
    leaks = assert_no_parser_derived_values(body)
    assert leaks == [], f"parser-derived values leaked: {leaks}"


def test_ki67_20pct_never_becomes_67_end_to_end(client):
    body = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": AUDITED_REPORT, "patient_id_hash": "b" * 64},
    ).json()

    def hunt(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if "ki67" in str(k).lower() and v in (67, "67", 67.0):
                    return f"{k}={v!r}"
                got = hunt(v)
                if got:
                    return got
        elif isinstance(node, (list, tuple)):
            for v in node:
                got = hunt(v)
                if got:
                    return got
        return None

    assert hunt(body) is None, f"the 67 error survived: {hunt(body)}"


def test_treatment_output_proves_it_consumed_no_parser_value(client):
    """A therapy request carrying only the audited free text must not produce
    options, because every input it could use is parser-derived."""
    body = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": AUDITED_REPORT, "patient_id_hash": "b" * 64},
    ).json()
    assert body.get("dss_prognosis") in (None, {}, [])
    assert body.get("arbiter_score") is None
    assert body["provenance"]["model_state"] in {"unavailable", "placeholder"}, (
        body["provenance"]["model_state"]
    )


def test_no_regex_or_fusion_state_is_ever_reported(client):
    from oncology_arbiter.api.schemas import RETIRED_MODEL_STATE_VALUES

    body = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": AUDITED_REPORT, "patient_id_hash": "b" * 64},
    ).json()
    assert body["provenance"]["model_state"] not in RETIRED_MODEL_STATE_VALUES


def test_independently_completed_image_stage_may_remain_visible(client):
    """Directive 3 explicitly allows image stages to survive a parser refusal.
    A biopsy request with no image should therefore show a *skipped* phikon
    stage, not a suppressed one -- the refusal is scoped to the parser."""
    body = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": AUDITED_REPORT, "patient_id_hash": "b" * 64},
    ).json()
    stages = {s["stage"]: s["status"] for s in body["stage_receipts"]}
    assert "phikon_embedding" in stages
    assert stages["phikon_embedding"].startswith(("skipped", "succeeded", "failed_required"))


def test_parser_derived_field_set_covers_the_two_audited_errors():
    assert "her2_value" in PARSER_DERIVED_FIELDS
    assert "ki67_pct" in PARSER_DERIVED_FIELDS
    assert "her2_status" in PARSER_DERIVED_FIELDS
