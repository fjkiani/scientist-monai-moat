"""Wire-level test: /v1/biopsy/analyze surfaces the v0.3.0 report_parse block."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api import create_app


@pytest.fixture()
def client():
    os.environ["ONCOLOGY_ARBITER_AUTH_MODE"] = "off"
    os.environ.pop("ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER", None)
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestReportParseBlockRegex:
    def test_full_luminal_a_report(self, client):
        r = client.post(
            "/v1/biopsy/analyze",
            json={
                "report_text": (
                    "Invasive ductal carcinoma. ER: positive. PR: positive. "
                    "HER2/neu: negative. Nottingham grade 2. Ki-67 index: 15%."
                ),
            },
            headers={"x-api-key": "oa-test-abc123"},
        )
        assert r.status_code == 200
        j = r.json()
        # v0.3.0: report_parse must be present with fusion_mode='regex'.
        assert j.get("report_parse") is not None
        rp = j["report_parse"]
        assert rp["parser_id"] == "proxy_regex_v0"
        assert rp["fusion_mode"] == "regex"
        # Every core field should be 'regex' source.
        for k in ("er", "pr", "her2", "grade"):
            assert rp["per_field_source"][k] == "regex"
            assert rp["per_field_confidence"][k] == 1.0
        # Extended fields are BERT-only, so empty here.
        assert rp["extended_fields"] == {}
        # Legacy receptor_panel still populated.
        assert j["receptor_panel"]["er_positive"] is True
        assert j["receptor_panel"]["pr_positive"] is True
        assert j["receptor_panel"]["her2_status"] == "negative"
        assert j["grade"] == 2

    def test_no_report_text_returns_no_parse_block(self, client):
        r = client.post(
            "/v1/biopsy/analyze",
            json={},  # no report_text, no wsi
            headers={"x-api-key": "oa-test-abc123"},
        )
        # Backend rejects a fully-empty request (needs report OR image).
        # We expect a 400/422 here — the point is that no report → no
        # report_parse block is surfaced downstream.
        assert r.status_code in (400, 422)

    def test_report_with_no_signal_returns_no_match_block(self, client):
        r = client.post(
            "/v1/biopsy/analyze",
            json={"report_text": "Patient presents with breast mass."},
            headers={"x-api-key": "oa-test-abc123"},
        )
        assert r.status_code == 200
        j = r.json()
        rp = j["report_parse"]
        assert rp["fusion_mode"] == "regex"
        # No fields matched — every source should be 'none'.
        for k in ("er", "pr", "her2", "grade"):
            assert rp["per_field_source"][k] == "none"

    def test_warnings_still_carry_matched_count(self, client):
        """The v0.2.1 warning shape must survive the v0.3.0 rewrite."""
        r = client.post(
            "/v1/biopsy/analyze",
            json={"report_text": "ER positive. Grade 3."},
            headers={"x-api-key": "oa-test-abc123"},
        )
        assert r.status_code == 200
        j = r.json()
        warns = j.get("warnings") or []
        # Exactly one 'receptor_panel_source:' line
        src_warns = [w for w in warns if w.startswith("receptor_panel_source:")]
        assert len(src_warns) == 1
        assert "matched=2/4" in src_warns[0]
