"""Wire-level test: /v1/biopsy/analyze and the retired regex report parser.

The three `TestReportParseBlockRegex` cases that lived here asserted
`report_parse.parser_id == "proxy_regex_v0"` and `fusion_mode == "regex"` --
i.e. they pinned the regex pathology parser as a *successful* parse source.
That capability is retired: ClinicalBERT is the only report-parsing backend,
and when it is unavailable the stage fails rather than degrading to regex.
Those cases are migrated (not silently dropped) in
artifacts/audit/retired_test_migration.json; their replacements are
tests/unit/test_production_specialist_contracts.py::
  test_biopsy_source_calls_only_clinicalbert_for_report_parsing
  test_biopsy_clinicalbert_failure_has_no_regex_fallback

What remains here is the one case that was never about the regex parser: an
empty request must be rejected outright.
"""

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


class TestBiopsyRequestShape:
    def test_no_report_text_returns_no_parse_block(self, client):
        r = client.post(
            "/v1/biopsy/analyze",
            json={},  # no report_text, no wsi
            headers={"x-api-key": "oa-test-abc123"},
        )
        # Backend rejects a fully-empty request (needs report OR image).
        # We expect a 400/422 here -- the point is that no report -> no
        # report_parse block is surfaced downstream.
        assert r.status_code in (400, 422)

    def test_unparseable_report_yields_no_parse_block_and_a_failed_stage(self, client):
        """The retired path returned a regex block here; the live path fails."""
        r = client.post(
            "/v1/biopsy/analyze",
            json={
                "report_text": (
                    "Invasive ductal carcinoma. ER: positive. PR: positive. "
                    "HER2/neu: negative. Nottingham grade 2. Ki-67 index: 15%."
                )
            },
            headers={"x-api-key": "oa-test-abc123"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["report_parse"] is None
        assert body["pipeline_status"] == "failed_required_stage"
        receipt = next(
            x for x in body["stage_receipts"]
            if x["stage"] == "clinicalbert_pathology_parse"
        )
        assert receipt["status"] == "failed_required"
        # No regex-sourced receptor values may appear despite the report text
        # containing "ER: positive", "PR: positive", "HER2/neu: negative".
        assert body["grade"] is None
        panel = body.get("receptor_panel")
        if panel is not None:
            assert panel.get("er_positive") is None
            assert panel.get("pr_positive") is None
            assert panel.get("her2_status") is None
        assert not [
            w for w in (body.get("warnings") or [])
            if w.startswith("receptor_panel_source:")
        ]
