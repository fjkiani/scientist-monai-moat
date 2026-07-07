"""Unit tests for the v0.3.0 fused parser.

The BERT weights ship separately and are gated by env — these tests
only exercise the regex-only path (identical to v0.2.1 wire) and the
fallback behaviour when BERT is disabled. Fused/BERT-mode integration
tests live in `test_clinicalbert_e2e.py` and are marked
`slow_ml` so `pytest -m "not slow_ml"` skips them.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from oncology_arbiter.nlp.report_parser_v2 import (
    FusedParsedReport,
    parse_pathology_report_v2,
)


# --------------------------------------------------------------------------- #
# Regex-only default (BERT env unset)


class TestRegexOnlyDefault:
    """When ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER is unset, we MUST
    fall back to the v0.2.1 regex parser — this is the safety floor."""

    def test_er_pr_her2_grade_matched(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER", None)
            r = parse_pathology_report_v2(
                "ER positive. PR negative. HER2 3+. Nottingham grade 2."
            )
        assert isinstance(r, FusedParsedReport)
        assert r.fusion_mode == "regex"
        assert r.parser_id == "proxy_regex_v0"
        assert r.er.value is True and r.er.match_state == "matched"
        assert r.pr.value is False and r.pr.match_state == "matched"
        assert r.her2.value == "positive" and r.her2.match_state == "matched"
        assert r.grade.value == 2 and r.grade.match_state == "matched"
        assert r.er.source == "regex"
        assert r.extended_fields == {}

    def test_no_match_when_no_signal(self):
        """A report that mentions nothing recognisable → all fields no_match."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER", None)
            r = parse_pathology_report_v2("Patient presents with breast mass.")
        assert r.fusion_mode == "regex"
        assert r.er.match_state == "no_match"
        assert r.pr.match_state == "no_match"
        assert r.her2.match_state == "no_match"
        assert r.grade.match_state == "no_match"

    def test_her2_2plus_is_ambiguous(self):
        """HER2 2+ is equivocal by IHC (needs FISH). Regex must flag."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER", None)
            r = parse_pathology_report_v2("HER2 2+")
        assert r.her2.value == "equivocal"
        assert r.her2.match_state == "ambiguous"


class TestBertEnvDisabled:
    """If the env flag is set to 0/false, the fused parser must NOT try to
    load BERT weights — otherwise a mis-flag on the server crashes biopsy."""

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_various_falsey_values_stay_regex(self, val):
        with patch.dict(os.environ,
                        {"ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER": val}):
            r = parse_pathology_report_v2("ER positive.")
        assert r.fusion_mode == "regex"
        assert r.er.value is True
        # No BERT was consulted so extended_fields must be empty.
        assert r.extended_fields == {}


class TestExplicitMode:
    """Callers can pass mode="regex" to force regex regardless of env."""

    def test_forced_regex_ignores_env(self):
        with patch.dict(os.environ,
                        {"ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER": "1",
                         "ONCOLOGY_ARBITER_CLINICALBERT_FUSION": "fused"}):
            r = parse_pathology_report_v2("ER positive.", mode="regex")
        assert r.fusion_mode == "regex"
        assert r.er.value is True

    def test_confidence_is_1_for_regex_match(self):
        r = parse_pathology_report_v2("ER positive.", mode="regex")
        assert r.er.confidence == 1.0
        assert r.er.source == "regex"

    def test_confidence_is_0_for_regex_no_match(self):
        r = parse_pathology_report_v2("Nothing here.", mode="regex")
        assert r.er.confidence == 0.0
        assert r.er.source == "none"
