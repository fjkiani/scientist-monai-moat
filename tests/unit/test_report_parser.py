"""
Unit tests for src/oncology_arbiter/models/report_parser.py

Grouping (kept intentional so a diff-in-a-single-field is easy to spot):

- ParsedReport contract & totality  (5 tests)
- ER field                          (6 tests)
- PR field                          (5 tests)
- HER2 field                        (10 tests)
- Grade field                       (7 tests)
- Multi-field / summary-wins        (4 tests)
- JSON serialization                (2 tests)

Total: 39 tests. All are pure-Python string parsing; no I/O, no network.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from oncology_arbiter.models.report_parser import (
    MATCH_STATE_AMBIGUOUS,
    MATCH_STATE_MATCHED,
    MATCH_STATE_NO_MATCH,
    PARSER_ID,
    ParsedReport,
    ParseField,
    parse_pathology_report,
)


# ---------------------------------------------------------------------------
# ParsedReport contract & totality
# ---------------------------------------------------------------------------


class TestReportContract:
    def test_empty_string_returns_all_no_match(self):
        r = parse_pathology_report("")
        assert r.er.match_state == MATCH_STATE_NO_MATCH
        assert r.pr.match_state == MATCH_STATE_NO_MATCH
        assert r.her2.match_state == MATCH_STATE_NO_MATCH
        assert r.grade.match_state == MATCH_STATE_NO_MATCH

    def test_none_input_returns_all_no_match(self):
        r = parse_pathology_report(None)
        assert r.er.match_state == MATCH_STATE_NO_MATCH
        assert r.pr.match_state == MATCH_STATE_NO_MATCH
        assert r.her2.match_state == MATCH_STATE_NO_MATCH
        assert r.grade.match_state == MATCH_STATE_NO_MATCH

    def test_whitespace_only_returns_all_no_match(self):
        r = parse_pathology_report("   \n\n\t  ")
        for f in (r.er, r.pr, r.her2, r.grade):
            assert f.match_state == MATCH_STATE_NO_MATCH
            assert f.value is None
            assert f.matched_text is None
            assert f.span is None

    def test_parser_id_is_proxy_regex_v0(self):
        r = parse_pathology_report("ER: positive.")
        assert r.parser_id == PARSER_ID == "proxy_regex_v0"

    def test_parsed_report_is_dataclass_asdict_safe(self):
        r = parse_pathology_report("ER positive. HER2 3+.")
        d = asdict(r)
        assert set(d.keys()) == {"er", "pr", "her2", "grade", "parser_id"}
        assert set(d["er"].keys()) == {"value", "match_state", "matched_text", "span"}


# ---------------------------------------------------------------------------
# ER field
# ---------------------------------------------------------------------------


class TestER:
    def test_er_positive_colon(self):
        r = parse_pathology_report("Estrogen Receptor: Positive (95%).")
        assert r.er.value is True
        assert r.er.match_state == MATCH_STATE_MATCHED

    def test_er_negative_short(self):
        r = parse_pathology_report("ER: negative.")
        assert r.er.value is False
        assert r.er.match_state == MATCH_STATE_MATCHED

    def test_er_pos_abbrev(self):
        r = parse_pathology_report("ER pos, PR pos, HER2 neg.")
        assert r.er.value is True
        assert r.er.match_state == MATCH_STATE_MATCHED

    def test_er_neg_abbrev(self):
        r = parse_pathology_report("ER neg.")
        assert r.er.value is False

    def test_er_missing_returns_no_match(self):
        r = parse_pathology_report(
            "Invasive ductal carcinoma. HER2 negative. Grade 2."
        )
        assert r.er.match_state == MATCH_STATE_NO_MATCH
        assert r.er.value is None

    def test_er_case_insensitive(self):
        r = parse_pathology_report("estrogen RECEPTOR POSITIVE")
        assert r.er.value is True


# ---------------------------------------------------------------------------
# PR field
# ---------------------------------------------------------------------------


class TestPR:
    def test_pr_positive(self):
        r = parse_pathology_report("Progesterone Receptor: Positive (80%).")
        assert r.pr.value is True
        assert r.pr.match_state == MATCH_STATE_MATCHED

    def test_pr_negative(self):
        r = parse_pathology_report("PR: negative")
        assert r.pr.value is False
        assert r.pr.match_state == MATCH_STATE_MATCHED

    def test_pr_missing(self):
        r = parse_pathology_report("ER positive. HER2 negative.")
        assert r.pr.match_state == MATCH_STATE_NO_MATCH

    def test_pr_abbrev_pos(self):
        r = parse_pathology_report("PR pos.")
        assert r.pr.value is True

    def test_pr_not_confused_by_pr_in_word(self):
        # 'prognosis' contains 'pr' — must NOT match because it's not a word boundary.
        r = parse_pathology_report("Prognosis: uncertain.")
        assert r.pr.match_state == MATCH_STATE_NO_MATCH


# ---------------------------------------------------------------------------
# HER2 field
# ---------------------------------------------------------------------------


class TestHER2:
    def test_her2_positive_word(self):
        r = parse_pathology_report("HER2: positive.")
        assert r.her2.value == "positive"
        assert r.her2.match_state == MATCH_STATE_MATCHED

    def test_her2_negative_word(self):
        r = parse_pathology_report("HER2: negative.")
        assert r.her2.value == "negative"
        assert r.her2.match_state == MATCH_STATE_MATCHED

    def test_her2_ihc_3plus_is_positive(self):
        r = parse_pathology_report("HER2 3+ by IHC.")
        assert r.her2.value == "positive"
        assert r.her2.match_state == MATCH_STATE_MATCHED

    def test_her2_ihc_0_is_negative(self):
        r = parse_pathology_report("HER2 IHC 0.")
        assert r.her2.value == "negative"
        assert r.her2.match_state == MATCH_STATE_MATCHED

    def test_her2_ihc_1plus_is_negative(self):
        r = parse_pathology_report("HER2/neu: 1+ (negative).")
        # Last-match-wins picks the "negative" word after the score.
        assert r.her2.value == "negative"

    def test_her2_ihc_2plus_is_equivocal_and_ambiguous(self):
        r = parse_pathology_report("HER2 2+.")
        assert r.her2.value == "equivocal"
        assert r.her2.match_state == MATCH_STATE_AMBIGUOUS

    def test_her2_equivocal_word_is_ambiguous(self):
        r = parse_pathology_report("HER2 equivocal, pending FISH.")
        assert r.her2.value == "equivocal"
        assert r.her2.match_state == MATCH_STATE_AMBIGUOUS

    def test_her2_missing(self):
        r = parse_pathology_report("ER positive. PR positive. Grade 2.")
        assert r.her2.match_state == MATCH_STATE_NO_MATCH

    def test_her2_variant_spelling_her_2_neu(self):
        r = parse_pathology_report("HER-2/neu: positive.")
        assert r.her2.value == "positive"

    def test_her2_amplified_is_positive(self):
        r = parse_pathology_report("HER2 amplified by FISH.")
        assert r.her2.value == "positive"
        assert r.her2.match_state == MATCH_STATE_MATCHED


# ---------------------------------------------------------------------------
# Grade field
# ---------------------------------------------------------------------------


class TestGrade:
    def test_grade_arabic_2(self):
        r = parse_pathology_report("Nottingham Grade: 2.")
        assert r.grade.value == 2
        assert r.grade.match_state == MATCH_STATE_MATCHED

    def test_grade_arabic_3(self):
        r = parse_pathology_report("Histologic grade: 3.")
        assert r.grade.value == 3

    def test_grade_roman_ii(self):
        r = parse_pathology_report("Grade II.")
        assert r.grade.value == 2
        assert r.grade.match_state == MATCH_STATE_MATCHED

    def test_grade_roman_iii(self):
        r = parse_pathology_report("Nottingham grade III of III.")
        assert r.grade.value == 3

    def test_grade_x_of_3_phrasing(self):
        r = parse_pathology_report("Grade 2 of 3.")
        assert r.grade.value == 2

    def test_grade_missing(self):
        r = parse_pathology_report("ER positive. HER2 negative.")
        assert r.grade.match_state == MATCH_STATE_NO_MATCH

    def test_grade_out_of_range_ignored(self):
        # "Grade 5" is not a valid Nottingham grade; regex should not match.
        r = parse_pathology_report("Grade 5.")
        assert r.grade.match_state == MATCH_STATE_NO_MATCH


# ---------------------------------------------------------------------------
# Multi-field / summary-wins behaviour
# ---------------------------------------------------------------------------


class TestMultiField:
    def test_luminal_a_demo_case(self):
        """The exact canned demo example we ship for the tumor board."""
        demo = (
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
        r = parse_pathology_report(demo)
        assert r.er.value is True and r.er.match_state == MATCH_STATE_MATCHED
        assert r.pr.value is True and r.pr.match_state == MATCH_STATE_MATCHED
        assert r.her2.value == "negative" and r.her2.match_state == MATCH_STATE_MATCHED
        assert r.grade.value == 2 and r.grade.match_state == MATCH_STATE_MATCHED

    def test_triple_negative_case(self):
        text = (
            "Invasive ductal carcinoma. ER negative. PR negative. "
            "HER2 0 by IHC. Grade 3."
        )
        r = parse_pathology_report(text)
        assert r.er.value is False
        assert r.pr.value is False
        assert r.her2.value == "negative"
        assert r.grade.value == 3

    def test_her2_positive_case(self):
        text = "ER positive. PR negative. HER2 3+. Grade 3."
        r = parse_pathology_report(text)
        assert r.er.value is True
        assert r.pr.value is False
        assert r.her2.value == "positive"

    def test_last_summary_mention_wins(self):
        # Body says "positive"; summary at end says "negative" — final report.
        text = (
            "Preliminary review: HER2 positive by IHC.\n"
            "Impression: On further review with FISH, HER2 negative.\n"
        )
        r = parse_pathology_report(text)
        assert r.her2.value == "negative"


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_asdict_roundtrips_through_json(self):
        r = parse_pathology_report("ER positive. HER2 3+. Grade II.")
        d = asdict(r)
        # spans are tuples; json.dumps will emit lists. That's fine for the
        # over-the-wire contract because the UI just uses them for highlight.
        s = json.dumps(d, default=str)
        back = json.loads(s)
        assert back["er"]["value"] is True
        assert back["her2"]["value"] == "positive"
        assert back["grade"]["value"] == 2

    def test_span_is_tuple_of_two_ints(self):
        r = parse_pathology_report("ER: positive.")
        assert r.er.span is not None
        assert isinstance(r.er.span, tuple)
        assert len(r.er.span) == 2
        assert all(isinstance(x, int) for x in r.er.span)
