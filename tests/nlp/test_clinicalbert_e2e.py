"""End-to-end tests for the trained Bio_ClinicalBERT report parser (v1).

These tests load the actual 430 MB trained checkpoint and run real
inference against representative test-set report styles. They're
marked ``@pytest.mark.models`` so ``pytest -m "not models"`` skips them
in fast dev loops.

Weights resolution follows ``ClinicalBertReportParser._resolve_default_model_dir``:
  1. env ``ONCOLOGY_ARBITER_CLINICALBERT_DIR``
  2. ``/workspace/models/report_parser_clinicalbert_v1``
  3. ``/mnt/results/models/report_parser_clinicalbert_v1``

Any of those with ``label_map.json`` counts; if none present, the test
module skips at import time with a clear message.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


# -------------------------------------------------------------------- #
# Skip the whole module unless weights are reachable via one of the
# three resolution paths.  This mirrors the parser's own resolver so a
# CI without weights just skips cleanly.
# -------------------------------------------------------------------- #
def _has_weights() -> bool:
    env = os.environ.get("ONCOLOGY_ARBITER_CLINICALBERT_DIR")
    candidates = [
        Path(env) if env else None,
        Path("/workspace/models/report_parser_clinicalbert_v1"),
        Path("/mnt/results/models/report_parser_clinicalbert_v1"),
    ]
    return any(
        c is not None and (c / "label_map.json").exists()
        for c in candidates
    )


pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        not _has_weights(),
        reason="ClinicalBERT v1 weights unavailable (checked env + /workspace + /mnt/results)",
    ),
]


# -------------------------------------------------------------------- #
# Fixture: singleton parser (loading is ~2s, run once per module)
# -------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bert_parser():
    from oncology_arbiter.nlp.clinicalbert_parser import ClinicalBertReportParser
    return ClinicalBertReportParser.get()


# -------------------------------------------------------------------- #
# Tests
# -------------------------------------------------------------------- #
class TestLoadsWeights:
    """The trained weights load and produce v1 provenance."""

    def test_parser_id_and_provenance(self, bert_parser):
        r = bert_parser.parse("ER positive. PR positive. HER2 3+. Grade 2.")
        d = r.as_dict()
        assert d["parser_id"] == "clinicalbert_v1"
        assert d["provenance"] == "SYNTHETIC-v0.3.0"

    def test_default_resolver_picks_a_real_dir(self):
        """The module-level _DEFAULT_MODEL_DIR must resolve to a
        directory that actually contains label_map.json."""
        from oncology_arbiter.nlp.clinicalbert_parser import _DEFAULT_MODEL_DIR
        assert (_DEFAULT_MODEL_DIR / "label_map.json").exists(), (
            f"Resolver picked {_DEFAULT_MODEL_DIR} but no label_map.json there"
        )


class TestBeatsRegexOnAmbiguousPhrasing:
    """Fusion should rescue reports where the regex parser fails but
    the model recognizes the phrasing. These specific phrases live in
    the training corpus and are exactly what v1 was trained to catch."""

    @pytest.mark.parametrize(
        "text, field, expected",
        [
            # ER: "strong nuclear positivity" — regex misses (no "positive")
            (
                "ER: strong nuclear positivity. PR: positive. HER2: 3+. Grade 2.",
                "er",
                True,
            ),
            # PR: "moderate to strong" — regex misses
            (
                "ER: positive. PR: moderate to strong. HER2: 1+. Grade 2.",
                "pr",
                True,
            ),
            # PR: "no staining" — regex misses (needs "negative")
            (
                "ER: negative. PR: no staining. HER2: 0. Grade 1.",
                "pr",
                False,
            ),
        ],
    )
    def test_fused_mode_rescues_regex(self, text, field, expected):
        with patch.dict(
            os.environ,
            {
                "ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER": "1",
                "ONCOLOGY_ARBITER_CLINICALBERT_FUSION": "fused",
            },
            clear=False,
        ):
            from oncology_arbiter.nlp.report_parser_v2 import (
                parse_pathology_report_v2,
            )
            r = parse_pathology_report_v2(text)
        assert r.fusion_mode == "fused"
        assert getattr(r, field).value is expected, (
            f"Expected {field}={expected}, got {getattr(r, field).value} "
            f"(source={getattr(r, field).source})"
        )
        # Fusion must attribute this to BERT since regex couldn't extract it
        assert getattr(r, field).source in {"clinicalbert", "fused"}


class TestStageAndExtendedFields:
    """The BERT layer is the ONLY path to extended fields (t/n/m stage,
    tumor size, ki67, margin, LVI). Fusion must expose them under
    `extended_fields` when BERT is enabled."""

    def test_extended_fields_populated(self):
        # Use corpus-style phrasing.  v1 was trained on templates that
        # include a "Pathologic stage:" header — off-template inputs
        # like a bare "pT2 pN1 pM0" at end-of-sentence sometimes get
        # the T/N/M prefix trimmed off; see model card §"Known failure
        # modes" for details.
        text = (
            "Invasive ductal carcinoma. ER: positive. PR: positive. HER2: 0. "
            "Nottingham grade 2. Ki-67: 15%. Tumor size: 12 mm. "
            "Pathologic stage: pT2, N1, M0. Margins negative. LVI absent."
        )
        with patch.dict(
            os.environ,
            {
                "ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER": "1",
                "ONCOLOGY_ARBITER_CLINICALBERT_FUSION": "fused",
            },
            clear=False,
        ):
            from oncology_arbiter.nlp.report_parser_v2 import (
                parse_pathology_report_v2,
            )
            r = parse_pathology_report_v2(text)

        assert r.fusion_mode == "fused"
        ext = r.extended_fields
        # Stage extraction: model recognizes T2/N1/M0 spans; value is
        # canonicalized by _stage() as uppercase "T2"/"N1"/"M0".  Allow
        # `ambiguous` state because stage confidence sometimes sits below
        # the 0.6 threshold (see model card strict-view section).
        assert "t_stage" in ext
        assert ext["t_stage"].match_state in {"matched", "ambiguous"}
        assert "n_stage" in ext
        assert "m_stage" in ext
        # Ki-67 is a high-confidence field (val_acc = 1.0 in v1).
        assert "ki67_pct" in ext and ext["ki67_pct"].value == 15
        # Margin/LVI recall is only 0.65 / 0.41 in v1 (see model card),
        # so we only assert the field slot exists — value may be None.
        assert "margin" in ext
        assert "lvi" in ext


class TestEdgeInputsDoNotCrash:
    """The fused pipeline must handle empty / whitespace / very-short
    inputs without raising.  These are the shapes the API receives when
    a user pastes fragments or the frontend sends a partial submit."""

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "no meaningful content",
            "ER positive.",  # only one field
            "focal weak positivity, clinical significance uncertain",  # phrase-only
        ],
    )
    def test_fused_handles_degenerate_input(self, text):
        with patch.dict(
            os.environ,
            {
                "ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER": "1",
                "ONCOLOGY_ARBITER_CLINICALBERT_FUSION": "fused",
            },
            clear=False,
        ):
            from oncology_arbiter.nlp.report_parser_v2 import (
                parse_pathology_report_v2,
            )
            r = parse_pathology_report_v2(text)
        # Must return a FusedParsedReport in fused mode, never raise.
        assert r.fusion_mode == "fused"
        # Every field slot must carry a valid source (never crash into
        # a None value; the design says value=None + source="none" for
        # unset).
        for field_name in ("er", "pr", "her2", "grade"):
            f = getattr(r, field_name)
            assert f.source in {"regex", "clinicalbert", "fused", "disagreement", "none"}
            assert f.match_state in {"matched", "ambiguous", "no_match"}
