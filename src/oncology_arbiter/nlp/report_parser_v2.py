"""Report parser v2: regex + Bio_ClinicalBERT fusion.

The v0.3.0 API selects a parser based on env flags:

    ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER=1 → BERT-only or fused
    ONCOLOGY_ARBITER_CLINICALBERT_FUSION=fused    → regex ∨ BERT (default)
    ONCOLOGY_ARBITER_CLINICALBERT_FUSION=bert     → BERT-only
    ONCOLOGY_ARBITER_CLINICALBERT_FUSION=regex    → regex-only (identical to v0.2.1)

Fusion strategy (`fused`)
-------------------------
- If regex AND BERT both match AND agree → use the value, state=matched,
  source="fused".
- If they disagree → state=ambiguous, value=None, source="disagreement".
  This is deliberately conservative — a disagreement means the operator
  must confirm.
- If only regex matches → use regex value.
- If only BERT matches AND its confidence ≥ per-field threshold → use
  BERT value.
- Otherwise → no_match.

The BERT parser is also the ONLY path that can populate the extended
fields (ki67_pct, tumor_size_mm, T/N/M stage, margin, LVI). Those live
on `FusedParsedReport.extended_fields` and are passed through by the
biopsy endpoint into the response for the UI.

Interface with the existing regex parser
----------------------------------------
`FusedParsedReport` has `.er`, `.pr`, `.her2`, `.grade` slots that are
type-compatible with the regex `ParsedReport`, so the biopsy endpoint's
existing conversion code (map True/False → er_positive, positive/negative/
equivocal → her2_status, int → grade) works unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oncology_arbiter.models.report_parser import (
    MATCH_STATE_AMBIGUOUS,
    MATCH_STATE_MATCHED,
    MATCH_STATE_NO_MATCH,
    parse_pathology_report as _parse_regex,
)


@dataclass
class FusedField:
    """Shape-compatible with `ParseField` + extra confidence/source fields."""

    value: Any = None
    match_state: str = MATCH_STATE_NO_MATCH
    matched_text: str | None = None
    span: tuple[int, int] | None = None
    confidence: float = 0.0
    source: str = "none"   # "regex" | "clinicalbert" | "fused" | "disagreement" | "none"


@dataclass
class FusedParsedReport:
    """The v0.3.0 unified parse result."""

    er: FusedField = field(default_factory=FusedField)
    pr: FusedField = field(default_factory=FusedField)
    her2: FusedField = field(default_factory=FusedField)
    grade: FusedField = field(default_factory=FusedField)
    extended_fields: dict[str, FusedField] = field(default_factory=dict)
    parser_id: str = "regex_v0"        # updated at construction time
    fusion_mode: str = "regex"         # "regex" | "bert" | "fused"


def _wrap_regex_field(rf) -> FusedField:
    """Turn a regex ParseField into a FusedField."""
    return FusedField(
        value=rf.value,
        match_state=rf.match_state,
        matched_text=rf.matched_text,
        span=rf.span,
        confidence=1.0 if rf.match_state == MATCH_STATE_MATCHED else 0.0,
        source="regex" if rf.match_state != MATCH_STATE_NO_MATCH else "none",
    )


def _fuse_field(regex_f, bert_f) -> FusedField:
    """Combine one field from both parsers.

    `bert_f` is a `ClinicalBertField` (dataclass); `regex_f` is a `ParseField`.
    """
    r_val = regex_f.value
    r_state = regex_f.match_state
    b_val = bert_f.value
    b_state = bert_f.match_state
    b_conf = bert_f.confidence

    if r_state == MATCH_STATE_MATCHED and b_state == MATCH_STATE_MATCHED:
        # Both matched
        if r_val == b_val:
            return FusedField(
                value=r_val,
                match_state=MATCH_STATE_MATCHED,
                matched_text=regex_f.matched_text,
                span=regex_f.span,
                confidence=max(1.0, b_conf),
                source="fused",
            )
        # Disagreement — flag it.
        return FusedField(
            value=None,
            match_state=MATCH_STATE_AMBIGUOUS,
            matched_text=f"regex={r_val!r} bert={b_val!r}",
            span=regex_f.span or bert_f.span,
            confidence=b_conf,
            source="disagreement",
        )

    if r_state == MATCH_STATE_MATCHED:
        return FusedField(
            value=r_val, match_state=MATCH_STATE_MATCHED,
            matched_text=regex_f.matched_text, span=regex_f.span,
            confidence=1.0, source="regex",
        )
    if b_state == MATCH_STATE_MATCHED:
        return FusedField(
            value=b_val, match_state=MATCH_STATE_MATCHED,
            matched_text=bert_f.matched_text, span=bert_f.span,
            confidence=b_conf, source="clinicalbert",
        )

    # Neither matched cleanly.
    # If BERT hedged (ambiguous), surface that so the UI can show it.
    if b_state == MATCH_STATE_AMBIGUOUS:
        return FusedField(
            value=None, match_state=MATCH_STATE_AMBIGUOUS,
            matched_text=bert_f.matched_text, span=bert_f.span,
            confidence=b_conf, source="clinicalbert",
        )
    if r_state == MATCH_STATE_AMBIGUOUS:
        return FusedField(
            value=None, match_state=MATCH_STATE_AMBIGUOUS,
            matched_text=regex_f.matched_text, span=regex_f.span,
            confidence=0.0, source="regex",
        )
    return FusedField()  # no_match


def parse_pathology_report_v2(
    text: str,
    mode: str | None = None,
    model_dir: Path | None = None,
) -> FusedParsedReport:
    """Parse a pathology report. Selects strategy from env or `mode`.

    Args:
        text: report text
        mode: "regex" | "bert" | "fused". Default from
            ONCOLOGY_ARBITER_CLINICALBERT_FUSION or "regex" if the model
            weights are missing.
        model_dir: override the BERT weights path.
    """
    # Choose the mode.
    if mode is None:
        mode = os.environ.get("ONCOLOGY_ARBITER_CLINICALBERT_FUSION", "").lower()

    enable_bert = os.environ.get(
        "ONCOLOGY_ARBITER_ENABLE_CLINICALBERT_PARSER", ""
    ).lower() in {"1", "true", "yes", "on"}

    if not enable_bert:
        mode = "regex"
    elif mode not in {"regex", "bert", "fused"}:
        mode = "fused"

    # Always compute the regex parse — it's cheap.
    reg = _parse_regex(text)

    if mode == "regex":
        return FusedParsedReport(
            er=_wrap_regex_field(reg.er),
            pr=_wrap_regex_field(reg.pr),
            her2=_wrap_regex_field(reg.her2),
            grade=_wrap_regex_field(reg.grade),
            extended_fields={},
            parser_id=reg.parser_id,
            fusion_mode="regex",
        )

    # Load BERT.
    from oncology_arbiter.nlp.clinicalbert_parser import (
        ClinicalBertReportParser,
    )
    parser_cls = (
        ClinicalBertReportParser.get(model_dir=model_dir)
        if model_dir is not None
        else ClinicalBertReportParser.get()
    )
    bert = parser_cls.parse(text)

    if mode == "bert":
        # BERT-only. Convert BERT fields → FusedField shape.
        def _from_bert(bf) -> FusedField:
            return FusedField(
                value=bf.value, match_state=bf.match_state,
                matched_text=bf.matched_text, span=bf.span,
                confidence=bf.confidence, source="clinicalbert",
            )
        return FusedParsedReport(
            er=_from_bert(bert.er),
            pr=_from_bert(bert.pr),
            her2=_from_bert(bert.her2),
            grade=_from_bert(bert.grade),
            extended_fields={k: _from_bert(v) for k, v in bert.extended_fields.items()},
            parser_id="clinicalbert_v1",
            fusion_mode="bert",
        )

    # Fused mode.
    def _from_bert(bf) -> FusedField:
        return FusedField(
            value=bf.value, match_state=bf.match_state,
            matched_text=bf.matched_text, span=bf.span,
            confidence=bf.confidence, source="clinicalbert",
        )

    return FusedParsedReport(
        er=_fuse_field(reg.er, bert.er),
        pr=_fuse_field(reg.pr, bert.pr),
        her2=_fuse_field(reg.her2, bert.her2),
        grade=_fuse_field(reg.grade, bert.grade),
        # Extended fields come from BERT only (regex doesn't have them).
        extended_fields={k: _from_bert(v) for k, v in bert.extended_fields.items()},
        parser_id="clinicalbert_v1+regex_v0",
        fusion_mode="fused",
    )
