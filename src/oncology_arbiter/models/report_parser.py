"""
Pathology report parser (proxy_regex_v0).

Purpose
-------
Extract a small, well-defined set of receptor / grade signals from a free-text
pathology report so the biopsy endpoint can populate the receptor panel it
today leaves empty. Downstream (therapy_reason, case_full) can then branch on
real values instead of silently defaulting to TNBC.

**This is a proxy heuristic, not a validated clinical NLP model.**
- It runs on the API side and its output MUST be surfaced to the user as
  parse-state pills that the user confirms (or overrides) before therapy is
  called. See PLAN.md v0.2.1 §1.2 for the honesty contract.
- It only extracts: ER, PR, HER2, Nottingham grade.
- It does NOT extract stage, size, Ki-67, node status, or any other field.
- Ambiguity ("HER2 2+", "ER equivocal") is preserved as a distinct state so
  the UI can flag it rather than silently coercing to a boolean.

Regex design notes
------------------
- Case-insensitive throughout (re.IGNORECASE).
- Matching is line- and phrase-scoped: patterns look forward at most ~40 chars
  after the receptor keyword, and we bail on the first \\n or ';' inside the
  window. This is to avoid picking up "ER negative" from "ER: positive.
  Neighboring lymph node: negative" style prose.
- Multiple mentions of the same field: last-match-wins. Reports often restate
  the summary at the bottom (`Impression:` / `Summary:` blocks), and the
  summary is closer to what the pathologist actually concluded than a mid-
  report snippet.
- HER2 has richer possible values because IHC scoring is standard practice:
    * 3+                 -> positive
    * 0, 1+              -> negative
    * 2+                 -> equivocal   (NCCN says reflex to FISH; we do NOT
                                         auto-upgrade — a proxy cannot know
                                         the FISH result)
    * "positive"         -> positive
    * "negative"         -> negative
    * "equivocal"        -> equivocal
    * bare "+" / "-"     -> DELIBERATELY NOT MATCHED (too ambiguous — could
                            mean IHC score or overall status; we make the
                            user confirm)

Return contract
---------------
`parse_pathology_report(text)` returns a `ParsedReport` with four
`ParseField[T]` entries. Every ParseField has:

    value: T | None            # None if no_match or ambiguous
    match_state: str           # matched | ambiguous | no_match
    matched_text: str | None   # the raw substring we matched on (for UI)
    span: tuple[int,int] | None  # (start, end) in the input for highlighting

The output is deliberately JSON-friendly (dataclasses.asdict works) so the
biopsy endpoint can put it straight into the response for the UI to render.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

__all__ = [
    "ParsedReport",
    "ParseField",
    "parse_pathology_report",
    "MATCH_STATE_MATCHED",
    "MATCH_STATE_AMBIGUOUS",
    "MATCH_STATE_NO_MATCH",
    "PARSER_ID",
]

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

PARSER_ID: str = "proxy_regex_v0"

MATCH_STATE_MATCHED: str = "matched"
MATCH_STATE_AMBIGUOUS: str = "ambiguous"
MATCH_STATE_NO_MATCH: str = "no_match"

MatchState = Literal["matched", "ambiguous", "no_match"]
Her2Value = Literal["positive", "negative", "equivocal"]


# ---------------------------------------------------------------------------
# Dataclasses (JSON-serializable via dataclasses.asdict)
# ---------------------------------------------------------------------------


@dataclass
class ParseField:
    """A single parsed field with provenance."""

    value: Optional[object] = None
    match_state: MatchState = MATCH_STATE_NO_MATCH
    matched_text: Optional[str] = None
    span: Optional[Tuple[int, int]] = None


@dataclass
class ParsedReport:
    """
    Aggregate result of parse_pathology_report.

    All four fields are always present with a well-defined match_state so the
    caller does not have to distinguish "missing" from "not applicable".
    """

    er: ParseField = field(default_factory=ParseField)
    pr: ParseField = field(default_factory=ParseField)
    her2: ParseField = field(default_factory=ParseField)
    grade: ParseField = field(default_factory=ParseField)
    parser_id: str = PARSER_ID


# ---------------------------------------------------------------------------
# Regex library
# ---------------------------------------------------------------------------
#
# All patterns are compiled with re.IGNORECASE. We use non-capturing groups
# for the keyword so `m.group(1)` is always the *value* substring.

# Keyword -> value separator: allow at most ~40 chars between keyword and
# value, provided the intervening chars are punctuation, whitespace, or short
# qualifier tokens ("status", "by IHC", "IHC score", "score of"). We reject
# any newline in the separator to keep matches inside a single pathology line
# — a "\n" between keyword and value almost always means we jumped topic.
#
# The value tokens themselves are terminated by a lookahead over any
# non-word / punctuation char (or end-of-string), rather than \b, because
# "\b" does NOT hold after a "+" in "3+" (both "+" and " " are non-word).
_SEP = (
    r"[^\n\S]*"                          # 0+ inline whitespace
    r"(?:"
    r"[:\-,]"                            # a punctuation separator
    r"|"
    r"(?:by\s+|for\s+|is\s+|score\s+of\s+)"  # qualifier prefix
    r"|"
    r"(?:status|expression|result|by\s+ihc|ihc(?:\s+score)?)"  # qualifier
    r"[^\n\S]*"
    r")*"
    r"[^\n\S]*"
)

_VALUE_END = r"(?=[\s.,;)\]\/]|$)"  # lookahead: whitespace, punctuation, EOL


# ER / PR
_ER_PATTERN = re.compile(
    r"\b(?:er|estrogen\s+receptor)\b"
    + _SEP
    + r"(positive|negative|neg|pos)"
    + _VALUE_END,
    re.IGNORECASE,
)

_PR_PATTERN = re.compile(
    r"\b(?:pr|progesterone\s+receptor)\b"
    + _SEP
    + r"(positive|negative|neg|pos)"
    + _VALUE_END,
    re.IGNORECASE,
)

# HER2 accepts a broader vocabulary (IHC scores + plain status).
_HER2_PATTERN = re.compile(
    r"\b(?:her[\s\-]?2(?:/neu)?|c[\s\-]?erb[\s\-]?b[\s\-]?2)\b"
    + _SEP
    + r"(3\+|2\+|1\+|positive|negative|equivocal|amplified|non[\s\-]?amplified|neg|pos|0)"
    + _VALUE_END,
    re.IGNORECASE,
)

# Nottingham grade. Accepts arabic (1-3), roman (I / II / III), and the
# "grade X of 3" phrasing pathologists sometimes use.
_GRADE_PATTERN = re.compile(
    r"\b(?:nottingham\s+)?(?:histologic\s+)?grade\b"
    + _SEP
    + r"([123]|i{1,3})(?:\s*(?:of|/)\s*(?:3|iii))?"
    + _VALUE_END,
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Value normalizers
# ---------------------------------------------------------------------------


def _normalize_bool_receptor(raw: str) -> Tuple[Optional[bool], MatchState]:
    """Map ER/PR raw values to a bool.

    'positive' / 'pos' -> True, matched
    'negative' / 'neg' -> False, matched
    Anything else (defensive) -> None, ambiguous
    """
    t = raw.strip().lower()
    if t in ("positive", "pos"):
        return True, MATCH_STATE_MATCHED
    if t in ("negative", "neg"):
        return False, MATCH_STATE_MATCHED
    return None, MATCH_STATE_AMBIGUOUS


def _normalize_her2(raw: str) -> Tuple[Optional[Her2Value], MatchState]:
    """Map HER2 raw values to positive/negative/equivocal.

    Special IHC-score rules:
        3+           -> positive, matched
        0 / 1+       -> negative, matched
        2+           -> equivocal, ambiguous  (needs FISH; proxy cannot decide)
        equivocal    -> equivocal, ambiguous
        positive     -> positive, matched
        negative     -> negative, matched
        amplified    -> positive, matched
        non-amplified-> negative, matched
    """
    t = raw.strip().lower().replace("  ", " ")
    # Score-based
    if t == "3+":
        return "positive", MATCH_STATE_MATCHED
    if t in ("0", "1+"):
        return "negative", MATCH_STATE_MATCHED
    if t == "2+":
        return "equivocal", MATCH_STATE_AMBIGUOUS
    # Status words
    if t in ("positive", "pos"):
        return "positive", MATCH_STATE_MATCHED
    if t in ("negative", "neg"):
        return "negative", MATCH_STATE_MATCHED
    if t == "equivocal":
        return "equivocal", MATCH_STATE_AMBIGUOUS
    if t in ("amplified",):
        return "positive", MATCH_STATE_MATCHED
    if t.replace("-", "").replace(" ", "") == "nonamplified":
        return "negative", MATCH_STATE_MATCHED
    return None, MATCH_STATE_AMBIGUOUS


_ROMAN_TO_INT = {"i": 1, "ii": 2, "iii": 3}


def _normalize_grade(raw: str) -> Tuple[Optional[int], MatchState]:
    """Map grade token to integer 1..3.

    Arabic: '1' | '2' | '3' -> matched.
    Roman:  'i' | 'ii' | 'iii' (case-insensitive) -> matched.
    Anything else -> None, ambiguous.
    """
    t = raw.strip().lower()
    if t in ("1", "2", "3"):
        return int(t), MATCH_STATE_MATCHED
    if t in _ROMAN_TO_INT:
        return _ROMAN_TO_INT[t], MATCH_STATE_MATCHED
    return None, MATCH_STATE_AMBIGUOUS


# ---------------------------------------------------------------------------
# Field extractor
# ---------------------------------------------------------------------------


def _extract(pattern: re.Pattern[str], text: str) -> Optional[re.Match[str]]:
    """Return the LAST match of `pattern` in `text`, or None.

    Pathology reports often restate the receptor result in a summary block at
    the bottom (`Impression:` etc.). The summary is what the pathologist
    concluded, so it should win over an in-body mention.
    """
    last: Optional[re.Match[str]] = None
    for m in pattern.finditer(text):
        last = m
    return last


def _build_field_receptor(
    match: Optional[re.Match[str]],
) -> ParseField:
    """Build a ParseField for ER or PR from a match object."""
    if match is None:
        return ParseField(
            value=None,
            match_state=MATCH_STATE_NO_MATCH,
            matched_text=None,
            span=None,
        )
    raw = match.group(1)
    value, state = _normalize_bool_receptor(raw)
    return ParseField(
        value=value,
        match_state=state,
        matched_text=match.group(0),
        span=match.span(),
    )


def _build_field_her2(match: Optional[re.Match[str]]) -> ParseField:
    if match is None:
        return ParseField(
            value=None,
            match_state=MATCH_STATE_NO_MATCH,
            matched_text=None,
            span=None,
        )
    raw = match.group(1)
    value, state = _normalize_her2(raw)
    return ParseField(
        value=value,
        match_state=state,
        matched_text=match.group(0),
        span=match.span(),
    )


def _build_field_grade(match: Optional[re.Match[str]]) -> ParseField:
    if match is None:
        return ParseField(
            value=None,
            match_state=MATCH_STATE_NO_MATCH,
            matched_text=None,
            span=None,
        )
    raw = match.group(1)
    value, state = _normalize_grade(raw)
    return ParseField(
        value=value,
        match_state=state,
        matched_text=match.group(0),
        span=match.span(),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_pathology_report(text: Optional[str]) -> ParsedReport:
    """Parse `text` and return a ParsedReport.

    Behaviour on edge inputs:
        None / empty / whitespace-only  -> all four fields = no_match.

    The function is total: it never raises for any string input. Callers can
    trust the returned object shape.
    """
    if not text or not text.strip():
        return ParsedReport()

    return ParsedReport(
        er=_build_field_receptor(_extract(_ER_PATTERN, text)),
        pr=_build_field_receptor(_extract(_PR_PATTERN, text)),
        her2=_build_field_her2(_extract(_HER2_PATTERN, text)),
        grade=_build_field_grade(_extract(_GRADE_PATTERN, text)),
    )
