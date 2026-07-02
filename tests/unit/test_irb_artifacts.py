"""Verifies the three IRB-readiness artifacts under artifacts/reports/.

The artifacts are institution-agnostic templates with bracketed placeholders.
These tests verify structural completeness (all required sections present),
correct citation of the honesty constants from src/oncology_arbiter/__init__.py,
and — for the SQL ledger — that the schema actually parses in SQLite and has
all the required columns.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

import oncology_arbiter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = REPO_ROOT / "artifacts" / "reports"
PROTOCOL = REPORTS / "irb_protocol_template.md"
LEDGER_SQL = REPORTS / "ai_prediction_ledger_schema.sql"
CONSENT = REPORTS / "informed_consent_template.md"


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def protocol_text() -> str:
    if not PROTOCOL.exists():
        pytest.fail(f"IRB protocol not found: {PROTOCOL}")
    return PROTOCOL.read_text()


REQUIRED_PROTOCOL_SECTIONS = [
    r"^## 1\. Background & Significance",
    r"^## 2\. Specific Aims",
    r"^## 3\. Study Population",
    r"^## 4\. Data Sources & Data Use Agreements",
    r"^## 5\. Model Description",
    r"^## 6\. Statistical Analysis Plan",
    r"^## 7\. Risk / Benefit Assessment",
    r"^## 8\. Informed Consent Process",
    r"^## 9\. Data Security & HIPAA",
    r"^## 10\. Adverse Event & Deviation Reporting",
    r"^## 11\. References",
]


@pytest.mark.parametrize("section_re", REQUIRED_PROTOCOL_SECTIONS)
def test_protocol_has_required_section(protocol_text: str, section_re: str) -> None:
    """Every one of the 11 required protocol sections must be present."""
    assert re.search(section_re, protocol_text, re.MULTILINE), (
        f"Protocol missing required section: {section_re}"
    )


def test_protocol_references_ruo_disclaimer(protocol_text: str) -> None:
    """The protocol MUST cite the RUO_DISCLAIMER symbol (not merely the phrase).

    This is a structural link: institutions using this template are put on notice
    that the exact language is imported from src/oncology_arbiter/__init__.py.
    """
    assert "RUO_DISCLAIMER" in protocol_text, (
        "Protocol must reference the RUO_DISCLAIMER symbol from oncology_arbiter"
    )


def test_protocol_references_auroc_caveat(protocol_text: str) -> None:
    """Similarly, the AUROC_CAVEAT symbol must be cited."""
    assert "AUROC_CAVEAT" in protocol_text, (
        "Protocol must reference the AUROC_CAVEAT symbol from oncology_arbiter"
    )


def test_protocol_cites_hubbard_2011_with_pmid(protocol_text: str) -> None:
    """Section 1 must cite the Hubbard 2011 false-positive paper by PMID and DOI.

    This is the anchor for the FP-recall and FP-biopsy figures used to justify
    the study's clinical premise.
    """
    assert "PMID: 22007042" in protocol_text or "PMID 22007042" in protocol_text
    assert "10.7326/0003-4819-155-8-201110180-00004" in protocol_text


def test_protocol_states_power_analysis(protocol_text: str) -> None:
    """Section 6 must specify the prespecified power analysis parameters."""
    assert "5" in protocol_text and "recall" in protocol_text.lower()
    # α = 0.05, power = 0.80
    assert "0.05" in protocol_text
    assert re.search(r"power\s*=\s*0\.8", protocol_text, re.IGNORECASE) is not None


def test_protocol_uses_placeholders_not_fabricated_identifiers(protocol_text: str) -> None:
    """Institutional fields must be bracketed placeholders, not invented names."""
    required_placeholders = [
        "[[SPONSOR_INSTITUTION]]",
        "[[PI_NAME]]",
        "[[IRB_NUMBER]]",
        "[[STUDY_SITE]]",
    ]
    for ph in required_placeholders:
        assert ph in protocol_text, f"Missing placeholder: {ph}"


# ---------------------------------------------------------------------------
# Ledger schema tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ledger_conn() -> sqlite3.Connection:
    """Open an in-memory SQLite DB and execute the schema. If it fails to parse,
    that's an immediate test failure — the schema is not valid DDL."""
    if not LEDGER_SQL.exists():
        pytest.fail(f"Ledger schema not found: {LEDGER_SQL}")
    conn = sqlite3.connect(":memory:")
    conn.executescript(LEDGER_SQL.read_text())
    return conn


def test_ledger_schema_is_valid_sqlite(ledger_conn: sqlite3.Connection) -> None:
    """The schema must load without error into a real SQLite connection."""
    # The fixture itself would have raised if the schema were invalid.
    tables = {
        row[0]
        for row in ledger_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "predictions" in tables


REQUIRED_LEDGER_COLUMNS = [
    "case_id",
    "input_dicom_sha256",
    "code_sha",
    "model_versions_json",
    "prediction_json",
    "radiologist_final_call",
    "biopsy_outcome",
    "days_to_outcome",
    "created_at",
    "updated_at",
]


def test_ledger_has_required_columns(ledger_conn: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in ledger_conn.execute("PRAGMA table_info(predictions)").fetchall()
    }
    for col in REQUIRED_LEDGER_COLUMNS:
        assert col in columns, f"predictions table missing column: {col}"


def test_ledger_case_id_is_primary_key(ledger_conn: sqlite3.Connection) -> None:
    pk_cols = {
        row[1]
        for row in ledger_conn.execute("PRAGMA table_info(predictions)").fetchall()
        if row[5] == 1
    }
    assert pk_cols == {"case_id"}, f"Expected PK=case_id, got {pk_cols}"


def test_ledger_has_audit_trail_table(ledger_conn: sqlite3.Connection) -> None:
    """A prediction_updates table is required for append-only mutation history."""
    tables = {
        row[0]
        for row in ledger_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "prediction_updates" in tables


def test_ledger_accepts_valid_row(ledger_conn: sqlite3.Connection) -> None:
    """Round-trip: insert a well-formed row and read it back."""
    ledger_conn.execute(
        """
        INSERT INTO predictions
            (case_id, input_dicom_sha256, code_sha,
             model_versions_json, prediction_json,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "test-case-001",
            "a" * 64,                 # 64-char SHA-256
            "b" * 40,                 # 40-char git SHA
            '{"l4a_screening":"c"}',
            '{"arbiter_score":0.42}',
            "2026-07-01T00:00:00Z",
            "2026-07-01T00:00:00Z",
        ),
    )
    row = ledger_conn.execute(
        "SELECT case_id FROM predictions WHERE case_id = ?", ("test-case-001",)
    ).fetchone()
    assert row == ("test-case-001",)


def test_ledger_rejects_short_sha256(ledger_conn: sqlite3.Connection) -> None:
    """The CHECK constraint on input_dicom_sha256 length=64 must fire."""
    with pytest.raises(sqlite3.IntegrityError):
        ledger_conn.execute(
            """
            INSERT INTO predictions
                (case_id, input_dicom_sha256, code_sha,
                 model_versions_json, prediction_json,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "test-bad-sha",
                "short-sha",           # wrong length
                "b" * 40,
                "{}",
                "{}",
                "2026-07-01T00:00:00Z",
                "2026-07-01T00:00:00Z",
            ),
        )


# ---------------------------------------------------------------------------
# Consent template tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def consent_text() -> str:
    if not CONSENT.exists():
        pytest.fail(f"Consent template not found: {CONSENT}")
    return CONSENT.read_text()


HIPAA_508_ELEMENTS = [
    # (short label, keyword or phrase that must appear)
    ("information to be disclosed", "Information to be used or disclosed"),
    ("purpose of disclosure",       "Purpose of the use or disclosure"),
    ("recipient of disclosure",     "Recipient of the disclosure"),
    ("expiration",                  "Expiration"),
    ("right to revoke",             "Right to revoke"),
    ("treatment not conditioned",   "Treatment not conditioned"),
    ("potential for re-disclosure", "Potential for re-disclosure"),
]


@pytest.mark.parametrize("label,keyword", HIPAA_508_ELEMENTS)
def test_consent_has_hipaa_508_element(consent_text: str, label: str, keyword: str) -> None:
    assert keyword in consent_text, f"Consent missing HIPAA §164.508 element: {label} ({keyword!r})"


def test_consent_declares_ruo(consent_text: str) -> None:
    """The consent must explicitly declare RESEARCH USE ONLY status."""
    assert "RESEARCH USE ONLY" in consent_text
    assert "not been approved" in consent_text.lower() or "not approved" in consent_text.lower()


def test_consent_declares_no_clinical_use(consent_text: str) -> None:
    """The consent must state that AI predictions will not be shared with the treating radiologist during the study."""
    assert (
        "will not depend on this system" in consent_text
        or "will not be shared with your radiologist" in consent_text.lower()
        or "will not be shared with your treating" in consent_text.lower()
    )


def test_consent_has_signature_line(consent_text: str) -> None:
    """A physical signature line is required for informed consent."""
    assert re.search(r"signature", consent_text, re.IGNORECASE) is not None
    assert "Date:" in consent_text
