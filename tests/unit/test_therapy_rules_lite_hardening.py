"""Hardening tests for L4c NCCN-lite therapy rules engine (v0.2).

Covers the pieces added on top of the branch-behavior tests in
`test_therapy_rules_lite.py`:

  * SHA-256 fingerprint of the on-disk rules file is stable and surfaces on
    every ``TherapyRulesResult``.
  * ``rules_model_id`` and ``branch_id`` fields are populated.
  * ``_load_rules`` validates ``model_id``, ``source_document_url``, per-branch
    required fields, non-empty ``recommended`` list, no duplicate ``branch_id``,
    and JSON-vs-code coverage.
  * ``strict=True`` input validation rejects grade / receptor_status / stage /
    menopausal_status drift with ``InvalidInputError`` (a ``ValueError``
    subclass).
  * ``menopausal_status="unknown"`` for HR+/HER2- flows a dedicated safe-default
    path that emits Tamoxifen + a menopause-evaluation workup, with a warning.

Fully deterministic — no network, no filesystem writes outside a scratch dir.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oncology_arbiter.models.therapy_rules_lite import (
    _CACHED_RULES,
    _CACHED_SHA256,
    _COVERED_BRANCH_IDS,
    _EXPECTED_RULES_MODEL_ID,
    _EXPECTED_RULES_URL,
    _RULES_PATH,
    InvalidInputError,
    NCCN_URL,
    RulesetIntegrityError,
    TherapyOption,
    TherapyRulesResult,
    _load_rules,
    apply_nccn_lite_rules,
    rules_sha256_hex,
)


# --------------------------------------------------------------------------- #
# 1. Fingerprint stability
# --------------------------------------------------------------------------- #


def test_rules_sha256_matches_disk() -> None:
    live = hashlib.sha256(_RULES_PATH.read_bytes()).hexdigest()
    assert _CACHED_SHA256 == live
    assert rules_sha256_hex() == live
    assert len(_CACHED_SHA256) == 64
    assert all(c in "0123456789abcdef" for c in _CACHED_SHA256)


def test_every_result_carries_rules_fingerprint() -> None:
    cases = [
        (dict(receptor_status={"ER": True, "PR": True, "HER2": False},
              grade=1, stage="T1N0M0", subtype="DCIS"),
         "dcis"),
        (dict(receptor_status={"ER": True, "PR": True, "HER2": False},
              grade=2, stage="T4N2M1"),
         "metastatic"),
        (dict(receptor_status={"ER": True, "PR": True, "HER2": True},
              grade=2, stage="T2N0M0"),
         "her2_positive"),
        (dict(receptor_status={"ER": False, "PR": False, "HER2": False},
              grade=3, stage="T2N1M0"),
         "triple_negative"),
        (dict(receptor_status={"ER": True, "PR": True, "HER2": False},
              grade=2, stage="T1N0M0", menopausal_status="postmenopausal"),
         "hr_positive_her2_negative"),
    ]
    for kwargs, expected_branch in cases:
        r = apply_nccn_lite_rules(**kwargs)
        assert r.rules_sha256 == _CACHED_SHA256, expected_branch
        assert r.rules_model_id == _EXPECTED_RULES_MODEL_ID, expected_branch
        assert r.branch_id == expected_branch


# --------------------------------------------------------------------------- #
# 2. Load-time integrity guards
# --------------------------------------------------------------------------- #


@pytest.fixture()
def clean_rules_dict() -> dict:
    return json.loads(_RULES_PATH.read_text())


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(data))
    return p


def test_load_rules_rejects_wrong_model_id(tmp_path: Path, clean_rules_dict: dict) -> None:
    clean_rules_dict["model_id"] = "nccn-lite-v99"
    p = _write(tmp_path, clean_rules_dict)
    with pytest.raises(RulesetIntegrityError, match="model_id"):
        _load_rules(p)


def test_load_rules_rejects_wrong_source_url(tmp_path: Path, clean_rules_dict: dict) -> None:
    clean_rules_dict["source_document_url"] = "https://example.com/breast.pdf"
    p = _write(tmp_path, clean_rules_dict)
    with pytest.raises(RulesetIntegrityError, match="source_document_url"):
        _load_rules(p)


def test_load_rules_rejects_missing_branch_field(tmp_path: Path, clean_rules_dict: dict) -> None:
    del clean_rules_dict["rule_branches"][0]["nccn_section"]
    p = _write(tmp_path, clean_rules_dict)
    with pytest.raises(RulesetIntegrityError, match="nccn_section"):
        _load_rules(p)


def test_load_rules_rejects_empty_recommended(tmp_path: Path, clean_rules_dict: dict) -> None:
    clean_rules_dict["rule_branches"][0]["recommended"] = []
    p = _write(tmp_path, clean_rules_dict)
    with pytest.raises(RulesetIntegrityError, match="empty recommended"):
        _load_rules(p)


def test_load_rules_rejects_duplicate_branch_id(tmp_path: Path, clean_rules_dict: dict) -> None:
    first_bid = clean_rules_dict["rule_branches"][0]["branch_id"]
    clean_rules_dict["rule_branches"][1]["branch_id"] = first_bid
    p = _write(tmp_path, clean_rules_dict)
    with pytest.raises(RulesetIntegrityError, match="duplicate branch_id"):
        _load_rules(p)


def test_load_rules_rejects_uncovered_branch_id(tmp_path: Path, clean_rules_dict: dict) -> None:
    clean_rules_dict["rule_branches"].append({
        "branch_id": "lobular_pleomorphic_specialcase",
        "nccn_section": "BINV-Z",
        "trigger": "invented",
        "recommended": [{"name": "x", "citation_url": NCCN_URL}],
    })
    p = _write(tmp_path, clean_rules_dict)
    with pytest.raises(RulesetIntegrityError, match="does not cover"):
        _load_rules(p)


def test_load_rules_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        _load_rules(missing)


def test_json_branch_ids_are_subset_of_code_coverage() -> None:
    assert _CACHED_RULES is not None
    ids_in_json = {b["branch_id"] for b in _CACHED_RULES["rule_branches"]}
    unknown = ids_in_json - _COVERED_BRANCH_IDS
    assert not unknown, f"JSON has branch_ids not covered in code: {unknown}"


# --------------------------------------------------------------------------- #
# 3. strict=True input validation
# --------------------------------------------------------------------------- #


def test_strict_rejects_missing_receptor_key() -> None:
    with pytest.raises(InvalidInputError, match="receptor_status missing"):
        apply_nccn_lite_rules(
            receptor_status={"ER": True, "PR": True},
            grade=2,
            stage="T1N0M0",
            strict=True,
        )


def test_strict_rejects_non_bool_receptor_value() -> None:
    with pytest.raises(InvalidInputError, match="must be bool") as info:
        apply_nccn_lite_rules(
            receptor_status={"ER": "yes", "PR": True, "HER2": False},  # type: ignore[dict-item]
            grade=2,
            stage="T1N0M0",
            strict=True,
        )
    assert "ER" in str(info.value)


def test_strict_rejects_out_of_range_grade() -> None:
    with pytest.raises(InvalidInputError, match=r"grade must be in \[1, 3\]"):
        apply_nccn_lite_rules(
            receptor_status={"ER": True, "PR": True, "HER2": False},
            grade=7,
            stage="T1N0M0",
            strict=True,
        )


def test_strict_rejects_bool_masquerading_as_grade() -> None:
    with pytest.raises(InvalidInputError, match="grade must be int"):
        apply_nccn_lite_rules(
            receptor_status={"ER": True, "PR": True, "HER2": False},
            grade=True,  # type: ignore[arg-type]
            stage="T1N0M0",
            strict=True,
        )


def test_strict_rejects_malformed_stage() -> None:
    with pytest.raises(InvalidInputError, match="does not match TNM"):
        apply_nccn_lite_rules(
            receptor_status={"ER": True, "PR": True, "HER2": False},
            grade=2,
            stage="stage-two",
            strict=True,
        )


def test_strict_accepts_metastatic_M1_shortcut() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": True, "PR": True, "HER2": False},
        grade=2,
        stage="M1",
        strict=True,
    )
    assert r.branch_id == "metastatic"


def test_strict_rejects_bad_menopausal_status() -> None:
    with pytest.raises(InvalidInputError, match="menopausal_status"):
        apply_nccn_lite_rules(
            receptor_status={"ER": True, "PR": True, "HER2": False},
            grade=2,
            stage="T1N0M0",
            menopausal_status="peri-menopausal-ish",
            strict=True,
        )


def test_non_strict_permissive_by_default() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": True, "PR": True, "HER2": False},
        grade=2,
        stage="stage-two-with-bad-format",
        menopausal_status="ambiguous",
    )
    assert isinstance(r, TherapyRulesResult)


# --------------------------------------------------------------------------- #
# 4. menopausal_status="unknown" safe-default branch
# --------------------------------------------------------------------------- #


def test_menopausal_unknown_emits_tamoxifen_and_workup() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": True, "PR": True, "HER2": False},
        grade=2,
        stage="T1N0M0",
        menopausal_status="unknown",
    )
    names = [o.name for o in r.recommended_options]
    assert any("Tamoxifen" in n for n in names)
    assert any("Menopause status evaluation" in n for n in names)
    assert r.branch_id == "hr_positive_her2_negative"
    assert any("menopausal_status=unknown" in w for w in r.warnings)


def test_menopausal_unknown_does_not_recommend_ai() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": True, "PR": True, "HER2": False},
        grade=2,
        stage="T1N0M0",
        menopausal_status="unknown",
    )
    ai_names = [o.name for o in r.recommended_options if "Aromatase" in o.name]
    assert not ai_names, f"Should not recommend AI without menopause status, got {ai_names}"


# --------------------------------------------------------------------------- #
# 5. Envelope + coverage invariants
# --------------------------------------------------------------------------- #


def test_every_option_carries_nccn_citation_and_section() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": True, "PR": True, "HER2": True},
        grade=2, stage="T2N0M0",
    )
    for o in r.recommended_options + r.not_recommended:
        assert o.citation_url == NCCN_URL, o.name
        assert o.nccn_section, f"{o.name!r} missing nccn_section"


def test_coverage_set_is_exactly_json() -> None:
    assert _CACHED_RULES is not None
    ids_in_json = frozenset(b["branch_id"] for b in _CACHED_RULES["rule_branches"])
    assert _COVERED_BRANCH_IDS == ids_in_json
