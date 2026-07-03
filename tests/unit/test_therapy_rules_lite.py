"""Unit tests for L4c NCCN-lite therapy rules engine.

Fully deterministic — no network, no HF calls.
"""
from __future__ import annotations

import pytest

from oncology_arbiter.models.therapy_rules_lite import (
    NCCN_URL,
    TherapyOption,
    TherapyRulesResult,
    THERAPY_RULES_PROXY_WARNING,
    apply_nccn_lite_rules,
)


# --------------------------------------------------------------------------- #
# 1. HR+/HER2- postmenopausal → AI first-line
# --------------------------------------------------------------------------- #


def test_hr_positive_her2_negative_postmenopausal_recommends_ai() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": True, "PR": True, "HER2": False},
        grade=2,
        stage="T1N0M0",
        menopausal_status="postmenopausal",
        subtype="IDC",
    )
    names = [o.name.lower() for o in r.recommended_options]
    assert any("aromatase" in n for n in names), f"expected AI; got {names}"
    assert r.model_state == "proxy_rules_lite"


# --------------------------------------------------------------------------- #
# 2. HR+/HER2- premenopausal → tamoxifen
# --------------------------------------------------------------------------- #


def test_hr_positive_her2_negative_premenopausal_recommends_tamoxifen() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": True, "PR": False, "HER2": False},
        grade=1,
        stage="T1N0M0",
        menopausal_status="premenopausal",
        subtype="IDC",
    )
    names = [o.name.lower() for o in r.recommended_options]
    assert any("tamoxifen" in n for n in names), f"expected tamoxifen; got {names}"


# --------------------------------------------------------------------------- #
# 3. HER2+ → trastuzumab + TCHP
# --------------------------------------------------------------------------- #


def test_her2_positive_recommends_trastuzumab() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": True, "PR": True, "HER2": True},
        grade=2,
        stage="T2N1M0",
    )
    names = [o.name.lower() for o in r.recommended_options]
    assert any("trastuzumab" in n for n in names), f"expected trastuzumab; got {names}"
    assert any("tchp" in n for n in names), f"expected TCHP; got {names}"


# --------------------------------------------------------------------------- #
# 4. Triple negative → chemo, NOT endocrine
# --------------------------------------------------------------------------- #


def test_triple_negative_recommends_chemo_not_endocrine() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": False, "PR": False, "HER2": False},
        grade=3,
        stage="T2N0M0",
    )
    rec_names = [o.name.lower() for o in r.recommended_options]
    not_rec_names = [o.name.lower() for o in r.not_recommended]
    assert any("chemotherapy" in n or "ac-t" in n for n in rec_names), (
        f"expected chemotherapy; got {rec_names}"
    )
    assert any("endocrine" in n for n in not_rec_names), (
        f"expected endocrine in not_recommended; got {not_rec_names}"
    )


# --------------------------------------------------------------------------- #
# 5. DCIS → surgery+radiation, NO chemo
# --------------------------------------------------------------------------- #


def test_dcis_recommends_surgery_and_radiation_not_chemo() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": True, "PR": False, "HER2": False},
        grade=1,
        stage="Tis",
        subtype="DCIS",
    )
    rec_names = [o.name.lower() for o in r.recommended_options]
    not_rec_names = [o.name.lower() for o in r.not_recommended]
    assert any("lumpectomy" in n and "radiation" in n for n in rec_names), (
        f"expected lumpectomy+radiation; got {rec_names}"
    )
    assert any("chemotherapy" in n for n in not_rec_names), (
        f"expected chemo in not_recommended for DCIS; got {not_rec_names}"
    )


# --------------------------------------------------------------------------- #
# 6. Metastatic → biomarker-directed + palliative
# --------------------------------------------------------------------------- #


def test_metastatic_recommends_biomarker_and_palliative() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": True, "PR": False, "HER2": False},
        grade=2,
        stage="T3N1M1",
    )
    names = [o.name.lower() for o in r.recommended_options]
    assert any("biomarker" in n for n in names), f"expected biomarker; got {names}"
    assert any("palliative" in n for n in names), f"expected palliative; got {names}"


# --------------------------------------------------------------------------- #
# 7. All recommendations carry NCCN citation URL
# --------------------------------------------------------------------------- #


def test_all_options_have_nccn_citation_url() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": True, "PR": True, "HER2": False},
        grade=3,
        stage="T2N0M0",
        menopausal_status="postmenopausal",
        subtype="IDC",
    )
    for option in r.recommended_options + r.not_recommended:
        assert option.citation_url == NCCN_URL, (
            f"option {option.name} has citation_url={option.citation_url}, "
            f"expected {NCCN_URL}"
        )
        assert option.nccn_section, f"option {option.name} missing nccn_section"


# --------------------------------------------------------------------------- #
# 8. Proxy warning always present
# --------------------------------------------------------------------------- #


def test_therapy_proxy_warning_always_emitted() -> None:
    r = apply_nccn_lite_rules(
        receptor_status={"ER": False, "PR": False, "HER2": True},
        grade=2,
        stage="T1N0M0",
    )
    assert THERAPY_RULES_PROXY_WARNING in r.warnings, (
        "rules-lite proxy warning MUST be emitted on every call"
    )
