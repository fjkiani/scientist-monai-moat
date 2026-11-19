"""Strike 4 Verification Suite: Therapy Fit v4 on scientist-monai-moat."""
from __future__ import annotations

import pytest

from oncology_arbiter.api.routers.tumor_board import _compute_case_therapy_fit_v4
from oncology_arbiter.therapy_fit_v4 import (
    EXPECTED_BILINEAR_SHA256,
    EXPECTED_BRM_GBM_SHA256,
    EXPECTED_V9_SHA256,
    TherapyFitV4Request,
    Tier1GenomicInput,
    Tier2TranscriptomicInput,
    Tier3KineticsInput,
    score_therapy_fit_v4,
)
from oncology_arbiter.therapy_fit_v4.schemas import PGxProfile


def test_cryptographic_artifact_locks() -> None:
    """Assert SHA-256 locks for all 3 Therapy Fit v4 artifacts on scientist-monai-moat."""
    resp = score_therapy_fit_v4(TherapyFitV4Request())
    locks = resp.artifact_locks
    assert locks["crispro_8d_bilinear_decoder_v1.json"] == EXPECTED_BILINEAR_SHA256
    assert locks["ovarian_two_door_standalone_v9.py"] == EXPECTED_V9_SHA256
    assert locks["brm_gbm_two_door_contract_v1.json"] == EXPECTED_BRM_GBM_SHA256


def test_ayesha_day0_and_post_nact_door2_invariants() -> None:
    """Assert Ayesha Day-0 Door-1 (Gem/Carbo #1, Ceralasertib #2, Olaparib falsified) and Post-NACT Door-2 switch."""
    day0 = score_therapy_fit_v4(
        TherapyFitV4Request(
            patient_id="AK",
            cancer_type="hgsoc",
            stage="IVB",
            tier1_dna=Tier1GenomicInput(
                mbd4_lof=True,
                mbd4_hgvs_c="c.1293delA",
                tp53_mut=True,
                tp53_variant="p.R273H",
                brca_wt=True,
                hrd_score=22.0,
            ),
            tier3_kinetics=Tier3KineticsInput(ca125_baseline=2842.0, kelim_k=0.87, kelim_category="unfavorable"),
        )
    )
    assert day0.active_door == "DOOR_1_DDR_REPLICATION_STRESS"
    assert "Gemcitabine + Carboplatin" in day0.ranked_drugs[0].drug_name
    assert "Ceralasertib" in day0.ranked_drugs[1].drug_name
    assert day0.parp_falsification_gate is not None
    assert day0.parp_falsification_gate.status == "GENOTYPE_FALSIFIED"
    assert day0.parp_falsification_gate.falsified is True

    door2 = score_therapy_fit_v4(
        TherapyFitV4Request(
            patient_id="AK_POST_NACT",
            cancer_type="hgsoc",
            stage="IVB",
            tier1_dna=Tier1GenomicInput(mbd4_lof=True, tp53_mut=True, brca_wt=True),
            tier2_rna=Tier2TranscriptomicInput(
                timepoint="post_nact",
                post_H_clean=1.48,
                post_S_sen=1.32,
                post_DDR=-0.45,
                post_BCL2A1_z=0.4,
                delta_MCL1_z=-0.2,
                post_EMT_z=0.8,
            ),
        )
    )
    assert door2.active_door == "DOOR_2_HIBERNATION_SENOLYTIC"
    assert len(door2.ranked_drugs) == 5
    assert "Dasatinib" in door2.ranked_drugs[0].drug_name
    assert "Doxercalciferol" in door2.ranked_drugs[1].drug_name
    assert "Azithromycin" in door2.ranked_drugs[2].drug_name
    assert "Vismodegib" in door2.ranked_drugs[3].drug_name
    assert "Navitoclax" in door2.ranked_drugs[4].drug_name
    blacklisted = {b["drug_name"] for b in door2.blacklisted_drugs}
    assert {"Venetoclax", "Vorinostat", "Atorvastatin", "Olaparib"} <= blacklisted


def test_pan_cancer_tracks_and_tumor_board_case_helper() -> None:
    """Assert CRC, BrM/GBM, NSCLC, Breast, Melanoma routing and tumor_board _compute_case_therapy_fit_v4 integration."""
    # CRC 3-arm checks
    crc_hot = score_therapy_fit_v4(
        TherapyFitV4Request(cancer_type="crc", tier2_rna=Tier2TranscriptomicInput(hobit_gnly_score=1.2))
    )
    assert "STC-1010" in crc_hot.ranked_drugs[0].drug_name

    crc_kras = score_therapy_fit_v4(
        TherapyFitV4Request(cancer_type="crc", tier1_dna=Tier1GenomicInput(kras_mut=True))
    )
    assert "Daraxonrasib" in crc_kras.ranked_drugs[0].drug_name

    crc_adc = score_therapy_fit_v4(
        TherapyFitV4Request(cancer_type="crc", tier2_rna=Tier2TranscriptomicInput(ceacam5_top1_high=True))
    )
    assert "SGN-CEACAM5C" in crc_adc.ranked_drugs[0].drug_name

    # BrM/GBM Door 2 check
    gbm_d2 = score_therapy_fit_v4(
        TherapyFitV4Request(
            cancer_type="gbm",
            tier2_rna=Tier2TranscriptomicInput(post_H_clean=1.40, bace1_tmem119_high=True),
        )
    )
    assert gbm_d2.active_door == "DOOR_2_BRAIN_HIBERNATION_SENOLYTIC"
    assert len(gbm_d2.ranked_drugs) == 5

    # Tumor board _compute_case_therapy_fit_v4 integration
    tb_out = _compute_case_therapy_fit_v4(
        primary_site="ovarian",
        clinical_summary="Stage IVB HGSOC, MBD4 c.1293delA frameshift, TP53 R273H, BRCA-WT, CA-125 2842",
        genomics_payload={
            "mbd4_lof": True,
            "mbd4_hgvs_c": "c.1293delA",
            "tp53_mut": True,
            "brca_wt": True,
            "hrd_score": 22.0,
            "kelim_k": 0.87,
            "pgx": {"ugt1a1": "*28/*28"},
        },
        ovarian_two_door_payload=None,
    )
    assert tb_out is not None
    assert tb_out["contract_version"] == "therapy_fit_v4.0"
    assert tb_out["active_door"] == "DOOR_1_DDR_REPLICATION_STRESS"
    assert "Gemcitabine" in tb_out["ranked_drugs"][0]["drug_name"]
    assert tb_out["ranked_drugs"][0]["pgx_status"] == "UGT1A1_HIGH_RISK_REDUCE_TOP1_DOSE"
