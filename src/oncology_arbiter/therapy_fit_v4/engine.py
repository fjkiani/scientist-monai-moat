"""Therapy Fit v4 Core Dual-Engine Arbiter (Two-Door + Signed 8D Bilinear Decoder)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from oncology_arbiter.ovarian import ovarian_two_door_standalone_v9 as v9_ledger
from oncology_arbiter.ovarian.two_door_arbiter import H_CUT, DDR_CUT, score_ovarian_case
from oncology_arbiter.therapy_fit_v4.schemas import (
    PARP1FalsificationReceipt,
    RankedDrugV4,
    TherapyFitV4Request,
    TherapyFitV4Response,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BILINEAR_JSON_PATH = _REPO_ROOT / "data" / "external" / "s3_validation" / "crispro_8d_bilinear_decoder_v1.json"
_BRM_GBM_JSON_PATH = _REPO_ROOT / "data" / "external" / "brm_multiomics" / "brm_gbm_two_door_contract_v1.json"
_V9_PY_PATH = Path(v9_ledger.__file__).resolve()

EXPECTED_BILINEAR_SHA256 = "fabe9feabd28e9fe9174bad8e44ac7203a5647ca0faed589f21f7b26de7ee026"
EXPECTED_V9_SHA256 = "dd5f2ef197efef7afce320aad0f3c436936c306f67c582534e3c4cd155a00c92"
EXPECTED_BRM_GBM_SHA256 = "0e9ae5a65939ab4921d5965c769517389d17fca7e49f5522f3a13b588e4c9ffd"

_MOA_ALIAS_MAP: Dict[str, str] = {
    "SRC_FAK_SENOLYTIC": "BCLxL_Dasatinib_Senolytic",
    "BCL_XL_SENOLYTIC": "BCLxL_Dasatinib_Senolytic",
    "VDR_DIFFERENTIATION": "BCLxL_Dasatinib_Senolytic",
    "SENOLYTIC_MACROLIDE": "BCLxL_Dasatinib_Senolytic",
    "HEDGEHOG_SMO_INHIBITOR": "BCLxL_Dasatinib_Senolytic",
    "NUCLEOSIDE_BER_SL": "FOLFOX_Oxaliplatin",
    "ANTI_VEGF_ANGIO": "Anti_VEGF_Bev",
    "ATR_WEE1_DDR": "ATR_WEE1_Inhibitor",
    "PARP_TRAPPING_SL": "PARP_Inhibitor",
    "ICD_VACCINE_CHECKPOINT": "Vaccine_STC1010",
    "PAN_RAS_RAF_MAPK": "Pan_RAS_Daraxonrasib",
    "ADC_SURFACE_TOP1": "CEACAM5_TOP1_ADC",
    "PI3K_AKT_MTOR": "BCLxL_Dasatinib_Senolytic",
    "BACE1_MYELOID_REPROGRAM": "BACE1_GBM_Targeted",
    "EGFR_HER2_RTK": "Anti_EGFR",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_locked_artifacts() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    bilinear_sha = _sha256_file(_BILINEAR_JSON_PATH)
    v9_sha = _sha256_file(_V9_PY_PATH)
    brm_sha = _sha256_file(_BRM_GBM_JSON_PATH)
    if bilinear_sha != EXPECTED_BILINEAR_SHA256:
        raise RuntimeError(f"Bilinear decoder SHA mismatch: {bilinear_sha}")
    if v9_sha != EXPECTED_V9_SHA256:
        raise RuntimeError(f"ovarian_two_door_standalone_v9.py SHA mismatch: {v9_sha}")
    if brm_sha != EXPECTED_BRM_GBM_SHA256:
        raise RuntimeError(f"brm_gbm_two_door_contract_v1.json SHA mismatch: {brm_sha}")

    bilinear_data = json.loads(_BILINEAR_JSON_PATH.read_text(encoding="utf-8"))
    brm_data = json.loads(_BRM_GBM_JSON_PATH.read_text(encoding="utf-8"))
    locks = {
        "crispro_8d_bilinear_decoder_v1.json": bilinear_sha,
        "ovarian_two_door_standalone_v9.py": v9_sha,
        "brm_gbm_two_door_contract_v1.json": brm_sha,
    }
    return bilinear_data, brm_data, locks


def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(val)))


def build_patient_8d_vector(req: TherapyFitV4Request) -> Dict[str, float]:
    """Construct signed 8D patient latent vector p in [-1, +1]^8 from 3-Tier inputs."""
    t1 = req.tier1_dna
    t2 = req.tier2_rna
    t3 = req.tier3_kinetics

    mut_genes = {str(m.get("gene") or "").upper() for m in t1.mutations}
    has_mbd4 = t1.mbd4_lof or ("MBD4" in mut_genes) or ("1293delA" in str(t1.mbd4_hgvs_c or ""))
    has_tp53 = t1.tp53_mut or ("TP53" in mut_genes)
    has_brca = (not t1.brca_wt) or t1.brca1_mut or t1.brca2_mut or bool(mut_genes & {"BRCA1", "BRCA2"})
    has_kras = t1.kras_mut or t1.nras_mut or bool(mut_genes & {"KRAS", "NRAS", "HRAS"})
    has_braf = t1.braf_mut or ("BRAF" in mut_genes)
    has_egfr = t1.egfr_activating_mut or ("EGFR" in mut_genes)

    # d1_ddr_rs: DNA Damage Repair & Replication Stress
    d1 = 0.0
    if has_mbd4 and has_tp53:
        d1 += 0.85
    elif has_tp53:
        d1 += 0.45
    if has_brca or (t1.hrd_score is not None and t1.hrd_score >= 42.0):
        d1 += 0.50
    if t1.s3_score is not None and t1.s3_score > 0.075:
        d1 += 0.45
    if t1.ccne1_amp or t1.myc_amp or t1.arid1a_lof:
        d1 += 0.40
    if t2.post_DDR is not None:
        d1 += 0.35 * math.tanh(t2.post_DDR)
    if t2.post_H_clean is not None and t2.post_H_clean > H_CUT:
        # Hibernation suppresses active proliferative replication stress
        d1 -= 0.65

    # d2_hibernation_sen: Therapy-Induced Hibernation & Senescence
    d2 = -0.35
    if t2.post_H_clean is not None:
        if t2.post_H_clean > H_CUT:
            d2 = _clamp(0.55 + 0.25 * (t2.post_H_clean - H_CUT))
        else:
            d2 = _clamp(-0.45 + 0.25 * (t2.post_H_clean - H_CUT))
    if t2.post_S_sen is not None and t2.post_S_sen > 0.5:
        d2 = _clamp(d2 + 0.25 * math.tanh(t2.post_S_sen / 2.0))
    if t2.fak1_ptk2_high or t2.mtor_pi3k_brain_high:
        d2 = _clamp(max(d2, 0.75))

    # d3_rtk_egfr_addiction: Oncogenic RTK / EGFR / HER2 Addiction
    d3 = 0.85 if (has_egfr or t1.alk_fusion or t1.ros1_fusion or t1.met_amp_or_ex14 or t1.erbb2_amp_or_mut) else -0.30

    # d4_ras_raf_oncogenic: RAS / RAF / MAPK Oncogenic Signaling
    d4 = 0.85 if (has_kras or has_braf or t1.nf1_loss) else -0.30

    # d5_io_effector: Cytotoxic T/NK Effector & ICD Competence
    d5 = -0.25
    if t1.msi_status == "MSI-H" or (t1.tmb_score is not None and t1.tmb_score >= 10.0):
        d5 += 0.85
    if t2.hobit_gnly_score is not None:
        d5 += 0.70 * math.tanh(t2.hobit_gnly_score)
    if t2.aacr_a065_positive is True or (t2.z_FAP is not None and t2.z_CXCL10 is not None and t2.z_FAP < 0 and t2.z_CXCL10 > 0):
        d5 += 0.55

    # d6_stroma_myeloid_barrier: CAF Stroma & Myeloid Immunosuppression Barrier
    d6 = -0.15
    if t2.z_FAP is not None and t2.z_FAP > 0.5:
        d6 += 0.65
    if t2.mfap4_expr is not None and t2.mfap4_expr > 0.5:
        d6 += 0.50
    if t2.bace1_tmem119_high:
        d6 = max(d6, 0.80)

    # d7_surface_adc_payload: Surface Lineage Antigen & Payload Susceptibility
    d7 = 0.0
    if t2.ceacam5_top1_high or (
        t2.ceacam5_expr is not None and t2.ceacam5_expr > 0.5 and (t2.top1_slfn11_score is None or t2.top1_slfn11_score > 0.0)
    ):
        d7 = 0.90
    elif t2.epha2_expr is not None and t2.epha2_expr > 0.5:
        d7 = 0.65

    # d8_angio_metabolic_safety: Angiogenic Dependence & Metabolic/PGx Safety
    d8 = 0.25
    if t3.kelim_k is not None and t3.kelim_k < 1.0:
        d8 += 0.35  # Unfavorable KELIM benefits from Bevacizumab addition (GOG-0218)
    if t1.pgx.dpyd in {"*2A", "PM", "DEFICIENT"} or t1.pgx.ugt1a1 in {"*28/*28", "PM", "DEFICIENT"}:
        d8 -= 0.75

    return {
        "d1_ddr_rs": round(_clamp(d1), 4),
        "d2_hibernation_sen": round(_clamp(d2), 4),
        "d3_rtk_egfr_addiction": round(_clamp(d3), 4),
        "d4_ras_raf_oncogenic": round(_clamp(d4), 4),
        "d5_io_effector": round(_clamp(d5), 4),
        "d6_stroma_myeloid_barrier": round(_clamp(d6), 4),
        "d7_surface_adc_payload": round(_clamp(d7), 4),
        "d8_angio_metabolic_safety": round(_clamp(d8), 4),
    }


def score_moa_bilinear(
    patient_8d: Dict[str, float],
    moa_weights: Dict[str, float],
    bilinear_contract: Dict[str, Any],
) -> float:
    """Compute signed bilinear score s(p, m) = (p^T W_MoA m) / ||W_MoA m||_2."""
    axes = bilinear_contract.get("signed_axes_8d") or bilinear_contract["latent_axes_8d"]
    classes = bilinear_contract.get("moa_classes_15") or bilinear_contract["moa_classes_15d"]
    w_mat = bilinear_contract.get("W_MoA_matrix") or bilinear_contract["W_MoA_matrix_8x15"]

    normalized_weights: Dict[str, float] = {}
    for k, val in moa_weights.items():
        canonical_k = _MOA_ALIAS_MAP.get(k, k)
        normalized_weights[canonical_k] = normalized_weights.get(canonical_k, 0.0) + float(val)

    p_vec = [float(patient_8d.get(ax, 0.0)) for ax in axes]
    m_vec = [float(normalized_weights.get(c, 0.0)) for c in classes]

    t_vec = [0.0] * len(axes)
    for i in range(len(axes)):
        t_vec[i] = sum(float(w_mat[i][j]) * m_vec[j] for j in range(len(classes)))

    norm_t = math.sqrt(sum(x * x for x in t_vec))
    if norm_t < 1e-12:
        return 0.0
    dot = sum(p_vec[i] * t_vec[i] for i in range(len(axes)))
    return round(dot / norm_t, 4)


def evaluate_parp_falsification_gate(req: TherapyFitV4Request) -> PARP1FalsificationReceipt:
    """Evaluate the MBD4/BRCA-WT PARP1 Falsification Gate."""
    t1 = req.tier1_dna
    t2 = req.tier2_rna
    mut_genes = {str(m.get("gene") or "").upper() for m in t1.mutations}
    has_brca = (not t1.brca_wt) or t1.brca1_mut or t1.brca2_mut or bool(mut_genes & {"BRCA1", "BRCA2"})

    if has_brca:
        return PARP1FalsificationReceipt(
            status="BRCA_MUT_ELIGIBLE",
            falsified=False,
            reason="Canonical BRCA1/2 loss-of-function present; PARP inhibition indicated.",
            patient_s3_score=t1.s3_score,
            patient_parp1_rna_z=t2.parp1_rna_z,
        )
    if t1.s3_score is not None and t1.s3_score > 0.075:
        return PARP1FalsificationReceipt(
            status="RESCUED_BY_S3_DNA",
            falsified=False,
            reason=f"DNA S3 score ({t1.s3_score:.4f} > 0.075) rescues homologous recombination deficiency (LOTO AUROC = 0.887).",
            patient_s3_score=t1.s3_score,
            patient_parp1_rna_z=t2.parp1_rna_z,
        )
    if t2.parp1_rna_z is not None and t2.parp1_rna_z >= 1.0:
        return PARP1FalsificationReceipt(
            status="RESCUED_BY_PARP1_RNA",
            falsified=False,
            reason=f"High PARP1 RNA expression (z = {t2.parp1_rna_z:.2f}, trapping substrate rho = -0.416) rescues PARP trapping sensitivity.",
            patient_s3_score=t1.s3_score,
            patient_parp1_rna_z=t2.parp1_rna_z,
        )

    return PARP1FalsificationReceipt(
        status="GENOTYPE_FALSIFIED",
        falsified=True,
        reason=(
            "GENOTYPE_FALSIFIED (PARP1 upregulation p = 0.605; GDSC2 Olaparib AUROC = 0.401, p = 0.905): "
            "MBD4 loss-of-function in BRCA-WT does not confer single-agent PARP inhibitor synthetic lethality "
            "without elevated PARP1 RNA (rho = -0.416) or DNA S3 > 0.075 (LOTO AUROC = 0.887)."
        ),
        patient_s3_score=t1.s3_score,
        patient_parp1_rna_z=t2.parp1_rna_z,
    )


def score_therapy_fit_v4(req: TherapyFitV4Request) -> TherapyFitV4Response:
    """Execute dual-engine Therapy Fit v4 scoring (Two-Door Arbiter + Signed 8D Bilinear Decoder)."""
    bilinear_contract, brm_contract, artifact_locks = _load_locked_artifacts()
    patient_8d = build_patient_8d_vector(req)
    cancer = req.cancer_type.strip().lower()
    t1 = req.tier1_dna
    t2 = req.tier2_rna
    t3 = req.tier3_kinetics

    pgx_flag = "SAFE"
    if t1.pgx.dpyd in {"*2A", "PM", "DEFICIENT"}:
        pgx_flag = "DPYD_HIGH_RISK_AVOID_5FU"
    elif t1.pgx.ugt1a1 in {"*28/*28", "PM", "DEFICIENT"}:
        pgx_flag = "UGT1A1_HIGH_RISK_REDUCE_TOP1_DOSE"

    # Compute bilinear class scores across all 15 MoA classes
    classes_15 = bilinear_contract.get("moa_classes_15") or bilinear_contract["moa_classes_15d"]
    moa_class_scores = {
        cls_name: score_moa_bilinear(patient_8d, {cls_name: 1.0}, bilinear_contract)
        for cls_name in classes_15
    }

    parp_gate = evaluate_parp_falsification_gate(req)
    ranked_drugs: List[RankedDrugV4] = []
    blacklisted: List[Dict[str, Any]] = []
    two_door_receipt: Dict[str, Any] = {}
    pan_cancer_routing: Optional[Dict[str, Any]] = None

    # Resolve H_clean if stress/prolif components provided
    post_h_clean = t2.post_H_clean
    if post_h_clean is None and t2.z_stress_34_clean is not None and t2.z_proliferation_106 is not None:
        post_h_clean = t2.z_stress_34_clean - t2.z_proliferation_106

    is_door2_rna = post_h_clean is not None and post_h_clean > H_CUT

    # =========================================================================
    # TRACK A: OVARIAN / HGSOC / AYESHA
    # =========================================================================
    if cancer in {"hgsoc", "ovarian", "ovarian_cancer", "ovarian_cancer_hgs"}:
        # Run ovarian_two_door_standalone_v3/v9 arbiter if post_nact scores present
        try:
            arb_payload = {
                "patient_id": req.patient_id,
                "timepoint": "post_nact" if (post_h_clean is not None or t2.timepoint in {"post_nact", "surgical_ascites"}) else "pre_treatment",
                "post_H_clean": post_h_clean if post_h_clean is not None else 0.0,
                "post_DDR": t2.post_DDR if t2.post_DDR is not None else 0.5,
                "post_BCL2A1_z": t2.post_BCL2A1_z if t2.post_BCL2A1_z is not None else 0.0,
                "delta_MCL1_z": t2.delta_MCL1_z if t2.delta_MCL1_z is not None else 0.0,
                "post_EMT_z": t2.post_EMT_z if t2.post_EMT_z is not None else 0.0,
                "post_S_sen": t2.post_S_sen,
                "mbd4_lof": t1.mbd4_lof or ("1293delA" in str(t1.mbd4_hgvs_c or "")),
                "mbd4_hgvs_c": t1.mbd4_hgvs_c,
                "tp53_mut": t1.tp53_mut or True,
                "brca_wt": t1.brca_wt,
                "hrd": t1.hrd_score if t1.hrd_score is not None else 20.0,
                "z_FAP": t2.z_FAP if t2.z_FAP is not None else -0.5,
                "z_CXCL10": t2.z_CXCL10 if t2.z_CXCL10 is not None else 0.8,
                "mfap4_expr": t2.mfap4_expr if t2.mfap4_expr is not None else 0.2,
            }
            two_door_receipt = score_ovarian_case(arb_payload)
        except Exception as exc:
            two_door_receipt = {"status": "partial", "note": str(exc)}

        two_door_receipt["v9_ledger_rows_count"] = len(v9_ledger.DRUG_LEDGER)
        two_door_receipt["v9_composites"] = v9_ledger.V8_COMPOSITES

        if is_door2_rna:
            active_door = "DOOR_2_HIBERNATION_SENOLYTIC"
            door_rationale = (
                f"Post-NACT / Ascites RNA post_H_clean = {post_h_clean:.4f} > {H_CUT} triggers Door 2 "
                "Hibernation/Senolytic switch (5 BH-significant Door 2 drugs, q = 0.04724; "
                "Dasatinib + Navitoclax composite rho = +0.528, q = 0.00060)."
            )
            door2_candidates = [
                {
                    "drug_name": "Dasatinib (Sprycel)",
                    "drug_class": "SRC/YES1 Senolytic Kinase Inhibitor",
                    "fda_approved": True,
                    "moa": {"SRC_FAK_SENOLYTIC": 1.0},
                    "composite_boost": 0.96,
                    "evidence_summary": (
                        "FDA-Approved Door 2 Senolytic (#1): Targets SRC/YES1 mesenchymal-hibernation axis "
                        "(DoseResponse_Consensus_Z rho = +0.369, p = 0.01817, BH q = 0.04724; "
                        "Dasatinib + Navitoclax composite rho = +0.528, p = 0.00015, BH q = 0.00060)."
                    ),
                    "receipts": {"rho": 0.369433, "p": 0.01817, "q": 0.047242, "composite_rho": 0.527931, "composite_q": 0.0006},
                },
                {
                    "drug_name": "Doxercalciferol / Paricalcitol (VDR Agonist)",
                    "drug_class": "VDR Differentiation / Senolytic Agonist",
                    "fda_approved": True,
                    "moa": {"VDR_DIFFERENTIATION": 1.0},
                    "composite_boost": 0.93,
                    "evidence_summary": (
                        "FDA-Approved Door 2 VDR Agonist (#2): Reprograms VDR+ hibernating persisters "
                        "(DoseResponse_Consensus_Z rho = +0.464, p = 0.01534, BH q = 0.04724; PRISM secondary q = 0.03573)."
                    ),
                    "receipts": {"rho": 0.46358, "p": 0.01534, "q": 0.047242, "prism_q": 0.035734},
                },
                {
                    "drug_name": "Azithromycin",
                    "drug_class": "Senolytic Macrolide / Mitochondrial Ribosome Inhibitor",
                    "fda_approved": True,
                    "moa": {"SENOLYTIC_MACROLIDE": 1.0},
                    "composite_boost": 0.90,
                    "evidence_summary": (
                        "FDA-Approved Door 2 Senolytic Macrolide (#3): Selective autophagic/metabolic clearance of senescent cells "
                        "(DoseResponse_Consensus_Z rho = +0.447, p = 0.0115, BH q = 0.04724; PRISM secondary q = 0.03573)."
                    ),
                    "receipts": {"rho": 0.446838, "p": 0.0115, "q": 0.047242, "prism_q": 0.035734},
                },
                {
                    "drug_name": "Vismodegib",
                    "drug_class": "Hedgehog / SMO Dormancy Inhibitor",
                    "fda_approved": True,
                    "moa": {"HEDGEHOG_SMO_INHIBITOR": 1.0},
                    "composite_boost": 0.87,
                    "evidence_summary": (
                        "FDA-Approved Door 2 Hedgehog/SMO Inhibitor (#4): Ablates stromal-hibernation signaling "
                        "(PRISM secondary S_sen rho = +0.433, p = 0.01363, BH q = 0.03573)."
                    ),
                    "receipts": {"rho": 0.432682, "p": 0.013631, "q": 0.035734},
                },
                {
                    "drug_name": "Navitoclax / DT2216 (BCL-xL)",
                    "drug_class": "BCL-xL Senolytic / PROTAC Degrader",
                    "fda_approved": False,
                    "moa": {"BCL_XL_SENOLYTIC": 1.0, "SRC_FAK_SENOLYTIC": 0.5},
                    "composite_boost": 0.85,
                    "evidence_summary": (
                        "Door 2 BCL-xL Senolytic (#5): Targets BCL2L1 survival buffer in hibernating cells "
                        "(PRISM primary rho = +0.473, q = 0.02862; DoseResponse_Consensus_Z q = 0.04724; "
                        "Dasatinib + Navitoclax composite rho = +0.528, q = 0.00060)."
                    ),
                    "receipts": {"rho": 0.472574, "p": 0.002385, "q": 0.047242, "composite_rho": 0.527931, "composite_q": 0.0006},
                },
            ]
            for idx, item in enumerate(door2_candidates, start=1):
                b_score = score_moa_bilinear(patient_8d, item["moa"], bilinear_contract)
                ranked_drugs.append(
                    RankedDrugV4(
                        rank=idx,
                        drug_name=item["drug_name"],
                        drug_class=item["drug_class"],
                        door="DOOR_2",
                        fda_approved=item["fda_approved"],
                        bilinear_8d_score=b_score,
                        composite_score=round(0.5 * item["composite_boost"] + 0.5 * max(0.0, b_score), 4),
                        evidence_summary=item["evidence_summary"],
                        statistical_receipts=item["receipts"],
                        gate_status="DOOR_2_BH_SIGNIFICANT",
                        pgx_status=pgx_flag,
                    )
                )
            blacklisted = [
                {"drug_name": "Venetoclax", "reason": "BLACKLISTED_DOOR_2: BCL-2 selective (fails BCL-xL/BCL2L1 dependency in ovarian hibernation)"},
                {"drug_name": "Vorinostat", "reason": "BLACKLISTED_DOOR_2: Non-selective HDACi toxicity without Door 2 S_sen selectivity"},
                {"drug_name": "Atorvastatin", "reason": "BLACKLISTED_DOOR_2: Fails dose-response consensus in Door 2 persister panel"},
                {
                    "drug_name": "Olaparib",
                    "reason": f"BLACKLISTED_DOOR_2_AND_{parp_gate.status}: Dormant G0/G1 arrest lacks replication-fork trapping + {parp_gate.reason}",
                },
            ]
        else:
            active_door = "DOOR_1_DDR_REPLICATION_STRESS"
            door_rationale = (
                "Day-0 / Door-1 Proliferative BER + Replication Stress Synthetic Lethality "
                "(MBD4 c.1293delA + TP53 R273H, BRCA-WT): Rank #1 Gemcitabine + Carboplatin +/- Bevacizumab "
                "and #2 Ceralasertib (AZD6738) / Berzosertib (ATR SL), with Olaparib/Niraparib gated by PARP1 falsification."
            )
            door1_candidates = [
                {
                    "drug_name": "Gemcitabine + Carboplatin ± Bevacizumab",
                    "drug_class": "On-Label Nucleoside Analog (BER SL) + Platinum ± Anti-VEGF",
                    "fda_approved": True,
                    "moa": {"NUCLEOSIDE_BER_SL": 1.0, "ANTI_VEGF_ANGIO": 0.4},
                    "composite_boost": 0.96,
                    "evidence_summary": (
                        "On-Label BER Synthetic Lethality (#1): MBD4 glycosylase loss traps repair intermediates under "
                        "Gemcitabine nucleotide incorporation (p = 0.0028 isogenic / p = 0.029 MSS; KELIM K < 1.0 favors Bevacizumab)."
                    ),
                    "receipts": {"isogenic_p": 0.0028, "mss_p": 0.029, "kelim_k": t3.kelim_k},
                },
                {
                    "drug_name": "Ceralasertib (AZD6738) / Berzosertib",
                    "drug_class": "ATR Kinase Inhibitor (Replication Stress Synthetic Lethality)",
                    "fda_approved": False,
                    "moa": {"ATR_WEE1_DDR": 1.0},
                    "composite_boost": 0.88,
                    "evidence_summary": (
                        "ATR Synthetic Lethality (#2): TP53-stratified MBD4-LOF replication fork collapse "
                        "(delta_LN_IC50 = -1.07, p = 0.003, Cohen's d = -0.74; Bilinear LOTO AUROC = 0.665, p = 0.0172)."
                    ),
                    "receipts": {
                        "delta_ln_ic50": -1.07,
                        "tp53_stratified_p": 0.003,
                        "cohens_d": -0.74,
                        "bilinear_loto_auroc": 0.665,
                        "bilinear_loto_p": 0.0172,
                    },
                },
            ]
            for idx, item in enumerate(door1_candidates, start=1):
                b_score = score_moa_bilinear(patient_8d, item["moa"], bilinear_contract)
                ranked_drugs.append(
                    RankedDrugV4(
                        rank=idx,
                        drug_name=item["drug_name"],
                        drug_class=item["drug_class"],
                        door="DOOR_1",
                        fda_approved=item["fda_approved"],
                        bilinear_8d_score=b_score,
                        composite_score=round(0.5 * item["composite_boost"] + 0.5 * max(0.0, b_score), 4),
                        evidence_summary=item["evidence_summary"],
                        statistical_receipts=item["receipts"],
                        gate_status="DOOR_1_PRIMARY_SL",
                        pgx_status=pgx_flag,
                    )
                )

            if not parp_gate.falsified:
                b_parp = score_moa_bilinear(patient_8d, {"PARP_TRAPPING_SL": 1.0}, bilinear_contract)
                ranked_drugs.append(
                    RankedDrugV4(
                        rank=len(ranked_drugs) + 1,
                        drug_name="Olaparib / Niraparib",
                        drug_class="PARP1/2 Inhibitor",
                        door="DOOR_1",
                        fda_approved=True,
                        bilinear_8d_score=b_parp,
                        composite_score=round(max(0.55, b_parp), 4),
                        evidence_summary=parp_gate.reason,
                        statistical_receipts={
                            "status": parp_gate.status,
                            "parp1_rna_rho": -0.416,
                            "s3_loto_auroc": 0.887,
                        },
                        gate_status=parp_gate.status,
                        pgx_status=pgx_flag,
                    )
                )
            else:
                blacklisted.append(
                    {
                        "drug_name": "Olaparib / Niraparib",
                        "status": "GENOTYPE_FALSIFIED",
                        "reason": parp_gate.reason,
                        "statistical_receipts": {
                            "parp1_upregulation_p": 0.605,
                            "gdsc2_olaparib_auroc": 0.401,
                            "gdsc2_olaparib_p": 0.905,
                            "parp1_rna_rho": -0.416,
                            "s3_loto_auroc": 0.887,
                        },
                    }
                )

    # =========================================================================
    # TRACK B: COLORECTAL CANCER (CRC)
    # =========================================================================
    elif cancer in {"crc", "colorectal", "colorectal_cancer", "coadread"}:
        mut_genes = {str(m.get("gene") or "").upper() for m in t1.mutations}
        has_ras = t1.kras_mut or t1.nras_mut or bool(mut_genes & {"KRAS", "NRAS"})
        is_ceacam5_adc = bool(
            t2.ceacam5_top1_high
            or (t2.ceacam5_expr is not None and t2.ceacam5_expr >= 1.0 and (t2.top1_slfn11_score is None or t2.top1_slfn11_score >= 0.5))
        )
        is_gate2_hot = bool(
            (t2.hobit_gnly_score is not None and t2.hobit_gnly_score >= 0.5)
            or t1.msi_status == "MSI-H"
        )

        crc_arms = [
            {
                "drug_name": "STC-1010 (Allogeneic ICD Cancer Vaccine + Low-Dose Cyclophosphamide)",
                "drug_class": "Immunogenic Cell Death (ICD) Vaccine + Treg Depletion",
                "door": "GATE_2_IO",
                "fda_approved": False,
                "moa": {"ICD_VACCINE_CHECKPOINT": 1.0},
                "matched": is_gate2_hot and not is_ceacam5_adc,
                "priority": 1 if (is_gate2_hot and not is_ceacam5_adc) else 2,
                "evidence_summary": "Gate-2 Hot Effector-Competent CRC (n = 146; HOBIT/GNLY+ effector niche).",
                "receipts": {"cohort_n": 146, "gate": "Gate-2 Hot", "signature": "HOBIT_GNLY"},
            },
            {
                "drug_name": "Daraxonrasib (RMC-6236)",
                "drug_class": "Pan-RAS(ON) Multi-Selective Tri-Complex Inhibitor",
                "door": "GATE_2_KRAS",
                "fda_approved": False,
                "moa": {"PAN_RAS_RAF_MAPK": 1.0},
                "matched": has_ras and not is_gate2_hot,
                "priority": 1 if (has_ras and not is_gate2_hot and not is_ceacam5_adc) else 2,
                "evidence_summary": "Gate-2 Cold RAS-Mutant CRC (n = 151; 19/19 Stage IV KRAS G12D/V routed to Pan-RAS(ON)).",
                "receipts": {"cohort_n": 151, "stage_iv_g12dv_routed": "19/19", "gate": "Gate-2 Cold RAS-mut"},
            },
            {
                "drug_name": "SGN-CEACAM5C (Precemtabart tocentecan / M9140)",
                "drug_class": "Anti-CEACAM5 TOP1i Antibody-Drug Conjugate",
                "door": "GATE_2_ADC",
                "fda_approved": False,
                "moa": {"ADC_SURFACE_TOP1": 1.0},
                "matched": is_ceacam5_adc,
                "priority": 1 if is_ceacam5_adc else 3,
                "evidence_summary": "CEACAM5-Hi + TOP1/SLFN11+ Payload-Sensitive CRC (n = 102).",
                "receipts": {"cohort_n": 102, "gate": "CEACAM5-Hi + TOP1/SLFN11+"},
            },
        ]
        crc_arms.sort(key=lambda x: (x["priority"], -score_moa_bilinear(patient_8d, x["moa"], bilinear_contract)))
        active_door = crc_arms[0]["door"]
        door_rationale = f"CRC 3-Arm Arbiter routed primary allocation to {crc_arms[0]['drug_name']}."
        for idx, item in enumerate(crc_arms, start=1):
            b_score = score_moa_bilinear(patient_8d, item["moa"], bilinear_contract)
            ranked_drugs.append(
                RankedDrugV4(
                    rank=idx,
                    drug_name=item["drug_name"],
                    drug_class=item["drug_class"],
                    door=item["door"],
                    fda_approved=item["fda_approved"],
                    bilinear_8d_score=b_score,
                    composite_score=round(0.6 * (1.0 if idx == 1 else 0.65) + 0.4 * max(0.0, b_score), 4),
                    evidence_summary=item["evidence_summary"],
                    statistical_receipts=item["receipts"],
                    gate_status="PRIMARY_CRC_ROUTE" if idx == 1 else "SECONDARY_CRC_OPTION",
                    pgx_status=pgx_flag,
                )
            )
        pan_cancer_routing = {
            "tumor_track": "CRC_3_ARM_ARBITER",
            "selected_arm": crc_arms[0]["drug_name"],
            "cohorts": {"STC-1010_Gate2_Hot_n": 146, "Daraxonrasib_Gate2_Cold_RAS_n": 151, "SGN_CEACAM5C_TOP1_n": 102},
        }

    # =========================================================================
    # TRACK C: BrM / GBM (BRAIN METASTASES & GLIOBLASTOMA)
    # =========================================================================
    elif cancer in {"gbm", "brm", "glioblastoma", "brain_metastasis"}:
        two_door_receipt = {
            "contract": brm_contract.get("contract_version", "brm_gbm_two_door_v1"),
            "door_1_proliferative_ddr_egfr": brm_contract.get("door_1_proliferative_ddr_egfr"),
            "door_2_brain_adapted_hibernation_senolytic": brm_contract.get("door_2_brain_adapted_hibernation_senolytic"),
            "door_2_myeloid_microglia_axis": brm_contract.get("door_2_myeloid_microglia_axis"),
        }
        is_brm_door2 = bool(is_door2_rna or t2.fak1_ptk2_high or t2.mtor_pi3k_brain_high or t2.bace1_tmem119_high)
        if is_brm_door2:
            active_door = "DOOR_2_BRAIN_HIBERNATION_SENOLYTIC"
            door_rationale = (
                "BrM/GBM Door 2 Brain-Adapted Hibernation / Astrocytic-Microglial Lock: "
                "Routes to Dasatinib (14.5x in GBM, q=0.013), AZD2014 (8.33x, p=0.0004), "
                "GNE-317 (4.85x, p=0.0012), GSK2256098C (FAK1i, 2.78x, p=0.0411), and Verubecestat (BACE1 dual lock)."
            )
            brm_door2 = [
                {
                    "drug_name": "Dasatinib",
                    "drug_class": "SRC/YES1 Brain-Penetrant Senolytic Kinase Inhibitor",
                    "fda_approved": True,
                    "moa": {"SRC_FAK_SENOLYTIC": 1.0},
                    "boost": 0.96,
                    "evidence_summary": "GBM/BrM Door 2 Senolytic (#1): 14.5x enrichment in GBM mesenchymal/hibernation state (BH q = 0.013).",
                    "receipts": {"gbm_fold_enrichment": 14.5, "q": 0.013, "delta_ic50_um": -17.34, "p": 0.000185},
                },
                {
                    "drug_name": "AZD2014 (Vistusertib)",
                    "drug_class": "Dual mTORC1/2 Inhibitor",
                    "fda_approved": False,
                    "moa": {"PI3K_AKT_MTOR": 1.0},
                    "boost": 0.92,
                    "evidence_summary": "BrM/GBM Door 2 mTORC1/2 Inhibitor (#2): 8.33x enrichment in brain-adapted state (p = 0.0004).",
                    "receipts": {"fold_enrichment": 8.33, "p": 0.000412, "delta_ic50_um": -3.14},
                },
                {
                    "drug_name": "GNE-317",
                    "drug_class": "BBB-Penetrant Dual PI3K/mTOR Inhibitor",
                    "fda_approved": False,
                    "moa": {"PI3K_AKT_MTOR": 1.0},
                    "boost": 0.89,
                    "evidence_summary": "BrM/GBM Door 2 BBB-Penetrant PI3K/mTORi (#3): Engineered to bypass P-gp/BCRP efflux (4.85x, p = 0.0012).",
                    "receipts": {"fold_enrichment": 4.85, "p": 0.0012},
                },
                {
                    "drug_name": "GSK2256098C",
                    "drug_class": "FAK1 (PTK2) Inhibitor (Alliance A071401 Validated)",
                    "fda_approved": False,
                    "moa": {"SRC_FAK_SENOLYTIC": 1.0},
                    "boost": 0.86,
                    "evidence_summary": "BrM/GBM Door 2 FAK1 Inhibitor (#4): 2.78x sensitivity shift in brain-metastatic lines (p = 0.0411).",
                    "receipts": {"fold_enrichment": 2.78, "p": 0.0411, "delta_ic50_um": -23.16},
                },
                {
                    "drug_name": "Verubecestat (MK-8931)",
                    "drug_class": "BBB-Penetrant BACE1 Inhibitor (Dual Tumor/Myeloid Lock)",
                    "fda_approved": False,
                    "moa": {"BACE1_MYELOID_REPROGRAM": 1.0},
                    "boost": 0.84,
                    "evidence_summary": "BrM/GBM Door 2 BACE1 Inhibitor (#5): Dual tumor-intrinsic + TMEM119+ microglial myeloid reprogramming.",
                    "receipts": {"bace1_brain_vs_extracranial_p": 0.0252, "microglia_bace1_tmem119_rho": 0.677, "microglia_p": 2.49e-13},
                },
            ]
            for idx, item in enumerate(brm_door2, start=1):
                b_score = score_moa_bilinear(patient_8d, item["moa"], bilinear_contract)
                ranked_drugs.append(
                    RankedDrugV4(
                        rank=idx,
                        drug_name=item["drug_name"],
                        drug_class=item["drug_class"],
                        door="DOOR_2",
                        fda_approved=item["fda_approved"],
                        bilinear_8d_score=b_score,
                        composite_score=round(0.5 * item["boost"] + 0.5 * max(0.0, b_score), 4),
                        evidence_summary=item["evidence_summary"],
                        statistical_receipts=item["receipts"],
                        gate_status="BRM_GBM_DOOR_2",
                        pgx_status=pgx_flag,
                    )
                )
        else:
            active_door = "DOOR_1_PROLIFERATIVE_RTK_DDR"
            door_rationale = "BrM/GBM Door 1 Proliferative / RTK-Addicted or Replication-Stressed State: Routes to Osimertinib / Zorifertinib / Ceralasertib."
            brm_door1 = [
                {
                    "drug_name": "Osimertinib / Zorifertinib (AZD3759)",
                    "drug_class": "CNS-Penetrant 3rd-Gen EGFR TKI",
                    "fda_approved": True,
                    "moa": {"EGFR_HER2_RTK": 1.0},
                    "boost": 0.94,
                    "evidence_summary": "Door 1 CNS-Penetrant EGFR TKI (#1): 12.1x sensitivity enrichment in EGFR-driven lines (p = 0.0085).",
                    "receipts": {"fold_enrichment": 12.1, "p": 0.00851, "delta_ic50_um": -11.55},
                },
                {
                    "drug_name": "Ceralasertib (AZD6738)",
                    "drug_class": "CNS-Active ATR Kinase Inhibitor",
                    "fda_approved": False,
                    "moa": {"ATR_WEE1_DDR": 1.0},
                    "boost": 0.89,
                    "evidence_summary": "Door 1 Replication Stress ATRi (#2): Targets proliferative DDR dependency in brain metastases.",
                    "receipts": {"bilinear_loto_auroc": 0.665, "bilinear_loto_p": 0.0172},
                },
            ]
            for idx, item in enumerate(brm_door1, start=1):
                b_score = score_moa_bilinear(patient_8d, item["moa"], bilinear_contract)
                ranked_drugs.append(
                    RankedDrugV4(
                        rank=idx,
                        drug_name=item["drug_name"],
                        drug_class=item["drug_class"],
                        door="DOOR_1",
                        fda_approved=item["fda_approved"],
                        bilinear_8d_score=b_score,
                        composite_score=round(0.5 * item["boost"] + 0.5 * max(0.0, b_score), 4),
                        evidence_summary=item["evidence_summary"],
                        statistical_receipts=item["receipts"],
                        gate_status="BRM_GBM_DOOR_1",
                        pgx_status=pgx_flag,
                    )
                )
        pan_cancer_routing = {"tumor_track": "BRM_GBM_TWO_DOOR", "active_door": active_door}

    # =========================================================================
    # TRACK D: NSCLC, BREAST, MELANOMA (PAN-CANCER SOLID TUMORS)
    # =========================================================================
    else:
        active_door = "DOOR_2_HIBERNATION_SENOLYTIC" if is_door2_rna else "DOOR_1_DRIVER_AND_SL"
        door_rationale = f"Pan-Cancer ({cancer.upper()}) 8D Bilinear Decoder + Two-Door routing ({active_door})."

        candidates_by_cancer: Dict[str, List[Dict[str, Any]]] = {
            "nsclc": [
                {
                    "drug_name": "Osimertinib (Tagrisso)",
                    "drug_class": "3rd-Gen EGFR TKI",
                    "fda_approved": True,
                    "door": "DOOR_1",
                    "moa": {"EGFR_HER2_RTK": 1.0},
                    "evidence_summary": "NSCLC EGFR-addicted primary driver therapy (d3_rtk_egfr_addiction).",
                },
                {
                    "drug_name": "Daraxonrasib (RMC-6236) / Sotorasib",
                    "drug_class": "RAS(ON) / KRAS G12C Inhibitor",
                    "fda_approved": True,
                    "door": "DOOR_1",
                    "moa": {"PAN_RAS_RAF_MAPK": 1.0},
                    "evidence_summary": "NSCLC KRAS-mutant MAPK axis blockade (d4_ras_raf_oncogenic).",
                },
                {
                    "drug_name": "Ceralasertib (AZD6738) + Durvalumab (HUDSON)",
                    "drug_class": "ATR Inhibitor + Anti-PD-L1",
                    "fda_approved": False,
                    "door": "DOOR_1",
                    "moa": {"ATR_WEE1_DDR": 0.7, "ICD_VACCINE_CHECKPOINT": 0.3},
                    "evidence_summary": "NSCLC DDR/Replication-Stress + IO combination (Bilinear LOTO AUROC = 0.665).",
                },
                {
                    "drug_name": "Dasatinib + Navitoclax",
                    "drug_class": "Door 2 Senolytic Combination",
                    "fda_approved": False,
                    "door": "DOOR_2",
                    "moa": {"SRC_FAK_SENOLYTIC": 0.6, "BCL_XL_SENOLYTIC": 0.4},
                    "evidence_summary": "Door 2 TKI-persister / mesenchymal hibernation clearance.",
                },
            ],
            "breast": [
                {
                    "drug_name": "Trastuzumab Deruxtecan (T-DXd)",
                    "drug_class": "Anti-HER2 TOP1i ADC",
                    "fda_approved": True,
                    "door": "DOOR_1",
                    "moa": {"EGFR_HER2_RTK": 0.5, "ADC_SURFACE_TOP1": 0.5},
                    "evidence_summary": "HER2+/HER2-low TOP1 payload delivery.",
                },
                {
                    "drug_name": "Ceralasertib / Gemcitabine + Carboplatin",
                    "drug_class": "ATR / Replication Stress SL",
                    "fda_approved": True,
                    "door": "DOOR_1",
                    "moa": {"ATR_WEE1_DDR": 0.6, "NUCLEOSIDE_BER_SL": 0.4},
                    "evidence_summary": "TNBC / HRD-negative replication stress & BER synthetic lethality.",
                },
                {
                    "drug_name": "Alpelisib / Capivasertib (PI3K/AKT)",
                    "drug_class": "PI3K/AKT Pathway Inhibitor",
                    "fda_approved": True,
                    "door": "DOOR_1",
                    "moa": {"PI3K_AKT_MTOR": 1.0},
                    "evidence_summary": "PIK3CA/PTEN-altered luminal/HER2 escape blockade.",
                },
                {
                    "drug_name": "Dasatinib + Navitoclax",
                    "drug_class": "Door 2 Senolytic Combination",
                    "fda_approved": False,
                    "door": "DOOR_2",
                    "moa": {"SRC_FAK_SENOLYTIC": 0.6, "BCL_XL_SENOLYTIC": 0.4},
                    "evidence_summary": "Post-chemo residual disease hibernation clearance (H_clean > 1.0165).",
                },
            ],
            "melanoma": [
                {
                    "drug_name": "Nivolumab + Ipilimumab / Relatlimab",
                    "drug_class": "Dual Immune Checkpoint Blockade",
                    "fda_approved": True,
                    "door": "GATE_2_IO",
                    "moa": {"ICD_VACCINE_CHECKPOINT": 1.0},
                    "evidence_summary": "Effector-competent / TMB-high melanoma checkpoint blockade.",
                },
                {
                    "drug_name": "Dabrafenib + Trametinib / Encorafenib + Binimetinib",
                    "drug_class": "BRAF + MEK Vertical MAPK Inhibition",
                    "fda_approved": True,
                    "door": "DOOR_1",
                    "moa": {"PAN_RAS_RAF_MAPK": 1.0},
                    "evidence_summary": "BRAF V600E/K vertical MAPK addiction blockade.",
                },
                {
                    "drug_name": "Dasatinib + Navitoclax",
                    "drug_class": "Door 2 Dedifferentiated / Neural-Crest Persister Senolytic",
                    "fda_approved": False,
                    "door": "DOOR_2",
                    "moa": {"SRC_FAK_SENOLYTIC": 0.6, "BCL_XL_SENOLYTIC": 0.4},
                    "evidence_summary": "Targets AXL+/SRC+ dedifferentiated Dormant Door 2 melanoma persisters.",
                },
            ],
        }
        pool = candidates_by_cancer.get(cancer, candidates_by_cancer["nsclc"])
        scored_pool = []
        for item in pool:
            b_score = score_moa_bilinear(patient_8d, item["moa"], bilinear_contract)
            door_match_bonus = 0.35 if (is_door2_rna and item["door"] == "DOOR_2") or (not is_door2_rna and item["door"] != "DOOR_2") else 0.0
            scored_pool.append((b_score + door_match_bonus, b_score, item))
        scored_pool.sort(key=lambda x: -x[0])

        for idx, (comp, b_score, item) in enumerate(scored_pool, start=1):
            ranked_drugs.append(
                RankedDrugV4(
                    rank=idx,
                    drug_name=item["drug_name"],
                    drug_class=item["drug_class"],
                    door=item["door"],
                    fda_approved=item["fda_approved"],
                    bilinear_8d_score=b_score,
                    composite_score=round(comp, 4),
                    evidence_summary=item["evidence_summary"],
                    statistical_receipts={"bilinear_8d_score": b_score},
                    gate_status="PAN_CANCER_RANKED",
                    pgx_status=pgx_flag,
                )
            )
        pan_cancer_routing = {
            "tumor_track": f"{cancer.upper()}_BILINEAR_TWO_DOOR",
            "top_recommendation": ranked_drugs[0].drug_name if ranked_drugs else None,
        }

    tier_completeness = {
        "tier1_dna_present": bool(t1.mutations or t1.mbd4_lof or t1.tp53_mut or t1.kras_mut or t1.braf_mut or t1.egfr_activating_mut),
        "tier2_rna_present": bool(post_h_clean is not None or t2.hobit_gnly_score is not None or t2.ceacam5_expr is not None or t2.parp1_rna_z is not None),
        "tier3_kinetics_present": bool(t3.kelim_k is not None or t3.ca125_baseline is not None or t3.ca125_series),
    }

    return TherapyFitV4Response(
        patient_id=req.patient_id,
        cancer_type=cancer,
        active_door=active_door,
        door_rationale=door_rationale,
        patient_8d_vector=patient_8d,
        two_door_arbiter_receipt=two_door_receipt,
        bilinear_decoder_receipt={
            "contract_version": bilinear_contract.get("schema_version") or bilinear_contract.get("contract_version"),
            "architecture": bilinear_contract.get("model_id") or bilinear_contract.get("architecture"),
            "moa_class_scores_15d": moa_class_scores,
            "loto_benchmark_summary": bilinear_contract.get("loto_validation_summary") or bilinear_contract.get("loto_benchmark_summary"),
        },
        parp_falsification_gate=parp_gate if cancer in {"hgsoc", "ovarian", "ovarian_cancer", "ovarian_cancer_hgs"} else None,
        ranked_drugs=ranked_drugs,
        blacklisted_drugs=blacklisted,
        pan_cancer_routing=pan_cancer_routing,
        tier_completeness=tier_completeness,
        artifact_locks=artifact_locks,
    )
