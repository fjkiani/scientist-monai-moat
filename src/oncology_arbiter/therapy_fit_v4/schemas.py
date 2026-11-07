"""Unified 3-Tier Input & Output Contracts for Therapy Fit v4."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class PGxProfile(BaseModel):
    dpyd: Optional[str] = Field(default=None, description="DPYD genotype/phenotype (e.g. WT, *2A, PM)")
    ugt1a1: Optional[str] = Field(default=None, description="UGT1A1 genotype/phenotype (e.g. *1/*1, *28/*28)")
    cyp2d6: Optional[str] = Field(default=None, description="CYP2D6 phenotype")
    tpmt: Optional[str] = Field(default=None, description="TPMT phenotype")
    nudt15: Optional[str] = Field(default=None, description="NUDT15 phenotype")
    alerts: List[str] = Field(default_factory=list)


class Tier1GenomicInput(BaseModel):
    """Tier 1: DNA Mutations / CNA / S3 signature + PGx."""
    mutations: List[Dict[str, Any]] = Field(default_factory=list, description="Somatic/germline mutations")
    cna: Dict[str, Any] = Field(default_factory=dict, description="Copy number alterations")
    mbd4_lof: bool = False
    mbd4_hgvs_c: Optional[str] = None
    tp53_mut: bool = False
    tp53_variant: Optional[str] = None
    brca_wt: bool = True
    brca1_mut: bool = False
    brca2_mut: bool = False
    hrd_score: Optional[float] = None
    s3_score: Optional[float] = Field(default=None, description="SBS3/S3 homologous recombination signature (>0.075 threshold)")
    ccne1_amp: bool = False
    myc_amp: bool = False
    kras_mut: bool = False
    kras_variant: Optional[str] = None
    nras_mut: bool = False
    braf_mut: bool = False
    braf_variant: Optional[str] = None
    egfr_activating_mut: bool = False
    egfr_variant: Optional[str] = None
    alk_fusion: bool = False
    ros1_fusion: bool = False
    met_amp_or_ex14: bool = False
    erbb2_amp_or_mut: bool = False
    pik3ca_mut: bool = False
    pten_loss: bool = False
    nf1_loss: bool = False
    arid1a_lof: bool = False
    msi_status: Literal["MSS", "MSI-H", "MSI-L", "Unknown"] = "MSS"
    tmb_score: Optional[float] = None
    pgx: PGxProfile = Field(default_factory=PGxProfile)


class Tier2TranscriptomicInput(BaseModel):
    """Tier 2: Tumor RNA Two-Door & Pan-Cancer Expression Signatures."""
    timepoint: Literal["pre_treatment", "post_nact", "surgical_ascites", "relapse"] = "pre_treatment"
    post_H_clean: Optional[float] = Field(default=None, description="Hibernation/Dormancy score H_clean (>1.01654 = Door 2)")
    post_S_sen: Optional[float] = Field(default=None, description="Senolytic sensitivity index S_sen")
    post_DDR: Optional[float] = Field(default=None, description="Post-NACT DDR proliferation score (>0.22866 = Door 1)")
    post_BCL2A1_z: Optional[float] = None
    delta_MCL1_z: Optional[float] = None
    post_EMT_z: Optional[float] = None
    z_stress_34_clean: Optional[float] = None
    z_proliferation_106: Optional[float] = None
    z_BCL2L1: Optional[float] = None
    z_PMAIP1: Optional[float] = None
    z_MCL1: Optional[float] = None
    z_FAP: Optional[float] = None
    z_CXCL10: Optional[float] = None
    mfap4_expr: Optional[float] = Field(default=None, description="MFAP4 stroma expression")
    aacr_a065_positive: Optional[bool] = Field(default=None, description="AACR A065 z(FAP)<0 & z(CXCL10)>0")
    hobit_gnly_score: Optional[float] = Field(default=None, description="HOBIT/ZNF683 + GNLY effector score (CRC Gate-2)")
    ceacam5_expr: Optional[float] = Field(default=None, description="CEACAM5 surface expression z-score")
    top1_slfn11_score: Optional[float] = Field(default=None, description="TOP1/SLFN11 payload sensitivity z-score")
    ceacam5_top1_high: Optional[bool] = Field(default=None, description="Combined CEACAM5-Hi + TOP1/SLFN11+")
    parp1_rna_z: Optional[float] = Field(default=None, description="PARP1 RNA z-score (rho=-0.416 with Olaparib IC50)")
    vdr_expr: Optional[float] = None
    epha2_expr: Optional[float] = None
    bace1_tmem119_high: Optional[bool] = Field(default=None, description="BrM/GBM BACE1 + TMEM119 myeloid lock")
    fak1_ptk2_high: Optional[bool] = Field(default=None, description="PTK2/FAK1 brain adaptation")
    mtor_pi3k_brain_high: Optional[bool] = Field(default=None, description="PI3K/mTOR brain met addiction")


class Tier3KineticsInput(BaseModel):
    """Tier 3: CA-125 KELIM Kinetics."""
    ca125_baseline: Optional[float] = None
    kelim_k: Optional[float] = Field(default=None, description="Standardized KELIM elimination rate constant K")
    kelim_category: Literal["favorable", "unfavorable", "unknown"] = "unknown"
    kinetics_score: Optional[float] = None
    platinum_sensitivity: Literal["sensitive", "intermediate", "resistant", "refractory", "unknown"] = "unknown"
    ca125_series: List[Dict[str, Any]] = Field(default_factory=list)


class TherapyFitV4Request(BaseModel):
    """Unified 3-Tier Input Contract for Therapy Fit v4."""
    patient_id: str = "AK"
    cancer_type: str = Field(default="hgsoc", description="hgsoc | ovarian | crc | gbm | brm | nsclc | breast | melanoma")
    stage: str = "IVB"
    tier1_dna: Tier1GenomicInput = Field(default_factory=Tier1GenomicInput)
    tier2_rna: Tier2TranscriptomicInput = Field(default_factory=Tier2TranscriptomicInput)
    tier3_kinetics: Tier3KineticsInput = Field(default_factory=Tier3KineticsInput)


class RankedDrugV4(BaseModel):
    rank: int
    drug_name: str
    drug_class: str
    door: Literal["DOOR_1", "DOOR_2", "GATE_2_IO", "GATE_2_KRAS", "GATE_2_ADC", "PAN_CANCER", "CONTRAINDICATED"]
    fda_approved: bool = False
    bilinear_8d_score: float
    composite_score: float
    evidence_summary: str
    statistical_receipts: Dict[str, Any] = Field(default_factory=dict)
    gate_status: str = "ELIGIBLE"
    pgx_status: str = "SAFE"


class PARP1FalsificationReceipt(BaseModel):
    drug_names: List[str] = Field(default_factory=lambda: ["Olaparib", "Niraparib", "Rucaparib"])
    status: Literal["GENOTYPE_FALSIFIED", "RESCUED_BY_PARP1_RNA", "RESCUED_BY_S3_DNA", "BRCA_MUT_ELIGIBLE", "HRD_ELIGIBLE"]
    falsified: bool
    reason: str
    parp1_upregulation_p: float = 0.605
    gdsc2_olaparib_auroc: float = 0.401
    gdsc2_olaparib_p: float = 0.905
    parp1_rna_rho: float = -0.416
    s3_loto_auroc: float = 0.887
    s3_threshold: float = 0.075
    patient_s3_score: Optional[float] = None
    patient_parp1_rna_z: Optional[float] = None


class TherapyFitV4Response(BaseModel):
    contract_version: str = "therapy_fit_v4.0"
    ruo_disclaimer: str = "RESEARCH USE ONLY (RUO). Preclinical & translational prioritization engine; not an FDA-cleared diagnostic."
    patient_id: str
    cancer_type: str
    active_door: str
    door_rationale: str
    patient_8d_vector: Dict[str, float]
    two_door_arbiter_receipt: Dict[str, Any]
    bilinear_decoder_receipt: Dict[str, Any]
    parp_falsification_gate: Optional[PARP1FalsificationReceipt] = None
    ranked_drugs: List[RankedDrugV4]
    blacklisted_drugs: List[Dict[str, Any]]
    pan_cancer_routing: Optional[Dict[str, Any]] = None
    tier_completeness: Dict[str, Any]
    artifact_locks: Dict[str, str]
