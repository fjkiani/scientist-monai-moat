"""8D bilinear decoder constants (axes, MoA class names, RUO strings)."""
from __future__ import annotations

CONTRACT_STATUS = "evidence_bound_exploratory_prioritization"
RUO_DISCLAIMER = (
    "RESEARCH USE ONLY (RUO). Supervised signed bilinear 8D trial decoding and "
    "multi-trial responder routing are exploratory research prioritization models, "
    "not FDA-cleared clinical diagnostics or treatment orders."
)

# ---------------------------------------------------------------------------
# 1. Canonical Signed 8D Patient State Axes (p_i in [-1, +1]^8)
# ---------------------------------------------------------------------------

SIGNED_AXES_8D: tuple[str, ...] = (
    "d1_ddr_rs",
    "d2_hibernation_sen",
    "d3_rtk_egfr_addiction",
    "d4_ras_raf_oncogenic",
    "d5_io_effector",
    "d6_stroma_myeloid_barrier",
    "d7_surface_adc_payload",
    "d8_angio_metabolic_safety",
)

# 15 Core Therapeutic Mechanism-of-Action (MoA) Classes (K = 15)
MOA_CLASSES_15: tuple[str, ...] = (
    "FOLFOX_Oxaliplatin",          # 0: Oxaliplatin + 5-FU/LV DNA crosslinking & ICD priming
    "FOLFIRI_Irinotecan_TOP1",     # 1: Irinotecan / SN-38 TOP1 inhibition
    "Anti_VEGF_Bev",               # 2: Bevacizumab / Aflibercept / VEGFR TKI (Regorafenib/Cabozantinib/Zanzalintinib)
    "Anti_EGFR",                   # 3: Panitumumab / Cetuximab / Erlotinib (repelled when RAS/RAF-mut via d3 < 0)
    "Anti_PD1_PDL1",               # 4: Pembrolizumab / Nivolumab / Durvalumab / Atezolizumab / Avelumab
    "Anti_CTLA4",                  # 5: Ipilimumab / Tremelimumab / Botensilimab
    "Vaccine_STC1010",             # 6: STC-1010 DNFB-haptenated whole-cell / NeoAg / CEA / mKRAS vaccine
    "Pan_RAS_Daraxonrasib",        # 7: Daraxonrasib (RMC-6236) RAS(ON) multi-selective tri-complex inhibitor
    "BRAF_Encorafenib",            # 8: Encorafenib BRAF V600E inhibitor (BREAKWATER)
    "MEK_Cobimetinib",             # 9: Cobimetinib / Binimetinib MEK inhibitor (IMblaze370)
    "ATR_WEE1_Inhibitor",          # 10: Ceralasertib / Berzosertib (ATR) & Adavosertib (WEE1) + Gemcitabine
    "PARP_Inhibitor",              # 11: Olaparib PARP1/2 trapping
    "BCLxL_Dasatinib_Senolytic",   # 12: Navitoclax / ABT-737 (BCL-xL) + Dasatinib (SRC/EPHA2) + Doxercalciferol
    "CEACAM5_TOP1_ADC",            # 13: Precemtabart tocentecan / Tusamitamab sonditecan / SAR408701 / IMMU-130
    "BACE1_GBM_Targeted",          # 14: Verubecestat (MK-8931) BACE1 dual-compartment / GBM Stupp-escape context
)

# v2 extension (9 cols) appended after MOA_CLASSES_15 — crispro_8d_bilinear_decoder_v2.json
MOA_CLASSES_9_EXT: tuple[str, ...] = (
    "VDR_Paricalcitol",
    "FAK_ROCK_Inhibitor",
    "mTORC2_Inhibitor",
    "PRMT5_MAT2A_MTAP",
    "EP300_CREBBP",
    "CDK2_PKMYT1",
    "MCL1_BCLxL_TW37",
    "FOLR1_ADC",
    "CDH6_ADC",
)

MOA_CLASSES_24: tuple[str, ...] = MOA_CLASSES_15 + MOA_CLASSES_9_EXT
