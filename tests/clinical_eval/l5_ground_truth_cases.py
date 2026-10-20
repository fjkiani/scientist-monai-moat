"""Official 20-case L5 Co-Scientist clinical ground-truth fixture.

15 phenotype / NCCN Category-1 preferred maps + 5 landmark Phase 3 head-to-heads.
"""
from __future__ import annotations

from typing import Any, TypedDict


class GroundTruthCase(TypedDict, total=False):
    case_id: str
    kind: str  # phenotype | landmark
    patient_context: dict[str, Any]
    preferred_regimen: str
    preferred_aliases: list[str]
    contraindicated_regimens: list[str]
    regimen_a: str
    regimen_b: str
    expected_winner: str  # A | B
    landmark_trial: str
    seed_urls: list[str]


_PUBMED = {
    "trastuzumab": "https://pubmed.ncbi.nlm.nih.gov/27650830/",
    "pertuzumab": "https://pubmed.ncbi.nlm.nih.gov/25176067/",
    "tdm1": "https://pubmed.ncbi.nlm.nih.gov/23020162/",
    "tdxd": "https://pubmed.ncbi.nlm.nih.gov/34449463/",
    "ribociclib": "https://pubmed.ncbi.nlm.nih.gov/27717303/",
    "tailorx": "https://pubmed.ncbi.nlm.nih.gov/29860917/",
    "ai": "https://pubmed.ncbi.nlm.nih.gov/18463340/",
    "chemo": "https://pubmed.ncbi.nlm.nih.gov/15625331/",
}


def _ev(url: str, text: str) -> dict[str, str]:
    return {"url": url, "quoted_text": text, "source": "pubmed"}


# --------------------------------------------------------------------------- #
# 15 phenotype cases

CLINICAL_GROUND_TRUTH_CASES: list[GroundTruthCase] = [
    {
        "case_id": "pheno_01_hr_pos_her2_neg_postmeno",
        "kind": "phenotype",
        "patient_context": {
            "ER": "POSITIVE", "PR": "POSITIVE", "HER2": "NEGATIVE",
            "menopausal_status": "post", "stage": "IIA", "grade": 2,
        },
        "preferred_regimen": "Aromatase inhibitor (letrozole)",
        "preferred_aliases": ["aromatase inhibitor", "letrozole", "anastrozole", "exemestane"],
        "contraindicated_regimens": [
            "Tamoxifen monotherapy for TNBC",
            "Nab-paclitaxel monotherapy early HR+",
        ],
        "seed_urls": [_PUBMED["ai"]],
    },
    {
        "case_id": "pheno_02_hr_pos_her2_neg_premeno",
        "kind": "phenotype",
        "patient_context": {
            "ER": "POSITIVE", "PR": "POSITIVE", "HER2": "NEGATIVE",
            "menopausal_status": "pre", "stage": "IB", "grade": 2,
        },
        "preferred_regimen": "Tamoxifen +/- OFS",
        "preferred_aliases": ["tamoxifen", "ovarian function suppression", "ofs"],
        "contraindicated_regimens": [
            "Aromatase inhibitor alone without OFS in premenopausal",
            "Trastuzumab without HER2+",
        ],
        "seed_urls": [_PUBMED["ai"]],
    },
    {
        "case_id": "pheno_03_her2_pos_hr_neg",
        "kind": "phenotype",
        "patient_context": {
            "ER": "NEGATIVE", "PR": "NEGATIVE", "HER2": "POSITIVE",
            "menopausal_status": "post", "stage": "IIA", "grade": 3,
        },
        "preferred_regimen": "Trastuzumab + Pertuzumab + Taxane",
        "preferred_aliases": ["trastuzumab", "pertuzumab", "thp", "dual her2"],
        "contraindicated_regimens": [
            "Tamoxifen for TNBC",
            "Aromatase inhibitor alone",
        ],
        "seed_urls": [_PUBMED["trastuzumab"], _PUBMED["pertuzumab"]],
    },
    {
        "case_id": "pheno_04_her2_pos_hr_pos",
        "kind": "phenotype",
        "patient_context": {
            "ER": "POSITIVE", "PR": "POSITIVE", "HER2": "POSITIVE",
            "menopausal_status": "post", "stage": "IIB", "grade": 3,
        },
        "preferred_regimen": "Trastuzumab + Pertuzumab + Taxane + Endocrine",
        "preferred_aliases": ["trastuzumab", "pertuzumab", "endocrine", "thp"],
        "contraindicated_regimens": [
            "Endocrine therapy alone without anti-HER2",
            "Nab-paclitaxel monotherapy early HR+",
        ],
        "seed_urls": [_PUBMED["trastuzumab"], _PUBMED["pertuzumab"]],
    },
    {
        "case_id": "pheno_05_tnbc_early",
        "kind": "phenotype",
        "patient_context": {
            "ER": "NEGATIVE", "PR": "NEGATIVE", "HER2": "NEGATIVE",
            "menopausal_status": "pre", "stage": "IIA", "grade": 3,
        },
        "preferred_regimen": "Anthracycline/taxane chemotherapy",
        "preferred_aliases": ["chemotherapy", "anthracycline", "taxane", "ac-t", "ddac"],
        "contraindicated_regimens": [
            "Tamoxifen for TNBC",
            "Trastuzumab without HER2+",
            "Aromatase inhibitor alone",
        ],
        "seed_urls": [_PUBMED["chemo"]],
    },
    {
        "case_id": "pheno_06_tnbc_advanced",
        "kind": "phenotype",
        "patient_context": {
            "ER": "NEGATIVE", "PR": "NEGATIVE", "HER2": "NEGATIVE",
            "menopausal_status": "post", "stage": "IV", "grade": 3,
        },
        "preferred_regimen": "Systemic chemotherapy (sacituzumab / taxane class)",
        "preferred_aliases": ["chemotherapy", "sacituzumab", "taxane", "carboplatin"],
        "contraindicated_regimens": [
            "Tamoxifen for TNBC",
            "Aromatase inhibitor alone",
        ],
        "seed_urls": [_PUBMED["chemo"]],
    },
    {
        "case_id": "pheno_07_hr_pos_node_pos_postmeno",
        "kind": "phenotype",
        "patient_context": {
            "ER": "POSITIVE", "PR": "POSITIVE", "HER2": "NEGATIVE",
            "menopausal_status": "post", "stage": "IIIA", "grade": 2, "nodes": "N1",
        },
        "preferred_regimen": "Aromatase inhibitor +/- CDK4/6",
        "preferred_aliases": ["aromatase", "letrozole", "ribociclib", "abemaciclib", "cdk4"],
        "contraindicated_regimens": [
            "Nab-paclitaxel monotherapy early HR+",
            "Trastuzumab without HER2+",
        ],
        "seed_urls": [_PUBMED["ai"], _PUBMED["ribociclib"]],
    },
    {
        "case_id": "pheno_08_her2_low_hr_pos",
        "kind": "phenotype",
        "patient_context": {
            "ER": "POSITIVE", "PR": "POSITIVE", "HER2": "LOW",
            "menopausal_status": "post", "stage": "IV", "grade": 2,
        },
        "preferred_regimen": "Endocrine +/- CDK4/6 (consider T-DXd later line)",
        "preferred_aliases": ["endocrine", "cdk4", "letrozole", "t-dxd", "trastuzumab deruxtecan"],
        "contraindicated_regimens": [
            "Tamoxifen for TNBC",
            "Dual HER2 blockade as if IHC3+",
        ],
        "seed_urls": [_PUBMED["ai"], _PUBMED["tdxd"]],
    },
    {
        "case_id": "pheno_09_dcis_hr_pos",
        "kind": "phenotype",
        "patient_context": {
            "ER": "POSITIVE", "PR": "POSITIVE", "HER2": "NEGATIVE",
            "histology": "DCIS", "stage": "0", "grade": 2,
        },
        "preferred_regimen": "Endocrine risk-reduction (tamoxifen or AI)",
        "preferred_aliases": ["tamoxifen", "aromatase", "endocrine", "risk-reduction"],
        "contraindicated_regimens": [
            "Neoadjuvant multi-agent chemo for pure DCIS",
            "Trastuzumab without invasive HER2+",
        ],
        "seed_urls": [_PUBMED["ai"]],
    },
    {
        "case_id": "pheno_10_elderly_hr_pos",
        "kind": "phenotype",
        "patient_context": {
            "ER": "POSITIVE", "PR": "POSITIVE", "HER2": "NEGATIVE",
            "menopausal_status": "post", "age": 82, "stage": "IA", "grade": 1,
        },
        "preferred_regimen": "Aromatase inhibitor (de-escalate chemo)",
        "preferred_aliases": ["aromatase", "letrozole", "endocrine", "de-escalat"],
        "contraindicated_regimens": [
            "Aggressive multi-agent chemo without indication",
            "Trastuzumab without HER2+",
        ],
        "seed_urls": [_PUBMED["ai"]],
    },
    {
        "case_id": "pheno_11_her2_pos_residual",
        "kind": "phenotype",
        "patient_context": {
            "ER": "NEGATIVE", "PR": "NEGATIVE", "HER2": "POSITIVE",
            "residual_disease": True, "stage": "II", "grade": 3,
        },
        "preferred_regimen": "T-DM1 (ado-trastuzumab emtansine)",
        "preferred_aliases": ["t-dm1", "ado-trastuzumab", "kadcyla", "emtansine"],
        "contraindicated_regimens": [
            "Tamoxifen for TNBC",
            "Continue trastuzumab alone when residual disease warrants T-DM1",
        ],
        "seed_urls": [_PUBMED["tdm1"]],
    },
    {
        "case_id": "pheno_12_hr_pos_high_risk_premeno",
        "kind": "phenotype",
        "patient_context": {
            "ER": "POSITIVE", "PR": "NEGATIVE", "HER2": "NEGATIVE",
            "menopausal_status": "pre", "stage": "IIB", "grade": 3, "nodes": "N1",
        },
        "preferred_regimen": "Chemo then endocrine +/- OFS",
        "preferred_aliases": ["chemotherapy", "tamoxifen", "ofs", "endocrine"],
        "contraindicated_regimens": [
            "AI alone without OFS in premenopausal",
            "Trastuzumab without HER2+",
        ],
        "seed_urls": [_PUBMED["chemo"], _PUBMED["ai"]],
    },
    {
        "case_id": "pheno_13_inflammatory_her2_pos",
        "kind": "phenotype",
        "patient_context": {
            "ER": "NEGATIVE", "PR": "NEGATIVE", "HER2": "POSITIVE",
            "inflammatory": True, "stage": "IIIB", "grade": 3,
        },
        "preferred_regimen": "Neoadjuvant THP then surgery",
        "preferred_aliases": ["trastuzumab", "pertuzumab", "neoadjuvant", "thp"],
        "contraindicated_regimens": [
            "Surgery-first without systemic neoadjuvant",
            "Tamoxifen for TNBC",
        ],
        "seed_urls": [_PUBMED["trastuzumab"], _PUBMED["pertuzumab"]],
    },
    {
        "case_id": "pheno_14_brca_tnbc",
        "kind": "phenotype",
        "patient_context": {
            "ER": "NEGATIVE", "PR": "NEGATIVE", "HER2": "NEGATIVE",
            "BRCA1": True, "stage": "IIA", "grade": 3,
        },
        "preferred_regimen": "Platinum-containing chemotherapy +/- PARP consideration",
        "preferred_aliases": ["platinum", "carboplatin", "chemotherapy", "parp", "olaparib"],
        "contraindicated_regimens": [
            "Tamoxifen for TNBC",
            "Aromatase inhibitor alone",
        ],
        "seed_urls": [_PUBMED["chemo"]],
    },
    {
        "case_id": "pheno_15_hr_pos_oligomet",
        "kind": "phenotype",
        "patient_context": {
            "ER": "POSITIVE", "PR": "POSITIVE", "HER2": "NEGATIVE",
            "menopausal_status": "post", "stage": "IV", "grade": 2,
            "sites": ["bone"],
        },
        "preferred_regimen": "AI + CDK4/6 inhibitor",
        "preferred_aliases": ["ribociclib", "palbociclib", "abemaciclib", "letrozole", "cdk4"],
        "contraindicated_regimens": [
            "Nab-paclitaxel monotherapy as first-line endocrine-sensitive bone-only",
            "Trastuzumab without HER2+",
        ],
        "seed_urls": [_PUBMED["ribociclib"], _PUBMED["ai"]],
    },
    # ----------------------------------------------------------------------- #
    # 5 landmark Phase 3 head-to-heads
    {
        "case_id": "landmark_16_cleopatra",
        "kind": "landmark",
        "landmark_trial": "CLEOPATRA",
        "patient_context": {
            "ER": "NEGATIVE", "PR": "NEGATIVE", "HER2": "POSITIVE",
            "setting": "metastatic first-line", "stage": "IV",
        },
        "regimen_a": "Trastuzumab + Pertuzumab + Docetaxel (THP)",
        "regimen_b": "Trastuzumab + Docetaxel (TH)",
        "expected_winner": "A",
        "preferred_regimen": "Trastuzumab + Pertuzumab + Docetaxel (THP)",
        "preferred_aliases": ["pertuzumab", "thp", "dual"],
        "contraindicated_regimens": [],
        "seed_urls": [_PUBMED["pertuzumab"]],
    },
    {
        "case_id": "landmark_17_katherine",
        "kind": "landmark",
        "landmark_trial": "KATHERINE",
        "patient_context": {
            "ER": "NEGATIVE", "PR": "NEGATIVE", "HER2": "POSITIVE",
            "residual_disease": True, "setting": "adjuvant post-neoadjuvant",
        },
        "regimen_a": "T-DM1",
        "regimen_b": "Trastuzumab",
        "expected_winner": "A",
        "preferred_regimen": "T-DM1",
        "preferred_aliases": ["t-dm1", "ado-trastuzumab", "kadcyla"],
        "contraindicated_regimens": [],
        "seed_urls": [_PUBMED["tdm1"]],
    },
    {
        "case_id": "landmark_18_destiny_breast03",
        "kind": "landmark",
        "landmark_trial": "DESTINY-Breast03",
        "patient_context": {
            "ER": "POSITIVE", "PR": "POSITIVE", "HER2": "POSITIVE",
            "setting": "metastatic previously treated", "stage": "IV",
        },
        "regimen_a": "Trastuzumab deruxtecan (T-DXd)",
        "regimen_b": "T-DM1",
        "expected_winner": "A",
        "preferred_regimen": "Trastuzumab deruxtecan (T-DXd)",
        "preferred_aliases": ["t-dxd", "deruxtecan", "enhertu"],
        "contraindicated_regimens": [],
        "seed_urls": [_PUBMED["tdxd"], _PUBMED["tdm1"]],
    },
    {
        "case_id": "landmark_19_monaleesa2",
        "kind": "landmark",
        "landmark_trial": "MONALEESA-2",
        "patient_context": {
            "ER": "POSITIVE", "PR": "POSITIVE", "HER2": "NEGATIVE",
            "menopausal_status": "post", "setting": "metastatic first-line", "stage": "IV",
        },
        "regimen_a": "Letrozole + Ribociclib",
        "regimen_b": "Letrozole alone",
        "expected_winner": "A",
        "preferred_regimen": "Letrozole + Ribociclib",
        "preferred_aliases": ["ribociclib", "cdk4", "letrozole + ribociclib"],
        "contraindicated_regimens": [],
        "seed_urls": [_PUBMED["ribociclib"]],
    },
    {
        "case_id": "landmark_20_tailorx",
        "kind": "landmark",
        "landmark_trial": "TAILORx",
        "patient_context": {
            "ER": "POSITIVE", "PR": "POSITIVE", "HER2": "NEGATIVE",
            "recurrence_score": "low", "nodes": "N0", "stage": "I",
        },
        "regimen_a": "Endocrine therapy alone",
        "regimen_b": "Chemo + Endocrine therapy",
        "expected_winner": "A",
        "preferred_regimen": "Endocrine therapy alone",
        "preferred_aliases": ["endocrine alone", "endocrine therapy alone", "no chemo"],
        "contraindicated_regimens": [],
        "seed_urls": [_PUBMED["tailorx"]],
    },
]


assert len(CLINICAL_GROUND_TRUTH_CASES) == 20
assert sum(1 for c in CLINICAL_GROUND_TRUTH_CASES if c["kind"] == "phenotype") == 15
assert sum(1 for c in CLINICAL_GROUND_TRUTH_CASES if c["kind"] == "landmark") == 5


def build_envelopes_for_case(case: GroundTruthCase) -> tuple[dict, dict, dict, set[str]]:
    """Build screening/biopsy/therapy envelopes with preferred confidence boost."""
    ctx = case["patient_context"]
    urls = list(case.get("seed_urls") or [])
    seen = set(urls)
    evidence = [_ev(u, "ground-truth citation") for u in urls] or [
        _ev(_PUBMED["chemo"], "fallback citation")
    ]
    if not urls:
        seen.add(_PUBMED["chemo"])

    screening = {
        "provenance": {"model_state": "loaded_medsiglip", "model_name": "gt"},
        "findings": [{"label": "suspicious_mass", "score": 0.8}],
        "evidence": evidence[:1],
    }
    biopsy = {
        "provenance": {"model_state": "loaded_biopsy_probe", "model_name": "gt"},
        "subtype_prediction": "IDC",
        "confidence": 0.85,
        "receptor_panel": {
            "ER": ctx.get("ER"), "PR": ctx.get("PR"), "HER2": ctx.get("HER2"),
        },
        "grade": ctx.get("grade"),
        "evidence": evidence[:1],
    }

    if case["kind"] == "landmark":
        options = [
            {
                "regimen": case["regimen_a"],
                "line_of_therapy": 1,
                "confidence": 0.92 if case["expected_winner"] == "A" else 0.55,
                "evidence": evidence[:1],
            },
            {
                "regimen": case["regimen_b"],
                "line_of_therapy": 1,
                "confidence": 0.92 if case["expected_winner"] == "B" else 0.55,
                "evidence": evidence[:1],
            },
        ]
    else:
        options = [
            {
                "regimen": case["preferred_regimen"],
                "line_of_therapy": 1,
                "confidence": 0.93,
                "evidence": evidence[:1],
            },
        ]
        for i, bad in enumerate(case.get("contraindicated_regimens") or []):
            options.append({
                "regimen": bad,
                "line_of_therapy": 2 + i,
                "confidence": 0.15,
                "evidence": evidence[:1],
            })

    therapy = {
        "provenance": {"model_state": "loaded_therapy_arbiter_v1", "model_name": "gt"},
        "recommended_options": options[:3],
        "evidence": evidence[:1],
    }
    return screening, biopsy, therapy, seen
