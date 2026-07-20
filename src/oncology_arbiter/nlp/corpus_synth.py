"""
Synthetic pathology-report corpus generator for Bio_ClinicalBERT training.

Purpose
-------
Real breast-pathology corpora (MIMIC-CXR-JPG reports, i2b2/n2c2 shared tasks,
BC5CDR, BRONCO) all require credentialed / DUA-gated access. To avoid
credential gating in the sandbox and to keep every training example auditable,
we synthesize reports from a rule-based template engine that (a) randomizes
phrasing so the model does not memorize a canonical template, (b) emits
token-level BIO tags automatically so labels cannot drift from the surface
text, and (c) covers pathology-report edge cases the regex parser misses.

Design honesty
--------------
1. All reports are labeled `SYNTHETIC-v0.3.0`. The saved model card
   (docs/proofs/report_parser_clinicalbert_v1_metrics.json) MUST disclose
   this.
2. Phrasing is drawn from real-world templates seen in the CAP breast
   pathology synoptic and NCCN commentary; the sampled numeric values (ER
   percent, Ki-67 percent, tumor size, node counts) are randomized within
   clinically plausible ranges.
3. We deliberately introduce hedged / equivocal / contradicted phrasing
   (~10-15% of reports) so the trained model learns to abstain instead of
   silently coercing.
4. No real patient data. No real institution names. No real dates.

Entities (BIO labels)
---------------------
Breast entities (v0.3.0, unchanged):
- ER, PR                            -> value in {positive, negative,
                                                  equivocal, unknown}
- HER2                              -> value in {positive, negative,
                                                  equivocal, unknown}
- KI67_PCT                          -> percent (0-100)
- GRADE                             -> Nottingham grade 1|2|3
- T_STAGE / N_STAGE / M_STAGE       -> TNM tokens ("T2", "N1", "M0")
- TUMOR_SIZE_MM                     -> mm
- MARGIN                            -> {negative, positive, close}
- LVI                               -> {present, absent}

NSCLC entities (v0.3.1, added when cancer_type in {"nsclc", "mixed"}):
- KRAS, EGFR, BRAF                  -> value in {mutated, wild_type}
- ALK, ROS1                         -> value in {fusion_positive, negative}
- HER2_AMP                          -> value in {amplified, not_amplified}
- MET                               -> value in {mutated, not_detected}
- MSI                               -> value in {msi_high, mss}
- PD_L1_TPS                         -> percent (0-100)
- TMB                               -> mut/Mb (float, typical 0-40)

The BIO tag scheme is:
    B-ER_VALUE, I-ER_VALUE, B-PR_VALUE, ..., B-KRAS, I-KRAS, ...

Non-entity tokens get "O".
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Vocabulary of BIO labels the trained model must know about.
ENTITY_TYPES = [
    # breast (v0.3.0)
    "ER_VALUE",
    "PR_VALUE",
    "HER2_VALUE",
    "KI67_PCT",
    "GRADE",
    "T_STAGE",
    "N_STAGE",
    "M_STAGE",
    "TUMOR_SIZE_MM",
    "MARGIN",
    "LVI",
    # NSCLC (v0.3.1)
    "KRAS",
    "EGFR",
    "ALK",
    "ROS1",
    "PD_L1_TPS",
    "TMB",
    "MSI",
    "HER2_AMP",
    "BRAF",
    "MET",
]

BIO_LABELS = ["O"] + [f"B-{e}" for e in ENTITY_TYPES] + [f"I-{e}" for e in ENTITY_TYPES]
LABEL2ID = {l: i for i, l in enumerate(BIO_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# ------------------------------------------------------------------
# Phrasing pools
# ------------------------------------------------------------------
ER_POS_PHRASES = [
    "ER: positive",
    "Estrogen receptor: positive",
    "Estrogen receptor (ER): positive",
    "ER: POSITIVE",
    "Estrogen receptor: strongly positive",
    "ER: focally positive",
    "ER: positive in 95% of tumor cells",
    "ER: positive (Allred score 7)",
    "ER: positive in >90% of nuclei",
    "Estrogen receptor: strong nuclear positivity",
]
ER_NEG_PHRASES = [
    "ER: negative",
    "Estrogen receptor: negative",
    "ER: NEGATIVE",
    "Estrogen receptor (ER): negative",
    "ER: no nuclear staining",
    "ER: negative (<1% of tumor cells)",
    "Estrogen receptor: negative",
]
ER_EQUIV_PHRASES = [
    "ER: equivocal",
    "Estrogen receptor: weakly positive, 1-5% of tumor cells",
    "ER: focal weak positivity, clinical significance uncertain",
    "ER: 1% weakly positive (borderline)",
]

PR_POS_PHRASES = [
    "PR: positive",
    "Progesterone receptor: positive",
    "PR (Progesterone receptor): positive",
    "PR: positive in 80% of tumor cells",
    "Progesterone receptor: moderate to strong",
]
PR_NEG_PHRASES = [
    "PR: negative",
    "Progesterone receptor: negative",
    "PR: negative in tumor cells",
    "Progesterone receptor: no staining",
]
PR_EQUIV_PHRASES = [
    "PR: equivocal",
    "Progesterone receptor: weakly positive, ~2% of cells",
    "PR: focal weak positivity",
]

HER2_POS_PHRASES = [
    "HER2: positive",
    "HER2 (IHC): 3+",
    "HER2/neu: 3+ (positive)",
    "HER2: positive (IHC 3+)",
    "HER2 IHC: 3+, positive",
    "HER2/neu: positive by IHC",
    "HER2: positive; FISH confirms amplification",
]
HER2_NEG_PHRASES = [
    "HER2: negative",
    "HER2 (IHC): 1+",
    "HER2 (IHC): 0",
    "HER2: negative (IHC 1+)",
    "HER2/neu: negative",
    "HER2 IHC: 0, negative",
]
HER2_EQUIV_PHRASES = [
    "HER2: 2+ (equivocal)",
    "HER2 (IHC): 2+, reflex FISH pending",
    "HER2/neu: equivocal, awaiting FISH",
    "HER2 IHC: 2+, FISH not yet resulted",
]

GRADE_PHRASES = {
    1: [
        "Nottingham grade: 1",
        "Modified Bloom-Richardson grade: 1 (well differentiated)",
        "Nottingham grade 1 (3+3+1=7... wait grade 1 total 3-5)",  # deliberate weird phrasing removed below
        "Nottingham combined histologic grade: 1",
        "Nottingham histologic grade: 1 (low)",
    ],
    2: [
        "Nottingham grade: 2",
        "Nottingham histologic grade: 2 (moderately differentiated)",
        "Modified Bloom-Richardson grade: 2",
        "Nottingham combined histologic grade: 2 of 3",
    ],
    3: [
        "Nottingham grade: 3",
        "Nottingham histologic grade: 3 (poorly differentiated)",
        "Modified Bloom-Richardson grade: 3",
        "Nottingham combined histologic grade: 3 of 3 (high)",
    ],
}
# Drop the deliberately broken filler above so we don't teach the model garbage.
GRADE_PHRASES[1] = [p for p in GRADE_PHRASES[1] if "wait" not in p]

KI67_PHRASES = [
    "Ki-67 proliferation index: {pct}%",
    "Ki-67: {pct}%",
    "Ki67: {pct}%",
    "Proliferation index (Ki-67): {pct}%",
    "Ki-67 labeling index: {pct}%",
    "Ki-67 (MIB-1): {pct}%",
]
KI67_UNKNOWN_PHRASES = [
    "Ki-67: not performed",
    "Ki-67: pending",
]

TUMOR_SIZE_PHRASES = [
    "Tumor size: {mm} mm",
    "Invasive tumor size: {mm} mm",
    "Greatest tumor dimension: {mm} mm",
    "Tumor measures {mm} mm in greatest dimension",
    "Size of invasive component: {mm} mm",
]

MARGIN_PHRASES = {
    "negative": [
        "Margins: negative",
        "Margins: uninvolved by invasive carcinoma",
        "Surgical margins: negative for tumor",
        "All margins negative",
    ],
    "positive": [
        "Margins: positive",
        "Surgical margins: positive at posterior margin",
        "Margins: involved by invasive carcinoma",
    ],
    "close": [
        "Margins: close (< 2 mm)",
        "Margins: closest 1 mm at deep margin",
        "Anterior margin: 1 mm from invasive tumor",
    ],
}

LVI_PHRASES = {
    "present": [
        "Lymphovascular invasion: present",
        "Lymphovascular invasion: identified",
        "LVI: present",
    ],
    "absent": [
        "Lymphovascular invasion: absent",
        "LVI: not identified",
        "Lymphovascular invasion: not identified",
    ],
}

T_STAGE_TEMPLATE = [
    "Pathologic stage: pT{t}, N{n}, M{m}",
    "Stage: T{t}N{n}M{m}",
    "Final stage: pT{t} pN{n} pM{m}",
    "TNM stage: T{t}N{n}M{m}",
]

# ------------------------------------------------------------------
# NSCLC phrasing pools (v0.3.1)
# ------------------------------------------------------------------
# Each NSCLC entity has a positive/mutated pool and a negative/wild-type pool
# ~10 variants each. Percent-like scores (PD-L1 TPS, TMB) have templated
# numeric surfaces so BIO labels stay on the value substring.

KRAS_MUT_PHRASES = [
    "KRAS G12C: DETECTED",
    "KRAS G12C: DETECTED (variant allele frequency 34%)",
    "KRAS: G12C mutation identified",
    "KRAS codon 12 mutation (G12C)",
    "KRAS mutation: G12C",
    "KRAS: p.G12C, VAF 27%",
    "KRAS G12D mutation identified",
    "KRAS G12V: detected",
    "KRAS: G13D",
    "KRAS mutation: positive (G12C)",
]
KRAS_WT_PHRASES = [
    "KRAS: wild-type",
    "KRAS: wild type",
    "KRAS: no pathogenic variant",
    "KRAS exons 2-4: wild-type",
    "KRAS: no mutation detected",
    "KRAS: NOT DETECTED",
    "KRAS: negative for pathogenic variant",
    "KRAS mutation status: wild-type",
    "KRAS: no activating mutation",
    "KRAS: wildtype",
]

EGFR_MUT_PHRASES = [
    "EGFR exon 19 deletion",
    "EGFR L858R mutation identified",
    "EGFR: exon 19 deletion (E746_A750del)",
    "EGFR mutation: L858R",
    "EGFR: L858R detected",
    "EGFR exon 20 insertion",
    "EGFR: T790M resistance mutation",
    "EGFR: exon 19 deletion detected",
    "EGFR mutation: exon 21 L858R",
    "EGFR: activating mutation (L858R)",
]
EGFR_WT_PHRASES = [
    "EGFR: wild-type",
    "EGFR: wild type",
    "EGFR: no activating mutation",
    "EGFR: no mutation detected",
    "EGFR: NOT DETECTED",
    "EGFR mutation status: wild-type",
    "EGFR: negative for activating mutation",
    "EGFR exons 18-21: wild-type",
    "EGFR: no pathogenic variant",
    "EGFR: wildtype",
]

ALK_POS_PHRASES = [
    "ALK (D5F3): positive",
    "ALK fusion: positive",
    "ALK: EML4-ALK fusion detected",
    "ALK rearrangement: DETECTED",
    "ALK IHC (D5F3): positive",
    "ALK: fusion positive by FISH",
    "ALK: positive (EML4-ALK)",
    "ALK rearrangement: identified",
    "ALK: fusion detected",
    "ALK (D5F3): strongly positive",
]
ALK_NEG_PHRASES = [
    "ALK (D5F3): negative",
    "ALK fusion: negative",
    "ALK: no rearrangement",
    "ALK IHC: negative",
    "ALK: NOT DETECTED",
    "ALK rearrangement: not identified",
    "ALK: negative by FISH",
    "ALK: no fusion detected",
    "ALK (D5F3): no staining",
    "ALK: negative",
]

ROS1_POS_PHRASES = [
    "ROS1 (D4D6): positive",
    "ROS1 rearrangement: DETECTED",
    "ROS1: fusion positive",
    "ROS1: CD74-ROS1 fusion identified",
    "ROS1 IHC (D4D6): positive",
    "ROS1: positive by FISH",
    "ROS1 rearrangement: identified",
    "ROS1: fusion detected",
    "ROS1: positive (SDC4-ROS1)",
    "ROS1 (D4D6): strongly positive",
]
ROS1_NEG_PHRASES = [
    "ROS1 (D4D6): negative",
    "ROS1 rearrangement: negative",
    "ROS1: no rearrangement",
    "ROS1 IHC: negative",
    "ROS1: NOT DETECTED",
    "ROS1 rearrangement: not identified",
    "ROS1: negative by FISH",
    "ROS1: no fusion detected",
    "ROS1: negative",
    "ROS1 (D4D6): no staining",
]

BRAF_MUT_PHRASES = [
    "BRAF V600E: detected",
    "BRAF V600E mutation identified",
    "BRAF: V600E",
    "BRAF mutation: V600E",
    "BRAF: p.V600E, VAF 41%",
    "BRAF V600E: DETECTED",
    "BRAF: V600K mutation",
    "BRAF: non-V600 mutation identified",
    "BRAF mutation: positive (V600E)",
    "BRAF: activating mutation (V600E)",
]
BRAF_WT_PHRASES = [
    "BRAF: wild-type",
    "BRAF: wild type",
    "BRAF: no mutation detected",
    "BRAF: NOT DETECTED",
    "BRAF V600: wild-type",
    "BRAF: negative for pathogenic variant",
    "BRAF mutation status: wild-type",
    "BRAF: no activating mutation",
    "BRAF: no pathogenic variant",
    "BRAF: wildtype",
]

MET_POS_PHRASES = [
    "MET exon 14 skipping: detected",
    "MET: exon 14 skipping mutation",
    "MET amplification: detected",
    "MET: exon 14 skipping identified",
    "MET exon 14: DETECTED",
    "MET: amplified (copy number 8)",
    "MET: exon 14 skipping mutation identified",
    "MET amplification: positive",
    "MET: high-level amplification",
    "MET: exon 14 skipping (VAF 22%)",
]
MET_NEG_PHRASES = [
    "MET exon 14 skipping: not detected",
    "MET: no exon 14 skipping",
    "MET amplification: not detected",
    "MET: not amplified",
    "MET: NOT DETECTED",
    "MET exon 14: negative",
    "MET: wild-type",
    "MET amplification: negative",
    "MET: no pathogenic variant",
    "MET: no mutation detected",
]

HER2_AMP_POS_PHRASES = [
    "HER2 amplification: detected",
    "HER2: amplified",
    "HER2 (ERBB2): amplified",
    "HER2 amplification: DETECTED",
    "HER2: gene amplification identified",
    "HER2 (ERBB2) amplification: positive",
    "HER2: high-level amplification (copy number 12)",
    "ERBB2 amplification: detected",
    "HER2 amplification: positive by FISH",
    "HER2 (ERBB2): high-level amplification",
]
HER2_AMP_NEG_PHRASES = [
    "HER2 amplification: not detected",
    "HER2: not amplified",
    "HER2 (ERBB2): not amplified",
    "HER2 amplification: NOT DETECTED",
    "HER2: no gene amplification",
    "HER2 (ERBB2) amplification: negative",
    "HER2: not amplified (copy number 2)",
    "ERBB2 amplification: not detected",
    "HER2 amplification: negative by FISH",
    "HER2 (ERBB2): normal copy number",
]

MSI_HIGH_PHRASES = [
    "MSI: MSI-H",
    "MSI-H",
    "MSI: MSI-High (high instability)",
    "Microsatellite instability: MSI-H",
    "MSI status: MSI-High",
    "MSI: high instability",
    "Microsatellite instability: high",
    "MSI: MSI-H (unstable)",
    "Microsatellite instability status: MSI-H",
    "MSI: high",
]
MSI_STABLE_PHRASES = [
    "MSI: MSS (stable)",
    "MSI: MSS",
    "MSI: microsatellite stable",
    "Microsatellite instability: MSS",
    "MSI status: stable",
    "MSI: stable",
    "MSI: microsatellite stable (MSS)",
    "Microsatellite instability: stable",
    "MSI status: MSS",
    "MSI: MSS (microsatellite stable)",
]

# PD-L1 TPS: value is a percent. We template the number and BIO-tag the "{pct}%".
PD_L1_TPS_PHRASES = [
    "PD-L1 (22C3): Tumor Proportion Score = {pct}%",
    "PD-L1 TPS: {pct}%",
    "PD-L1 (22C3) TPS: {pct}%",
    "PD-L1: high expression (TPS {pct}%)",
    "PD-L1 (22C3): {pct}%",
    "PD-L1 tumor proportion score: {pct}%",
    "PD-L1 TPS = {pct}%",
    "PD-L1: {pct}% (22C3)",
    "PD-L1 IHC (22C3): TPS {pct}%",
    "PD-L1 TPS ({pct}%)",
]

# TMB: value is mut/Mb. Template the number; BIO-tag the numeric surface.
TMB_PHRASES = [
    "TMB: {tmb} mut/Mb",
    "Tumor mutational burden: {tmb} mut/Mb",
    "TMB: {tmb} mutations per megabase",
    "TMB (mut/Mb): {tmb}",
    "Tumor mutational burden (TMB): {tmb} mut/Mb",
    "TMB: high ({tmb} mut/Mb)",
    "TMB: intermediate ({tmb} mut/Mb)",
    "TMB: low ({tmb} mut/Mb)",
    "Tumor mutational burden: {tmb}/Mb",
    "TMB score: {tmb} mut/Mb",
]

NSCLC_REPORT_HEADERS = [
    "SURGICAL PATHOLOGY REPORT — LUNG",
    "PATHOLOGY REPORT — NSCLC",
    "MOLECULAR PATHOLOGY REPORT — LUNG ADENOCARCINOMA",
    "COMPREHENSIVE GENOMIC PROFILING — NSCLC",
    "LUNG BIOPSY — FINAL DIAGNOSIS",
]

NSCLC_DIAGNOSIS_PROSE = [
    "Invasive lung adenocarcinoma, {side} upper lobe.",
    "Non-small cell lung carcinoma (NSCLC), {side} lower lobe.",
    "Lung adenocarcinoma with acinar and lepidic patterns, {side} lung.",
    "NSCLC, adenocarcinoma subtype, {side} upper lobe.",
    "Poorly differentiated lung adenocarcinoma, {side} lung.",
]


REPORT_HEADERS = [
    "SURGICAL PATHOLOGY REPORT",
    "PATHOLOGY REPORT — BREAST",
    "FINAL PATHOLOGY REPORT",
    "SYNOPTIC PATHOLOGY REPORT",
    "BREAST BIOPSY — FINAL DIAGNOSIS",
]

DIAGNOSIS_PROSE = [
    "Invasive ductal carcinoma of the {side} breast.",
    "Infiltrating ductal carcinoma, {side} breast.",
    "Invasive carcinoma of no special type (NST), {side} breast.",
    "Invasive lobular carcinoma, {side} breast.",
]

CLOSE_LINES = [
    "Reviewed at multidisciplinary tumor board.",
    "Recommend clinical correlation.",
    "Report finalized by attending pathologist.",
    "End of report.",
]


# ------------------------------------------------------------------
# Token / label helpers
# ------------------------------------------------------------------
_TOKEN_SPLIT_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?%?|[^\sA-Za-z0-9]")


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    """Whitespace + punctuation tokenizer with char offsets.

    Returns list of (token, start, end) where start/end are char offsets in
    the original text.
    """
    out: list[tuple[str, int, int]] = []
    for m in _TOKEN_SPLIT_RE.finditer(text):
        out.append((m.group(0), m.start(), m.end()))
    return out


@dataclass
class Entity:
    """One labeled span in a synthetic report."""

    entity_type: str      # e.g. "ER_VALUE"
    value: str            # canonicalized value ("positive" / "negative" / int / mm)
    char_start: int
    char_end: int
    surface: str


@dataclass
class SynthReport:
    """One synthetic report with tokens, BIO labels, and entities."""

    text: str
    tokens: list[str]
    labels: list[str]                         # BIO string labels
    entities: list[Entity]
    ground_truth: dict = field(default_factory=dict)
    provenance: str = "SYNTHETIC-v0.3.0"


def _apply_bio_labels(text: str, entities: list[Entity]) -> tuple[list[str], list[str]]:
    """Turn char-span entities into token-level BIO labels.

    Only tokens whose (start,end) fully lies inside the entity's char span
    get labeled. Punctuation between the header ("ER:") and the value
    ("positive") is not labeled, so the model learns to focus on the value
    tokens, not the header text.
    """
    toks = _tokenize(text)
    labels = ["O"] * len(toks)
    for e in entities:
        first = True
        for i, (_tok, s, ee) in enumerate(toks):
            if s >= e.char_start and ee <= e.char_end:
                labels[i] = ("B-" if first else "I-") + e.entity_type
                first = False
    return [t for t, _, _ in toks], labels


# ------------------------------------------------------------------
# Field emitters
# ------------------------------------------------------------------
def _emit_er(rng: random.Random, val: str) -> tuple[str, str]:
    """Return (phrase, value_substring). value_substring is what we BIO-tag."""
    if val == "positive":
        p = rng.choice(ER_POS_PHRASES)
    elif val == "negative":
        p = rng.choice(ER_NEG_PHRASES)
    else:
        p = rng.choice(ER_EQUIV_PHRASES)
    # We label the value token(s) after the colon or after "ER".
    # Find "positive"/"negative"/"equivocal"/"weakly positive"/"no nuclear staining" etc.
    return p, _extract_value_span(p, val, "ER")


def _emit_pr(rng: random.Random, val: str) -> tuple[str, str]:
    if val == "positive":
        p = rng.choice(PR_POS_PHRASES)
    elif val == "negative":
        p = rng.choice(PR_NEG_PHRASES)
    else:
        p = rng.choice(PR_EQUIV_PHRASES)
    return p, _extract_value_span(p, val, "PR")


def _emit_her2(rng: random.Random, val: str) -> tuple[str, str]:
    if val == "positive":
        p = rng.choice(HER2_POS_PHRASES)
    elif val == "negative":
        p = rng.choice(HER2_NEG_PHRASES)
    else:
        p = rng.choice(HER2_EQUIV_PHRASES)
    return p, _extract_value_span(p, val, "HER2")


def _extract_value_span(phrase: str, canonical: str, header: str) -> str:
    """Best-effort: return the substring that carries the semantic value.

    We prefer the longest matching alternative from a canonical set so
    that phrasings like "no nuclear staining" get labeled as the negative
    value span (not just "negative").
    """
    lower = phrase.lower()
    priority = {
        "positive": [
            "strong nuclear positivity",
            "positive in 95% of tumor cells",
            "positive in 80% of tumor cells",
            "positive in >90% of nuclei",
            "positive in tumor cells",
            "focally positive",
            "strongly positive",
            "moderate to strong",
            "3+ (positive)",
            "IHC 3+",
            "3+",
            "POSITIVE",
            "positive",
        ],
        "negative": [
            "no nuclear staining",
            "no staining",
            "negative in tumor cells",
            "negative (<1% of tumor cells)",
            "IHC 0, negative",
            "1+",
            "0",
            "NEGATIVE",
            "negative",
        ],
        "equivocal": [
            "weakly positive, 1-5% of tumor cells",
            "weakly positive, ~2% of cells",
            "1% weakly positive (borderline)",
            "focal weak positivity, clinical significance uncertain",
            "focal weak positivity",
            "2+ (equivocal)",
            "2+, reflex FISH pending",
            "2+, FISH not yet resulted",
            "equivocal, awaiting FISH",
            "equivocal",
        ],
    }
    for cand in priority[canonical]:
        idx = lower.find(cand.lower())
        if idx >= 0:
            return phrase[idx : idx + len(cand)]
    # Fallback: label the canonical word if we can find it.
    idx = lower.find(canonical)
    return phrase[idx : idx + len(canonical)] if idx >= 0 else canonical


def _emit_ki67(rng: random.Random) -> tuple[str, int | None, str | None]:
    """Return (line, pct_value, value_span). value_span is the substring
    that carries the % number (e.g. '35%')."""
    if rng.random() < 0.05:
        return rng.choice(KI67_UNKNOWN_PHRASES), None, None
    pct = int(rng.gauss(20, 15))
    pct = max(1, min(95, pct))
    tmpl = rng.choice(KI67_PHRASES)
    line = tmpl.format(pct=pct)
    span = f"{pct}%"
    return line, pct, span


def _emit_grade(rng: random.Random) -> tuple[str, int, str]:
    grade = rng.choice([1, 2, 3])
    line = rng.choice(GRADE_PHRASES[grade])
    return line, grade, str(grade)


def _emit_tumor_size(rng: random.Random) -> tuple[str, float, str]:
    mm = round(rng.uniform(4, 65), 1)
    tmpl = rng.choice(TUMOR_SIZE_PHRASES)
    line = tmpl.format(mm=mm)
    return line, mm, f"{mm} mm"


def _emit_margin(rng: random.Random) -> tuple[str, str, str]:
    val = rng.choices(["negative", "close", "positive"], weights=[0.75, 0.15, 0.10])[0]
    line = rng.choice(MARGIN_PHRASES[val])
    return line, val, val


def _emit_lvi(rng: random.Random) -> tuple[str, str, str]:
    val = rng.choices(["absent", "present"], weights=[0.70, 0.30])[0]
    line = rng.choice(LVI_PHRASES[val])
    return line, val, val


def _emit_tnm(rng: random.Random) -> tuple[str, dict, dict]:
    """Return (line, gt_dict, spans_dict).

    spans_dict maps entity_type -> substring to tag.
    """
    t = rng.choice(["1a", "1b", "1c", "2", "3", "4"])
    n = rng.choice(["0", "1mi", "1", "2", "3"])
    m = rng.choice(["0", "0", "0", "1"])  # M0 heavily weighted
    tmpl = rng.choice(T_STAGE_TEMPLATE)
    line = tmpl.format(t=t, n=n, m=m)
    return (
        line,
        {"t_stage": t, "n_stage": n, "m_stage": m},
        {"T_STAGE": f"T{t}", "N_STAGE": f"N{n}", "M_STAGE": f"M{m}"},
    )


# ------------------------------------------------------------------
# NSCLC field emitters (v0.3.1)
# ------------------------------------------------------------------
# Each returns (phrase, canonical_value, span_surface). span_surface is the
# substring of `phrase` that carries the semantic value and gets BIO-tagged.

def _emit_binary_marker(
    rng: random.Random,
    pos_phrases: list[str],
    neg_phrases: list[str],
    pos_pool: list[str],
    neg_pool: list[str],
    canonical_pos: str,
    canonical_neg: str,
    pos_weight: float = 0.35,
) -> tuple[str, str, str]:
    """Sample a binary genomic marker phrase and return the value-carrying span.

    pos_pool / neg_pool are the substrings we prefer to BIO-tag (longest match
    first) so the model learns to attend to the value words rather than the
    entity name.
    """
    is_pos = rng.random() < pos_weight
    phrase = rng.choice(pos_phrases if is_pos else neg_phrases)
    val = canonical_pos if is_pos else canonical_neg
    lower = phrase.lower()
    # longest-first match
    pool = pos_pool if is_pos else neg_pool
    for cand in pool:
        idx = lower.find(cand.lower())
        if idx >= 0:
            return phrase, val, phrase[idx : idx + len(cand)]
    # fallback: label the canonical
    idx = lower.find(val.lower())
    if idx >= 0:
        return phrase, val, phrase[idx : idx + len(val)]
    return phrase, val, val


_KRAS_POS_POOL = [
    "G12C: DETECTED (variant allele frequency 34%)",
    "G12C mutation identified",
    "codon 12 mutation (G12C)",
    "positive (G12C)",
    "p.G12C, VAF 27%",
    "G12D mutation identified",
    "G12C: DETECTED",
    "G12V: detected",
    "G12C",
    "G12D",
    "G12V",
    "G13D",
]
_KRAS_NEG_POOL = [
    "no pathogenic variant",
    "no activating mutation",
    "no mutation detected",
    "exons 2-4: wild-type",
    "wild-type",
    "wild type",
    "wildtype",
    "NOT DETECTED",
    "negative for pathogenic variant",
]
_EGFR_POS_POOL = [
    "exon 19 deletion (E746_A750del)",
    "exon 19 deletion detected",
    "exon 19 deletion",
    "exon 20 insertion",
    "T790M resistance mutation",
    "L858R mutation identified",
    "activating mutation (L858R)",
    "exon 21 L858R",
    "L858R detected",
    "L858R",
]
_EGFR_NEG_POOL = [
    "no activating mutation",
    "no mutation detected",
    "no pathogenic variant",
    "negative for activating mutation",
    "exons 18-21: wild-type",
    "wild-type",
    "wild type",
    "wildtype",
    "NOT DETECTED",
]
_ALK_POS_POOL = [
    "EML4-ALK fusion detected",
    "rearrangement: DETECTED",
    "fusion positive by FISH",
    "IHC (D5F3): positive",
    "(D5F3): strongly positive",
    "(D5F3): positive",
    "positive (EML4-ALK)",
    "rearrangement: identified",
    "fusion: positive",
    "fusion detected",
    "positive",
]
_ALK_NEG_POOL = [
    "no rearrangement",
    "no fusion detected",
    "rearrangement: not identified",
    "negative by FISH",
    "IHC: negative",
    "(D5F3): no staining",
    "(D5F3): negative",
    "fusion: negative",
    "NOT DETECTED",
    "negative",
]
_ROS1_POS_POOL = [
    "CD74-ROS1 fusion identified",
    "SDC4-ROS1",
    "rearrangement: DETECTED",
    "positive by FISH",
    "IHC (D4D6): positive",
    "(D4D6): strongly positive",
    "(D4D6): positive",
    "positive (SDC4-ROS1)",
    "rearrangement: identified",
    "fusion positive",
    "fusion detected",
    "positive",
]
_ROS1_NEG_POOL = [
    "no rearrangement",
    "no fusion detected",
    "rearrangement: not identified",
    "rearrangement: negative",
    "negative by FISH",
    "IHC: negative",
    "(D4D6): no staining",
    "(D4D6): negative",
    "NOT DETECTED",
    "negative",
]
_BRAF_POS_POOL = [
    "V600E mutation identified",
    "V600E: DETECTED",
    "V600E: detected",
    "non-V600 mutation identified",
    "activating mutation (V600E)",
    "positive (V600E)",
    "p.V600E, VAF 41%",
    "mutation: V600E",
    "V600K mutation",
    "V600E",
    "V600K",
]
_BRAF_NEG_POOL = [
    "no mutation detected",
    "no activating mutation",
    "no pathogenic variant",
    "negative for pathogenic variant",
    "V600: wild-type",
    "wild-type",
    "wild type",
    "wildtype",
    "NOT DETECTED",
]
_MET_POS_POOL = [
    "exon 14 skipping mutation identified",
    "exon 14 skipping mutation",
    "exon 14 skipping identified",
    "exon 14 skipping (VAF 22%)",
    "exon 14 skipping: detected",
    "exon 14: DETECTED",
    "amplification: detected",
    "amplification: positive",
    "amplified (copy number 8)",
    "high-level amplification",
    "amplified",
]
_MET_NEG_POOL = [
    "no exon 14 skipping",
    "exon 14 skipping: not detected",
    "amplification: not detected",
    "amplification: negative",
    "no pathogenic variant",
    "no mutation detected",
    "not amplified",
    "exon 14: negative",
    "wild-type",
    "NOT DETECTED",
]
_HER2_AMP_POS_POOL = [
    "amplification: DETECTED",
    "amplification: detected",
    "amplification: positive by FISH",
    "gene amplification identified",
    "high-level amplification (copy number 12)",
    "high-level amplification",
    "(ERBB2): high-level amplification",
    "(ERBB2): amplified",
    "amplified",
]
_HER2_AMP_NEG_POOL = [
    "amplification: not detected",
    "amplification: NOT DETECTED",
    "amplification: negative by FISH",
    "no gene amplification",
    "not amplified (copy number 2)",
    "normal copy number",
    "(ERBB2): not amplified",
    "not amplified",
]
_MSI_HIGH_POOL = [
    "MSI-High (high instability)",
    "MSI-High",
    "MSI-H (unstable)",
    "MSI-H",
    "high instability",
    "high",
]
_MSI_STABLE_POOL = [
    "microsatellite stable (MSS)",
    "microsatellite stable",
    "MSS (stable)",
    "MSS (microsatellite stable)",
    "MSS",
    "stable",
]


def _emit_kras(rng):
    return _emit_binary_marker(
        rng, KRAS_MUT_PHRASES, KRAS_WT_PHRASES,
        _KRAS_POS_POOL, _KRAS_NEG_POOL, "mutated", "wild_type",
    )

def _emit_egfr(rng):
    return _emit_binary_marker(
        rng, EGFR_MUT_PHRASES, EGFR_WT_PHRASES,
        _EGFR_POS_POOL, _EGFR_NEG_POOL, "mutated", "wild_type",
    )

def _emit_alk(rng):
    return _emit_binary_marker(
        rng, ALK_POS_PHRASES, ALK_NEG_PHRASES,
        _ALK_POS_POOL, _ALK_NEG_POOL, "fusion_positive", "negative",
        pos_weight=0.15,
    )

def _emit_ros1(rng):
    return _emit_binary_marker(
        rng, ROS1_POS_PHRASES, ROS1_NEG_PHRASES,
        _ROS1_POS_POOL, _ROS1_NEG_POOL, "fusion_positive", "negative",
        pos_weight=0.10,
    )

def _emit_braf(rng):
    return _emit_binary_marker(
        rng, BRAF_MUT_PHRASES, BRAF_WT_PHRASES,
        _BRAF_POS_POOL, _BRAF_NEG_POOL, "mutated", "wild_type",
        pos_weight=0.15,
    )

def _emit_met(rng):
    return _emit_binary_marker(
        rng, MET_POS_PHRASES, MET_NEG_PHRASES,
        _MET_POS_POOL, _MET_NEG_POOL, "mutated", "not_detected",
        pos_weight=0.15,
    )

def _emit_her2_amp(rng):
    return _emit_binary_marker(
        rng, HER2_AMP_POS_PHRASES, HER2_AMP_NEG_PHRASES,
        _HER2_AMP_POS_POOL, _HER2_AMP_NEG_POOL, "amplified", "not_amplified",
        pos_weight=0.20,
    )

def _emit_msi(rng):
    return _emit_binary_marker(
        rng, MSI_HIGH_PHRASES, MSI_STABLE_PHRASES,
        _MSI_HIGH_POOL, _MSI_STABLE_POOL, "msi_high", "mss",
        pos_weight=0.10,
    )


def _emit_pd_l1_tps(rng) -> tuple[str, int, str]:
    """Return (phrase, pct_value, value_span). value_span is '{pct}%'."""
    # Bimodal: many 0-5%, many 50-95%, some middle.
    r = rng.random()
    if r < 0.30:
        pct = rng.randint(0, 5)
    elif r < 0.55:
        pct = rng.randint(6, 49)
    else:
        pct = rng.randint(50, 95)
    tmpl = rng.choice(PD_L1_TPS_PHRASES)
    line = tmpl.format(pct=pct)
    span = f"{pct}%"
    return line, pct, span


def _emit_tmb(rng) -> tuple[str, float, str]:
    """Return (phrase, tmb_value, value_span)."""
    # Log-uniform-ish: many low, tail high.
    if rng.random() < 0.6:
        tmb = round(rng.uniform(0.2, 9.9), 1)
    else:
        tmb = round(rng.uniform(10.0, 38.0), 1)
    tmpl = rng.choice(TMB_PHRASES)
    line = tmpl.format(tmb=tmb)
    span = f"{tmb}"
    return line, tmb, span


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def generate_report(
    rng: random.Random,
    seed: int | None = None,
    cancer_type: str = "breast",
) -> SynthReport:
    """Generate one synthetic pathology report.

    Parameters
    ----------
    rng : random.Random
        Seeded RNG.
    seed : int | None
        Optional re-seed for reproducible single-report generation.
    cancer_type : {"breast", "nsclc", "mixed"}
        - "breast" (default, backward compatible): only breast entities.
        - "nsclc": breast entities are still emitted so the label space is
          shared, but the header + diagnosis are NSCLC, and the 10 NSCLC
          entities are added.
        - "mixed": ~50/50 sample of breast vs. NSCLC per report.

    Field values are randomized; entities are guaranteed to be non-overlapping
    because each field is emitted on its own line.
    """
    if seed is not None:
        rng.seed(seed)

    if cancer_type == "mixed":
        this_cancer = "nsclc" if rng.random() < 0.5 else "breast"
    elif cancer_type in ("breast", "nsclc"):
        this_cancer = cancer_type
    else:
        raise ValueError(f"unknown cancer_type: {cancer_type!r}")

    # Sample breast field values first so ground_truth is deterministic.
    er_val = rng.choices(["positive", "negative", "equivocal"], weights=[0.6, 0.3, 0.1])[0]
    pr_val = rng.choices(["positive", "negative", "equivocal"], weights=[0.55, 0.35, 0.1])[0]
    her2_val = rng.choices(["positive", "negative", "equivocal"], weights=[0.2, 0.65, 0.15])[0]

    er_line, er_span = _emit_er(rng, er_val)
    pr_line, pr_span = _emit_pr(rng, pr_val)
    her2_line, her2_span = _emit_her2(rng, her2_val)
    ki67_line, ki67_val, ki67_span = _emit_ki67(rng)
    grade_line, grade_val, grade_span = _emit_grade(rng)
    size_line, size_val, size_span = _emit_tumor_size(rng)
    margin_line, margin_val, margin_span = _emit_margin(rng)
    lvi_line, lvi_val, lvi_span = _emit_lvi(rng)
    tnm_line, tnm_gt, tnm_spans = _emit_tnm(rng)

    # NSCLC field values (only emitted when this_cancer == "nsclc").
    nsclc_emitters = None
    if this_cancer == "nsclc":
        nsclc_emitters = {
            "KRAS":      _emit_kras(rng),
            "EGFR":      _emit_egfr(rng),
            "ALK":       _emit_alk(rng),
            "ROS1":      _emit_ros1(rng),
            "BRAF":      _emit_braf(rng),
            "MET":       _emit_met(rng),
            "HER2_AMP":  _emit_her2_amp(rng),
            "MSI":       _emit_msi(rng),
        }
        pd_l1_line, pd_l1_val, pd_l1_span = _emit_pd_l1_tps(rng)
        tmb_line, tmb_val, tmb_span = _emit_tmb(rng)
    else:
        pd_l1_line = pd_l1_val = pd_l1_span = None
        tmb_line = tmb_val = tmb_span = None

    # Assemble the report. We shuffle the FIELD block so we don't teach the
    # model a fixed template position.
    if this_cancer == "nsclc":
        header = rng.choice(NSCLC_REPORT_HEADERS)
        diagnosis = rng.choice(NSCLC_DIAGNOSIS_PROSE).format(side=rng.choice(["left", "right"]))
    else:
        header = rng.choice(REPORT_HEADERS)
        diagnosis = rng.choice(DIAGNOSIS_PROSE).format(side=rng.choice(["left", "right"]))
    field_lines = [
        er_line,
        pr_line,
        her2_line,
        ki67_line,
        grade_line,
        size_line,
        margin_line,
        lvi_line,
        tnm_line,
    ]
    if nsclc_emitters is not None:
        for _etype, (line, _val, _span) in nsclc_emitters.items():
            field_lines.append(line)
        field_lines.append(pd_l1_line)
        field_lines.append(tmb_line)
    rng.shuffle(field_lines)
    close = rng.choice(CLOSE_LINES)

    parts = [header, "", diagnosis, ""] + field_lines + ["", close]
    text = "\n".join(parts)

    # Build entities by locating each labeled span within the assembled text.
    # We use find() with a running cursor so we only match once, and we
    # search from the start of the corresponding field line to avoid the
    # tumor-size percentage colliding with Ki-67 percentage etc.
    entities: list[Entity] = []

    def _find_span_after(needle: str, after_line_start: int) -> tuple[int, int] | None:
        if not needle:
            return None
        idx = text.find(needle, after_line_start)
        if idx < 0:
            return None
        return idx, idx + len(needle)

    line_starts: dict[str, int] = {}
    for line in field_lines:
        line_starts[line] = text.find(line)

    def _entity(entity_type: str, value: str, needle: str | None, line: str) -> None:
        if needle is None:
            return
        span = _find_span_after(needle, line_starts.get(line, 0))
        if span is None:
            return
        entities.append(
            Entity(
                entity_type=entity_type,
                value=str(value),
                char_start=span[0],
                char_end=span[1],
                surface=text[span[0] : span[1]],
            )
        )

    _entity("ER_VALUE", er_val, er_span, er_line)
    _entity("PR_VALUE", pr_val, pr_span, pr_line)
    _entity("HER2_VALUE", her2_val, her2_span, her2_line)
    if ki67_val is not None:
        _entity("KI67_PCT", str(ki67_val), ki67_span, ki67_line)
    _entity("GRADE", str(grade_val), grade_span, grade_line)
    _entity("TUMOR_SIZE_MM", str(size_val), size_span, size_line)
    _entity("MARGIN", margin_val, margin_span, margin_line)
    _entity("LVI", lvi_val, lvi_span, lvi_line)
    for etype, surf in tnm_spans.items():
        _entity(etype, tnm_gt[etype.lower()], surf, tnm_line)

    nsclc_gt: dict = {}
    if nsclc_emitters is not None:
        for etype, (line, val, span) in nsclc_emitters.items():
            _entity(etype, val, span, line)
            nsclc_gt[etype.lower()] = val
        _entity("PD_L1_TPS", str(pd_l1_val), pd_l1_span, pd_l1_line)
        nsclc_gt["pd_l1_tps"] = pd_l1_val
        _entity("TMB", str(tmb_val), tmb_span, tmb_line)
        nsclc_gt["tmb"] = tmb_val

    tokens, labels = _apply_bio_labels(text, entities)

    ground_truth = {
        "cancer_type": this_cancer,
        "er": er_val,
        "pr": pr_val,
        "her2": her2_val,
        "ki67_pct": ki67_val,
        "grade": grade_val,
        "tumor_size_mm": size_val,
        "margin": margin_val,
        "lvi": lvi_val,
        **tnm_gt,
        **nsclc_gt,
    }

    return SynthReport(
        text=text,
        tokens=tokens,
        labels=labels,
        entities=entities,
        ground_truth=ground_truth,
        provenance="SYNTHETIC-v0.3.1" if this_cancer == "nsclc" else "SYNTHETIC-v0.3.0",
    )


def generate_corpus(
    n: int,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    cancer_type: str = "breast",
) -> dict[str, list[SynthReport]]:
    """Generate n synthetic reports and return train/val/test splits.

    Parameters
    ----------
    cancer_type : {"breast", "nsclc", "mixed"}
        Passed through to `generate_report`. Default "breast" preserves
        v0.3.0 backward compatibility. Use "mixed" for a corpus that trains
        one model to handle both breast and NSCLC reports.
    """
    rng = random.Random(seed)
    reports = [generate_report(rng, cancer_type=cancer_type) for _ in range(n)]
    rng.shuffle(reports)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = reports[:n_test]
    val = reports[n_test : n_test + n_val]
    train = reports[n_test + n_val :]
    return {"train": train, "val": val, "test": test}


def save_split(split: list[SynthReport], path: Path) -> None:
    """Dump a split to JSONL (one report per line)."""
    with path.open("w") as f:
        for r in split:
            f.write(
                json.dumps(
                    {
                        "text": r.text,
                        "tokens": r.tokens,
                        "labels": r.labels,
                        "entities": [
                            {
                                "entity_type": e.entity_type,
                                "value": e.value,
                                "char_start": e.char_start,
                                "char_end": e.char_end,
                                "surface": e.surface,
                            }
                            for e in r.entities
                        ],
                        "ground_truth": r.ground_truth,
                        "provenance": r.provenance,
                    }
                )
                + "\n"
            )


def _cli() -> None:
    """Standalone CLI: generate + save a corpus to JSONL splits.

    Example:
        python -m oncology_arbiter.nlp.corpus_synth \\
            --cancer mixed --n-reports 4000 --seed 42 \\
            --out-dir /mnt/shared-workspace/shared/nsclc_corpus_v031
    """
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--cancer", choices=["breast", "nsclc", "mixed"], default="breast")
    p.add_argument("--n-reports", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = generate_corpus(
        args.n_reports,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        cancer_type=args.cancer,
    )
    for split, reports in corpus.items():
        path = out_dir / f"{split}.jsonl"
        save_split(reports, path)
        print(f"[corpus] {split}: {len(reports)} reports -> {path}")

    # Provenance manifest.
    manifest = {
        "cancer_type": args.cancer,
        "n_reports": args.n_reports,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "provenance": "SYNTHETIC-v0.3.1" if args.cancer != "breast" else "SYNTHETIC-v0.3.0",
        "entity_types": ENTITY_TYPES,
        "n_bio_labels": len(BIO_LABELS),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[corpus] manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    _cli()
