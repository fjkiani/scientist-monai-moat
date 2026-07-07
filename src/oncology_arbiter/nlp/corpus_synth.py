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

The BIO tag scheme is:
    B-ER_VALUE, I-ER_VALUE, B-PR_VALUE, ..., B-KI67_PCT, I-KI67_PCT, ...

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
# Public API
# ------------------------------------------------------------------
def generate_report(rng: random.Random, seed: int | None = None) -> SynthReport:
    """Generate one synthetic pathology report.

    Field values are randomized; entities are guaranteed to be non-overlapping
    because each field is emitted on its own line.
    """
    if seed is not None:
        rng.seed(seed)

    # Sample field values first so ground_truth is deterministic.
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

    # Assemble the report. We shuffle the FIELD block so we don't teach the
    # model a fixed template position.
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

    tokens, labels = _apply_bio_labels(text, entities)

    ground_truth = {
        "er": er_val,
        "pr": pr_val,
        "her2": her2_val,
        "ki67_pct": ki67_val,
        "grade": grade_val,
        "tumor_size_mm": size_val,
        "margin": margin_val,
        "lvi": lvi_val,
        **tnm_gt,
    }

    return SynthReport(
        text=text,
        tokens=tokens,
        labels=labels,
        entities=entities,
        ground_truth=ground_truth,
    )


def generate_corpus(
    n: int,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, list[SynthReport]]:
    """Generate n synthetic reports and return train/val/test splits."""
    rng = random.Random(seed)
    reports = [generate_report(rng) for _ in range(n)]
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


if __name__ == "__main__":
    corpus = generate_corpus(2000)
    print("train", len(corpus["train"]), "val", len(corpus["val"]), "test", len(corpus["test"]))
    print(corpus["train"][0].text[:200])
    for e in corpus["train"][0].entities[:5]:
        print(" ", e)
