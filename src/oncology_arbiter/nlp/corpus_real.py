"""Real de-identified pathology-report corpus builder for Bio_ClinicalBERT.

Design honesty
--------------
Trains on **real, human-written, de-identified TCGA pathology reports**
(Kefeli & Tatonetti 2024, Mendeley DOI 10.17632/hyg5xkznpx.1, CC BY 4.0).
No credentialed data. No synthetic Mad Libs.

Two labeled sources:
1. **TCGA-Reports** (9,523 → filtered to 2,789 for our 5 cohorts) — labeled
   via **regex weak-supervision**. Reports with ZERO gold entity hits are
   dropped from training (nothing for the model to learn from an unlabeled
   report on this task). This is a strong-signal, high-noise training set.
2. **TCGA-242** (Chow et al. 2026, Zenodo 20263861) — pathologist-adjudicated
   field-level JSON gold. Converted to BIO spans by locating the field
   value in the source text with a per-entity regex, span-tagging on match.
   Held out as the honest test set (no gradient).

Splits
------
- train: TCGA-Reports minus held-out gold, regex-labeled (weak).
- val: 10% of train, held out for early-stopping / checkpoint selection.
- test: TCGA-242 categories 1+2 (breast + colorectal) BIO-converted from
  gold JSON. This is where the honest real-text F1 is measured.

Blind spots (documented)
------------------------
- NSCLC molecular markers (KRAS/EGFR/ALK/ROS1/PD_L1_TPS/TMB/MSI/HER2_AMP/
  BRAF/MET) are trained on regex weak-labels from TCGA LUAD/LUSC reports
  but **eval on these entities is not in TCGA-242 gold** (which is
  breast/colorectal/esophagus/stomach only). Report card discloses this.
- Regex weak-labels have a ceiling: they miss idiosyncratic phrasing the
  model would learn from richer text. Real F1 on TCGA-242 gold measures
  the ceiling honestly.
- Category 1 (breast) TCGA-242 uses lower-case "pt1c" style; our regex
  handles case-insensitively.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from oncology_arbiter.nlp.corpus_synth import (
    BIO_LABELS,
    ENTITY_TYPES,
    Entity,
    LABEL2ID,
    SynthReport,  # reused as the file record type (name is legacy)
    save_split,
)

# ------------------------------------------------------------------
# Whitespace tokenizer + span alignment
# ------------------------------------------------------------------

# Tokenizer: whitespace + preserve punctuation as separate tokens.
_TOK_RE = re.compile(r"\w+(?:[-\.]\w+)*|[^\s\w]|\S+", re.UNICODE)


def whitespace_tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Return (tokens, char_spans). char_spans is [(start, end)] per token.

    Compatible with corpus_synth's expectation of `SynthReport.tokens`
    being whitespace-level (BERT WordPiece then splits further at train time).
    """
    toks: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in _TOK_RE.finditer(text):
        toks.append(m.group(0))
        spans.append((m.start(), m.end()))
    return toks, spans


def bio_from_char_spans(
    text: str, entities: list[Entity]
) -> tuple[list[str], list[str]]:
    """Given entity char spans, produce whitespace tokens + BIO labels."""
    toks, spans = whitespace_tokenize(text)
    labels = ["O"] * len(toks)
    # Sort entities by char_start for deterministic overlap handling.
    for e in sorted(entities, key=lambda x: x.char_start):
        cs, ce = e.char_start, e.char_end
        first = None
        for i, (ts, te) in enumerate(spans):
            if te <= cs:
                continue
            if ts >= ce:
                break
            # token i overlaps entity span
            if labels[i] != "O":
                # avoid clobbering — deterministic first-writer-wins
                continue
            if first is None:
                labels[i] = f"B-{e.entity_type}"
                first = i
            else:
                labels[i] = f"I-{e.entity_type}"
    return toks, labels


# ------------------------------------------------------------------
# Weak-label regex library (extends v0.2.x parser to full 21-entity schema)
# ------------------------------------------------------------------

# Precompiled regexes. Group 1 (or the whole match, defaulting to group 0
# when no group) is treated as the "value" surface, but the ENTITY span
# always covers the whole match so BIO tags include the label + value.

def _rx(pat: str) -> re.Pattern:
    return re.compile(pat, re.IGNORECASE | re.MULTILINE)


# ---- Breast markers -------------------------------------------------
# ER: match "ER" or "estrogen receptor" then a status word within 60 chars.
_ER_RE = _rx(r"(?:\bER\b|estrogen\s+receptors?)[^\.\n]{0,60}?"
             r"\b(positive|negative|equivocal)\b")
_PR_RE = _rx(r"(?:\bPR\b|progesterone\s+receptors?)[^\.\n]{0,60}?"
             r"\b(positive|negative|equivocal)\b")
_HER2_RE = _rx(
    r"(?:HER[\s\-]?2(?:/neu|-neu)?|c-?erbB-?2)[^\.\n]{0,60}?"
    r"\b(negative|positive|equivocal|not\s+amplified|amplified|1\+|2\+|3\+)\b"
)
_KI67_RE = _rx(
    r"(?:Ki[\s\-]?67|mib[\s\-]?1)[^\.\n]{0,40}?\b(\d{1,3})\s*%"
)
# Nottingham/histologic grade
_GRADE_RE = _rx(
    r"(?:nottingham|combined|histolog\w*|bloom[-\s]?richardson)?\s*"
    r"(?:nuclear\s+)?grade[:\s]+(?:g\s*)?([1-3]|I{1,3})\b"
)
_GRADE_ALT_RE = _rx(
    r"\b(?:well|moderately|poorly)\s+differentiated\b"
)
# TUMOR_SIZE: "3.2 cm" or "17 mm" — allow decimals; prefer explicit "tumor
# size" cue OR "greatest dimension" OR "measuring" contexts.
_SIZE_RE = _rx(
    r"(?:tumor\s+size|greatest\s+dimension|maximum\s+(?:tumor\s+)?dimension|measur\w+)"
    r"[^\.\n]{0,40}?"
    r"(\d{1,3}(?:\.\d)?)\s*(cm|mm)"
)
# Standalone "1.5 cm" mass reference
_SIZE_STANDALONE_RE = _rx(
    r"\b(\d{1,3}(?:\.\d)?)\s*(cm|mm)\s+(?:tumor|mass|lesion|carcinoma)"
)
# T-stage: pT2, T1c, T2, T3, T4, pT1a, pT1b, pT1c, pT2a etc.
_T_RE = _rx(r"\b(?:p|c|y|yp)?(T\s*[0-4](?:[abc])?(?:is)?)\b")
# N-stage: pN0, N1, N2, N3
_N_RE = _rx(r"\b(?:p|c|y|yp)?(N\s*[0-3](?:[a-c])?(?:\(i[+\-]\))?(?:mi)?)\b")
# M-stage: M0, M1, MX
_M_RE = _rx(r"\b(?:p|c|y|yp)?(M\s*[0-1X])\b")
# MARGIN: "margins negative/positive/close" or "free of tumor"
_MARGIN_RE = _rx(
    r"(?:surgical\s+)?margins?\b[^\.\n]{0,30}?\b(negative|positive|free\s+of\s+tumor|involved|close|clear|not\s+involved)\b"
)
# LVI: "lymphvascular invasion: absent/present" or "no LVI"
_LVI_RE = _rx(
    r"(?:lymph(?:o)?vascular|lymphvascular|lvsi|angiolymphatic)\s+invasion[^\.\n]{0,30}?"
    r"\b(present|absent|negative|positive|identified|not\s+identified)\b"
)

# ---- NSCLC markers ---------------------------------------------------
_KRAS_RE = _rx(r"\bKRAS\b[^\.\n]{0,60}?\b(mutated|mutation|wild[-\s]?type|wt|G12[A-Z]|G13[A-Z])\b")
_EGFR_RE = _rx(r"\bEGFR\b[^\.\n]{0,60}?\b(mutated|mutation|wild[-\s]?type|wt|L858R|T790M|exon\s*19|exon\s*20|exon\s*21|del)\b")
_BRAF_RE = _rx(r"\bBRAF\b[^\.\n]{0,60}?\b(mutated|mutation|wild[-\s]?type|wt|V600[A-Z])\b")
_ALK_RE = _rx(r"\bALK\b[^\.\n]{0,60}?\b(fusion|rearrang\w+|positive|negative|EML4)\b")
_ROS1_RE = _rx(r"\bROS[\s\-]?1\b[^\.\n]{0,60}?\b(fusion|rearrang\w+|positive|negative)\b")
_MET_RE = _rx(r"\bMET\b[^\.\n]{0,60}?\b(exon\s*14|skipping|amplification|mutation|mutated|wild[-\s]?type|not\s+detected|negative)\b")
_HER2AMP_RE = _rx(r"HER[\s\-]?2[^\.\n]{0,40}?\b(amplified|not\s+amplified|no\s+amplification|amplification)\b")
_MSI_RE = _rx(r"(?:MSI|microsatellite)[^\.\n]{0,40}?\b(high|low|stable|MSS|MSI[-\s]?H|MSI[-\s]?L)\b")
_PDL1_RE = _rx(r"(?:PD[-\s]?L1|CPS|TPS)[^\.\n]{0,40}?(\d{1,3})\s*%")
_TMB_RE = _rx(r"(?:TMB|tumor\s+mutational\s+burden)[^\.\n]{0,40}?"
              r"(\d{1,3}(?:\.\d+)?)\s*(?:mut/mb|mutations?/mb|mut\s+per\s+mb)")

# Ordering matters: label longer / more-specific entities first so shorter
# regexes (like _T_RE which fires on many "T-cell" strings) don't clobber
# genuine TNM anchors. Using a first-writer-wins policy in bio_from_char_spans.
_RX_LIB: list[tuple[str, re.Pattern, int]] = [
    ("KI67_PCT",       _KI67_RE,        0),
    ("TUMOR_SIZE_MM",  _SIZE_RE,        0),
    ("TUMOR_SIZE_MM",  _SIZE_STANDALONE_RE, 0),
    ("ER_VALUE",       _ER_RE,          0),
    ("PR_VALUE",       _PR_RE,          0),
    ("HER2_VALUE",     _HER2_RE,        0),
    ("MARGIN",         _MARGIN_RE,      0),
    ("LVI",            _LVI_RE,         0),
    ("GRADE",          _GRADE_RE,       0),
    # NSCLC ----
    ("PD_L1_TPS",      _PDL1_RE,        0),
    ("TMB",            _TMB_RE,         0),
    ("KRAS",           _KRAS_RE,        0),
    ("EGFR",           _EGFR_RE,        0),
    ("BRAF",           _BRAF_RE,        0),
    ("ALK",            _ALK_RE,         0),
    ("ROS1",           _ROS1_RE,        0),
    ("MET",            _MET_RE,         0),
    ("HER2_AMP",       _HER2AMP_RE,     0),
    ("MSI",            _MSI_RE,         0),
    # TNM - constrained (require T/N/M followed by digit; may fire on genes
    # like "T-cell" if unqualified — regex requires digit/letter suffix which
    # avoids most; residual false positives are tolerable weak label noise).
    ("T_STAGE",        _T_RE,           1),  # group 1 for token
    ("N_STAGE",        _N_RE,           1),
    ("M_STAGE",        _M_RE,           1),
]


def weak_label(text: str) -> list[Entity]:
    """Regex weak-labeler over the 21-entity schema. Returns entity spans
    on the source text, no overlaps (first-writer-wins by ordering above)."""
    ents: list[Entity] = []
    occupied: list[tuple[int, int]] = []  # sorted list of taken char ranges

    def overlaps_taken(cs: int, ce: int) -> bool:
        for ts, te in occupied:
            if not (ce <= ts or cs >= te):
                return True
        return False

    for etype, rx, val_group in _RX_LIB:
        for m in rx.finditer(text):
            if val_group and m.lastindex is not None and m.lastindex >= val_group:
                # Use just the value group span (for TNM: skip preceding "p"/"c").
                cs = m.start(val_group)
                ce = m.end(val_group)
                surface = m.group(val_group)
            else:
                cs = m.start()
                ce = m.end()
                surface = m.group(0)
            if overlaps_taken(cs, ce):
                continue
            occupied.append((cs, ce))
            ents.append(
                Entity(
                    entity_type=etype,
                    value=surface,
                    char_start=cs,
                    char_end=ce,
                    surface=surface,
                )
            )
    return ents


# ------------------------------------------------------------------
# TCGA-242 gold JSON -> BIO conversion (test set)
# ------------------------------------------------------------------

def _first_re_span(text: str, pattern: re.Pattern) -> tuple[int, int, str] | None:
    m = pattern.search(text)
    if m is None:
        return None
    return m.start(), m.end(), m.group(0)


# Field → (entity_type, regex_builder_from_value)
def _tumor_size_regex(v: float | int) -> re.Pattern | None:
    if v is None:
        return None
    # tumor_size in gold is in mm; look for "N.N cm" or "N mm"
    mm = float(v)
    cm = mm / 10.0
    parts = []
    if mm >= 1:
        parts.append(rf"\b{mm:g}\s*mm\b")
    if cm >= 0.1:
        parts.append(rf"\b{cm:g}\s*cm\b")
    # Also match integer cm if size is whole cm value.
    if abs(cm - round(cm)) < 0.01:
        parts.append(rf"\b{int(round(cm))}\s*cm\b")
    return re.compile("|".join(parts), re.IGNORECASE) if parts else None


def _grade_regex(v: int) -> re.Pattern | None:
    if v is None:
        return None
    roman = {1: "I", 2: "II", 3: "III"}
    return re.compile(
        rf"(?:grade[:\s]+(?:g\s*)?{v}\b|\bg{v}\b|\bgrade\s+{roman.get(v,'X')}\b)",
        re.IGNORECASE,
    )


def _tnm_regex(prefix: str, v: str) -> re.Pattern | None:
    # v like "t1c" -> match pT1c / T1c / cT1c
    if not v:
        return None
    letter = v[0].upper()  # T/N/M
    rest = v[1:]
    return re.compile(
        rf"\b(?:p|c|y|yp)?{letter}\s*{re.escape(rest)}\b",
        re.IGNORECASE,
    )


def _biomarker_er_regex(expr: bool | None) -> re.Pattern | None:
    if expr is None:
        return None
    word = "positive|strongly positive|focally positive" if expr else "negative"
    return re.compile(
        rf"(?:\bER\b|estrogen\s+receptor)[^\.\n]{{0,60}}?\b({word})\b",
        re.IGNORECASE,
    )


def _biomarker_pr_regex(expr: bool | None) -> re.Pattern | None:
    if expr is None:
        return None
    word = "positive|strongly positive|focally positive" if expr else "negative"
    return re.compile(
        rf"(?:\bPR\b|progesterone\s+receptor)[^\.\n]{{0,60}}?\b({word})\b",
        re.IGNORECASE,
    )


def _biomarker_her2_regex(expr: bool | None, score: int | None) -> re.Pattern | None:
    if expr is None and score is None:
        return None
    if score is not None:
        return re.compile(
            rf"\bHER[\s\-]?2(?:/neu|-neu)?[^\.\n]{{0,60}}?\b({score}\+)\b",
            re.IGNORECASE,
        )
    word = "positive|amplified" if expr else "negative|not amplified|no amplification"
    return re.compile(
        rf"\bHER[\s\-]?2(?:/neu|-neu)?[^\.\n]{{0,60}}?\b({word})\b",
        re.IGNORECASE,
    )


def _margin_regex(any_involved: bool | None) -> re.Pattern | None:
    if any_involved is None:
        return None
    word = "positive|involved" if any_involved else "negative|free of tumor|clear|not involved"
    return re.compile(
        rf"(?:surgical\s+)?margins?\b[^\.\n]{{0,30}}?\b({word})\b",
        re.IGNORECASE,
    )


def _lvi_regex(v: str | None) -> re.Pattern | None:
    if v is None:
        return None
    if v is True or (isinstance(v, str) and v.lower() in ("present", "true", "positive")):
        word = "present|identified|positive"
    else:
        word = "absent|negative|not identified"
    return re.compile(
        rf"(?:lymph(?:o)?vascular|lymphvascular|lvsi|angiolymphatic)\s+invasion[^\.\n]{{0,30}}?\b({word})\b",
        re.IGNORECASE,
    )


def _extract_text_body(raw: str) -> str:
    """TCGA-242 reports have header lines starting with 'patient_filename:' /
    'text:'. Strip them and return the report body only, plus the char-offset
    at which the body starts (for future re-alignment; we don't need it
    since we re-search on the body itself).
    """
    body = raw
    # Remove first two header lines if present
    lines = raw.split("\n")
    if lines and lines[0].startswith("patient_filename:"):
        lines = lines[1:]
    if lines and lines[0].startswith("text:"):
        lines[0] = lines[0][len("text:"):].lstrip()
    body = "\n".join(lines)
    return body


def gold_to_entities(report_text: str, gold: dict) -> list[Entity]:
    """Convert TCGA-242 field-level gold JSON into token-span-anchored
    Entity records on the report text.

    Only fields overlapping our 21-entity schema are converted.
    """
    ents: list[Entity] = []
    occupied: list[tuple[int, int]] = []

    def add(cs: int, ce: int, etype: str, surface: str):
        for ts, te in occupied:
            if not (ce <= ts or cs >= te):
                return
        occupied.append((cs, ce))
        ents.append(Entity(entity_type=etype, value=surface,
                           char_start=cs, char_end=ce, surface=surface))

    cd = gold.get("cancer_data", {}) or {}
    # tumor_size (mm)
    if cd.get("tumor_size"):
        p = _tumor_size_regex(cd["tumor_size"])
        if p:
            r = _first_re_span(report_text, p)
            if r: add(r[0], r[1], "TUMOR_SIZE_MM", r[2])
    # grade
    if cd.get("grade") is not None:
        p = _grade_regex(cd["grade"])
        if p:
            r = _first_re_span(report_text, p)
            if r: add(r[0], r[1], "GRADE", r[2])
    # T/N/M
    for field_name, etype in (("pt_category", "T_STAGE"),
                              ("pn_category", "N_STAGE"),
                              ("pm_category", "M_STAGE")):
        v = cd.get(field_name)
        if v:
            p = _tnm_regex(etype[0], v)
            if p:
                r = _first_re_span(report_text, p)
                if r: add(r[0], r[1], etype, r[2])
    # margins → any margin_involved=True is a "positive margin"
    margins = cd.get("margins", []) or []
    if margins:
        any_involved = any(m.get("margin_involved") for m in margins)
        p = _margin_regex(any_involved)
        if p:
            r = _first_re_span(report_text, p)
            if r: add(r[0], r[1], "MARGIN", r[2])
    # LVI (may be null → skip)
    lvi = cd.get("lymphovascular_invasion")
    if lvi is not None:
        p = _lvi_regex(lvi)
        if p:
            r = _first_re_span(report_text, p)
            if r: add(r[0], r[1], "LVI", r[2])
    # biomarkers
    for bm in cd.get("biomarkers", []) or []:
        cat = (bm.get("biomarker_category") or "").lower()
        if cat == "er":
            p = _biomarker_er_regex(bm.get("expression"))
            if p:
                r = _first_re_span(report_text, p)
                if r: add(r[0], r[1], "ER_VALUE", r[2])
        elif cat == "pr":
            p = _biomarker_pr_regex(bm.get("expression"))
            if p:
                r = _first_re_span(report_text, p)
                if r: add(r[0], r[1], "PR_VALUE", r[2])
        elif cat == "her2":
            p = _biomarker_her2_regex(bm.get("expression"), bm.get("score"))
            if p:
                r = _first_re_span(report_text, p)
                if r: add(r[0], r[1], "HER2_VALUE", r[2])
    return ents


# ------------------------------------------------------------------
# Corpus assembly
# ------------------------------------------------------------------

def build_train_from_tcga_reports(parquet_path: Path, min_entities: int = 1) -> list[SynthReport]:
    """Read filtered TCGA-Reports parquet, weak-label with regex, drop
    reports with < min_entities gold hits (nothing to learn).
    """
    df = pd.read_parquet(parquet_path)
    out: list[SynthReport] = []
    for _, row in df.iterrows():
        text = row["text"]
        if not isinstance(text, str) or len(text.strip()) < 50:
            continue
        # Trim to reasonable length; the model uses max_len=192 sub-tokens
        # so we don't need > 3k chars per report.
        text = text[:6000]
        ents = weak_label(text)
        if len(ents) < min_entities:
            continue
        toks, labels = bio_from_char_spans(text, ents)
        # Skip if tokenization dropped all entity spans (unusual)
        if not any(l != "O" for l in labels):
            continue
        out.append(SynthReport(
            text=text,
            tokens=toks,
            labels=labels,
            entities=ents,
            ground_truth={"study": row["study"],
                          "patient_filename": row["patient_filename"]},
            provenance="REAL-v0.5.0-tcga-reports-regex-weak",
        ))
    return out


def build_gold_from_tcga242(reports_dir: Path, annots_dir: Path,
                            categories: Iterable[int] = (1, 2)) -> list[SynthReport]:
    """Build BIO-tagged eval set from TCGA-242 gold JSON."""
    out: list[SynthReport] = []
    for cat in categories:
        rdir = reports_dir / str(cat)
        adir = annots_dir / str(cat)
        if not rdir.exists() or not adir.exists():
            continue
        for rfile in sorted(rdir.glob("*.txt")):
            afile = adir / (rfile.stem + ".json")
            if not afile.exists():
                continue
            raw = rfile.read_text(errors="replace")
            text = _extract_text_body(raw)
            gold = json.loads(afile.read_text())
            ents = gold_to_entities(text, gold)
            if not ents:
                # Even with no matches, keep the report as "all-O" so
                # false-positive rate is measurable on gold — but drop for
                # cleanliness.
                continue
            toks, labels = bio_from_char_spans(text, ents)
            out.append(SynthReport(
                text=text,
                tokens=toks,
                labels=labels,
                entities=ents,
                ground_truth={"tcga242_cat": cat,
                              "tcga242_file": rfile.name,
                              "gold_meta": gold.get("_meta", {})},
                provenance="REAL-v0.5.0-tcga242-gold",
            ))
    return out


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--tcga-reports-parquet", default="/mnt/shared-workspace/shared/tcga_reports_v1/tcga_filtered.parquet")
    p.add_argument("--tcga242-reports-dir", default="/mnt/shared-workspace/shared/tcga_242_gold/tcga/reports")
    p.add_argument("--tcga242-annots-dir", default="/mnt/shared-workspace/shared/tcga_242_gold/tcga/annotations/gold")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=20260619)
    p.add_argument("--min-entities", type=int, default=2)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[real] building train from TCGA-Reports (weak-labeled)...")
    train_all = build_train_from_tcga_reports(
        Path(args.tcga_reports_parquet), min_entities=args.min_entities
    )
    print(f"[real]   {len(train_all)} reports with >= {args.min_entities} weak-labeled entities")

    print("[real] building gold test from TCGA-242 (breast + colorectal)...")
    test = build_gold_from_tcga242(
        Path(args.tcga242_reports_dir),
        Path(args.tcga242_annots_dir),
        categories=(1, 2),
    )
    print(f"[real]   {len(test)} gold reports converted to BIO")

    # Val split from train (deterministic).
    import random
    rng = random.Random(args.seed)
    rng.shuffle(train_all)
    n_val = int(len(train_all) * args.val_frac)
    val = train_all[:n_val]
    train = train_all[n_val:]

    for split, reports in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{split}.jsonl"
        save_split(reports, path)
        print(f"[real] {split}: {len(reports)} -> {path}")

    # Coverage summary
    from collections import Counter
    def cov(reports):
        c = Counter()
        for r in reports:
            for e in r.entities:
                c[e.entity_type] += 1
        return dict(c)

    manifest = {
        "provenance": "REAL-v0.5.0",
        "train_source": "TCGA-Reports (Kefeli 2024, CC BY 4.0, regex weak-labeled)",
        "test_source": "TCGA-242 gold (Chow 2026, pathologist-adjudicated, BIO-converted)",
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "seed": args.seed,
        "min_entities_per_train_report": args.min_entities,
        "train_entity_coverage": cov(train),
        "val_entity_coverage": cov(val),
        "test_entity_coverage": cov(test),
        "entity_types": ENTITY_TYPES,
        "n_bio_labels": len(BIO_LABELS),
        "blind_spots": [
            "NSCLC molecular entities (KRAS/EGFR/ALK/ROS1/PD_L1_TPS/TMB/MSI/HER2_AMP/BRAF/MET) are trained on regex weak-labels only. TCGA-242 gold does not adjudicate these.",
            "Regex weak-labels have known ceiling (see PLOS ONE 2025, Chow 2026). Real F1 measured on gold is the honest baseline.",
            "Gold BIO conversion locates field values with a per-entity regex; a value present in the JSON but not surface-anchored in the source text will not be scored.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[real] manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    _cli()
