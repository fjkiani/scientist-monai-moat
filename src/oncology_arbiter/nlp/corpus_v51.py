"""v0.5.1 multi-signal weak-supervision corpus builder for Bio_ClinicalBERT.

Rationale
---------
v0.5.0 shipped with regex-only weak labels (`corpus_real.weak_label`) and
converged at ~F1=0.064 on the TCGA-242 test set — the "regex ceiling" blind
spot the v0.5.1 plan explicitly targets. This module composes four labeling
functions (LFs) into a Snorkel LabelModel, resolving conflicts via learned
per-LF accuracies to produce hard BIO labels stronger than any single
signal.

Four LFs
--------
1. **LF-regex**: existing `corpus_real.weak_label` regex spans.
2. **LF-LLM**: openrouter/tencent/hy3:free annotations, accepted spans
   only (self-consistency merge, min_votes=2 of 3).
3. **LF-ontology**: dictionary lookup over NSCLC molecular biomarker terms
   (KRAS, EGFR, ALK, ROS1, HER2 amp, BRAF, MET, PD-L1, TMB, MSI, KI67) with
   context-anchored value extraction.
4. **LF-section**: section-conditional relabeling. Reports have canonical
   sections (FINAL DIAGNOSIS, MOLECULAR, STAGING). We downweight spans
   claimed by non-anchor sections (e.g. an EGFR mention in the CLINICAL
   HISTORY section should NOT be labeled as a positive molecular finding
   — that's a patient-history mention, not a report finding).

Outputs
-------
- `train.jsonl`  — v0.5.1 training corpus with soft-labeled BIO tags
- `val.jsonl`    — held-out validation from LLM-annotated `val_lm.jsonl`
- `test_breast_crc.jsonl` — TCGA-242 gold (unchanged from v0.5.0)
- `test_nsclc.jsonl` — NSCLC gold from hy3 annotations, converted to BIO
- `manifest.json` — Snorkel LF accuracies, coverage, conflict rates,
                     class distribution, and provenance stamp

Provenance stamp: `REAL-v0.5.1-snorkel-openrouter-llm`
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from oncology_arbiter.nlp.corpus_real import (
    weak_label as regex_weak_label,
    whitespace_tokenize,
)
from oncology_arbiter.nlp.corpus_synth import (
    BIO_LABELS,
    ENTITY_TYPES,
    Entity,
    LABEL2ID,
    SynthReport,
    save_split,
)
from oncology_arbiter.nlp.weak_supervision import ABSTAIN, LabelModel

logger = logging.getLogger(__name__)

# ================================================================
# LF-ontology — NSCLC molecular biomarker dictionary with value regex
# ================================================================
# Source: NCI Thesaurus (HGNC gene symbols) + HGNC/UniProt canonical
# aliases for the 10 NSCLC entity types.

_ONTOLOGY_TERMS: dict[str, list[str]] = {
    # NSCLC molecular
    "KRAS": [r"\bKRAS\b", r"\bK-RAS\b", r"Kirsten\s+ras"],
    "EGFR": [r"\bEGFR\b", r"epidermal\s+growth\s+factor\s+receptor", r"ERBB1", r"HER1"],
    "ALK": [r"\bALK\b", r"anaplastic\s+lymphoma\s+kinase"],
    "ROS1": [r"\bROS1\b", r"ROS\s+proto-oncogene\s+1"],
    "PD_L1_TPS": [r"\bPD-?L1\b", r"programmed\s+death\s+ligand\s+1", r"\bTPS\b", r"tumor\s+proportion\s+score"],
    "TMB": [r"\bTMB\b", r"tumor\s+mutational?\s+burden"],
    "MSI": [r"\bMSI\b", r"microsatellite\s+instability", r"MSI-H", r"MSI-L", r"MSS"],
    "HER2_AMP": [r"\bHER2\b", r"ERBB2", r"HER-2/neu", r"c-erbB-2"],
    "BRAF": [r"\bBRAF\b", r"v-raf.*B1"],
    "MET": [r"\bMET\b(?!\w)", r"c-MET", r"MET\s+exon\s+14", r"MET\s+amplification"],
    # Breast biomarkers — high-value ontology signals
    "ER_VALUE": [r"\bER\b(?!\w)", r"estrogen\s+receptor", r"ER-?positive", r"ER-?negative"],
    "PR_VALUE": [r"\bPR\b(?!\w)", r"progesterone\s+receptor", r"PR-?positive", r"PR-?negative"],
    "HER2_VALUE": [r"\bHER2\b", r"ERBB2", r"HER-?2/neu"],
    "KI67_PCT": [r"\bKi-?67\b", r"MKI67", r"proliferation\s+index"],
}
_ONTOLOGY_PATTERNS: dict[str, re.Pattern[str]] = {
    et: re.compile("|".join(terms), re.IGNORECASE) for et, terms in _ONTOLOGY_TERMS.items()
}


def ontology_weak_label(text: str) -> list[Entity]:
    """Emit entity spans for ontology-listed terms.

    Fires only on exact term matches (case-insensitive). Value normalization
    is left to LF-regex — this LF's role is to boost recall of *entity type
    recognition* on biomarkers, where the regex LF often misses because it
    only fires when a value follows the marker word.
    """
    ents: list[Entity] = []
    occupied: list[tuple[int, int]] = []

    def taken(cs: int, ce: int) -> bool:
        return any(not (ce <= ts or cs >= te) for ts, te in occupied)

    # Order matters: prefer molecular entities over breast (both use HER2).
    # NSCLC first — reports that mention both HER2 and EGFR in NSCLC context
    # get HER2_AMP not HER2_VALUE. Breast reports don't mention EGFR.
    priority = [
        "EGFR", "KRAS", "ALK", "ROS1", "BRAF", "MET",  # NSCLC drivers
        "PD_L1_TPS", "TMB", "MSI",                       # NSCLC IO
        "HER2_AMP",                                       # NSCLC HER2
        "KI67_PCT", "ER_VALUE", "PR_VALUE", "HER2_VALUE", # breast
    ]
    for etype in priority:
        rx = _ONTOLOGY_PATTERNS[etype]
        for m in rx.finditer(text):
            cs, ce = m.start(), m.end()
            if taken(cs, ce):
                continue
            occupied.append((cs, ce))
            ents.append(
                Entity(
                    entity_type=etype,
                    value=m.group(0),
                    char_start=cs,
                    char_end=ce,
                    surface=m.group(0),
                )
            )
    return ents


# ================================================================
# LF-section — section-conditional relabeling
# ================================================================
# Pathology reports have canonical anchor sections. Findings mentioned in
# CLINICAL HISTORY / PRE-OP DIAGNOSIS sections are patient-history, not
# report findings, and should NOT be labeled as positive entities.

# Anchor headers that indicate *findings* (label as-is).
# TCGA text is often flattened (single-line paragraphs with period-separated
# clauses), so we match at *start-of-line* OR after *sentence-ending punctuation*.
_HEADER_LEAD = r"(?:^|(?<=[\.\n]))\s*"
_ANCHOR_SECTIONS = re.compile(
    _HEADER_LEAD +
    r"(?:FINAL\s+(?:PATHOLOGIC\s+)?DIAGNOSIS|(?:PATHOLOGIC\s+)?DIAGNOSIS|"
    r"DIAGNOSTIC\s+INTERPRETATION|SUMMARY|MOLECULAR\s+FINDINGS|"
    r"MOLECULAR\s+STUDIES|IMMUNOHISTOCHEMISTRY|IMMUNOPHENOTYPE|"
    r"STAGING|SYNOPTIC\s+(?:REPORT|SUMMARY|FINDINGS)|"
    r"IHC\s+RESULTS?|MICROSCOPIC\s+(?:DESCRIPTION|FINDINGS)|"
    r"PATHOLOGY\s+REPORT-?SUMMARY|COMMENT|FINAL\s+DIAGNOSIS)"
    r"[:.\s]",
    re.IGNORECASE | re.MULTILINE,
)
# Non-anchor headers indicating history/preop/administrative (abstain).
_NON_ANCHOR_SECTIONS = re.compile(
    _HEADER_LEAD +
    r"(?:CLINICAL\s+(?:HISTORY|INFORMATION|INDICATION)|"
    r"PRE-?OP(?:ERATIVE)?\s+DIAGNOSIS|PATIENT\s+HISTORY|"
    r"BRIEF\s+CLINICAL\s+HISTORY|GROSS\s+DESCRIPTION|"
    r"SPECIMENS?\s+SUBMITTED|PROCEDURE\s+DATE|OPERATIVE\s+FINDINGS|"
    r"MEDICAL\s+HISTORY|POST-?OP(?:ERATIVE)?\s+DIAGNOSIS)"
    r"[:.\s]",
    re.IGNORECASE | re.MULTILINE,
)


def build_section_map(text: str) -> np.ndarray:
    """Return an int8 array of length len(text): 1 for anchor-section chars,
    -1 for non-anchor-section chars, 0 for unknown/default."""
    section = np.zeros(len(text), dtype=np.int8)
    # Find all section-header starts and mark state until the next header.
    events: list[tuple[int, int]] = []  # (pos, state) state=+1 or -1
    for m in _ANCHOR_SECTIONS.finditer(text):
        events.append((m.start(), 1))
    for m in _NON_ANCHOR_SECTIONS.finditer(text):
        events.append((m.start(), -1))
    events.sort()
    if not events:
        # No headers detected — treat whole doc as anchor. This is common for
        # short synoptic reports and for the plain-text TCGA reports without
        # ALL-CAPS section headings.
        section[:] = 1
        return section
    # Segment: from events[i].pos to events[i+1].pos-1 is state events[i].state.
    prev_pos, prev_state = 0, 1  # default beginning is anchor
    for pos, state in events:
        section[prev_pos:pos] = prev_state
        prev_pos, prev_state = pos, state
    section[prev_pos:] = prev_state
    return section


def section_weak_label(text: str, upstream_entities: list[Entity]) -> list[Entity]:
    """Section-conditional filter: return only upstream entities whose center
    falls inside an anchor section (or unknown). Entities in non-anchor
    sections are dropped (LF abstains — represented as no vote from this LF).

    This LF *doesn't* invent new spans; it acts as a conflict-resolution
    signal against the regex LF, which is credulous about section context.
    """
    if not upstream_entities:
        return []
    section = build_section_map(text)
    kept: list[Entity] = []
    for e in upstream_entities:
        mid = (e.char_start + e.char_end) // 2
        if mid >= len(section):
            continue
        if section[mid] >= 0:  # anchor (+1) or unknown (0)
            kept.append(e)
    return kept


# ================================================================
# LF-LLM — load hy3 annotations JSONL into a per-report entity map
# ================================================================


def load_llm_annotations(path: Path) -> dict[str, list[Entity]]:
    """Load an LLM-annotation JSONL (from `scripts/llm_annotate.py`) into
    a dict[report_id -> list[Entity]] using only accepted spans."""
    per_report: dict[str, list[Entity]] = {}
    if not path.exists():
        logger.warning("LLM annotation file not found: %s", path)
        return per_report
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rid = d.get("report_id") or d.get("id")
            if not rid:
                continue
            ents: list[Entity] = []
            for e in d.get("accepted") or []:
                try:
                    ents.append(
                        Entity(
                            entity_type=str(e["entity_type"]),
                            value=str(e.get("value") or e.get("text_span") or ""),
                            char_start=int(e["char_start"]),
                            char_end=int(e["char_end"]),
                            surface=str(e.get("text_span") or e.get("value") or ""),
                        )
                    )
                except (KeyError, ValueError, TypeError):
                    continue
            per_report[rid] = ents
    return per_report


# ================================================================
# Build LF vote matrix at whitespace-token level
# ================================================================


def _bio_labels_from_entities(
    text: str, entities: list[Entity]
) -> tuple[list[str], list[tuple[int, int]], list[str], int, int]:
    """Tokenize text and emit BIO labels from entity spans. Returns
    (tokens, char_spans, bio_labels, n_align_ok, n_align_miss)."""
    tokens, spans = whitespace_tokenize(text)
    labels = ["O"] * len(tokens)
    n_ok = 0
    n_miss = 0
    # Deterministic first-writer-wins on overlaps.
    for e in sorted(entities, key=lambda x: (x.char_start, -(x.char_end - x.char_start))):
        cs, ce = e.char_start, e.char_end
        first = None
        for i, (ts, te) in enumerate(spans):
            if te <= cs:
                continue
            if ts >= ce:
                break
            if labels[i] != "O":
                continue
            if first is None:
                labels[i] = f"B-{e.entity_type}"
                first = i
            else:
                labels[i] = f"I-{e.entity_type}"
        if first is None:
            n_miss += 1
        else:
            n_ok += 1
    return tokens, spans, labels, n_ok, n_miss


def build_lf_votes(
    text: str,
    llm_entities: list[Entity] | None,
) -> tuple[list[str], list[tuple[int, int]], np.ndarray, list[str]]:
    """Compute the LF vote matrix L for one report.

    Returns:
        tokens: whitespace tokens
        spans: token char spans
        L: (n_tokens, n_lfs) array of BIO-label-id votes with ABSTAIN=-1
        lf_names: LF names in column order
    """
    # LF-regex
    regex_ents = regex_weak_label(text)
    # LF-ontology
    ont_ents = ontology_weak_label(text)
    # LF-LLM (from external annotation)
    llm_ents = llm_entities or []
    # LF-section: section-filtered regex (uses regex spans, drops history-section ones)
    section_ents = section_weak_label(text, regex_ents)

    lf_ent_sets = [regex_ents, llm_ents, ont_ents, section_ents]
    lf_names = ["LF-regex", "LF-LLM", "LF-ontology", "LF-section"]

    # Get canonical tokens once from raw text.
    tokens, spans = whitespace_tokenize(text)
    n_toks = len(tokens)
    n_lfs = len(lf_ent_sets)
    L = np.full((n_toks, n_lfs), ABSTAIN, dtype=np.int64)

    for j, ents in enumerate(lf_ent_sets):
        if not ents:
            continue
        _, _, bio, _, _ = _bio_labels_from_entities(text, ents)
        # LF-regex and LF-section: emit full BIO. Any O tag is ABSTAIN (LF
        # doesn't claim to know background tokens are background). For other
        # LFs same treatment.
        for i, tag in enumerate(bio):
            if tag == "O":
                continue
            L[i, j] = LABEL2ID[tag]
    return tokens, spans, L, lf_names


# ================================================================
# Fit Snorkel LabelModel over all training reports
# ================================================================


def fit_label_model(
    reports: list[dict[str, Any]],
    llm_per_report: dict[str, list[Entity]],
    *,
    n_classes: int = len(BIO_LABELS),
) -> tuple[LabelModel, dict[str, Any], list[dict[str, Any]]]:
    """Compute LF votes over every training report, fit the label model,
    and return the model + stats + per-report token/label output.

    Returns:
        lm: fitted LabelModel
        stats_dict: dict with lf_accuracies/coverage/conflict, class dist,
                    provenance stamp
        per_report_out: list of dicts with tokens, labels (BIO strings),
                        char_spans, entities, provenance for each report
    """
    all_L_blocks: list[np.ndarray] = []
    per_report_tokens: list[list[str]] = []
    per_report_spans: list[list[tuple[int, int]]] = []
    per_report_meta: list[dict[str, Any]] = []
    lf_names: list[str] | None = None

    for rec in reports:
        text = rec["text"]
        rid = rec["report_id"]
        llm_ents = llm_per_report.get(rid, [])
        tokens, spans, L, names = build_lf_votes(text, llm_ents)
        if lf_names is None:
            lf_names = names
        all_L_blocks.append(L)
        per_report_tokens.append(tokens)
        per_report_spans.append(spans)
        per_report_meta.append({"text": text, "report_id": rid, "cancer": rec.get("cancer")})

    if not all_L_blocks:
        raise RuntimeError("no reports supplied to fit_label_model")

    L_full = np.concatenate(all_L_blocks, axis=0)
    logger.info("LF matrix shape=%s  ABSTAIN=%.1f%%", L_full.shape, 100 * (L_full == ABSTAIN).mean())

    lm = LabelModel(n_classes=n_classes, lf_names=lf_names)
    fit_stats = lm.fit(L_full)
    logger.info("LabelModel fit: %s", fit_stats)

    # Predict hard BIO labels per token, reshape back per report.
    y_hat = lm.predict(L_full)

    per_report_out: list[dict[str, Any]] = []
    idx = 0
    id2label = {v: k for k, v in LABEL2ID.items()}
    for i, meta in enumerate(per_report_meta):
        n_i = len(per_report_tokens[i])
        y_i = y_hat[idx : idx + n_i]
        bio_labels = [id2label[int(y)] for y in y_i]
        # Recover entity spans from BIO for the record.
        ents = _decode_spans_char(per_report_tokens[i], per_report_spans[i], bio_labels)
        per_report_out.append(
            {
                "text": meta["text"],
                "report_id": meta["report_id"],
                "cancer": meta.get("cancer"),
                "tokens": per_report_tokens[i],
                "labels": bio_labels,
                "entities": [asdict(e) for e in ents],
                "provenance": "REAL-v0.5.1-snorkel-openrouter-llm",
            }
        )
        idx += n_i

    # Class distribution
    from collections import Counter
    labelid_counts = Counter(int(y) for y in y_hat)
    class_dist = {id2label[k]: int(v) for k, v in labelid_counts.items()}

    stats_dict = {
        "lf_accuracies": fit_stats.lf_accuracies,
        "lf_coverage": fit_stats.lf_coverage,
        "lf_conflict": fit_stats.lf_conflict,
        "n_classes": fit_stats.n_classes,
        "n_lfs": fit_stats.n_lfs,
        "n_tokens": int(L_full.shape[0]),
        "class_distribution": class_dist,
        "provenance": "REAL-v0.5.1-snorkel-openrouter-llm",
        "abstain_rate": float((L_full == ABSTAIN).mean()),
    }
    return lm, stats_dict, per_report_out


def _decode_spans_char(
    tokens: list[str], spans: list[tuple[int, int]], labels: list[str]
) -> list[Entity]:
    """From BIO labels recover Entity spans with char coordinates."""
    out: list[Entity] = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B-"):
            etype = lab[2:]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{etype}":
                j += 1
            cs = spans[i][0]
            ce = spans[j - 1][1]
            surface = " ".join(tokens[i:j])
            out.append(Entity(entity_type=etype, value=surface, char_start=cs, char_end=ce, surface=surface))
            i = j
        else:
            i += 1
    return out


# ================================================================
# Data loading + orchestration
# ================================================================


def _load_reports_jsonl(path: Path, cancer_default: str | None = None) -> list[dict[str, Any]]:
    """Read a JSONL of pathology reports. Each row must have text + report_id."""
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append(
                {
                    "text": d["text"],
                    "report_id": d.get("report_id") or d.get("id"),
                    "cancer": d.get("cancer") or cancer_default,
                }
            )
    return rows


def _reports_to_synthreports(records: list[dict[str, Any]]) -> list[SynthReport]:
    """Convert corpus_v51 rows into the SynthReport dataclass that
    clinicalbert_train.py's dataset expects."""
    out: list[SynthReport] = []
    for r in records:
        ents = [Entity(**e) if not isinstance(e, Entity) else e for e in r.get("entities", [])]
        out.append(
            SynthReport(
                text=r["text"],
                tokens=r["tokens"],
                labels=r["labels"],
                entities=ents,
                ground_truth={"report_id": r["report_id"], "cancer": r.get("cancer")},
                provenance=r.get("provenance", "REAL-v0.5.1-snorkel-openrouter-llm"),
            )
        )
    return out


def build_gold_bio_from_llm(reports: list[dict[str, Any]], llm_per_report: dict[str, list[Entity]]) -> tuple[list[dict[str, Any]], int, int]:
    """For gold splits, use LLM accepted-entity spans directly (self-consistency
    already merged upstream) with BIO tokenization. No Snorkel — this is the
    held-out labels themselves. Returns (rows, n_align_ok, n_align_miss)."""
    out: list[dict[str, Any]] = []
    total_ok = 0
    total_miss = 0
    for r in reports:
        rid = r["report_id"]
        ents = llm_per_report.get(rid, [])
        tokens, _, labels, n_ok, n_miss = _bio_labels_from_entities(r["text"], ents)
        total_ok += n_ok
        total_miss += n_miss
        out.append(
            {
                "text": r["text"],
                "report_id": rid,
                "cancer": r.get("cancer"),
                "tokens": tokens,
                "labels": labels,
                "entities": [asdict(e) for e in ents],
                "provenance": "GOLD-v0.5.1-openrouter-llm-selfconsistency",
            }
        )
    return out, total_ok, total_miss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-reports", type=Path, required=True,
                        help="JSONL of training reports (raw text + report_id).")
    parser.add_argument("--nsclc-gold-reports", type=Path, required=True,
                        help="JSONL of NSCLC gold reports (text + report_id).")
    parser.add_argument("--llm-train", type=Path, required=True,
                        help="hy3 annotations for training corpus (n_runs=1).")
    parser.add_argument("--llm-nsclc-gold", type=Path, required=True,
                        help="hy3 annotations for NSCLC gold (n_runs=3, min_votes=2).")
    parser.add_argument("--v05-corpus-dir", type=Path, required=True,
                        help="Existing v0.5.0 corpus dir. We reuse its val.jsonl "
                             "and test.jsonl (TCGA-242 breast/CRC gold) directly.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(asctime)s] %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[corpus_v51] loading raw report JSONLs...")
    train_reports = _load_reports_jsonl(args.train_reports)
    nsclc_gold_reports = _load_reports_jsonl(args.nsclc_gold_reports, cancer_default="nsclc")

    logger.info("  train=%d  nsclc_gold=%d",
                len(train_reports), len(nsclc_gold_reports))

    logger.info("[corpus_v51] loading LLM annotations...")
    llm_train = load_llm_annotations(args.llm_train)
    llm_nsclc = load_llm_annotations(args.llm_nsclc_gold)
    logger.info("  llm_train reports: %d  llm_nsclc reports: %d",
                len(llm_train), len(llm_nsclc))

    # Fit Snorkel label model on the training corpus.
    logger.info("[corpus_v51] fitting Snorkel LabelModel on training corpus...")
    lm, stats, train_out = fit_label_model(train_reports, llm_train)

    # NSCLC gold: use hy3 accepted-span annotations directly.
    nsclc_out, nsclc_align_ok, nsclc_align_miss = build_gold_bio_from_llm(nsclc_gold_reports, llm_nsclc)
    for r in nsclc_out:
        r["cancer"] = "nsclc"
    bio_alignment_error_rate = float(nsclc_align_miss) / float(max(1, nsclc_align_ok + nsclc_align_miss))
    logger.info(
        "[corpus_v51] BIO alignment: nsclc_ok=%d nsclc_miss=%d rate=%.4f",
        nsclc_align_ok, nsclc_align_miss, bio_alignment_error_rate,
    )

    # Reuse v0.5.0's val + breast/CRC test verbatim (TCGA-242 pathologist gold).
    v05_val_path = args.v05_corpus_dir / "val.jsonl"
    v05_test_path = args.v05_corpus_dir / "test.jsonl"
    logger.info("[corpus_v51] reusing v0.5.0 val=%s test=%s", v05_val_path, v05_test_path)

    def _load_v05_split(p: Path, cancer_default: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                d.setdefault("cancer", cancer_default)
                out.append(d)
        return out

    val_out = _load_v05_split(v05_val_path, cancer_default=None)
    breast_crc_out = _load_v05_split(v05_test_path, cancer_default="breast_crc")

    # Write splits.
    def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    _write_jsonl(train_out, args.out_dir / "train.jsonl")
    _write_jsonl(val_out, args.out_dir / "val.jsonl")
    _write_jsonl(nsclc_out, args.out_dir / "test_nsclc.jsonl")
    _write_jsonl(breast_crc_out, args.out_dir / "test_breast_crc.jsonl")

    manifest = {
        "provenance": "REAL-v0.5.1-snorkel-openrouter-llm",
        "n_train_reports": len(train_out),
        "n_val_reports": len(val_out),
        "n_nsclc_gold_reports": len(nsclc_out),
        "n_breast_crc_gold_reports": len(breast_crc_out),
        "snorkel_stats": stats,
        "llm_provider": "openrouter",
        "llm_model": "tencent/hy3:free",
        "self_consistency_n_runs": 3,
        "self_consistency_min_votes": 2,
        "test_breast_crc_source": "reused v0.5.0 TCGA-242 pathologist-adjudicated gold",
        "test_nsclc_source": "hy3 self-consistency annotations (min_votes=2 of 3 runs)",
        "val_source": "reused v0.5.0 val (regex weak-labeled)",
        "bio_alignment_error_rate": bio_alignment_error_rate,
        "bio_alignment_ok": nsclc_align_ok,
        "bio_alignment_miss": nsclc_align_miss,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("[corpus_v51] wrote %s", args.out_dir)
    logger.info("[corpus_v51] manifest: %s", json.dumps(manifest, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
