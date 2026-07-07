"""Entity-level eval + fused-vs-regex agreement for trained Bio_ClinicalBERT
report parser.

Corpus JSONL shape (per corpus_synth.py):
    text          : str
    tokens        : list[str]  (whitespace tokens the training script used)
    labels        : list[str]  (BIO tag per token, in {"O", "B-<E>", "I-<E>"})
    entities      : list[dict] (gold spans with value + surface)
    ground_truth  : dict       (field-level roll-up ready for direct compare)
    provenance    : "SYNTHETIC-v0.3.0"

Two levels of eval, both matter:
  1. Entity RECOGNITION: for each entity type, does the model surface a
     matched value at all?  Compares parser output (matched vs. no_match)
     to gold_truth having a non-null value for that field.
  2. Entity VALUE ACCURACY: given the model matched something, is the
     canonicalized value equal to the gold value?

Fused-vs-regex agreement is computed on the 4 primary receptor-panel
fields (er, pr, her2, grade), which is what the biopsy API surfaces.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------ #
# Value canonicalization — normalize both parser output and gold to a
# comparable form. The corpus uses "positive"/"negative"/"equivocal" for
# receptors, integers for grade, floats for size, "positive"/"negative"
# for margin, "present"/"absent" for LVI.
# ------------------------------------------------------------------ #
def _canon_bool_receptor(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "positive" if v else "negative"
    s = str(v).strip().lower()
    if s in {"positive", "pos", "+", "true"}:
        return "positive"
    if s in {"negative", "neg", "-", "false"}:
        return "negative"
    if s in {"equivocal", "borderline"}:
        return "equivocal"
    return s


def _canon_her2(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"positive", "3+"}:
        return "positive"
    if s in {"negative", "0", "1+"}:
        return "negative"
    if s in {"equivocal", "2+"}:
        return "equivocal"
    return s


def _canon_grade(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _canon_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def _canon_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _canon_stage(v) -> str | None:
    if v is None:
        return None
    return str(v).strip().lower().lstrip("t").lstrip("n").lstrip("m").lstrip("p")


def _canon_margin(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if "positive" in s or "involved" in s:
        return "positive"
    if "negative" in s or "clear" in s:
        return "negative"
    return s


def _canon_lvi(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return "present" if v else "absent"
    s = str(v).strip().lower()
    if "present" in s or s in {"positive", "yes", "identified"}:
        return "present"
    if "absent" in s or s in {"negative", "no", "not identified"}:
        return "absent"
    return s


FIELD_CANON = {
    "er":           _canon_bool_receptor,
    "pr":           _canon_bool_receptor,
    "her2":         _canon_her2,
    "grade":        _canon_grade,
    "ki67_pct":     _canon_int,
    "tumor_size_mm": _canon_float,
    "t_stage":      _canon_stage,
    "n_stage":      _canon_stage,
    "m_stage":      _canon_stage,
    "margin":       _canon_margin,
    "lvi":          _canon_lvi,
}

# Map parser output → gold key.
PARSER_TO_GOLD = {
    "er": "er", "pr": "pr", "her2": "her2", "grade": "grade",
    "ki67_pct": "ki67_pct",
    "tumor_size_mm": "tumor_size_mm",
    "t_stage": "t_stage", "n_stage": "n_stage", "m_stage": "m_stage",
    "margin": "margin", "lvi": "lvi",
}


def _extract_parser_field(parser_dict: dict, field: str, *, strict: bool = False):
    """Pull value from parser output.

    strict=False (default): accept both matched AND ambiguous states as
    "the model recognized this span" — value may be None if the ambiguous
    state came from low confidence, in which case we surface matched_text
    as the value.  This mirrors how a real downstream consumer would use
    the parser: they read the field, see match_state != "no_match", and
    decide per-field whether the confidence is enough.

    strict=True: only matched — used for the "how confident is the model
    when it commits" number.
    """
    if field in ("er", "pr", "her2", "grade"):
        f = parser_dict.get(field, {})
    else:
        f = parser_dict.get("extended_fields", {}).get(field, {})
    if not f:
        return None
    state = f.get("match_state")
    if state == "no_match":
        return None
    if strict and state != "matched":
        return None
    v = f.get("value")
    if v is None and state == "ambiguous":
        # Model tagged the span but the canonicalizer refused to commit.
        # Surface matched_text so per-field canonicalizer downstream can
        # still evaluate it.
        return f.get("matched_text")
    return v


# ------------------------------------------------------------------ #
# Regex parser (for fused-vs-regex comparison)
# ------------------------------------------------------------------ #
def _regex_result(text: str) -> dict[str, str | int | float | None]:
    from oncology_arbiter.models.report_parser import parse_pathology_report
    r = parse_pathology_report(text)
    return {
        "er":    _canon_bool_receptor(r.er.value if r.er.value is not None else None),
        "pr":    _canon_bool_receptor(r.pr.value if r.pr.value is not None else None),
        "her2":  _canon_her2(r.her2.value if r.her2.value is not None else None),
        "grade": _canon_grade(r.grade.value if r.grade.value is not None else None),
    }


def _fused_result(text: str, parser_dict_out: dict) -> dict[str, str | int | float | None]:
    """Use the parser's own dict output, canonicalized to compare to gold."""
    return {
        "er":    _canon_bool_receptor(_extract_parser_field(parser_dict_out, "er")),
        "pr":    _canon_bool_receptor(_extract_parser_field(parser_dict_out, "pr")),
        "her2":  _canon_her2(_extract_parser_field(parser_dict_out, "her2")),
        "grade": _canon_grade(_extract_parser_field(parser_dict_out, "grade")),
    }


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--test-jsonl", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-len", default=512, type=int)
    ap.add_argument("--limit", default=None, type=int)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    print(f"[eval] loading parser from {args.ckpt_dir}", flush=True)
    from oncology_arbiter.nlp.clinicalbert_parser import ClinicalBertReportParser

    parser = ClinicalBertReportParser(model_dir=args.ckpt_dir)
    print(f"[eval] parser loaded, device={parser._device}, "
          f"labels={len(parser._label2id)}", flush=True)

    reports = _load_jsonl(args.test_jsonl)
    if args.limit:
        reports = reports[: args.limit]
    print(f"[eval] {len(reports)} reports in {args.test_jsonl.name}", flush=True)

    # ------------------------------------------------------------ #
    # 1. Recognition: TP/FP/FN per entity type — "did the model match
    #    a value?" vs "was a gold value present?"
    # 2. Value accuracy: given both matched, does canonical value agree?
    # ------------------------------------------------------------ #
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    correct_val: dict[str, int] = defaultdict(int)
    matched_both: dict[str, int] = defaultdict(int)
    # Strict view (matched-only).
    tp_strict: dict[str, int] = defaultdict(int)
    fp_strict: dict[str, int] = defaultdict(int)
    fn_strict: dict[str, int] = defaultdict(int)
    correct_val_strict: dict[str, int] = defaultdict(int)

    agree_matrix: dict[str, dict[str, int]] = {
        f: {"fused_correct_regex_correct": 0,
            "fused_correct_regex_wrong": 0,
            "fused_wrong_regex_correct": 0,
            "fused_wrong_regex_wrong": 0}
        for f in ["er", "pr", "her2", "grade"]
    }

    t0 = time.time()
    for i, rep in enumerate(reports):
        text = rep["text"]
        gt = rep.get("ground_truth", {})

        pred = parser.parse(text, max_len=args.max_len)
        parser_dict = pred.as_dict()

        for field in FIELD_CANON:
            gold_raw = gt.get(field)
            gold_canon = FIELD_CANON[field](gold_raw)

            # Two views: relaxed (matched|ambiguous) and strict (matched only).
            pred_raw = _extract_parser_field(parser_dict, field, strict=False)
            pred_canon = FIELD_CANON[field](pred_raw)

            gold_present = gold_canon is not None
            pred_present = pred_canon is not None

            if gold_present and pred_present:
                tp[field] += 1
                matched_both[field] += 1
                if gold_canon == pred_canon:
                    correct_val[field] += 1
            elif not gold_present and pred_present:
                fp[field] += 1
            elif gold_present and not pred_present:
                fn[field] += 1

            # Strict view (matched-only, for "commit confidence" number).
            pred_strict = _extract_parser_field(parser_dict, field, strict=True)
            pred_strict_canon = FIELD_CANON[field](pred_strict)
            if gold_present and pred_strict_canon is not None:
                tp_strict[field] += 1
                if gold_canon == pred_strict_canon:
                    correct_val_strict[field] += 1
            elif not gold_present and pred_strict_canon is not None:
                fp_strict[field] += 1
            elif gold_present and pred_strict_canon is None:
                fn_strict[field] += 1

        # Fused-vs-regex on 4 receptor-panel fields.
        try:
            reg = _regex_result(text)
            fused = _fused_result(text, parser_dict)
            for f in ["er", "pr", "her2", "grade"]:
                g = FIELD_CANON[f](gt.get(f))
                if g is None:
                    continue
                rc = reg.get(f) == g
                fc = fused.get(f) == g
                key = ("fused_correct" if fc else "fused_wrong") + "_" + \
                      ("regex_correct" if rc else "regex_wrong")
                agree_matrix[f][key] += 1
        except Exception as e:
            print(f"[warn] fused-vs-regex report {i}: {type(e).__name__}: {e}",
                  flush=True)

        if (i + 1) % 50 == 0:
            print(f"[eval] {i+1}/{len(reports)}  "
                  f"({(i+1)/(time.time()-t0):.1f} rep/s)", flush=True)

    dt = time.time() - t0

    # ---------------------------------------------------------------- #
    # Aggregate
    # ---------------------------------------------------------------- #
    per_field: dict[str, dict[str, Any]] = {}
    for field in FIELD_CANON:
        m = _prf(tp[field], fp[field], fn[field])
        m["support"] = tp[field] + fn[field]
        m["value_accuracy"] = (
            correct_val[field] / matched_both[field] if matched_both[field] else 0.0
        )
        m["matched_both_gold_and_pred"] = matched_both[field]
        m["value_correct"] = correct_val[field]
        # Strict view: matched-only.
        ms = _prf(tp_strict[field], fp_strict[field], fn_strict[field])
        m["strict"] = {
            "precision": ms["precision"],
            "recall": ms["recall"],
            "f1": ms["f1"],
            "tp": ms["tp"],
            "fp": ms["fp"],
            "fn": ms["fn"],
            "value_correct": correct_val_strict[field],
            "value_accuracy": (
                correct_val_strict[field] / tp_strict[field]
                if tp_strict[field] else 0.0
            ),
        }
        per_field[field] = m

    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    total_correct = sum(correct_val.values())
    total_matched = sum(matched_both.values())
    micro = _prf(total_tp, total_fp, total_fn)
    micro["support"] = total_tp + total_fn
    micro["value_accuracy"] = (
        total_correct / total_matched if total_matched else 0.0
    )
    micro["value_correct"] = total_correct
    micro["matched_both_gold_and_pred"] = total_matched
    # Strict micro.
    total_tp_s = sum(tp_strict.values())
    total_fp_s = sum(fp_strict.values())
    total_fn_s = sum(fn_strict.values())
    total_correct_s = sum(correct_val_strict.values())
    micro_strict = _prf(total_tp_s, total_fp_s, total_fn_s)
    micro_strict["value_correct"] = total_correct_s
    micro_strict["value_accuracy"] = (
        total_correct_s / total_tp_s if total_tp_s else 0.0
    )
    micro["strict"] = micro_strict

    nonempty = [m for m in per_field.values() if m["support"] > 0]
    macro = {
        "precision": sum(m["precision"] for m in nonempty) / len(nonempty) if nonempty else 0.0,
        "recall":    sum(m["recall"]    for m in nonempty) / len(nonempty) if nonempty else 0.0,
        "f1":        sum(m["f1"]        for m in nonempty) / len(nonempty) if nonempty else 0.0,
        "value_accuracy": (
            sum(m["value_accuracy"] for m in nonempty if m["matched_both_gold_and_pred"] > 0)
            / max(1, sum(1 for m in nonempty if m["matched_both_gold_and_pred"] > 0))
        ),
        "n_fields": len(nonempty),
    }

    metrics = {
        "ckpt_dir": str(args.ckpt_dir),
        "test_jsonl": str(args.test_jsonl),
        "n_reports": len(reports),
        "eval_seconds": dt,
        "micro": micro,
        "macro": macro,
        "per_field": per_field,
        "fused_vs_regex_agreement": agree_matrix,
    }

    (args.out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[eval] wrote {args.out_dir / 'test_metrics.json'}", flush=True)
    print(f"[eval] micro F1 = {micro['f1']:.4f}  "
          f"value-acc = {micro['value_accuracy']:.4f}  "
          f"macro F1 = {macro['f1']:.4f}  "
          f"({len(reports)} reports in {dt:.1f}s)", flush=True)

    # Human-readable summary.
    lines = [
        "# ClinicalBERT report parser — held-out test-set eval",
        "",
        f"- Checkpoint: `{args.ckpt_dir}`",
        f"- Test corpus: `{args.test_jsonl}` ({len(reports)} synthetic reports)",
        f"- Eval wall-clock: {dt:.1f}s ({dt/len(reports):.2f}s/report on CPU)",
        "",
        "## Micro (all fields pooled)",
        "",
        f"- **F1: {micro['f1']:.4f}**  (P={micro['precision']:.4f}, R={micro['recall']:.4f})",
        f"- **Value accuracy given match: {micro['value_accuracy']:.4f}**  "
        f"({micro['value_correct']}/{micro['matched_both_gold_and_pred']})",
        f"- Support: {micro['support']} gold field values across {len(reports)} reports.",
        "",
        "## Macro (unweighted across fields)",
        "",
        f"- F1: {macro['f1']:.4f}   (over {macro['n_fields']} fields with gold support)",
        f"- Value accuracy: {macro['value_accuracy']:.4f}",
        "",
        "## Per-field",
        "",
        "| Field | Support | P | R | F1 | Value-acc (given match) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field, m in sorted(per_field.items(), key=lambda kv: -kv[1]["support"]):
        if m["support"] == 0:
            continue
        lines.append(
            f"| {field} | {m['support']} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['value_accuracy']:.4f} ({m['value_correct']}/{m['matched_both_gold_and_pred']}) |"
        )
    lines += ["", "## Fused-vs-regex agreement (receptor panel)", ""]
    for field in ["er", "pr", "her2", "grade"]:
        m = agree_matrix[field]
        total = sum(m.values())
        if total == 0:
            continue
        lines += [
            f"### {field.upper()}  (n={total} reports with gold {field})",
            "",
            "|                 | regex correct | regex wrong |",
            "|-----------------|--------------:|------------:|",
            f"| **fused correct** | {m['fused_correct_regex_correct']} | {m['fused_correct_regex_wrong']} |",
            f"| **fused wrong**   | {m['fused_wrong_regex_correct']}   | {m['fused_wrong_regex_wrong']}   |",
            "",
            f"- Fused rescued regex (regex wrong → fused correct): "
            f"**{m['fused_correct_regex_wrong']}**",
            f"- Fused broke regex (regex correct → fused wrong): "
            f"**{m['fused_wrong_regex_correct']}**",
            f"- Net fusion delta on {field.upper()}: "
            f"**{m['fused_correct_regex_wrong'] - m['fused_wrong_regex_correct']:+d}** reports",
            "",
        ]

    (args.out_dir / "eval_summary.md").write_text("\n".join(lines))
    print(f"[eval] wrote {args.out_dir / 'eval_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
