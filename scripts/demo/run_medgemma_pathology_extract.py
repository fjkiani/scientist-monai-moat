#!/usr/bin/env python3
"""Smoke / batch MedGemma structured pathology extraction (RUO).

Examples::

    PYTHONPATH=src python3 scripts/demo/run_medgemma_pathology_extract.py \\
        --report fixtures/patient_sample_01/pathology_note.txt

    PYTHONPATH=src python3 scripts/demo/run_medgemma_pathology_extract.py \\
        --eval-n 8 --out artifacts/medgemma_pathology/smoke_eval.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from oncology_arbiter.nlp.medgemma_pathology_client import (  # noqa: E402
    MedGemmaPathologyError,
    extract_pathology,
)


def _normalize_pos_neg(surface: str) -> str:
    s = surface.lower()
    if "positive" in s or "pos" in s.split():
        return "POSITIVE"
    if "negative" in s or "neg" in s.split():
        return "NEGATIVE"
    if "equivocal" in s:
        return "EQUIVOCAL"
    return "UNKNOWN"


def _gold_from_entities(entities: list[dict[str, Any]]) -> dict[str, Any]:
    gold: dict[str, Any] = {
        "er_status": "UNKNOWN",
        "pr_status": "UNKNOWN",
        "her2_status": "UNKNOWN",
        "nottingham_grade": None,
        "tumor_size_mm": None,
    }
    for ent in entities:
        et = ent.get("entity_type")
        surface = str(ent.get("surface") or "")
        if et == "ER_VALUE":
            gold["er_status"] = _normalize_pos_neg(surface)
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", surface)
            if m:
                gold["er_percent"] = float(m.group(1))
        elif et == "PR_VALUE":
            gold["pr_status"] = _normalize_pos_neg(surface)
        elif et == "HER2_VALUE":
            if "3+" in surface:
                gold["her2_status"] = "3+"
            elif "2+" in surface:
                gold["her2_status"] = "2+"
            elif "1+" in surface:
                gold["her2_status"] = "1+"
            elif re.search(r"\b0\b", surface):
                gold["her2_status"] = "0"
            else:
                gold["her2_status"] = _normalize_pos_neg(surface)
        elif et == "GRADE":
            m = re.search(r"([123])", surface)
            if m:
                gold["nottingham_grade"] = int(m.group(1))
        elif et == "TUMOR_SIZE_MM":
            m = re.search(r"(\d+(?:\.\d+)?)\s*cm", surface, re.I)
            if m:
                gold["tumor_size_mm"] = float(m.group(1)) * 10.0
            else:
                m2 = re.search(r"(\d+(?:\.\d+)?)\s*mm", surface, re.I)
                if m2:
                    gold["tumor_size_mm"] = float(m2.group(1))
    return gold


def _load_report_text(report_id: str, csv_path: Path) -> str | None:
    # TCGA_Reports.csv: patient_filename,text (Mendeley/Zenodo dump).
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        text_key = next(
            (k for k in fields if k and k.lower() in {"text", "report", "report_text", "patient_reports"}),
            None,
        )
        id_keys = [
            k
            for k in fields
            if k
            and (
                "id" in k.lower()
                or "patient" in k.lower()
                or "barcode" in k.lower()
                or "filename" in k.lower()
            )
        ]
        if not text_key:
            return None
        needle = report_id.split(".")[0]
        for row in reader:
            blob = " ".join(str(row.get(k) or "") for k in id_keys)
            if needle in blob or report_id in blob:
                return str(row.get(text_key) or "")
    return None


def _field_match(pred: Any, gold: Any, *, field: str) -> bool | None:
    if gold in (None, "UNKNOWN") and field.endswith("_status"):
        return None  # not scorable
    if gold is None:
        return None
    if field in {"er_status", "pr_status"}:
        return str(pred).upper() == str(gold).upper()
    if field == "her2_status":
        p, g = str(pred).upper(), str(gold).upper()
        if p == g:
            return True
        # soft map POSITIVE↔3+/AMPLIFIED, NEGATIVE↔0/1+/NON_AMPLIFIED
        pos = {"POSITIVE", "3+", "AMPLIFIED"}
        neg = {"NEGATIVE", "0", "1+", "NON_AMPLIFIED"}
        if p in pos and g in pos:
            return True
        if p in neg and g in neg:
            return True
        return False
    if field == "nottingham_grade":
        return pred == gold
    if field == "tumor_size_mm":
        if pred is None:
            return False
        return abs(float(pred) - float(gold)) <= max(3.0, 0.2 * float(gold))
    return str(pred) == str(gold)


def run_eval(n: int, out: Path) -> dict[str, Any]:
    dataset = json.loads((ROOT / "artifacts/clinicalbert/clinicalbert_dataset_v2.json").read_text())
    split = json.loads((ROOT / "artifacts/clinicalbert/clinicalbert_split_v2.json").read_text())
    # TCGA-242 pathologist gold lives in test_sample_ids (not external_validation).
    test_ids = set(split.get("test_sample_ids") or [])
    csv_path = ROOT / "data/pathology_text/TCGA_Reports.csv"
    candidates = []
    for sample in dataset["samples"]:
        if sample["sample_id"] not in test_ids:
            continue
        ents = sample.get("structured_target") or []
        types = {e.get("entity_type") for e in ents if isinstance(e, dict)}
        if "ER_VALUE" in types or "HER2_VALUE" in types:
            candidates.append(sample)
    candidates = candidates[:n]
    rows = []
    tallies: dict[str, dict[str, int]] = {}
    for sample in candidates:
        report = None
        for key in (
            sample.get("report_id"),
            sample.get("patient_id"),
            sample.get("source_record"),
            sample.get("sample_id"),
        ):
            if key:
                report = _load_report_text(str(key), csv_path)
                if report:
                    break
        if not report:
            rows.append({"sample_id": sample["sample_id"], "status": "missing_report_text"})
            continue
        gold = _gold_from_entities(sample.get("structured_target") or [])
        try:
            pred_wrap = extract_pathology(report)
            pred = pred_wrap["extraction"]
            status = "ok"
        except MedGemmaPathologyError as exc:
            rows.append({"sample_id": sample["sample_id"], "status": "failed", "error": str(exc)})
            continue
        field_results = {}
        for field in ("er_status", "pr_status", "her2_status", "nottingham_grade", "tumor_size_mm"):
            matched = _field_match(pred.get(field), gold.get(field), field=field)
            field_results[field] = matched
            if matched is None:
                continue
            bucket = tallies.setdefault(field, {"correct": 0, "scored": 0})
            bucket["scored"] += 1
            if matched:
                bucket["correct"] += 1
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "status": status,
                "gold": gold,
                "pred": pred,
                "field_match": field_results,
                "revision_sha256": pred_wrap.get("revision_sha256"),
            }
        )
    metrics = {
        field: {
            "n": v["scored"],
            "accuracy": (v["correct"] / v["scored"]) if v["scored"] else None,
        }
        for field, v in tallies.items()
    }
    payload = {
        "disclaimer": "RUO — MedGemma structured pathology extract eval",
        "n_requested": n,
        "n_rows": len(rows),
        "metrics": metrics,
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Path to a single pathology report text file")
    parser.add_argument("--eval-n", type=int, default=0, help="Evaluate on N held-out labeled reports")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/medgemma_pathology/smoke_eval.json",
    )
    args = parser.parse_args()
    if args.report:
        text = args.report.read_text(encoding="utf-8", errors="replace")
        result = extract_pathology(text)
        print(json.dumps(result, indent=2))
        return 0
    if args.eval_n > 0:
        payload = run_eval(args.eval_n, args.out)
        print(json.dumps({"metrics": payload["metrics"], "n_rows": payload["n_rows"], "out": str(args.out)}, indent=2))
        return 0
    parser.error("provide --report or --eval-n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
