#!/usr/bin/env python3
"""Ghost Trial harness — multi-modal public-case stress run (RUO).

Runs available titanium paths on a batch of TCGA-BRCA-linked pathology reports
+ structured therapy features. Mammography/CT optional when fixtures absent.

    PYTHONPATH=src python3 scripts/demo/run_ghost_trial_harness.py --n 50 \\
        --out artifacts/ghost_trial/ledger_v1.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _load_reports(n: int) -> list[dict[str, Any]]:
    dataset = json.loads((ROOT / "artifacts/clinicalbert/clinicalbert_dataset_v2.json").read_text())
    split = json.loads((ROOT / "artifacts/clinicalbert/clinicalbert_split_v2.json").read_text())
    test_ids = set(split.get("test_sample_ids") or [])
    csv_path = ROOT / "data/pathology_text/TCGA_Reports.csv"
    index: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            fn = str(row.get("patient_filename") or "")
            index[fn] = str(row.get("text") or "")
            if "." in fn:
                index[fn.split(".")[0]] = index[fn]

    cases = []
    for sample in dataset["samples"]:
        if sample["sample_id"] not in test_ids:
            continue
        text = None
        for key in (sample.get("report_id"), sample.get("patient_id")):
            if key and str(key) in index:
                text = index[str(key)]
                break
            if key and str(key).split(".")[0] in index:
                text = index[str(key).split(".")[0]]
                break
        if not text:
            continue
        cases.append({"sample_id": sample["sample_id"], "patient_id": sample["patient_id"], "report_text": text})
        if len(cases) >= n:
            break
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/ghost_trial/ledger_v1.json")
    args = ap.parse_args()

    from oncology_arbiter.nlp.medgemma_pathology_client import extract_pathology
    from oncology_arbiter.arbiter.stage_therapy_wiring import load_stage_therapy_arbiter

    cases = _load_reports(args.n)
    arb = load_stage_therapy_arbiter()
    rows = []
    t0 = time.time()
    for case in cases:
        row: dict[str, Any] = {
            "sample_id": case["sample_id"],
            "patient_id": case["patient_id"],
            "disclaimer": "RUO ghost trial row",
        }
        try:
            ext = extract_pathology(case["report_text"])
            extraction = ext["extraction"]
            row["medgemma_pathology"] = {
                "status": "ok",
                "extraction": extraction,
                "revision_sha256": ext.get("revision_sha256"),
            }
            features = {
                "histology": "invasive_ductal",
                "grade": str(extraction.get("nottingham_grade") or "3"),
                "er_status_positive": extraction.get("er_status") == "POSITIVE",
                "pr_status_positive": extraction.get("pr_status") == "POSITIVE",
                "her2_status_positive": extraction.get("her2_status")
                in {"POSITIVE", "3+", "AMPLIFIED"},
                "node_status_positive": False,
                "brca_status_known_pathogenic": False,
                "tumor_size_norm": float(extraction.get("tumor_size_mm") or 20.0) / 50.0,
                "age_at_diagnosis_norm": 0.55,
                "ki67_norm": 0.2,
            }
            scored = arb.score(features)
            row["stage_therapy"] = {
                "status": "ok",
                "p_event": float(scored.p_positive),
                "risk_bucket": scored.risk_bucket,
                "recommendation": scored.recommendation,
            }
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        else:
            row["status"] = "ok"
        rows.append(row)
        print(json.dumps({"progress": len(rows), "n": len(cases), "last": row["sample_id"], "status": row["status"]}))

    ok = sum(1 for r in rows if r.get("status") == "ok")
    ledger = {
        "disclaimer": "RESEARCH USE ONLY — Ghost Trial harness v1; not clinical validation.",
        "n_requested": args.n,
        "n_cases": len(rows),
        "n_ok": ok,
        "elapsed_s": round(time.time() - t0, 3),
        "modalities_exercised": ["medgemma_pathology", "stage_therapy"],
        "modalities_absent": ["luna16_ct", "mammo_rsna", "phikon_wsi"],
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "n_ok": ok, "n_cases": len(rows)}, indent=2))
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
