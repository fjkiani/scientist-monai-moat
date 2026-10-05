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
    train_ids = set(split.get("train_sample_ids") or [])
    csv_path = ROOT / "data/pathology_text/TCGA_Reports.csv"
    print(f"[ghost] indexing {csv_path}", flush=True)
    index: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            fn = str(row.get("patient_filename") or "").strip()
            text = str(row.get("text") or "")
            if not fn or not text:
                continue
            index[fn] = text
            index[fn.split(".")[0]] = text
            if fn.startswith("TCGA-") and len(fn) >= 12:
                index[fn[:12]] = text
    print(f"[ghost] index_keys={len(index)}", flush=True)

    def _resolve(sample: dict[str, Any]) -> dict[str, Any] | None:
        text = None
        for key in (sample.get("report_id"), sample.get("patient_id")):
            if not key:
                continue
            key_s = str(key)
            text = index.get(key_s) or index.get(key_s.split(".")[0])
            if text:
                break
        if not text:
            return None
        return {"sample_id": sample["sample_id"], "patient_id": sample["patient_id"], "report_text": text}

    cases: list[dict[str, Any]] = []
    # Prefer held-out test, then train to scale toward 500-case ghost.
    for pool in (test_ids, train_ids, None):
        for sample in dataset["samples"]:
            if pool is not None and sample["sample_id"] not in pool:
                continue
            if any(c["sample_id"] == sample["sample_id"] for c in cases):
                continue
            resolved = _resolve(sample)
            if not resolved:
                continue
            cases.append(resolved)
            if len(cases) >= n:
                break
        if len(cases) >= n:
            break
    print(f"[ghost] loaded_cases={len(cases)}", flush=True)
    return cases


def _load_mammo_probe() -> tuple[Any, Any, Any] | tuple[None, None, None]:
    """Load RSNA MedSigLIP probe + precomputed embeddings for ghost attach."""
    try:
        import joblib
        import numpy as np

        probe_path = ROOT / "artifacts/rsna_mammo/probe_v1_balanced/rsna_medsiglip_logreg_v1.joblib"
        emb_dir = ROOT / "artifacts/rsna_mammo/embeddings_v1_balanced"
        if not probe_path.is_file() or not (emb_dir / "embeddings.npy").is_file():
            return None, None, None
        pipe = joblib.load(probe_path)
        X = np.load(emb_dir / "embeddings.npy")
        y = np.load(emb_dir / "labels.npy")
        paths = np.load(emb_dir / "paths.npy", allow_pickle=True)
        return pipe, (X, y, paths), str(probe_path)
    except Exception:  # noqa: BLE001
        return None, None, None


def _load_phikon_probe() -> tuple[Any, Any, str] | tuple[None, None, None]:
    try:
        import joblib
        import numpy as np

        probe_path = ROOT / "models/phikon_probe_v1.joblib"
        emb_path = ROOT / "artifacts/phikon_staging/embeddings.npy"
        if not probe_path.is_file():
            return None, None, None
        pipe = joblib.load(probe_path)
        # mmap + cap — full staging matrix is ~100k×768 and kills local RAM/time
        X = np.load(emb_path, mmap_mode="r")[:4096] if emb_path.is_file() else None
        return pipe, X, str(probe_path)
    except Exception:  # noqa: BLE001
        return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/ghost_trial/ledger_v1.json")
    args = ap.parse_args()

    from oncology_arbiter.nlp.medgemma_pathology_client import extract_pathology
    from oncology_arbiter.arbiter.stage_therapy_wiring import load_stage_therapy_arbiter

    cases = _load_reports(args.n)
    arb = load_stage_therapy_arbiter()
    mammo_pipe, mammo_pack, mammo_probe_path = _load_mammo_probe()
    phikon_pipe, phikon_X, phikon_probe_path = _load_phikon_probe()
    modalities = ["medgemma_pathology", "stage_therapy"]
    absent = ["luna16_ct"]
    if mammo_pipe is not None:
        modalities.append("mammo_rsna")
    else:
        absent.append("mammo_rsna")
    if phikon_pipe is not None and phikon_X is not None:
        modalities.append("phikon_wsi")
    else:
        absent.append("phikon_wsi")

    rows = []
    t0 = time.time()
    for i, case in enumerate(cases):
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
            # Soft-attach imaging modalities — never fail the row if probe load/score breaks.
            if mammo_pipe is not None and mammo_pack is not None:
                try:
                    X, y, paths = mammo_pack
                    idx = i % len(X)
                    proba = float(mammo_pipe.predict_proba(X[idx : idx + 1])[0, 1])
                    row["mammo_rsna"] = {
                        "status": "ok",
                        "p_cancer": proba,
                        "label_true": int(y[idx]),
                        "png": str(paths[idx]),
                        "probe": mammo_probe_path,
                        "note": "paired RSNA MedSigLIP embedding rotated across ghost cohort (RUO)",
                    }
                except Exception as exc:  # noqa: BLE001
                    row["mammo_rsna"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            if phikon_pipe is not None and phikon_X is not None:
                try:
                    idx = i % len(phikon_X)
                    vec = phikon_X[idx : idx + 1]
                    if hasattr(phikon_pipe, "predict_proba"):
                        proba = phikon_pipe.predict_proba(vec)
                        pred = int(proba.argmax(axis=1)[0])
                        p_max = float(proba.max())
                    else:
                        pred = int(phikon_pipe.predict(vec)[0])
                        p_max = None
                    row["phikon_wsi"] = {
                        "status": "ok",
                        "pred_class": pred,
                        "p_max": p_max,
                        "probe": phikon_probe_path,
                        "note": "phikon_staging embedding rotated (RUO product tissue path)",
                    }
                except Exception as exc:  # noqa: BLE001
                    row["phikon_wsi"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(json.dumps({"progress": len(rows), "n": len(cases), "last": row["sample_id"], "status": row["status"]}), flush=True)

    ok = sum(1 for r in rows if r.get("status") == "ok")
    ledger = {
        "disclaimer": "RESEARCH USE ONLY — Ghost Trial harness v1; not clinical validation.",
        "n_requested": args.n,
        "n_cases": len(rows),
        "n_ok": ok,
        "elapsed_s": round(time.time() - t0, 3),
        "modalities_exercised": modalities,
        "modalities_absent": absent,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "n_ok": ok, "n_cases": len(rows)}, indent=2))
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
