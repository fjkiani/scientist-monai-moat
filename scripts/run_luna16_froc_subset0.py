"""Run the real, in-repo LUNA16 FROC pipeline (scripts/luna16_froc.py) end-to-end
against real subset0 CT volumes: upload -> detect -> recover world coords ->
results.csv, resumable and checkpointed per series.

This is the execution driver for the pipeline module; see scripts/luna16_froc.py
for the actual upload/detect/coordinate-recovery logic and its docstring for the
full audit context (finding B_luna16_froc_vaporware, finding
F_luna16_infer_missing_resample_preprocessing_defect).

Usage::

    CASE_STORAGE_MODAL_URL=https://crispro--case-storage \\
    LUNA16_MODAL_DETECT_URL=https://crispro--luna16-detect.modal.run \\
        .venv/bin/python scripts/run_luna16_froc_subset0.py \\
        --subset-dir /workspace/luna16/subset0/subset0 \\
        --annotations-csv /workspace/luna16/csv/annotations.csv \\
        --out-dir /workspace/smm/artifacts/luna16_froc_subset0 \\
        [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luna16_froc import (  # noqa: E402
    read_mhd_volume,
    hu_array_to_dicom_series,
    upload_dicom_series,
    call_detect,
    detections_to_results_rows,
    round_trip_selftest,
    write_results_csv,
    CoordinateFrameError,
)


def load_annotations(path: Path) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["seriesuid"], []).append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-dir", type=Path, required=True)
    ap.add_argument("--annotations-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--case-storage-base-url", type=str, default=None,
                     help="defaults to $CASE_STORAGE_MODAL_URL")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--only-seriesuid", type=str, default=None,
                     help="process only this one seriesuid (debug/smoke)")
    args = ap.parse_args()

    import os
    base_url = args.case_storage_base_url or os.environ.get("CASE_STORAGE_MODAL_URL")
    if not base_url:
        raise SystemExit("set --case-storage-base-url or $CASE_STORAGE_MODAL_URL")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_path = out_dir / "processed_seriesuids.json"
    results_path = out_dir / "results.csv"
    selftest_path = out_dir / "selftest_log.jsonl"

    processed: dict[str, dict] = {}
    if processed_path.exists():
        processed = json.loads(processed_path.read_text())
        print(f"[resume] {len(processed)} series already processed")

    ann_by_series = load_annotations(args.annotations_csv)

    mhd_files = sorted(args.subset_dir.glob("*.mhd"))
    if args.only_seriesuid:
        mhd_files = [p for p in mhd_files if p.stem == args.only_seriesuid]
        if not mhd_files:
            raise SystemExit(f"--only-seriesuid {args.only_seriesuid!r} not found in {args.subset_dir}")
    elif args.limit:
        mhd_files = mhd_files[: args.limit]
    print(f"[gather] {len(mhd_files)} .mhd volumes in {args.subset_dir}")

    all_rows: list[dict] = []
    # Re-load any prior rows for already-processed series so results.csv stays complete on resume.
    if results_path.exists():
        with open(results_path, newline="") as f:
            all_rows = list(csv.DictReader(f))

    t_start = time.time()
    for i, mhd_path in enumerate(mhd_files):
        seriesuid = mhd_path.stem
        if seriesuid in processed:
            continue
        print(f"[{i+1}/{len(mhd_files)}] {seriesuid}")
        vol = None
        stage = "read_mhd_volume"
        try:
            vol = read_mhd_volume(mhd_path)
            stage = "round_trip_selftest_or_convert"

            # Mandatory precondition: prove coordinate recovery on THIS volume
            # before trusting any detection from it, if it has a real annotation.
            anns = ann_by_series.get(seriesuid, [])
            if anns:
                st = round_trip_selftest(vol, anns[0])
                with open(selftest_path, "a") as f:
                    f.write(json.dumps(st) + "\n")
                if not st["ok"]:
                    raise CoordinateFrameError(
                        f"{seriesuid}: round_trip_selftest FAILED, max_abs_error_mm={st['max_abs_error_mm']}"
                    )

            blobs = hu_array_to_dicom_series(vol)
            stage = "upload_dicom_series"
            t0 = time.time()
            upload_resp = upload_dicom_series(
                blobs, base_url=base_url, uploader_note=f"luna16_froc_subset0:{seriesuid}"
            )
            case_id = upload_resp.get("case_id")
            if not case_id:
                raise RuntimeError(f"{seriesuid}: upload response missing case_id: {upload_resp!r}")
            upload_s = time.time() - t0

            stage = "call_detect"
            t0 = time.time()
            detect_resp = call_detect(case_id, request_id=str(uuid.uuid4()), top_n=args.top_n)
            detect_s = time.time() - t0

            rows = detections_to_results_rows(seriesuid, detect_resp, image=vol["image"])
            all_rows.extend(rows)
            write_results_csv(all_rows, results_path)

            processed[seriesuid] = {
                "case_id": case_id,
                "n_detections": len(rows),
                "n_real_annotations": len(anns),
                "upload_seconds": round(upload_s, 2),
                "detect_seconds": round(detect_s, 2),
            }
            processed_path.write_text(json.dumps(processed, indent=2))
            print(
                f"    case_id={case_id}  n_detections={len(rows)}  n_real_annotations={len(anns)}  "
                f"upload={upload_s:.1f}s  detect={detect_s:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001 - record honestly and CONTINUE.
            # Deliberate design choice (post-mortem on the first full-run attempt,
            # which crashed the entire 89-series batch on the very first server-side
            # HTTP 500 encountered): a single series's real, reproducible failure
            # (confirmed non-transient via a targeted retry -- see
            # docs/proofs/luna16_froc_subset0_large_volume_failure_diagnosis.json)
            # must not block honest measurement of the other 88 series. The failure
            # is recorded VERBATIM below (no mocking, no downgrading to a fake
            # success) and the series is skipped; it is NOT silently dropped because
            # its exact exception string is permanently on disk in
            # processed_seriesuids.json and is included as "failed" (not "excluded")
            # in every summary count this script prints.
            print(f"    FAILED at stage={stage}: {type(exc).__name__}: {exc}")
            processed[seriesuid] = {
                "error": f"{type(exc).__name__}: {exc}",
                "failed_stage": stage,
                "shape_dhw": list(vol["shape_dhw"]) if vol is not None else None,
                "spacing_xyz": list(vol["spacing_xyz"]) if vol is not None else None,
            }
            processed_path.write_text(json.dumps(processed, indent=2))
            continue

    elapsed = time.time() - t_start
    n_ok = sum(1 for v in processed.values() if "error" not in v)
    n_err = sum(1 for v in processed.values() if "error" in v)
    print(f"[done] {n_ok} ok, {n_err} failed, {len(all_rows)} total detections, wall={elapsed:.0f}s")


if __name__ == "__main__":
    main()
