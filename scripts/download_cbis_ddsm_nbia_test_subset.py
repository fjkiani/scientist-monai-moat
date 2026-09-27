"""Download the CBIS-DDSM held-out TEST split from the canonical TCIA/NBIA source.

This is deliberately independent of ``scripts/embed_cbis_ddsm.py`` /
``scripts/train_cbis_ddsm_logreg.py``, which trained
``models/cbis_ddsm_logreg_v1.joblib`` on a THIRD-PARTY HuggingFace mirror
(``dbaek111/CBIS-DDSM_1024``) of *derived, downsampled 1024x1024 PNGs* — not
the original DICOM pixel data, and not obtained via NBIA (see
``docs/proofs/cbis_ddsm_logreg_v1_metrics.json``, honesty_caveats).

Audit finding (Wave-1 item 1, locked order): the user's directive asked for
"NBIA image subsets" specifically. This script pulls real, full-resolution
mammograms directly from the canonical TCIA/NBIA REST API
(``services.cancerimagingarchive.net``) for the official CBIS-DDSM upstream
**test** split ONLY, so that a downstream evaluation of the existing frozen
probe against this cohort is a genuine, leakage-free, independent-provenance
cross-cohort validation -- not a re-derivation of the same HF-PNG numbers.

Source of truth for labels: the official case-description CSVs
(``mass_case_description_test_set.csv``, ``calc_case_description_test_set.csv``),
downloaded from the TCIA wiki attachment URLs (CC BY 3.0). The
``image file path`` column embeds the real ``SeriesInstanceUID`` as its
second-to-last path component; NBIA's ``getImage?SeriesInstanceUID=...``
returns a ZIP containing the real full-resolution DICOM for that series
(confirmed this session: 4808x3024, 16-bit, Modality=MG, SeriesInstanceUID
matches the CSV exactly).

Case-level label convention (standard, documented): an image with ANY
MALIGNANT abnormality is labeled malignant=1; BENIGN and
BENIGN_WITHOUT_CALLBACK are both malignant=0. A handful of images carry
BOTH a malignant and a benign abnormality (multiple lesions on one
mammogram) -- these are labeled malignant=1 (conservative: any malignancy
present).

Output
------
``--out-dir``/manifest.csv        one row per unique full-mammogram series
``--out-dir``/dicoms/<series_uid>.dcm   the real DICOM pixel data

Resumable: rows already present in manifest.csv with an existing on-disk
DICOM are skipped on re-run.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List
from urllib import error as urllib_error
from urllib import request as urllib_request

NBIA_GET_IMAGE = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={uid}"

CSV_LABEL_URLS = {
    "mass_case_description_test_set.csv": (
        "https://wiki.cancerimagingarchive.net/download/attachments/22516629/"
        "mass_case_description_test_set.csv?version=1&modificationDate=1506796343175&api=v2"
    ),
    "calc_case_description_test_set.csv": (
        "https://wiki.cancerimagingarchive.net/download/attachments/22516629/"
        "calc_case_description_test_set.csv?version=1&modificationDate=1506796343686&api=v2"
    ),
}


def ensure_label_csvs(raw_dir: Path) -> Dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in CSV_LABEL_URLS.items():
        dest = raw_dir / name
        if not dest.exists():
            print(f"[csv] downloading {name}")
            req = urllib_request.Request(url, headers={"User-Agent": "oncology-arbiter-audit/1.0"})
            with urllib_request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
        out[name] = dest
    return out


def parse_test_csvs(csv_paths: Dict[str, Path]) -> List[Dict[str, Any]]:
    """Parse mass+calc test CSVs, dedup by SeriesInstanceUID -> case-level rows."""
    by_series: Dict[str, Dict[str, Any]] = {}
    for name, path in csv_paths.items():
        abnormality_type = "mass" if name.startswith("mass") else "calc"
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                img_path = row["image file path"]
                parts = img_path.split("/")
                series_uid = parts[-2]
                study_uid = parts[-3]
                is_malignant = 1 if row["pathology"] == "MALIGNANT" else 0
                if series_uid not in by_series:
                    by_series[series_uid] = {
                        "series_uid": series_uid,
                        "study_uid": study_uid,
                        "patient_id": row["patient_id"],
                        "abnormality_type": abnormality_type,
                        "left_or_right": row["left or right breast"],
                        "view": row["image view"],
                        "pathology_raw_values": set(),
                        "malignant": 0,
                        "n_abnormality_rows": 0,
                    }
                entry = by_series[series_uid]
                entry["pathology_raw_values"].add(row["pathology"])
                entry["malignant"] = max(entry["malignant"], is_malignant)
                entry["n_abnormality_rows"] += 1
    rows = list(by_series.values())
    for r in rows:
        r["pathology_raw_values"] = "|".join(sorted(r["pathology_raw_values"]))
    return rows


def download_one_series(series_uid: str, dest_path: Path, *, timeout: float = 60.0, max_retries: int = 3) -> Dict[str, Any]:
    """Download+extract one series via NBIA getImage. Returns status dict."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return {"series_uid": series_uid, "status": "already_on_disk", "bytes": dest_path.stat().st_size}
    url = NBIA_GET_IMAGE.format(uid=series_uid)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib_request.Request(url, headers={"User-Agent": "oncology-arbiter-audit/1.0"})
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                zip_bytes = resp.read()
            zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
            # NBIA getImage zips are confirmed (this session, manual test) to
            # contain exactly {"LICENSE", "<name>.dcm"}. Filter by explicit
            # exclusion of LICENSE, NOT by a "no dot in filename" heuristic --
            # "LICENSE" itself has no dot and was wrongly matched by an
            # earlier version of this filter, causing the license TEXT file
            # to be saved as the ".dcm" output (caught via a file-size sanity
            # check: all downloads were an identical, suspiciously-small 2788
            # bytes instead of the expected ~10-30 MB real pixel data).
            dcm_names = [n for n in zf.namelist() if n.upper() != "LICENSE"]
            if not dcm_names:
                raise RuntimeError(f"zip for {series_uid} had no non-LICENSE member: {zf.namelist()}")
            if len(dcm_names) > 1:
                raise RuntimeError(f"zip for {series_uid} had >1 non-LICENSE member, ambiguous: {zf.namelist()}")
            dcm_bytes = zf.read(dcm_names[0])
            if len(dcm_bytes) < 100_000:
                raise RuntimeError(
                    f"zip for {series_uid} member {dcm_names[0]!r} is only {len(dcm_bytes)} bytes "
                    f"-- too small to be a real full-resolution mammogram DICOM, refusing to save"
                )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(dcm_bytes)
            return {"series_uid": series_uid, "status": "downloaded", "bytes": len(dcm_bytes), "zip_members": zf.namelist()}
        except (urllib_error.URLError, urllib_error.HTTPError, zipfile.BadZipFile, RuntimeError) as exc:
            last_err = str(exc)
            time.sleep(min(2 ** attempt, 10))
    return {"series_uid": series_uid, "status": "failed", "error": last_err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("/mnt/shared-workspace/cbis_ddsm/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/shared-workspace/cbis_ddsm/nbia_test_subset"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="cap number of series (for smoke tests)")
    args = ap.parse_args()

    csv_paths = ensure_label_csvs(args.raw_dir)
    rows = parse_test_csvs(csv_paths)
    print(f"[parse] {len(rows)} unique full-mammogram series in test split")
    labels = Counter(r["malignant"] for r in rows)
    print(f"[parse] malignant={labels[1]}  benign={labels[0]}")

    if args.limit:
        rows = rows[: args.limit]
        print(f"[limit] capped to {len(rows)} series")

    dicom_dir = args.out_dir / "dicoms"
    dicom_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(download_one_series, r["series_uid"], dicom_dir / f"{r['series_uid']}.dcm"): r["series_uid"]
            for r in rows
        }
        done_n = 0
        for fut in as_completed(futs):
            su = futs[fut]
            res = fut.result()
            results[su] = res
            done_n += 1
            if done_n % 25 == 0 or done_n == len(rows):
                elapsed = time.time() - t0
                n_fail = sum(1 for r in results.values() if r["status"] == "failed")
                print(f"[dl {done_n:>4d}/{len(rows)}]  elapsed={elapsed:.0f}s  failed_so_far={n_fail}")

    n_ok = sum(1 for r in results.values() if r["status"] in ("downloaded", "already_on_disk"))
    n_fail = sum(1 for r in results.values() if r["status"] == "failed")
    print(f"[done] ok={n_ok}  failed={n_fail}  wall={time.time()-t0:.0f}s")

    manifest_path = args.out_dir / "manifest.csv"
    fieldnames = [
        "series_uid", "study_uid", "patient_id", "abnormality_type", "left_or_right",
        "view", "pathology_raw_values", "malignant", "n_abnormality_rows",
        "download_status", "dicom_path", "dicom_bytes", "error",
    ]
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            res = results.get(r["series_uid"], {"status": "not_attempted"})
            dpath = dicom_dir / f"{r['series_uid']}.dcm"
            w.writerow({
                **r,
                "download_status": res.get("status"),
                "dicom_path": str(dpath) if res.get("status") in ("downloaded", "already_on_disk") else "",
                "dicom_bytes": res.get("bytes", ""),
                "error": res.get("error", ""),
            })
    print(f"[manifest] wrote {manifest_path}")
    if n_fail:
        failed_uids = [su for su, r in results.items() if r["status"] == "failed"]
        print(f"[FAILURES] {n_fail} series failed to download (see manifest 'error' column): {failed_uids[:10]}{'...' if n_fail > 10 else ''}")
        sys.exit(1 if n_fail == len(rows) else 0)  # only hard-fail if EVERYTHING failed


if __name__ == "__main__":
    main()
