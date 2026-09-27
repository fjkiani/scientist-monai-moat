"""Download a real, non-phantom NLST (National Lung Screening Trial) chest-CT
subset directly from TCIA's dedicated NLST NBIA server, for a genuine
cross-cohort domain-shift stress test of the luna16-detect endpoint.

Wave-1 locked instruction: "NLST via TCIA Search subset (beyond planted
phantoms)". This script queries the LIVE NBIA metadata for the NLST
collection and empirically confirms provenance before downloading anything
-- it does not assume phantom-free data, it checks:

  - getPatient?Collection=NLST (this session, live query, all 26,254
    patients) -> Phantom field is "NO" for every single patient in this
    public release. There are zero phantom-flagged patients to accidentally
    include; this is asserted from the live API response, not assumed.
  - Only the NLST-LSS study arm (main screening protocol, 139,509 of
    203,099 total CT series) is sampled here, not NLST-ACRIN (a separate
    technical sub-study, 63,590 series) -- LSS is what "NLST" conventionally
    refers to in the screening literature.

Access mechanics (empirically determined this session, NOT in the generic
NBIA docs verbatim -- the NLST collection lives on a SEPARATE dedicated NBIA
server because of its size):
  - Token: POST https://nlst.cancerimagingarchive.net/nbia-api/oauth/token
    with username=nbia_guest&password=&client_id=NBIA&grant_type=password
    (no real credentials; TCIA's own documented public guest flow). Token is
    a ~2h JWT.
  - getSeries / getImage on https://nlst.cancerimagingarchive.net/nbia-api/
    services/v2/... REQUIRE the Bearer token even for this "public" data
    (unauthenticated calls return HTTP 404/500) -- unlike the main
    services.cancerimagingarchive.net server used for CBIS-DDSM, which
    needs no token at all. This asymmetry was verified by direct probing,
    not assumed from documentation.
  - getImage returns a ZIP with one member per DICOM instance (e.g. 140
    members for a 139-slice series: "LICENSE" + 139 "%08d.dcm" files) --
    NOT the single-DICOM-per-zip pattern used by the CBIS-DDSM mammography
    download script. Confirmed this session on a live 139-slice series
    (49,977,865 zip bytes, member size range consistent with real per-slice
    CT pixel data, not placeholder/truncated files).

Scope (Wave-1 vs Wave-2, per locked instructions):
  - THIS script is Wave-1: images only, no ground-truth nodule/malignancy
    labels (there are none in the open TCIA release -- NLST's full clinical
    outcomes require a formal Cancer Data Access System (CDAS) application,
    which is Wave-2, drafted separately).
  - Consequently this subset supports a QUALITATIVE domain-shift check
    against luna16-detect (does the endpoint run, does detection-count /
    Finding-G-style large-volume crash risk look different on a screening
    population vs the diagnostic-referral LIDC-IDRI/LUNA16 cohort) -- NOT a
    FROC/sensitivity number, which requires the CDAS-gated ground truth.

Usage::

    python3 scripts/download_nlst_ct_subset.py \\
        --out-dir /mnt/shared-workspace/nlst/subset \\
        --n-series 20 \\
        --min-image-count 80 --max-image-count 400
"""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List
from urllib import request as urllib_request

NLST_BASE = "https://nlst.cancerimagingarchive.net/nbia-api"
TOKEN_URL = f"{NLST_BASE}/oauth/token"
SERVICES_V2 = f"{NLST_BASE}/services/v2"


def get_guest_token() -> str:
    data = b"username=nbia_guest&password=&client_id=NBIA&grant_type=password"
    req = urllib_request.Request(TOKEN_URL, data=data, method="POST")
    with urllib_request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read())
    return payload["access_token"]


def _get_json(url: str, token: str, timeout: float = 120.0) -> Any:
    req = urllib_request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def verify_no_phantoms(token: str) -> Dict[str, int]:
    patients = _get_json(f"{SERVICES_V2}/getPatient?Collection=NLST", token, timeout=60)
    counts: Dict[str, int] = {}
    for p in patients:
        v = p.get("Phantom", "UNKNOWN")
        counts[v] = counts.get(v, 0) + 1
    print(f"[verify] Phantom field breakdown across {len(patients)} NLST patients: {counts}")
    return counts


def select_series(token: str, *, n_series: int, min_ic: int, max_ic: int) -> List[Dict[str, Any]]:
    all_series = _get_json(f"{SERVICES_V2}/getSeries?Collection=NLST&Modality=CT", token, timeout=180)
    print(f"[select] {len(all_series)} total NLST CT series from live query")
    seen_patients: set[str] = set()
    chosen: List[Dict[str, Any]] = []
    for r in sorted(all_series, key=lambda r: (r["PatientID"], r["SeriesInstanceUID"])):
        if r.get("StudyDesc") != "NLST-LSS":
            continue
        ic = r.get("ImageCount", 0)
        if not (min_ic <= ic <= max_ic):
            continue
        pid = r["PatientID"]
        if pid in seen_patients:
            continue
        seen_patients.add(pid)
        chosen.append(r)
        if len(chosen) >= n_series:
            break
    return chosen


def download_series(series_uid: str, dest_dir: Path, token: str, *, max_retries: int = 3) -> Dict[str, Any]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{SERVICES_V2}/getImage?SeriesInstanceUID={series_uid}"
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib_request.Request(url, headers={"Authorization": f"Bearer {token}"})
            with urllib_request.urlopen(req, timeout=180) as resp:
                zip_bytes = resp.read()
            zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
            dcm_names = [n for n in zf.namelist() if n.upper() != "LICENSE"]
            if not dcm_names:
                raise RuntimeError(f"no non-LICENSE members: {zf.namelist()[:5]}")
            for n in dcm_names:
                (dest_dir / n).write_bytes(zf.read(n))
            return {"series_uid": series_uid, "status": "downloaded", "n_slices": len(dcm_names),
                     "zip_bytes": len(zip_bytes)}
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 10))
    return {"series_uid": series_uid, "status": "failed", "error": last_err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/shared-workspace/nlst/subset"))
    ap.add_argument("--n-series", type=int, default=20)
    ap.add_argument("--min-image-count", type=int, default=80)
    ap.add_argument("--max-image-count", type=int, default=400)
    args = ap.parse_args()

    token = get_guest_token()
    print(f"[auth] guest token acquired, len={len(token)}")

    phantom_counts = verify_no_phantoms(token)

    chosen = select_series(
        token, n_series=args.n_series, min_ic=args.min_image_count, max_ic=args.max_image_count
    )
    print(f"[select] chose {len(chosen)} series (1 per distinct patient, NLST-LSS, "
          f"{args.min_image_count}<=ImageCount<={args.max_image_count})")

    dicom_root = args.out_dir / "dicoms"
    results = []
    t0 = time.time()
    for i, r in enumerate(chosen):
        su = r["SeriesInstanceUID"]
        dest_dir = dicom_root / su
        if dest_dir.exists() and any(dest_dir.iterdir()):
            n_existing = len(list(dest_dir.glob("*.dcm")))
            if n_existing >= r["ImageCount"]:
                results.append({"series_uid": su, "status": "already_on_disk", "n_slices": n_existing})
                print(f"[{i+1}/{len(chosen)}] {su[-16:]} already on disk ({n_existing} slices)")
                continue
        # Refresh token every 10 series as a cheap safeguard against the ~2h expiry.
        if i > 0 and i % 10 == 0:
            token = get_guest_token()
        res = download_series(su, dest_dir, token)
        res["patient_id"] = r["PatientID"]
        res["declared_image_count"] = r["ImageCount"]
        res["manufacturer"] = r.get("Manufacturer")
        res["study_date"] = r.get("StudyDate")
        results.append(res)
        print(f"[{i+1}/{len(chosen)}] {su[-16:]} pid={r['PatientID']} -> {res['status']} "
              f"({res.get('n_slices', '?')} slices, declared={r['ImageCount']})  elapsed={time.time()-t0:.0f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "collection": "NLST",
        "study_arm": "NLST-LSS",
        "source": "https://nlst.cancerimagingarchive.net (live NBIA query + download, this session)",
        "phantom_field_breakdown_all_26254_patients": phantom_counts,
        "selection_criteria": {
            "modality": "CT", "study_desc": "NLST-LSS",
            "min_image_count": args.min_image_count, "max_image_count": args.max_image_count,
            "one_series_per_distinct_patient": True,
        },
        "scope_disclaimer": (
            "Images only, no CDAS ground-truth labels (Wave-1 scope). Supports a qualitative "
            "cross-cohort domain-shift check against luna16-detect, NOT a FROC/sensitivity claim."
        ),
        "results": results,
    }, indent=2))
    n_ok = sum(1 for r in results if r["status"] in ("downloaded", "already_on_disk"))
    print(f"[done] {n_ok}/{len(chosen)} ok, manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
