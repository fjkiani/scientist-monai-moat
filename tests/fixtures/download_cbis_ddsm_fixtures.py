"""Download the 5 CBIS-DDSM DICOM fixtures used by real-data tests.

The DICOMs are ~120 MB total and are excluded from git via `.gitignore`
(`tests/fixtures/cbis_ddsm/*.dcm`). Run this script once after cloning
to populate the fixture directory:

    python tests/fixtures/download_cbis_ddsm_fixtures.py

Source: helloerikaaa/cbis-ddsm-r on Hugging Face (CC-BY-NC 4.0).
No authentication required.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "helloerikaaa/cbis-ddsm-r"
REPO_TYPE = "dataset"
FIXTURE_DIR = Path(__file__).parent / "cbis_ddsm"

# (fixture filename, path within the HF repo)
FIXTURES = [
    ("Calc-Test_P_00038_LEFT_CC.dcm",
     "img/Calc-Test_P_00038_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.85935434310203356712688695661986996009/"
     "1.3.6.1.4.1.9590.100.1.2.374115997511889073021386151921807063992/00000001.dcm"),
    ("Calc-Test_P_00038_RIGHT_CC.dcm",
     "img/Calc-Test_P_00038_RIGHT_CC/1.3.6.1.4.1.9590.100.1.2.177706148911820252341905176394069228468/"
     "1.3.6.1.4.1.9590.100.1.2.263861248711313923336051913560309963304/00000001.dcm"),
    ("Calc-Test_P_00038_LEFT_MLO.dcm",
     "img/Calc-Test_P_00038_LEFT_MLO/1.3.6.1.4.1.9590.100.1.2.384159464510350889125645400702639717613/"
     "1.3.6.1.4.1.9590.100.1.2.174390361112646747718661211471328897934/00000001.dcm"),
    ("Calc-Test_P_00038_RIGHT_MLO.dcm",
     "img/Calc-Test_P_00038_RIGHT_MLO/"
     "1.3.6.1.4.1.9590.100.1.2.166810167510410155611211126134061906263/"
     "1.3.6.1.4.1.9590.100.1.2.313125398410123020722513144242180415293/00000001.dcm"),
    ("Mass-Test_P_00016_LEFT_CC.dcm",
     "img/Mass-Test_P_00016_LEFT_CC/1.3.6.1.4.1.9590.100.1.2.253817146311303055112013188031820262133/"
     "1.3.6.1.4.1.9590.100.1.2.221881145710257050212215210330168260107/00000001.dcm"),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = FIXTURE_DIR / "_dicom_metadata.json"
    if not meta_path.exists():
        print(f"WARNING: {meta_path} missing — sanity-check metadata may not be available",
              file=sys.stderr)

    results = {}
    for target_name, hf_path in FIXTURES:
        target = FIXTURE_DIR / target_name
        if target.exists():
            print(f"[skip] {target_name} (already present, {target.stat().st_size} bytes)")
            results[target_name] = {"status": "skipped", "sha256": sha256_of(target)}
            continue
        print(f"[fetch] {target_name} <- {hf_path}")
        cached = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=hf_path,
            cache_dir=str(FIXTURE_DIR / ".cache"),
        )
        # Copy (not symlink) so the fixture survives cache eviction
        target.write_bytes(Path(cached).read_bytes())
        results[target_name] = {
            "status": "downloaded",
            "bytes": target.stat().st_size,
            "sha256": sha256_of(target),
        }
        print(f"        wrote {target.stat().st_size} bytes")

    print("\nSummary:")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
