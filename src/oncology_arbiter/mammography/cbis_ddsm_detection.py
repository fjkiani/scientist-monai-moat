"""CBIS-DDSM detection pipeline: pair DICOMs to ROI masks, extract 2D bboxes.

CBIS-DDSM TCIA delivery structure
---------------------------------
Every case is stored as multiple *series*, keyed by `PatientID`:

  PatientID                          | SeriesDescription     | ROI index
  -----------------------------------|-----------------------|----------
  Mass-Training_P_01239_RIGHT_CC     | full mammogram images | -
  Mass-Training_P_01239_RIGHT_CC_1   | ROI mask images       | 1
  Mass-Training_P_01239_RIGHT_CC_2   | ROI mask images       | 2 (if two lesions)
  Mass-Training_P_01239_RIGHT_CC_1   | cropped images        | 1

The full mammogram carries the primary pixel data (uint16, 4000+ px on the
long side, BitsStored=16). The ROI mask series carries a binary silhouette
of the annotator's lesion contour at the SAME resolution as the mammogram
(BitsStored=8, PhotometricInterpretation=MONOCHROME2, pixel values 0/255).

TCIA's counts on the collection are:
  * 3103 full-mammogram series (one per breast+view+case)
  * 3565 ROI mask series (one per lesion — cases can have >1 lesion)
  *  107 cropped images (patch snapshots — not used for detection)

Case ID = PatientID with any trailing `_<int>` stripped.

BBox extraction
---------------
Each ROI mask is a full-size binary image. `bbox_from_roi_mask` returns the
axis-aligned rectangle enclosing the largest connected foreground component
(dropping any noise pixels < 0.1% of the image area). The bbox is normalized
to `[x_min, y_min, x_max, y_max]` in pixel coordinates on the SAME grid as
the full mammogram.

If multiple ROI masks exist for one case, each contributes one bbox on the
same mammogram (multi-lesion cases stay together in the dataset).

Lesion type labeling
--------------------
`PatientID` prefix carries the lesion class:
  * `Mass-...`  -> class "mass"       (label 0)
  * `Calc-...`  -> class "calcification" (label 1)

The v0.4.0-alpha detector treats these as a single lesion class ("lesion")
for RetinaNet training (matching LUNA16's single-class nodule setup) and
surfaces the finer mass/calc label on the metadata side.

Train/test split
----------------
CBIS-DDSM ships an official CBIS-DDSM split via `Training` / `Test` prefix
on `PatientID`. `build_case_manifest()` respects this — no random splitting.

Cross-reference
---------------
- CBIS-DDSM landing: https://www.cancerimagingarchive.net/collection/cbis-ddsm/
- Original DDSM: Lee et al. 2017 Sci Data 4:170177
- License: CC-BY-3.0
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# CBIS-DDSM PatientID pattern:
#   Mass-Training_P_01239_RIGHT_CC
#   Mass-Training_P_01239_RIGHT_CC_1  (with lesion index)
#   Calc-Test_P_00033_LEFT_MLO_2
_PATIENT_ID_RE = re.compile(
    r"^(?P<lesion_type>Mass|Calc)-"
    r"(?P<split>Training|Test)_"
    r"P_(?P<case_num>\d+)_"
    r"(?P<laterality>LEFT|RIGHT)_"
    r"(?P<view>CC|MLO)"
    r"(?:_(?P<lesion_idx>\d+))?$"
)


@dataclass
class CaseAnnotations:
    """One imaging case = one full mammogram + N ROI masks (lesions)."""

    case_id: str                            # "Mass-Training_P_01239_RIGHT_CC"
    lesion_type: str                        # "Mass" | "Calc"
    split: str                              # "Training" | "Test"
    case_num: int                           # 1239
    laterality: str                         # "LEFT" | "RIGHT"
    view: str                               # "CC" | "MLO"
    full_mammogram_uid: Optional[str] = None
    roi_mask_uids: List[str] = field(default_factory=list)   # in lesion-index order
    cropped_uids: List[str] = field(default_factory=list)


@dataclass
class BBox:
    """Axis-aligned 2D bounding box in mammogram pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: int         # 0 = lesion (single class for detector training)
    lesion_type: str   # "mass" | "calcification"

    def width(self) -> int:
        return self.x_max - self.x_min

    def height(self) -> int:
        return self.y_max - self.y_min

    def area(self) -> int:
        return self.width() * self.height()

    def to_xyxy(self) -> List[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


def parse_patient_id(patient_id: str) -> Optional[Dict[str, str]]:
    """Parse a CBIS-DDSM PatientID string into structured fields.

    Returns None on parse failure so callers can drop malformed series.
    """
    m = _PATIENT_ID_RE.match(patient_id)
    if not m:
        return None
    d = m.groupdict()
    d["case_num"] = int(d["case_num"])  # type: ignore[assignment]
    d["lesion_idx"] = int(d["lesion_idx"]) if d.get("lesion_idx") else 0  # type: ignore[assignment]
    return d


def build_case_manifest(series_manifest: List[Dict]) -> Dict[str, CaseAnnotations]:
    """Group a flat CBIS-DDSM series list into per-case annotations.

    Args:
        series_manifest: JSON output from `nbia.getSeries(collection='CBIS-DDSM')`.

    Returns:
        Dict of case_id -> CaseAnnotations.
    """
    cases: Dict[str, CaseAnnotations] = {}

    for s in series_manifest:
        pid = s.get("PatientID", "")
        desc = s.get("SeriesDescription", "").lower()
        uid = s.get("SeriesInstanceUID", "")

        parsed = parse_patient_id(pid)
        if not parsed:
            continue

        # Strip trailing _<lesion_idx> to derive the parent case id
        case_id = f"{parsed['lesion_type']}-{parsed['split']}_P_{parsed['case_num']:05d}_{parsed['laterality']}_{parsed['view']}"

        c = cases.setdefault(case_id, CaseAnnotations(
            case_id=case_id,
            lesion_type=parsed["lesion_type"],
            split=parsed["split"],
            case_num=parsed["case_num"],
            laterality=parsed["laterality"],
            view=parsed["view"],
        ))

        if "full mammogram" in desc:
            if c.full_mammogram_uid is None:
                c.full_mammogram_uid = uid
            # else: duplicate — CBIS-DDSM sometimes ships multiple takes; keep first
        elif "roi mask" in desc:
            c.roi_mask_uids.append(uid)
        elif "cropped" in desc:
            c.cropped_uids.append(uid)

    return cases


def bbox_from_roi_mask(mask_array, min_area_frac: float = 1e-3) -> Optional[BBox]:
    """Compute axis-aligned bbox from a binary ROI mask array.

    Args:
        mask_array: 2D uint8/uint16 array; foreground = nonzero pixels.
        min_area_frac: reject components smaller than this fraction of the
            image area (removes speckle / annotator noise).

    Returns:
        BBox with pixel coords, label=0 (single class), lesion_type placeholder.
        Returns None if the mask is empty or all components are below threshold.
    """
    import numpy as np
    from scipy import ndimage as ndi

    if mask_array is None or mask_array.size == 0:
        return None

    binary = mask_array > 0
    if not binary.any():
        return None

    labels, n = ndi.label(binary)
    if n == 0:
        return None

    sizes = ndi.sum(binary, labels, range(1, n + 1))
    total_area = float(binary.size)
    min_area = min_area_frac * total_area

    valid = sizes >= min_area
    if not valid.any():
        return None

    # Take the largest valid component
    best = int(np.argmax(sizes * valid)) + 1
    ys, xs = np.where(labels == best)

    return BBox(
        x_min=int(xs.min()),
        y_min=int(ys.min()),
        x_max=int(xs.max()) + 1,   # exclusive bound, xyxy convention
        y_max=int(ys.max()) + 1,
        label=0,
        lesion_type="unknown",     # caller fills from case metadata
    )


def bboxes_from_case_masks(
    case: CaseAnnotations,
    tcia_root: Path,
    read_dicom_pixels,
) -> List[BBox]:
    """Load every ROI mask for a case and return the list of bboxes.

    Args:
        case: CaseAnnotations from build_case_manifest.
        tcia_root: Root of the downloaded TCIA data
            (contains one directory per SeriesInstanceUID).
        read_dicom_pixels: callable(dicom_dir: Path) -> np.ndarray.
            Injected to avoid hardcoding a pydicom dependency here.

    Returns:
        List of BBox objects (one per ROI mask series). Empty list if the
        case has no valid masks.
    """
    lesion_label = "mass" if case.lesion_type == "Mass" else "calcification"

    bboxes: List[BBox] = []
    for uid in case.roi_mask_uids:
        mask_dir = tcia_root / uid
        if not mask_dir.exists():
            continue
        try:
            arr = read_dicom_pixels(mask_dir)
        except Exception:
            continue

        bb = bbox_from_roi_mask(arr)
        if bb is None:
            continue
        bb.lesion_type = lesion_label
        bboxes.append(bb)

    return bboxes


def coco_style_annotations(
    cases: List[CaseAnnotations],
    tcia_root: Path,
    read_dicom_pixels,
    out_path: Path,
) -> Dict[str, object]:
    """Emit a COCO-format annotations JSON compatible with MONAI's RetinaNet.

    Format:
        {
          "info": {...},
          "images": [
              {"id": <int>, "file_name": "<uid>/<uid>.dcm", "width": ..., "height": ...},
              ...
          ],
          "annotations": [
              {"id": <int>, "image_id": <int>, "bbox": [x, y, w, h], "category_id": 0, "iscrowd": 0},
              ...
          ],
          "categories": [{"id": 0, "name": "lesion"}],
        }
    """
    images: List[Dict] = []
    anns: List[Dict] = []
    img_id_of: Dict[str, int] = {}
    next_ann_id = 1

    for case in cases:
        if not case.full_mammogram_uid:
            continue
        # Load full mammogram to get width/height
        mammo_dir = tcia_root / case.full_mammogram_uid
        if not mammo_dir.exists():
            continue
        try:
            mammo = read_dicom_pixels(mammo_dir)
        except Exception:
            continue

        img_id = len(images) + 1
        img_id_of[case.case_id] = img_id
        images.append({
            "id": img_id,
            "file_name": f"{case.full_mammogram_uid}/full.dcm",
            "case_id": case.case_id,
            "width": int(mammo.shape[1]),
            "height": int(mammo.shape[0]),
            "split": case.split,
            "laterality": case.laterality,
            "view": case.view,
            "lesion_type": case.lesion_type,
        })

        bboxes = bboxes_from_case_masks(case, tcia_root, read_dicom_pixels)
        for bb in bboxes:
            anns.append({
                "id": next_ann_id,
                "image_id": img_id,
                "bbox": [bb.x_min, bb.y_min, bb.width(), bb.height()],
                "area": bb.area(),
                "category_id": 0,
                "iscrowd": 0,
                "lesion_type": bb.lesion_type,
            })
            next_ann_id += 1

    payload = {
        "info": {
            "description": "CBIS-DDSM bounding boxes derived from TCIA ROI mask series",
            "version": "v0.4.0-alpha",
            "n_images": len(images),
            "n_annotations": len(anns),
            "n_train_images": sum(1 for i in images if i["split"] == "Training"),
            "n_test_images": sum(1 for i in images if i["split"] == "Test"),
        },
        "images": images,
        "annotations": anns,
        "categories": [{"id": 0, "name": "lesion"}],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)

    return payload


__all__ = [
    "BBox",
    "CaseAnnotations",
    "bbox_from_roi_mask",
    "bboxes_from_case_masks",
    "build_case_manifest",
    "coco_style_annotations",
    "parse_patient_id",
]
