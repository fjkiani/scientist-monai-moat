"""Unit tests for `oncology_arbiter.mammography.cbis_ddsm_detection`.

These tests do NOT require the 163 GB TCIA corpus. They exercise:
    * `parse_patient_id` on all valid CBIS-DDSM PatientID forms.
    * `build_case_manifest` on synthetic manifest fixtures.
    * `bbox_from_roi_mask` on synthetic binary masks.
    * `coco_style_annotations` with a mock DICOM reader.

Real-data integration tests live in `tests/integration/` and are gated on
`OA_CBIS_TCIA_ROOT` env var pointing to a materialized TCIA download.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pytest

from oncology_arbiter.mammography.cbis_ddsm_detection import (
    BBox,
    CaseAnnotations,
    bbox_from_roi_mask,
    bboxes_from_case_masks,
    build_case_manifest,
    coco_style_annotations,
    parse_patient_id,
)


# --------------------------------------------------------------------------- #
# parse_patient_id
# --------------------------------------------------------------------------- #

class TestParsePatientId:
    def test_full_mammogram(self):
        d = parse_patient_id("Mass-Training_P_01239_RIGHT_CC")
        assert d is not None
        assert d["lesion_type"] == "Mass"
        assert d["split"] == "Training"
        assert d["case_num"] == 1239
        assert d["laterality"] == "RIGHT"
        assert d["view"] == "CC"
        assert d["lesion_idx"] == 0

    def test_roi_mask_with_index(self):
        d = parse_patient_id("Mass-Training_P_01239_RIGHT_CC_1")
        assert d is not None
        assert d["lesion_idx"] == 1

    def test_calc_test_mlo(self):
        d = parse_patient_id("Calc-Test_P_00033_LEFT_MLO_2")
        assert d is not None
        assert d["lesion_type"] == "Calc"
        assert d["split"] == "Test"
        assert d["laterality"] == "LEFT"
        assert d["view"] == "MLO"
        assert d["lesion_idx"] == 2

    def test_malformed_returns_none(self):
        assert parse_patient_id("random_stuff") is None
        assert parse_patient_id("Mass-P_01239_RIGHT_CC") is None       # missing split
        assert parse_patient_id("Mass-Training_P_LEFT_CC") is None     # missing case num
        assert parse_patient_id("") is None


# --------------------------------------------------------------------------- #
# build_case_manifest
# --------------------------------------------------------------------------- #

def _mk(pid: str, desc: str, uid: str) -> Dict[str, str]:
    return {
        "PatientID": pid,
        "SeriesDescription": desc,
        "SeriesInstanceUID": uid,
    }


class TestBuildCaseManifest:
    def test_single_case_single_lesion(self):
        manifest = [
            _mk("Mass-Training_P_00001_RIGHT_CC", "full mammogram images", "uid-full-1"),
            _mk("Mass-Training_P_00001_RIGHT_CC_1", "ROI mask images", "uid-mask-1"),
        ]
        cases = build_case_manifest(manifest)
        assert len(cases) == 1
        c = cases["Mass-Training_P_00001_RIGHT_CC"]
        assert c.full_mammogram_uid == "uid-full-1"
        assert c.roi_mask_uids == ["uid-mask-1"]
        assert c.lesion_type == "Mass"
        assert c.laterality == "RIGHT"

    def test_multi_lesion_grouping(self):
        # A case with two masks — both should end up on the same CaseAnnotations
        manifest = [
            _mk("Calc-Test_P_00033_LEFT_MLO", "full mammogram images", "uid-full-2"),
            _mk("Calc-Test_P_00033_LEFT_MLO_1", "ROI mask images", "uid-mask-2a"),
            _mk("Calc-Test_P_00033_LEFT_MLO_2", "ROI mask images", "uid-mask-2b"),
            _mk("Calc-Test_P_00033_LEFT_MLO_1", "cropped images", "uid-cropped-2"),
        ]
        cases = build_case_manifest(manifest)
        assert len(cases) == 1
        c = list(cases.values())[0]
        assert c.full_mammogram_uid == "uid-full-2"
        assert set(c.roi_mask_uids) == {"uid-mask-2a", "uid-mask-2b"}
        assert c.cropped_uids == ["uid-cropped-2"]
        assert c.lesion_type == "Calc"

    def test_malformed_skipped(self):
        manifest = [
            _mk("Mass-Training_P_00001_RIGHT_CC", "full mammogram images", "uid-full-1"),
            _mk("NOT_A_VALID_ID", "full mammogram images", "uid-junk"),
        ]
        cases = build_case_manifest(manifest)
        assert len(cases) == 1
        assert "Mass-Training_P_00001_RIGHT_CC" in cases

    def test_orphan_mask_still_creates_case(self):
        # A ROI mask without a matching full mammogram still creates the case
        # entry (with full_mammogram_uid=None). This mirrors the TCIA reality
        # that a small number of cases have missing full-mammogram series.
        manifest = [
            _mk("Mass-Training_P_00099_LEFT_CC_1", "ROI mask images", "uid-mask-99"),
        ]
        cases = build_case_manifest(manifest)
        assert len(cases) == 1
        c = list(cases.values())[0]
        assert c.full_mammogram_uid is None
        assert c.roi_mask_uids == ["uid-mask-99"]


# --------------------------------------------------------------------------- #
# bbox_from_roi_mask
# --------------------------------------------------------------------------- #

class TestBboxFromRoiMask:
    def test_single_rectangle(self):
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[30:70, 50:150] = 255
        bb = bbox_from_roi_mask(mask)
        assert bb is not None
        assert bb.x_min == 50
        assert bb.y_min == 30
        assert bb.x_max == 150   # exclusive bound
        assert bb.y_max == 70
        assert bb.label == 0

    def test_ellipse(self):
        # An ellipse — bbox should hug it, not be off-by-huge
        yy, xx = np.mgrid[0:200, 0:300]
        mask = (((xx - 150) / 60) ** 2 + ((yy - 100) / 40) ** 2 <= 1).astype(np.uint8)
        bb = bbox_from_roi_mask(mask)
        assert bb is not None
        assert 85 <= bb.x_min <= 95
        assert 205 <= bb.x_max <= 215
        assert 55 <= bb.y_min <= 65
        assert 135 <= bb.y_max <= 145

    def test_empty_mask_returns_none(self):
        assert bbox_from_roi_mask(np.zeros((100, 100), dtype=np.uint8)) is None

    def test_noise_below_threshold_rejected(self):
        # Two tiny 3x3 blobs, both below the 0.1% area threshold on a 100x100 image
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[5:8, 5:8] = 255   # 9 pixels (0.09% of 10k)
        mask[80:83, 80:83] = 255
        assert bbox_from_roi_mask(mask, min_area_frac=1e-3) is None

    def test_multiple_components_takes_largest(self):
        # Two components: a small one and a big one — big should win
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:30, 10:30] = 255       # small: 20x20 = 400 pixels
        mask[100:180, 100:180] = 255   # big: 80x80 = 6400 pixels
        bb = bbox_from_roi_mask(mask)
        assert bb is not None
        # Should hug the big component, not the small one
        assert bb.x_min == 100
        assert bb.y_min == 100

    def test_bbox_geometry(self):
        bb = BBox(x_min=10, y_min=20, x_max=110, y_max=170, label=0, lesion_type="mass")
        assert bb.width() == 100
        assert bb.height() == 150
        assert bb.area() == 15_000
        assert bb.to_xyxy() == [10, 20, 110, 170]


# --------------------------------------------------------------------------- #
# coco_style_annotations
# --------------------------------------------------------------------------- #

class TestCocoStyleAnnotations:
    def _fake_reader(self, mask_shape=(4600, 2800)) -> Callable[[Path], np.ndarray]:
        """Return a callable that returns a full-size mask when the dir name matches."""
        def read(dicom_dir: Path) -> np.ndarray:
            uid = dicom_dir.name
            # "full-*" uids -> empty full mammogram (we just need shape)
            if uid.startswith("uid-full-"):
                return np.zeros(mask_shape, dtype=np.uint16)
            # "mask-*" uids -> a rectangle in the middle
            arr = np.zeros(mask_shape, dtype=np.uint8)
            arr[1000:2000, 800:1600] = 255
            return arr
        return read

    def test_single_case_produces_one_image_one_ann(self, tmp_path: Path):
        cases = [
            CaseAnnotations(
                case_id="Mass-Training_P_00001_RIGHT_CC",
                lesion_type="Mass",
                split="Training",
                case_num=1,
                laterality="RIGHT",
                view="CC",
                full_mammogram_uid="uid-full-1",
                roi_mask_uids=["uid-mask-1"],
            ),
        ]
        # Create the (empty) directories so the reader is called
        (tmp_path / "uid-full-1").mkdir()
        (tmp_path / "uid-mask-1").mkdir()

        out_path = tmp_path / "annotations.json"
        payload = coco_style_annotations(cases, tmp_path, self._fake_reader(), out_path)

        assert payload["info"]["n_images"] == 1
        assert payload["info"]["n_annotations"] == 1
        assert payload["info"]["n_train_images"] == 1
        assert payload["info"]["n_test_images"] == 0

        img = payload["images"][0]
        assert img["case_id"] == "Mass-Training_P_00001_RIGHT_CC"
        assert img["width"] == 2800   # numpy shape[1]
        assert img["height"] == 4600  # numpy shape[0]

        ann = payload["annotations"][0]
        assert ann["image_id"] == img["id"]
        assert ann["bbox"] == [800, 1000, 800, 1000]  # [x, y, w, h] COCO format
        assert ann["category_id"] == 0
        assert ann["lesion_type"] == "mass"

        # Also written to disk
        with open(out_path) as f:
            reloaded = json.load(f)
        assert reloaded["info"]["n_images"] == 1

    def test_multi_lesion_case_produces_multiple_anns(self, tmp_path: Path):
        cases = [
            CaseAnnotations(
                case_id="Calc-Test_P_00033_LEFT_MLO",
                lesion_type="Calc",
                split="Test",
                case_num=33,
                laterality="LEFT",
                view="MLO",
                full_mammogram_uid="uid-full-2",
                roi_mask_uids=["uid-mask-2a", "uid-mask-2b"],
            ),
        ]
        for uid in ["uid-full-2", "uid-mask-2a", "uid-mask-2b"]:
            (tmp_path / uid).mkdir()

        payload = coco_style_annotations(
            cases, tmp_path, self._fake_reader(), tmp_path / "ann.json"
        )
        assert payload["info"]["n_images"] == 1
        assert payload["info"]["n_annotations"] == 2
        assert payload["info"]["n_test_images"] == 1
        for ann in payload["annotations"]:
            assert ann["lesion_type"] == "calcification"

    def test_missing_full_mammogram_skipped(self, tmp_path: Path):
        cases = [
            CaseAnnotations(
                case_id="Mass-Training_P_00099_LEFT_CC",
                lesion_type="Mass",
                split="Training",
                case_num=99,
                laterality="LEFT",
                view="CC",
                full_mammogram_uid=None,          # <- missing
                roi_mask_uids=["uid-mask-99"],
            ),
        ]
        payload = coco_style_annotations(
            cases, tmp_path, self._fake_reader(), tmp_path / "ann.json"
        )
        assert payload["info"]["n_images"] == 0
        assert payload["info"]["n_annotations"] == 0
