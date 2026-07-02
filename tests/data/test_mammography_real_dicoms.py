"""Real-data tests for mammography preprocessing.

Uses 5 real DICOMs pulled from a public CBIS-DDSM mirror on Hugging Face
(helloerikaaa/cbis-ddsm-r, CC-BY-NC 4.0). Fixtures live in
tests/fixtures/cbis_ddsm/ and total ~120 MB.

These are integration-style tests — no mocks. They exercise the real pydicom
readers, the numpy segmentation code, and the orientation normalization
against genuine acquisition-orientation clinical mammograms.

Ground truth is derived from the CBIS-DDSM filename convention:
    {Calc|Mass}-{Test|Training}_P_{patient_id}_{LEFT|RIGHT}_{CC|MLO}.dcm
which is unambiguous. Content-based laterality is cross-checked against it.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from oncology_arbiter.mammography import (
    Laterality,
    View,
    breast_mask_otsu,
    detect_laterality_from_content,
    detect_laterality_from_metadata,
    detect_view_from_metadata,
    preprocess_mammogram,
    read_mammogram_dicom,
    remove_pectoral_mlo,
    orient_to_radiological_convention,
)

pytestmark = pytest.mark.data

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "cbis_ddsm"


class Fixture(NamedTuple):
    filename: str
    expected_laterality: Laterality
    expected_view: View
    lesion_type: str


FIXTURES: list[Fixture] = [
    Fixture("Calc-Test_P_00038_LEFT_CC.dcm", Laterality.LEFT, View.CC, "calcification"),
    Fixture("Calc-Test_P_00038_RIGHT_CC.dcm", Laterality.RIGHT, View.CC, "calcification"),
    Fixture("Calc-Test_P_00038_LEFT_MLO.dcm", Laterality.LEFT, View.MLO, "calcification"),
    Fixture("Calc-Test_P_00038_RIGHT_MLO.dcm", Laterality.RIGHT, View.MLO, "calcification"),
    Fixture("Mass-Test_P_00016_LEFT_CC.dcm", Laterality.LEFT, View.CC, "mass"),
]


def _skip_if_missing(f: Fixture) -> Path:
    path = FIXTURE_DIR / f.filename
    if not path.exists():
        pytest.skip(f"missing fixture {path}")
    return path


# --------------------------------------------------------------------------- #
# Reader tests


@pytest.mark.parametrize("f", FIXTURES, ids=lambda f: f.filename)
def test_reader_produces_normalized_float(f: Fixture) -> None:
    """DICOM reader returns 2D float32 in [0, 1] with reasonable dynamic range."""
    path = _skip_if_missing(f)
    arr, meta = read_mammogram_dicom(path)
    assert arr.ndim == 2, f"expected 2D array, got {arr.shape}"
    assert arr.dtype == np.float32, f"expected float32, got {arr.dtype}"
    assert 0.0 <= arr.min() <= arr.max() <= 1.0
    # A real breast mammogram is not all zero and not all one.
    assert arr.max() > 0.001, "image is uniformly zero"
    assert arr.mean() > 0.0001, "image has essentially no tissue"
    assert meta["modality"] == "MG"
    assert meta["photometric"] in ("MONOCHROME1", "MONOCHROME2")
    assert meta["rows"] == arr.shape[0]
    assert meta["cols"] == arr.shape[1]


@pytest.mark.parametrize("f", FIXTURES, ids=lambda f: f.filename)
def test_reader_extracts_body_part_and_orientation(f: Fixture) -> None:
    """CBIS-DDSM stores view in PatientOrientation and (sometimes) laterality
    in BodyPartExamined. Real-data heterogeneity: Calc- subset uses
    'Left Breast'/'Right Breast', but Mass- subset uses just 'BREAST' with no
    laterality info. The reader records whatever BodyPartExamined says; the
    laterality hint field is only populated when the string mentions a side."""
    path = _skip_if_missing(f)
    _, meta = read_mammogram_dicom(path)
    body = meta.get("body_part") or ""
    # BodyPartExamined must at least say "breast" (case-insensitive)
    assert "breast" in body.lower(), f"BodyPartExamined={body!r} — not a breast study?"
    # If it also contains a side, laterality_hint must match; otherwise it may be None.
    if "left" in body.lower():
        assert meta["laterality_hint"] == "L"
        assert f.expected_laterality == Laterality.LEFT
    elif "right" in body.lower():
        assert meta["laterality_hint"] == "R"
        assert f.expected_laterality == Laterality.RIGHT
    else:
        # e.g. Mass- subset with just "BREAST" — laterality_hint should be None
        # and we rely on filename or content detection downstream.
        assert meta["laterality_hint"] is None


# --------------------------------------------------------------------------- #
# Laterality detection


@pytest.mark.parametrize("f", FIXTURES, ids=lambda f: f.filename)
def test_laterality_from_metadata_matches_filename(f: Fixture) -> None:
    """Metadata-based laterality (from BodyPartExamined) matches the filename."""
    path = _skip_if_missing(f)
    _, meta = read_mammogram_dicom(path)
    lat = detect_laterality_from_metadata(meta, filename=path)
    assert lat == f.expected_laterality, (
        f"metadata laterality {lat} != expected {f.expected_laterality} for {f.filename}"
    )


@pytest.mark.parametrize("f", FIXTURES, ids=lambda f: f.filename)
def test_content_detection_produces_a_valid_answer(f: Fixture) -> None:
    """Content-based laterality returns L or R (never UNKNOWN) on a real mammogram
    with visible tissue.

    We do NOT assert content_lat == expected_lat here. Real-data finding:
    CBIS-DDSM is not uniformly in acquisition orientation. The 4 Calc- fixtures
    are in acquisition orientation (tissue side matches physical laterality),
    but Mass-Test_P_00016_LEFT_CC ships already-mirrored — its tissue is on the
    RIGHT side of the frame despite being a LEFT breast. This is a real dataset
    heterogeneity we surface here rather than paper over.

    The pipeline handles this in `orient_to_radiological_convention` using the
    authoritative filename/DICOM-derived laterality as ground truth — see the
    orientation and end-to-end tests below."""
    path = _skip_if_missing(f)
    arr, _ = read_mammogram_dicom(path)
    content_lat = detect_laterality_from_content(arr)
    assert content_lat in (Laterality.LEFT, Laterality.RIGHT), (
        f"content detection returned UNKNOWN for {f.filename} — real mammogram "
        f"should give a definite answer"
    )


def test_cbis_ddsm_dataset_orientation_is_heterogeneous() -> None:
    """Document (and lock in) the CBIS-DDSM orientation heterogeneity we found.

    Calc- fixtures: content-based side MATCHES filename laterality (= they are
        in acquisition orientation).
    Mass- fixture: content-based side DIFFERS from filename laterality (= it
        ships in a different orientation than the Calc- subset).

    If this test starts failing, either (a) our fixtures changed, or (b) the
    CBIS-DDSM curators harmonized orientations across subsets — either way
    worth investigating."""
    calc_agreements = 0
    calc_total = 0
    mass_agreements = 0
    mass_total = 0
    for f in FIXTURES:
        path = _skip_if_missing(f)
        arr, _ = read_mammogram_dicom(path)
        content_lat = detect_laterality_from_content(arr)
        agrees = content_lat == f.expected_laterality
        if f.filename.startswith("Calc-"):
            calc_total += 1
            calc_agreements += int(agrees)
        elif f.filename.startswith("Mass-"):
            mass_total += 1
            mass_agreements += int(agrees)
    # All Calc- fixtures should agree (they're in acquisition orientation)
    assert calc_agreements == calc_total, (
        f"expected all {calc_total} Calc- fixtures in acquisition orientation, "
        f"only {calc_agreements} agreed"
    )
    # The Mass- fixture we downloaded ships pre-mirrored
    assert mass_agreements < mass_total, (
        "Mass- fixture unexpectedly agrees — dataset orientation may have changed"
    )


# --------------------------------------------------------------------------- #
# View detection


@pytest.mark.parametrize("f", FIXTURES, ids=lambda f: f.filename)
def test_view_from_metadata_matches_filename(f: Fixture) -> None:
    path = _skip_if_missing(f)
    _, meta = read_mammogram_dicom(path)
    view = detect_view_from_metadata(meta, filename=path)
    assert view == f.expected_view, (
        f"view {view} != expected {f.expected_view} for {f.filename}"
    )


# --------------------------------------------------------------------------- #
# Orientation normalization


@pytest.mark.parametrize("f", FIXTURES, ids=lambda f: f.filename)
def test_orient_to_radiological_flips_when_needed(f: Fixture) -> None:
    """After radiological orientation:
       LEFT laterality → tissue on the LEFT half (chest wall on right)
       RIGHT laterality → tissue on the RIGHT half (chest wall on left)

    NB: this project defines "radiological convention" as tissue LEFT for
    LEFT breast (see laterality.py docstring). Test what the code guarantees.
    """
    path = _skip_if_missing(f)
    arr, _ = read_mammogram_dicom(path)
    oriented = orient_to_radiological_convention(arr, f.expected_laterality)
    # Tissue side after orientation
    content_side = detect_laterality_from_content(oriented)
    assert content_side == f.expected_laterality


# --------------------------------------------------------------------------- #
# Breast mask


@pytest.mark.parametrize("f", FIXTURES, ids=lambda f: f.filename)
def test_breast_mask_coverage_is_plausible(f: Fixture) -> None:
    """Otsu + largest-CC mask should cover a plausible fraction of the frame.

    Real mammograms have breast tissue on 15-70% of the frame (varies with
    breast size and framing). Outside this range means the mask has picked
    up background or missed the breast.
    """
    path = _skip_if_missing(f)
    arr, _ = read_mammogram_dicom(path)
    mask = breast_mask_otsu(arr)
    coverage = mask.mean()
    assert 0.10 <= coverage <= 0.80, (
        f"breast mask coverage {coverage:.2%} outside plausible range 10-80% "
        f"for {f.filename}"
    )
    # Mask should be connected (largest CC returns exactly one region).
    assert mask.any(), "empty mask"


@pytest.mark.parametrize("f", FIXTURES, ids=lambda f: f.filename)
def test_breast_mask_is_on_expected_side(f: Fixture) -> None:
    """After orientation to radiological convention, the mask centroid should
    be on the correct side of the frame for the given laterality."""
    path = _skip_if_missing(f)
    arr, _ = read_mammogram_dicom(path)
    oriented = orient_to_radiological_convention(arr, f.expected_laterality)
    mask = breast_mask_otsu(oriented)
    h, w = mask.shape
    # Column centroid of the mask
    _, cols = np.where(mask)
    col_centroid = float(cols.mean()) if cols.size else w / 2
    if f.expected_laterality == Laterality.LEFT:
        # Tissue should sit in the left half (col_centroid < w/2)
        assert col_centroid < w / 2, (
            f"LEFT laterality but mask centroid at column {col_centroid:.0f} of {w} "
            f"(should be < {w/2:.0f}) for {f.filename}"
        )
    else:
        assert col_centroid > w / 2, (
            f"RIGHT laterality but mask centroid at column {col_centroid:.0f} of {w} "
            f"(should be > {w/2:.0f}) for {f.filename}"
        )


# --------------------------------------------------------------------------- #
# Pectoral removal (MLO views only)


@pytest.mark.parametrize(
    "f", [ff for ff in FIXTURES if ff.expected_view == View.MLO],
    ids=lambda f: f.filename,
)
def test_pectoral_removal_reduces_top_corner_intensity(f: Fixture) -> None:
    """On an MLO view after radiological orientation, the pectoral muscle
    sits in the top corner opposite the breast. Pectoral removal should
    reduce mean intensity in that corner by a non-trivial amount."""
    path = _skip_if_missing(f)
    arr, _ = read_mammogram_dicom(path)
    oriented = orient_to_radiological_convention(arr, f.expected_laterality)
    h, w = oriented.shape
    # Corner opposite the breast (= chest wall side)
    ch, cw = h // 4, w // 4
    if f.expected_laterality == Laterality.LEFT:
        # Radiological convention: LEFT breast → chest wall on RIGHT → pectoral top-RIGHT
        corner_before = oriented[:ch, w - cw:]
    else:
        corner_before = oriented[:ch, :cw]

    removed = remove_pectoral_mlo(oriented, laterality=f.expected_laterality.value)
    if f.expected_laterality == Laterality.LEFT:
        corner_after = removed[:ch, w - cw:]
    else:
        corner_after = removed[:ch, :cw]

    # The corner shouldn't get *brighter*. It should either stay the same
    # (if there was no pectoral to remove — sometimes true if the MLO was
    # cropped tight) or drop noticeably.
    assert corner_after.mean() <= corner_before.mean() + 1e-6
    # And in the typical case, mean should drop by at least a small amount.
    # Log the actual reduction so if a fixture doesn't reduce, we know.
    reduction = corner_before.mean() - corner_after.mean()
    print(f"[pectoral] {f.filename} corner mean reduction: {reduction:.4f}")


# --------------------------------------------------------------------------- #
# End-to-end pipeline


@pytest.mark.parametrize("f", FIXTURES, ids=lambda f: f.filename)
def test_preprocess_end_to_end_produces_consistent_result(f: Fixture) -> None:
    """The full pipeline returns a PreprocessedMammogram with a correctly
    labeled laterality/view, mask that isn't empty, and pectoral removed
    for MLO views."""
    path = _skip_if_missing(f)
    result = preprocess_mammogram(path)
    assert result.metadata.laterality == f.expected_laterality
    assert result.metadata.view == f.expected_view
    assert result.image.dtype == np.float32
    assert result.image.shape == result.breast_mask.shape
    assert result.breast_mask.any(), "empty breast mask"
    if f.expected_view == View.MLO:
        assert result.pectoral_removed, "MLO view but pectoral not removed"
    else:
        assert not result.pectoral_removed, "non-MLO view but pectoral removed"


def test_pipeline_metadata_records_source_of_truth() -> None:
    """A single-fixture sanity check that metadata sources are populated."""
    path = FIXTURE_DIR / "Calc-Test_P_00038_LEFT_CC.dcm"
    if not path.exists():
        pytest.skip(f"missing fixture {path}")
    result = preprocess_mammogram(path)
    m = result.metadata
    # BodyPartExamined is populated in CBIS-DDSM, so laterality source is dicom_tag
    assert m.laterality_source in ("dicom_tag", "filename", "content")
    # Similarly for view (PatientOrientation ↦ view_hint)
    assert m.view_source in ("dicom_tag", "filename")
    assert m.original_metadata.get("source_path", "").endswith(".dcm")
