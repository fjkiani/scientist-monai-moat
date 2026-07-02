"""Tests for oncology_arbiter.data.cbis_ddsm

Real-data properties MUST NOT be fabricated. This suite verifies:
  * Filename parser handles the CBIS `<Class>-<Split>_P_<PID>_<LAT>_<VIEW>` schema
  * `list_available_cases()` enumerates the 5 committed fixtures
  * `resolve_local_dicom_path` fails fast if the fixture is missing and
    `allow_download=False`, and invokes hf_hub_download with the correct
    repo/subfolder/filename when `allow_download=True`
  * `load_cbis_case` returns the exact tag values from the DICOM
  * `dataset_provenance` carries the CBIS-DDSM DOI verbatim
  * The five fixtures pass sanity: MG modality, MONOCHROME2, 16-bit, non-zero rows/cols

All positive-path tests use the REAL DICOMs on disk. No image data is
synthesised or mocked. If the fixtures are absent, real-data tests skip
rather than pass by trickery.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from oncology_arbiter.data.cbis_ddsm import (
    CBIS_DATASET_DOI,
    CBIS_HF_MIRROR_LICENSE,
    CBIS_HF_REPO_ID,
    CBIS_HF_REPO_TYPE,
    CBIS_LICENSE,
    CbisCase,
    dataset_provenance,
    default_fixture_dir,
    list_available_cases,
    load_cbis_case,
    parse_stem,
    resolve_local_dicom_path,
)


# --------------------------------------------------------------------------- #
# Fixture wiring

FIXTURE_DIR = default_fixture_dir()

# Ground-truth metadata pulled from tests/fixtures/cbis_ddsm/_dicom_metadata.json
EXPECTED_METADATA: dict[str, dict[str, object]] = {
    "Calc-Test_P_00038_LEFT_CC":  {"rows": 4616, "cols": 3016, "body": "Left Breast",  "orient": "CC"},
    "Calc-Test_P_00038_RIGHT_CC": {"rows": 4688, "cols": 2744, "body": "Right Breast", "orient": "CC"},
    "Calc-Test_P_00038_LEFT_MLO": {"rows": 4728, "cols": 3064, "body": "Left Breast",  "orient": "MLO"},
    "Calc-Test_P_00038_RIGHT_MLO": {"rows": 4720, "cols": 2928, "body": "Right Breast", "orient": "MLO"},
    "Mass-Test_P_00016_LEFT_CC":  {"rows": 4006, "cols": 1846, "body": "BREAST",       "orient": "CC"},
}


def _fixtures_present() -> bool:
    if not FIXTURE_DIR.is_dir():
        return False
    return all((FIXTURE_DIR / f"{stem}.dcm").is_file() for stem in EXPECTED_METADATA)


needs_fixtures = pytest.mark.skipif(
    not _fixtures_present(),
    reason="CBIS-DDSM DICOM fixtures not on disk; run download_cbis_ddsm_fixtures.py",
)


# --------------------------------------------------------------------------- #
# Provenance constants


def test_cbis_dataset_doi_verbatim() -> None:
    # DOI from tests/fixtures/cbis_ddsm/README.md and prior IRB anchors.
    assert CBIS_DATASET_DOI == "10.7937/K9/TCIA.2016.7O02S9CY"


def test_cbis_licenses_recorded() -> None:
    assert CBIS_LICENSE == "CC-BY-3.0"
    assert CBIS_HF_MIRROR_LICENSE == "CC-BY-NC-4.0"


def test_hf_repo_id_and_type() -> None:
    assert CBIS_HF_REPO_ID == "helloerikaaa/cbis-ddsm-r"
    assert CBIS_HF_REPO_TYPE == "dataset"


def test_dataset_provenance_dict_contains_doi_and_repo() -> None:
    prov = dataset_provenance()
    assert prov["dataset"] == "CBIS-DDSM"
    assert prov["dataset_doi"] == "10.7937/K9/TCIA.2016.7O02S9CY"
    assert prov["dataset_license"] == "CC-BY-3.0"
    assert prov["hf_repo_id"] == "helloerikaaa/cbis-ddsm-r"
    assert prov["hf_repo_type"] == "dataset"
    assert prov["hf_mirror_license"] == "CC-BY-NC-4.0"


# --------------------------------------------------------------------------- #
# Filename parsing


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("Calc-Test_P_00038_LEFT_CC", {
            "lesion_class": "Calc", "split": "Test", "patient_id": "P_00038",
            "laterality": "LEFT", "view": "CC",
        }),
        ("Mass-Test_P_00016_LEFT_CC", {
            "lesion_class": "Mass", "split": "Test", "patient_id": "P_00016",
            "laterality": "LEFT", "view": "CC",
        }),
        ("Calc-Training_P_12345_RIGHT_MLO", {
            "lesion_class": "Calc", "split": "Training", "patient_id": "P_12345",
            "laterality": "RIGHT", "view": "MLO",
        }),
    ],
)
def test_parse_stem_valid(stem: str, expected: dict) -> None:
    assert parse_stem(stem) == expected


@pytest.mark.parametrize(
    "bad_stem",
    [
        "",
        "not-a-cbis-file",
        "Calc-Test_P_XX_LEFT_CC",           # non-digit patient id
        "Cyst-Test_P_00038_LEFT_CC",        # unknown lesion class
        "Calc-Training_P_00038_LEFT_XYZ",   # unknown view
        "Calc-Test_P_00038_MIDDLE_CC",      # unknown laterality
        "Calc-Test_P_00038_LEFT_CC.dcm",    # extension included
    ],
)
def test_parse_stem_rejects_malformed(bad_stem: str) -> None:
    with pytest.raises(ValueError):
        parse_stem(bad_stem)


# --------------------------------------------------------------------------- #
# Enumeration


@needs_fixtures
def test_list_available_cases_returns_all_five() -> None:
    got = list_available_cases()
    assert set(got) == set(EXPECTED_METADATA)
    assert len(got) == 5


def test_list_available_cases_custom_dir_empty(tmp_path: Path) -> None:
    assert list_available_cases(tmp_path) == []


def test_default_fixture_dir_points_at_tests_fixtures() -> None:
    d = default_fixture_dir()
    assert d.name == "cbis_ddsm"
    assert d.parent.name == "fixtures"
    assert d.parent.parent.name == "tests"


# --------------------------------------------------------------------------- #
# resolve_local_dicom_path


@needs_fixtures
def test_resolve_local_path_returns_existing_file() -> None:
    p = resolve_local_dicom_path("Calc-Test_P_00038_LEFT_CC")
    assert os.path.isfile(p)
    assert p.endswith("Calc-Test_P_00038_LEFT_CC.dcm")


def test_resolve_local_path_missing_without_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_local_dicom_path(
            "Calc-Test_P_00038_LEFT_CC",
            fixture_dir=tmp_path,
            allow_download=False,
        )


def test_resolve_local_path_invalid_stem_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_local_dicom_path("not-a-cbis-file", fixture_dir=tmp_path)


def test_resolve_local_path_downloads_via_injected_hf(tmp_path: Path) -> None:
    """When allow_download=True and file missing, hf_hub_download is called
    with the right repo/subfolder/filename."""
    called: dict[str, object] = {}

    def fake_hf(*, repo_id, repo_type, filename, local_dir):
        called["repo_id"] = repo_id
        called["repo_type"] = repo_type
        called["filename"] = filename
        called["local_dir"] = local_dir
        # Simulate a successful download by writing a tiny file.
        # The real hf_hub_download returns the on-disk path; we mirror that
        # behaviour but do NOT try to make it a valid DICOM — this test
        # only verifies the download call arguments.
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"DICM-stub-not-a-real-dicom")
        return str(out)

    p = resolve_local_dicom_path(
        "Calc-Test_P_00038_LEFT_CC",
        fixture_dir=tmp_path,
        allow_download=True,
        hf_hub_download=fake_hf,
    )
    assert called["repo_id"] == "helloerikaaa/cbis-ddsm-r"
    assert called["repo_type"] == "dataset"
    assert called["filename"] == "img/Calc-Test_P_00038_LEFT_CC.dcm"
    assert Path(called["local_dir"]).resolve() == tmp_path.resolve()
    assert os.path.isfile(p)


# --------------------------------------------------------------------------- #
# load_cbis_case on REAL DICOMs


@needs_fixtures
@pytest.mark.parametrize("stem", list(EXPECTED_METADATA))
def test_load_cbis_case_pixel_metadata_matches_readme(stem: str) -> None:
    case = load_cbis_case(stem)
    exp = EXPECTED_METADATA[stem]
    assert isinstance(case, CbisCase)
    assert case.stem == stem
    assert case.rows == exp["rows"]
    assert case.columns == exp["cols"]
    assert case.bits_stored == 16
    assert case.photometric_interp == "MONOCHROME2"
    assert case.modality == "MG"
    assert case.body_part_examined == exp["body"]
    assert case.patient_orientation == exp["orient"]


@needs_fixtures
def test_load_cbis_case_laterality_and_view_come_from_filename() -> None:
    case = load_cbis_case("Calc-Test_P_00038_RIGHT_MLO")
    assert case.laterality_filename == "RIGHT"
    assert case.view_filename == "MLO"
    assert case.lesion_class == "Calc"
    assert case.patient_id == "P_00038"
    # HF mirror leaves ImageLaterality and ViewPosition empty across all fixtures.
    assert case.laterality_tag == ""
    assert case.view_tag == ""


@needs_fixtures
def test_load_cbis_case_records_dicom_path_that_exists() -> None:
    case = load_cbis_case("Mass-Test_P_00016_LEFT_CC")
    assert os.path.isfile(case.dicom_path)


@needs_fixtures
def test_load_cbis_case_mass_fixture_has_bare_breast_tag() -> None:
    """The Mass- fixture ships with BodyPartExamined="BREAST" (no laterality).
    This asymmetry is a real-data property that downstream code MUST NOT
    silently coerce. See tests/fixtures/cbis_ddsm/README.md."""
    case = load_cbis_case("Mass-Test_P_00016_LEFT_CC")
    assert case.body_part_examined == "BREAST"


@needs_fixtures
def test_metadata_json_matches_load_cbis_case() -> None:
    """Cross-check: the pinned _dicom_metadata.json inventory MUST agree with
    what pydicom actually reports for the five fixtures."""
    meta_path = FIXTURE_DIR / "_dicom_metadata.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text())
    for stem in EXPECTED_METADATA:
        got = load_cbis_case(stem)
        m = meta[stem]
        assert got.rows == m["Rows"]
        assert got.columns == m["Columns"]
        assert got.bits_stored == m["BitsStored"]
        assert got.photometric_interp == m["PhotometricInterpretation"]
        assert got.modality == m["Modality"]
        assert got.body_part_examined == m["BodyPartExamined"]
        assert got.patient_orientation == m["PatientOrientation"]


# --------------------------------------------------------------------------- #
# Honesty: no synthesis, no substitution


@needs_fixtures
def test_cases_are_real_dicom_files_on_disk() -> None:
    """No fixture may be smaller than 500 KB — real 16-bit mammograms at
    the sizes claimed in README are multiple MB. If any fixture is small,
    something was substituted for a stub."""
    for stem in EXPECTED_METADATA:
        p = FIXTURE_DIR / f"{stem}.dcm"
        assert p.is_file()
        assert p.stat().st_size > 500_000, (
            f"{p} is {p.stat().st_size} bytes — real CBIS DICOMs are multi-MB. "
            "Fixture may have been replaced with a stub."
        )


def test_module_does_not_hardcode_dicom_bytes() -> None:
    """Ingest module MUST NOT bake a DICOM into source (would look like
    a base64 blob or `b'\\x44\\x49\\x43\\x4d'` sequence outside a comment)."""
    import inspect
    from oncology_arbiter.data import cbis_ddsm as mod
    src = inspect.getsource(mod)
    # Non-comment DICM byte magic would be a red flag. Comments are fine —
    # this test lets docstring mentions through by looking for suspicious
    # literal bytestring assignments only.
    assert "b\"\\x44\\x49\\x43\\x4d\"" not in src
    assert "b'\\x44\\x49\\x43\\x4d'" not in src
    # No base64-ish long literal
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith("#") or s.startswith('"""') or s.startswith("'''"):
            continue
        # An accidental embedded blob would be a huge string literal on one line.
        assert len(s) < 2000, f"suspicious long literal line: {ln[:80]}..."
