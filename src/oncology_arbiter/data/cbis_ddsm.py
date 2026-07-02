"""Real-data ingest for CBIS-DDSM DICOM screening mammograms.

This module supplies the arbiter with a small, deterministic pool of
real DICOM screening mammograms drawn from the public
`helloerikaaa/cbis-ddsm-r` mirror of CBIS-DDSM on Hugging Face
(dataset DOI **10.7937/K9/TCIA.2016.7O02S9CY**, license **CC-BY 3.0**
for CBIS-DDSM itself; the HF mirror declares **CC-BY-NC 4.0**).

Design constraints:

* Never fabricate or synthesize DICOMs — every case returned by this
  module is a real 16-bit MONOCHROME2 mammogram pulled from Hugging Face.
* Never assume DICOM tags are populated — the HF mirror ships heterogeneous
  metadata (some fixtures have `BodyPartExamined="Left Breast"`, others
  have bare `"BREAST"`; `ImageLaterality` and `ViewPosition` are empty on
  the whole mirror). Filename hints are the fallback and are documented in
  `tests/fixtures/cbis_ddsm/README.md`.
* Never require Hugging Face authentication for CBIS ingest itself — the
  `helloerikaaa/cbis-ddsm-r` mirror is public. HAI-DEF gating applies only
  to Google model weights (see `oncology_arbiter.models.hai_def`).
* Never bundle DICOM binaries into the git repo — they are large and their
  license (CC-BY-NC) is incompatible with the repo's own license posture.
  All ingest is via `huggingface_hub.hf_hub_download` into a local cache
  directory that the caller controls.

Public entry points:

  * `CbisCase`             — dataclass describing one loaded case
  * `list_available_cases()` — enumerate the five committed fixture stems
  * `resolve_local_dicom_path()` — return the path to a fixture DICOM,
                                   downloading it via HF if missing
  * `load_cbis_case()`     — build a `CbisCase` from a local DICOM

Every function preserves the RUO honesty invariants: values are read
verbatim from the DICOM, laterality/view come from filename+tag fusion
with the tag-only value recorded so a downstream consumer can see the
disagreement.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Fixture inventory
#
# Verbatim from `tests/fixtures/cbis_ddsm/README.md`. If the fixture list on
# disk drifts, `list_available_cases()` still enumerates from the filesystem;
# these constants are the SOURCE OF TRUTH for downloads.
CBIS_HF_REPO_ID: str = "helloerikaaa/cbis-ddsm-r"
CBIS_HF_REPO_TYPE: str = "dataset"
CBIS_DATASET_DOI: str = "10.7937/K9/TCIA.2016.7O02S9CY"
CBIS_LICENSE: str = "CC-BY-3.0"
CBIS_HF_MIRROR_LICENSE: str = "CC-BY-NC-4.0"

# Filename stems in the committed fixture set (order = enumeration order).
# Filename format: `<Class>-<Split>_P_<PatientID>_<LATERALITY>_<VIEW>.dcm`
_FIXTURE_STEMS: tuple[str, ...] = (
    "Calc-Test_P_00038_LEFT_CC",
    "Calc-Test_P_00038_LEFT_MLO",
    "Calc-Test_P_00038_RIGHT_CC",
    "Calc-Test_P_00038_RIGHT_MLO",
    "Mass-Test_P_00016_LEFT_CC",
)


# --------------------------------------------------------------------------- #
# Data model


@dataclass(frozen=True)
class CbisCase:
    """One CBIS-DDSM DICOM loaded into memory (metadata only by default)."""

    stem: str                 # "Calc-Test_P_00038_LEFT_CC"
    dicom_path: str           # absolute path on disk
    lesion_class: str         # "Calc" | "Mass"   (from filename)
    split: str                # "Test"           (from filename)
    patient_id: str           # "P_00038"        (from filename)
    laterality_filename: str  # "LEFT" | "RIGHT" (from filename)
    view_filename: str        # "CC"  | "MLO"    (from filename)
    body_part_examined: str   # DICOM tag verbatim, may be "" or "BREAST" etc.
    rows: int                 # DICOM tag Rows
    columns: int              # DICOM tag Columns
    bits_stored: int          # DICOM tag BitsStored (16 across this mirror)
    photometric_interp: str   # "MONOCHROME2"
    modality: str             # "MG"
    #
    # Optional / advisory
    #
    laterality_tag: str = ""  # DICOM ImageLaterality tag (may be "" on this mirror)
    view_tag: str = ""        # DICOM ViewPosition tag  (may be "" on this mirror)
    patient_orientation: str = ""  # DICOM PatientOrientation tag ("CC"/"MLO" here)


# --------------------------------------------------------------------------- #
# Filename parsing


_FILENAME_RE = re.compile(
    r"^(?P<cls>Calc|Mass)-(?P<split>Test|Training)_P_(?P<pid>\d+)_"
    r"(?P<lat>LEFT|RIGHT)_(?P<view>CC|MLO)$"
)


def parse_stem(stem: str) -> dict[str, str]:
    """Split a fixture stem into components.

    >>> parse_stem("Calc-Test_P_00038_LEFT_CC")["laterality"]
    'LEFT'
    """
    m = _FILENAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"unrecognized CBIS fixture stem {stem!r}; expected "
            "'<Calc|Mass>-<Test|Training>_P_<digits>_<LEFT|RIGHT>_<CC|MLO>'"
        )
    return {
        "lesion_class": m.group("cls"),
        "split": m.group("split"),
        "patient_id": f"P_{m.group('pid')}",
        "laterality": m.group("lat"),
        "view": m.group("view"),
    }


def list_available_cases(fixture_dir: str | os.PathLike[str] | None = None) -> list[str]:
    """Return the fixture stems that are currently on disk.

    If `fixture_dir` is None, the canonical committed fixture directory
    `tests/fixtures/cbis_ddsm` (relative to the repo root) is used.

    Only files that both exist AND match the filename schema are returned;
    stems in `_FIXTURE_STEMS` that are not present on disk are silently
    skipped so callers can distinguish "fully synced" (=5) from "partial
    checkout" (<5) without a crash.
    """
    if fixture_dir is None:
        fixture_dir = default_fixture_dir()
    root = Path(fixture_dir)
    out: list[str] = []
    for stem in _FIXTURE_STEMS:
        if (root / f"{stem}.dcm").is_file():
            out.append(stem)
    return out


def default_fixture_dir() -> Path:
    """Return the canonical fixture directory (../tests/fixtures/cbis_ddsm)."""
    # This file lives at src/oncology_arbiter/data/cbis_ddsm.py
    # → repo root is three parents up.
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return repo_root / "tests" / "fixtures" / "cbis_ddsm"


# --------------------------------------------------------------------------- #
# Hugging Face download


def resolve_local_dicom_path(
    stem: str,
    *,
    fixture_dir: str | os.PathLike[str] | None = None,
    allow_download: bool = False,
    hf_hub_download: Any = None,
) -> str:
    """Return the absolute path to `<stem>.dcm`, optionally downloading it.

    Parameters
    ----------
    stem:
        Fixture stem, e.g. `"Calc-Test_P_00038_LEFT_CC"`. Must match the
        CBIS filename schema (validated).
    fixture_dir:
        Directory to look in (default: repo tests/fixtures/cbis_ddsm).
    allow_download:
        If True and the file is missing, pull it from Hugging Face
        (`helloerikaaa/cbis-ddsm-r`, subfolder `img`). If False and the
        file is missing, `FileNotFoundError` is raised.
    hf_hub_download:
        Optional callable overriding `huggingface_hub.hf_hub_download`
        (used in tests). Signature must be compatible with the real one.

    Returns
    -------
    str  — absolute path to the DICOM on the local filesystem.
    """
    # Validate stem shape before touching disk.
    parse_stem(stem)
    if fixture_dir is None:
        fixture_dir = default_fixture_dir()
    target = Path(fixture_dir) / f"{stem}.dcm"
    if target.is_file():
        return str(target.resolve())
    if not allow_download:
        raise FileNotFoundError(
            f"CBIS fixture {stem!r} not on disk at {target}. "
            "Re-run `python tests/fixtures/download_cbis_ddsm_fixtures.py` "
            "or call resolve_local_dicom_path(..., allow_download=True)."
        )
    # Lazy import so this module is importable without huggingface_hub.
    if hf_hub_download is None:
        from huggingface_hub import hf_hub_download as _hf  # type: ignore
        hf_hub_download = _hf
    # The HF mirror stores DICOMs under `img/<stem>.dcm`.
    downloaded = hf_hub_download(
        repo_id=CBIS_HF_REPO_ID,
        repo_type=CBIS_HF_REPO_TYPE,
        filename=f"img/{stem}.dcm",
        local_dir=str(Path(fixture_dir).resolve()),
    )
    return str(Path(downloaded).resolve())


# --------------------------------------------------------------------------- #
# DICOM loading


def load_cbis_case(
    stem: str,
    *,
    fixture_dir: str | os.PathLike[str] | None = None,
    pydicom_module: Any = None,
) -> CbisCase:
    """Read `<stem>.dcm` from disk and return a metadata-only CbisCase.

    Pixel data is intentionally NOT decoded here — that is preprocessing's
    job. This function is cheap enough to call inside a listing loop.

    Parameters
    ----------
    stem: fixture stem (validated via parse_stem)
    fixture_dir: directory containing the DICOM
    pydicom_module: injected `pydicom` for testing
    """
    path = resolve_local_dicom_path(stem, fixture_dir=fixture_dir)
    parsed = parse_stem(stem)
    if pydicom_module is None:
        import pydicom  # type: ignore
        pydicom_module = pydicom
    ds = pydicom_module.dcmread(path, stop_before_pixels=True, force=True)
    body_part = str(getattr(ds, "BodyPartExamined", "") or "")
    return CbisCase(
        stem=stem,
        dicom_path=path,
        lesion_class=parsed["lesion_class"],
        split=parsed["split"],
        patient_id=parsed["patient_id"],
        laterality_filename=parsed["laterality"],
        view_filename=parsed["view"],
        body_part_examined=body_part,
        rows=int(getattr(ds, "Rows", 0) or 0),
        columns=int(getattr(ds, "Columns", 0) or 0),
        bits_stored=int(getattr(ds, "BitsStored", 0) or 0),
        photometric_interp=str(getattr(ds, "PhotometricInterpretation", "") or ""),
        modality=str(getattr(ds, "Modality", "") or ""),
        laterality_tag=str(getattr(ds, "ImageLaterality", "") or ""),
        view_tag=str(getattr(ds, "ViewPosition", "") or ""),
        patient_orientation=str(getattr(ds, "PatientOrientation", "") or ""),
    )


def dataset_provenance() -> dict[str, str]:
    """Return provenance metadata for the CBIS ingest pipeline.

    Copied verbatim into every API response envelope that touches CBIS.
    """
    return {
        "dataset": "CBIS-DDSM",
        "dataset_doi": CBIS_DATASET_DOI,
        "dataset_license": CBIS_LICENSE,
        "hf_repo_id": CBIS_HF_REPO_ID,
        "hf_repo_type": CBIS_HF_REPO_TYPE,
        "hf_mirror_license": CBIS_HF_MIRROR_LICENSE,
    }


__all__ = [
    "CBIS_HF_REPO_ID",
    "CBIS_HF_REPO_TYPE",
    "CBIS_DATASET_DOI",
    "CBIS_LICENSE",
    "CBIS_HF_MIRROR_LICENSE",
    "CbisCase",
    "parse_stem",
    "list_available_cases",
    "default_fixture_dir",
    "resolve_local_dicom_path",
    "load_cbis_case",
    "dataset_provenance",
]
