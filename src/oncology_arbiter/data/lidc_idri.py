"""Real-data ingest for LIDC-IDRI CT series (worker-2 local cohort).

This module lets the arbiter enumerate and load CT series from a locally
mounted LIDC-IDRI cohort. Nothing here downloads the cohort — that has to
happen out-of-band (see ``idc-index`` docs). We just walk the on-disk tree
and hand back paths + a lightweight manifest.

Design constraints (mirror `cbis_ddsm.py`):

* Never fabricate DICOMs — every returned path points at a real file on
  the mounted cohort.
* Never bundle LIDC binaries into the repo — the full cohort is ~137 GB
  and its license (CC-BY-3.0) is compatible, but the size is not.
* Rebuild the manifest on demand by walking the directory tree so the
  loader still works when the pre-computed parquet mirror is missing.
* Filter for ``CT_<SeriesUID>`` subdirectories only — LIDC-IDRI stores
  segmentation (``SEG_*``), structured report (``SR_*``), digital
  radiograph (``DX_*``), and computed radiograph (``CR_*``) series next
  to each CT under the same StudyUID.

Public surface:
    LidcSeries                    — dataclass describing one CT series
    LidcCohortNotFound            — raised when the cohort root is missing
    dataset_provenance()          — the citations to embed in every
                                    response that uses LIDC data
    list_lidc_series(root, limit) — walk the tree, return series descriptors
    resolve_series_dir(patient_id, root) — first-CT-series-for-patient helper
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


# --------------------------------------------------------------------------- #
# Provenance & citation constants
#
# Every response using this dataset MUST embed these strings into its
# `provenance.dataset` field.  Verified against TCIA on 2026-07-04.

LIDC_COLLECTION_ID: str = "lidc_idri"
LIDC_TCIA_DOI: str = "10.7937/K9/TCIA.2015.LO9QL9SX"
LIDC_MEDICAL_PHYSICS_DOI: str = "10.1118/1.3528204"
LIDC_LICENSE: str = "CC-BY-3.0"
LIDC_N_PATIENTS: int = 1010
LIDC_N_IMAGES: int = 244_527

LIDC_CITATION: str = (
    "Armato SG III, McLennan G, Bidaut L, et al. (2015). "
    "Data from LIDC-IDRI. The Cancer Imaging Archive. "
    f"DOI: {LIDC_TCIA_DOI}. Publication: Armato SG III, et al. "
    "The Lung Image Database Consortium (LIDC) and Image Database "
    "Resource Initiative (IDRI): a completed reference database of "
    "lung nodules on CT scans. Med Phys 2011;38(2):915-931. "
    f"DOI: {LIDC_MEDICAL_PHYSICS_DOI}. "
    f"License: {LIDC_LICENSE}."
)


class LidcCohortNotFound(FileNotFoundError):
    """Raised when the configured LIDC cohort root does not exist."""


# --------------------------------------------------------------------------- #
# Data model


@dataclass(frozen=True)
class LidcSeries:
    """One LIDC-IDRI CT series descriptor."""

    patient_id: str            # "LIDC-IDRI-0001"
    study_uid: str             # DICOM StudyInstanceUID
    series_uid: str            # DICOM SeriesInstanceUID (unprefixed)
    series_dir: str            # absolute path to CT_<SeriesUID>/
    n_slices: int              # count of .dcm files in the dir

    def as_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "series_dir": self.series_dir,
            "n_slices": int(self.n_slices),
        }


# --------------------------------------------------------------------------- #
# Path resolution


def _cohort_root(root: Optional[str | Path]) -> Path:
    """Return the on-disk LIDC cohort root.

    Order of precedence:
      1. explicit `root` arg
      2. `$ONCOLOGY_ARBITER_LIDC_ROOT` env var
      3. default: `/workspace/lidc_cohort/lidc_idri`
    """
    if root is not None:
        p = Path(root)
    else:
        env = os.environ.get("ONCOLOGY_ARBITER_LIDC_ROOT", "")
        if env:
            p = Path(env)
        else:
            p = Path("/workspace/lidc_cohort/lidc_idri")
    if not p.exists():
        raise LidcCohortNotFound(
            f"LIDC cohort root does not exist: {p} "
            "(set ONCOLOGY_ARBITER_LIDC_ROOT or pass root explicitly)"
        )
    return p


def dataset_provenance() -> dict:
    """Return the provenance dict every LIDC-backed response embeds.

    The `citation` field is the human-readable reference. The DOI + license
    fields are the machine-readable ones a downstream aggregator can use to
    build a bibliography.
    """
    return {
        "collection_id": LIDC_COLLECTION_ID,
        "citation": LIDC_CITATION,
        "tcia_doi": LIDC_TCIA_DOI,
        "medical_physics_doi": LIDC_MEDICAL_PHYSICS_DOI,
        "license": LIDC_LICENSE,
        "n_patients_published": LIDC_N_PATIENTS,
        "n_images_published": LIDC_N_IMAGES,
    }


# --------------------------------------------------------------------------- #
# Walk


def _iter_ct_series_dirs(patient_dir: Path) -> Iterator[Path]:
    """Yield `CT_<uid>` subdirectories inside a patient directory.

    Skips `SEG_*`, `SR_*`, `DX_*`, `CR_*` siblings.
    """
    if not patient_dir.is_dir():
        return
    for study_dir in sorted(patient_dir.iterdir()):
        if not study_dir.is_dir():
            continue
        for series_dir in sorted(study_dir.iterdir()):
            if series_dir.is_dir() and series_dir.name.startswith("CT_"):
                yield series_dir


def _count_dicoms(series_dir: Path) -> int:
    n = 0
    for f in series_dir.iterdir():
        if f.is_file():
            n += 1
    return n


def list_lidc_series(
    root: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> list[LidcSeries]:
    """Enumerate CT series under the LIDC cohort root.

    Parameters
    ----------
    root : optional cohort root override (see `_cohort_root` for defaults)
    limit : if given, stop after `limit` series (useful for pilots)

    Returns
    -------
    list[LidcSeries]

    Raises
    ------
    LidcCohortNotFound if the cohort root does not exist.
    """
    base = _cohort_root(root)
    out: list[LidcSeries] = []
    for patient_dir in sorted(base.iterdir()):
        if not patient_dir.is_dir():
            continue
        if not patient_dir.name.startswith("LIDC-IDRI-"):
            continue
        for series_dir in _iter_ct_series_dirs(patient_dir):
            study_uid = series_dir.parent.name
            series_uid = series_dir.name[len("CT_"):]
            n_slices = _count_dicoms(series_dir)
            out.append(
                LidcSeries(
                    patient_id=patient_dir.name,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    series_dir=str(series_dir),
                    n_slices=n_slices,
                )
            )
            if limit is not None and len(out) >= int(limit):
                return out
    return out


def resolve_series_dir(
    patient_id: str,
    root: Optional[str | Path] = None,
) -> str:
    """Return the first CT series directory for `patient_id`.

    Parameters
    ----------
    patient_id : e.g. "LIDC-IDRI-0001"
    root : optional cohort root override

    Raises
    ------
    LidcCohortNotFound if the cohort root or patient dir is missing.
    FileNotFoundError if no CT_<uid> subdirectory is found for the patient.
    """
    base = _cohort_root(root)
    patient_dir = base / patient_id
    if not patient_dir.is_dir():
        raise LidcCohortNotFound(
            f"patient directory not found: {patient_dir}"
        )
    for series_dir in _iter_ct_series_dirs(patient_dir):
        return str(series_dir)
    raise FileNotFoundError(
        f"no CT_<uid> series subdirectory found for {patient_id} under {patient_dir}"
    )


__all__ = [
    "LidcCohortNotFound",
    "LidcSeries",
    "LIDC_CITATION",
    "LIDC_COLLECTION_ID",
    "LIDC_TCIA_DOI",
    "LIDC_MEDICAL_PHYSICS_DOI",
    "LIDC_LICENSE",
    "LIDC_N_PATIENTS",
    "LIDC_N_IMAGES",
    "dataset_provenance",
    "list_lidc_series",
    "resolve_series_dir",
]
