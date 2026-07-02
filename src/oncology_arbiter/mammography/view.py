"""View detection: CC vs MLO."""
from __future__ import annotations

from enum import Enum
from pathlib import Path


class View(str, Enum):
    CC = "CC"
    MLO = "MLO"
    UNKNOWN = "U"


def detect_view_from_metadata(
    metadata: dict,
    filename: str | Path | None = None,
) -> View:
    """Return view from DICOM tags or filename hints; UNKNOWN otherwise.

    CBIS-DDSM filenames encode the view as `..._CC` or `..._MLO` at the end
    of the study name.
    """
    v = metadata.get("view_hint")
    if v:
        u = v.upper()
        if u == "CC":
            return View.CC
        if u == "MLO":
            return View.MLO
    if filename is not None:
        stem = Path(filename).stem.upper()
        if stem.endswith("_CC") or "_CC." in stem or "_CC_" in stem:
            return View.CC
        if stem.endswith("_MLO") or "_MLO." in stem or "_MLO_" in stem:
            return View.MLO
    return View.UNKNOWN
