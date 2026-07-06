"""Server-side demo case fixture.

v0.2.2 adds ``GET /v1/demo/case`` — a fully-formed sample case (DICOM +
pathology report + patient context) so a user landing on the SPA can
click "Load demo case" and see the whole workflow run end-to-end
without hunting for a DICOM.

The DICOM is the smallest of the 5 CBIS-DDSM fixtures we already ship
(Mass-Test_P_00016_LEFT_CC.dcm, 14.1 MB). It is downloaded once on
first request from ``helloerikaaa/cbis-ddsm-r`` (HuggingFace,
CC-BY-NC 4.0, no auth) and cached to ``/tmp/oa-demo/demo.dcm`` for the
lifetime of the container.

The pathology text is the same luminal-A canned example the frontend
already carries (``LUMINAL_A_EXAMPLE`` in ``CaseViewTab.tsx``,
``BiopsyTab.tsx``); we keep the strings in lockstep so the demo runs
the same path a manual click would.

Endpoint contract: see ``GET /v1/demo/case`` handler in ``app.py``.
"""
from __future__ import annotations

import base64
import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Public constants — kept in lockstep with the frontend LUMINAL_A_EXAMPLE.

DEMO_REPORT_TEXT = (
    "Age: 58, postmenopausal\n"
    "Stage: T1N0M0\n"
    "Pathology:\n"
    "  Invasive ductal carcinoma of the right breast, 1.4 cm.\n"
    "  Estrogen Receptor: Positive (95%).\n"
    "  Progesterone Receptor: Positive (80%).\n"
    "  HER2/neu: Negative (IHC 1+).\n"
    "  Nottingham Grade: 2.\n"
    "  Ki-67 index: 12%."
)

DEMO_PATIENT_CONTEXT: dict[str, Any] = {
    "age": 58,
    "menopausal_status": "post",
    "stage_ct": "T1N0M0",
    "grade": 2,
    "notes": (
        "Demo case only. CBIS-DDSM public mammogram fixture stitched to a "
        "hand-written luminal-A pathology report. Not a real patient."
    ),
}

DEMO_WARNINGS: list[str] = [
    "This is a demonstration case for research use only.",
    "The DICOM is a public CBIS-DDSM training image (CC-BY-NC 4.0); "
    "the pathology text is synthetic.",
    "Results shown do NOT constitute medical advice.",
]

DEMO_DICOM_SOURCE = (
    "CBIS-DDSM (helloerikaaa/cbis-ddsm-r on HuggingFace, CC-BY-NC 4.0): "
    "Mass-Test_P_00016_LEFT_CC.dcm"
)

# --------------------------------------------------------------------------- #
# HF fetch config.

_HF_REPO_ID = "helloerikaaa/cbis-ddsm-r"
_HF_REPO_TYPE = "dataset"
# Real path on HF as of v0.2.2 (verified via HfApi.list_repo_files 2026-07-06).
# The Study/Series UIDs are CBIS-DDSM's — they are stable identifiers in the
# public dataset, not something we chose.
_HF_FILE_PATH = (
    "img/Mass-Test_P_00016_LEFT_CC/"
    "1.3.6.1.4.1.9590.100.1.2.416403281812750683720028031170500130104/"
    "1.3.6.1.4.1.9590.100.1.2.245063149211255120613007755642780114172/"
    "00000001.dcm"
)

# Where the cached DICOM lives on the running container. Overridable for tests.
_DEFAULT_CACHE_DIR = Path("/tmp/oa-demo")

# A repo-local fallback for offline runs (dev laptop, CI without network).
# If the CBIS fixture already exists in tests/fixtures/cbis_ddsm/ we prefer
# that copy over a network fetch.
_REPO_LOCAL_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "cbis_ddsm"
    / "Mass-Test_P_00016_LEFT_CC.dcm"
)

# The container is best served if the DICOM already exists; single-thread the
# first-request download so we don't kick off two HF fetches under load.
_download_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Errors + result shape.


class DemoFixtureUnavailable(RuntimeError):
    """Raised when the demo DICOM cannot be provided (no local copy AND
    no network access AND no HF cache). Maps to HTTP 503 upstream."""


@dataclass(frozen=True)
class DemoCase:
    dicom_bytes_b64: str
    dicom_source: str
    dicom_sha256: str
    dicom_size_bytes: int
    report_text: str
    patient_context: dict[str, Any]
    warnings: list[str]


# --------------------------------------------------------------------------- #
# Cache resolution.


def _resolve_cache_dir() -> Path:
    """Where to stage the demo DICOM. Overridable via env for tests."""
    override = os.environ.get("ONCOLOGY_ARBITER_DEMO_CACHE_DIR", "").strip()
    return Path(override) if override else _DEFAULT_CACHE_DIR


def _cached_dicom_path() -> Path:
    return _resolve_cache_dir() / "demo.dcm"


def _resolve_local_fixture() -> Path:
    """Repo-local test fixture, if present. Preferred over network fetch."""
    override = os.environ.get(
        "ONCOLOGY_ARBITER_DEMO_LOCAL_DICOM", ""
    ).strip()
    return Path(override) if override else _REPO_LOCAL_FIXTURE


# --------------------------------------------------------------------------- #
# DICOM sourcing.


def _download_from_hf(dest: Path) -> Path:
    """Download the fixture from HuggingFace to ``dest``. Uses hf_hub_download
    under the hood so its cache is reused across calls."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise DemoFixtureUnavailable(
            "huggingface_hub not installed; cannot download demo DICOM"
        ) from e

    # cache_dir is set to our own writable location so containers where
    # $HOME is unwritable (Render: HOME=/app, owned by root; process runs
    # as UID 10001 without write access) don't fail on the default
    # ~/.cache/huggingface directory. This is the same directory where we
    # stage the final demo.dcm copy — HF's cache lives at
    # <cache_dir>/<hub-style-blob-tree>/, our copy lives at
    # <cache_dir>/demo.dcm.
    hf_cache_dir = _resolve_cache_dir()
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        hf_path = hf_hub_download(
            repo_id=_HF_REPO_ID,
            repo_type=_HF_REPO_TYPE,
            filename=_HF_FILE_PATH,
            cache_dir=str(hf_cache_dir),
        )
    except Exception as e:  # ConnectionError, HTTPError, GatedRepoError, ...
        raise DemoFixtureUnavailable(
            f"HuggingFace download failed: {type(e).__name__}: {e}"
        ) from e

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Copy from HF cache into our stable location so subsequent requests can
    # skip the whole hf_hub_download machinery.
    src_bytes = Path(hf_path).read_bytes()
    dest.write_bytes(src_bytes)
    return dest


def _ensure_dicom_on_disk() -> Path:
    """Return a path to the demo DICOM, sourcing it in this precedence:

    1. ``_cached_dicom_path()`` — already staged in a previous request.
    2. ``_resolve_local_fixture()`` — repo-local fixture (dev/CI).
    3. HuggingFace download to ``_cached_dicom_path()``.

    Only one download runs at a time (``_download_lock``); concurrent
    callers wait for the first to finish and reuse its result.
    """
    cached = _cached_dicom_path()
    if cached.is_file() and cached.stat().st_size > 0:
        return cached

    with _download_lock:
        # Re-check under the lock — another thread may have populated it.
        if cached.is_file() and cached.stat().st_size > 0:
            return cached

        local = _resolve_local_fixture()
        if local.is_file() and local.stat().st_size > 0:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(local.read_bytes())
            return cached

        return _download_from_hf(cached)


# --------------------------------------------------------------------------- #
# Public API.


def build_demo_case() -> DemoCase:
    """Return a fully-formed DemoCase ready to be serialized by the API layer.

    Raises DemoFixtureUnavailable if the DICOM can't be produced.
    """
    dcm_path = _ensure_dicom_on_disk()
    dcm_bytes = dcm_path.read_bytes()
    if not dcm_bytes:
        raise DemoFixtureUnavailable(
            f"demo DICOM at {dcm_path} is empty"
        )
    sha256 = hashlib.sha256(dcm_bytes).hexdigest()
    b64 = base64.b64encode(dcm_bytes).decode("ascii")
    return DemoCase(
        dicom_bytes_b64=b64,
        dicom_source=DEMO_DICOM_SOURCE,
        dicom_sha256=sha256,
        dicom_size_bytes=len(dcm_bytes),
        report_text=DEMO_REPORT_TEXT,
        patient_context=dict(DEMO_PATIENT_CONTEXT),
        warnings=list(DEMO_WARNINGS),
    )


def prewarm_demo_case() -> Path | None:
    """Attempt to fetch the DICOM at startup so the first user request is fast.

    Silently returns None on any failure — startup must never crash because
    HF is offline. The endpoint will retry (and possibly still succeed) at
    request time.
    """
    try:
        return _ensure_dicom_on_disk()
    except Exception:
        return None
