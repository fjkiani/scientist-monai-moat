"""v0.2.2: demo_fixtures.build_demo_case() sources the CBIS-DDSM DICOM,
prefers a repo-local copy over HuggingFace, and packages it with the canon
luminal-A pathology text."""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from oncology_arbiter.api import demo_fixtures
from oncology_arbiter.api.demo_fixtures import (
    DEMO_DICOM_SOURCE,
    DEMO_PATIENT_CONTEXT,
    DEMO_REPORT_TEXT,
    DEMO_WARNINGS,
    DemoFixtureUnavailable,
    build_demo_case,
    prewarm_demo_case,
)


@pytest.fixture
def tmp_demo_cache(tmp_path, monkeypatch):
    """Point the demo cache at a tmp dir so tests don't touch /tmp/oa-demo."""
    cache = tmp_path / "demo-cache"
    monkeypatch.setenv("ONCOLOGY_ARBITER_DEMO_CACHE_DIR", str(cache))
    monkeypatch.setenv("ONCOLOGY_ARBITER_DEMO_LOCAL_DICOM", "")
    yield cache


@pytest.fixture
def fake_local_fixture(tmp_path, monkeypatch):
    """Provide a small fake DICOM as the local repo fixture."""
    local = tmp_path / "local.dcm"
    local.write_bytes(b"\x00\x00DICM_fake_v0.2.2" + b"\xa5" * 1024)
    monkeypatch.setenv("ONCOLOGY_ARBITER_DEMO_LOCAL_DICOM", str(local))
    return local


def test_build_demo_case_prefers_local_fixture_over_hf(
    tmp_demo_cache, fake_local_fixture,
):
    """When a local file is present, we must not call HuggingFace at all."""
    with patch(
        "oncology_arbiter.api.demo_fixtures._download_from_hf",
        side_effect=AssertionError("must not download from HF when local exists"),
    ) as p:
        case = build_demo_case()
    p.assert_not_called()

    # Round-trip check
    dicom_bytes = base64.b64decode(case.dicom_bytes_b64)
    assert dicom_bytes == fake_local_fixture.read_bytes()
    assert case.dicom_sha256 == hashlib.sha256(dicom_bytes).hexdigest()
    assert case.dicom_size_bytes == len(dicom_bytes)


def test_build_demo_case_falls_back_to_hf_when_no_local(tmp_demo_cache, monkeypatch):
    """No local fixture → we must call _download_from_hf and use its output."""
    payload = b"\x00\x00DICM_hf_download" + b"\xf1" * 2048

    def fake_download(dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return dest

    monkeypatch.setenv(
        "ONCOLOGY_ARBITER_DEMO_LOCAL_DICOM", "/nonexistent/path/that/does/not/exist.dcm"
    )
    with patch(
        "oncology_arbiter.api.demo_fixtures._download_from_hf",
        side_effect=fake_download,
    ) as p:
        case = build_demo_case()
    p.assert_called_once()

    assert base64.b64decode(case.dicom_bytes_b64) == payload


def test_build_demo_case_uses_cache_on_second_call(
    tmp_demo_cache, fake_local_fixture,
):
    """Second call must not re-read local file (cache is populated)."""
    build_demo_case()  # populates cache
    fake_local_fixture.write_bytes(b"CHANGED - must not be picked up")

    case2 = build_demo_case()
    # The cached bytes are the ORIGINAL fake fixture, not "CHANGED".
    assert b"CHANGED" not in base64.b64decode(case2.dicom_bytes_b64)


def test_build_demo_case_report_text_matches_luminal_a_example():
    """The report_text must be the exact luminal-A block the frontend ships.
    v0.2.2 wires this via demo_fixtures; if it drifts, the "Load demo case"
    button will populate a different report than the manual example button."""
    # These are the receptor markers the canned example advertises.
    for marker in [
        "Age: 58, postmenopausal",
        "T1N0M0",
        "Invasive ductal carcinoma",
        "Estrogen Receptor: Positive",
        "HER2/neu: Negative",
        "Nottingham Grade: 2",
        "Ki-67 index: 12%",
    ]:
        assert marker in DEMO_REPORT_TEXT


def test_build_demo_case_warnings_and_context_shape(
    tmp_demo_cache, fake_local_fixture,
):
    """Warnings must contain the RUO disclaimer + demo-only marker; context
    must include the fields the therapy endpoint needs to route to luminal-A."""
    case = build_demo_case()
    joined = " ".join(case.warnings).lower()
    assert "demo" in joined
    assert "not a real patient" in joined or "research use only" in joined
    assert case.patient_context["age"] == 58
    assert case.patient_context["menopausal_status"] == "post"
    assert case.dicom_source == DEMO_DICOM_SOURCE


def test_build_demo_case_raises_when_hf_unreachable(tmp_demo_cache, monkeypatch):
    """No cache, no local file, HF blows up → DemoFixtureUnavailable."""
    monkeypatch.setenv(
        "ONCOLOGY_ARBITER_DEMO_LOCAL_DICOM", "/nonexistent/local.dcm"
    )
    with patch(
        "oncology_arbiter.api.demo_fixtures._download_from_hf",
        side_effect=demo_fixtures.DemoFixtureUnavailable(
            "network unreachable in test"
        ),
    ):
        with pytest.raises(DemoFixtureUnavailable, match="network unreachable"):
            build_demo_case()


def test_prewarm_demo_case_swallows_errors(tmp_demo_cache, monkeypatch):
    """prewarm must never raise — startup path relies on this."""
    monkeypatch.setenv(
        "ONCOLOGY_ARBITER_DEMO_LOCAL_DICOM", "/nonexistent/local.dcm"
    )
    with patch(
        "oncology_arbiter.api.demo_fixtures._download_from_hf",
        side_effect=RuntimeError("boom"),
    ):
        result = prewarm_demo_case()
    assert result is None


def test_prewarm_demo_case_returns_path_on_success(
    tmp_demo_cache, fake_local_fixture,
):
    result = prewarm_demo_case()
    assert result is not None
    assert result.is_file()
    assert result.stat().st_size == fake_local_fixture.stat().st_size
