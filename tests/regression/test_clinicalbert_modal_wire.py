"""Regression: enforce MedGemma-primary pathology parser wiring.

The production contract requires:

  1. ``render.yaml`` configures the MedGemma specialist, not the retired
     ClinicalBERT parser as a production backend.
  2. The parser orchestrator routes to MedGemma by default and permits
     ClinicalBERT only through an explicit research override.
  3. The public schema package preserves the parsed-report wire fields.
  4. The legacy Modal client remains import-light for explicit research use.

The Modal client module exposes ``ClinicalBertModalClient``
     with a stdlib-only ``urllib`` import surface (Render free tier has no
     ``requests``; shipping a heavy dep would blow the 512 MB budget).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_YAML = REPO_ROOT / "render.yaml"
MODAL_CLIENT_PY = (
    REPO_ROOT / "src" / "oncology_arbiter" / "nlp" / "clinicalbert_modal_client.py"
)


@pytest.mark.regression
def test_render_yaml_declares_medgemma_primary_not_clinicalbert() -> None:
    """Production must route pathology extraction to MedGemma."""
    assert RENDER_YAML.exists(), f"render.yaml missing at {RENDER_YAML}"
    text = RENDER_YAML.read_text()
    assert re.search(
        r"^\s*-\s*key:\s*MEDGEMMA_MODAL_URL\s*\n\s*value:\s*\S+\s*$",
        text,
        re.MULTILINE,
    ), "render.yaml does not configure MEDGEMMA_MODAL_URL"
    assert not re.search(
        r"^\s*-\s*key:\s*(?:CLINICALBERT_BACKEND|CLINICALBERT_MODAL_URL)\s*$",
        text,
        re.MULTILINE,
    ), "ClinicalBERT must not be configured as the production pathology backend"


@pytest.mark.regression
def test_modal_client_module_exists_and_is_stdlib_only() -> None:
    """Modal client must exist and must not import 'requests' / heavy deps."""
    assert MODAL_CLIENT_PY.exists(), f"missing {MODAL_CLIENT_PY}"
    src = MODAL_CLIENT_PY.read_text()
    tree = ast.parse(src)

    class_names = {
        n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
    }
    assert "ClinicalBertModalClient" in class_names, (
        f"ClinicalBertModalClient class not defined in {MODAL_CLIENT_PY}; "
        f"found {class_names}"
    )

    banned = {"requests", "httpx", "aiohttp"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    overlap = imported & banned
    assert not overlap, (
        f"Modal client must be stdlib-only for Render free tier; "
        f"forbidden imports found: {overlap}"
    )
    # Positive check: urllib must be used to make the request.
    assert "urllib" in imported, (
        f"Modal client is expected to use urllib; imports were {imported}"
    )


@pytest.mark.regression
def test_pathology_orchestrator_routes_to_medgemma_by_default(monkeypatch) -> None:
    """A real pathology note must take the MedGemma branch by default."""
    from oncology_arbiter.api.orchestrator.pathology_parse import (
        PathologyParseOrchestrator,
    )

    observed: list[tuple[str, str, str]] = []

    def medgemma(report_text: str, *, request_id: str):
        observed.append(("medgemma", report_text, request_id))
        return None, None, None, {"status": "sentinel"}

    def clinicalbert(report_text: str, *, request_id: str):
        observed.append(("clinicalbert", report_text, request_id))
        return None, None, None, {"status": "sentinel"}

    monkeypatch.delenv("ONCOLOGY_FORCE_CLINICALBERT_NER", raising=False)
    monkeypatch.setattr(PathologyParseOrchestrator, "_parse_medgemma", medgemma)
    monkeypatch.setattr(PathologyParseOrchestrator, "_parse_clinicalbert", clinicalbert)

    note = (
        "Invasive ductal carcinoma, Nottingham grade 3. "
        "ER positive (95%), PR positive (70%), HER2 IHC 2+ equivocal."
    )
    PathologyParseOrchestrator.parse_biopsy_report(note, request_id="clinical-regression")

    assert observed == [("medgemma", note, "clinical-regression")]


@pytest.mark.regression
def test_nsclc_response_schema_carries_parsed_report() -> None:
    """The NsclcResponse envelope must expose parsed_report + parsed_report_provenance."""
    from oncology_arbiter.api.schemas import NsclcResponse

    field_names = set(NsclcResponse.model_fields)
    assert "parsed_report" in field_names, (
        f"NsclcResponse missing 'parsed_report' field; found {sorted(field_names)}"
    )
    assert "parsed_report_provenance" in field_names, (
        "NsclcResponse missing 'parsed_report_provenance' field; "
        f"found {sorted(field_names)}"
    )


@pytest.mark.regression
def test_modal_deployment_packages_identity_locked_sliding_window_runtime() -> None:
    """Legacy research deployment must package the measured v2 runtime."""
    modal_app_py = REPO_ROOT / "deploy" / "modal" / "clinicalbert_app.py"
    runtime_py = (
        REPO_ROOT
        / "src"
        / "oncology_arbiter"
        / "nlp"
        / "clinicalbert_runtime_v2.py"
    )
    assert modal_app_py.exists(), f"missing {modal_app_py}"
    assert runtime_py.exists(), f"missing {runtime_py}"

    src = modal_app_py.read_text()
    runtime_src = runtime_py.read_text()
    ast.parse(src)
    ast.parse(runtime_src)

    assert ".add_local_file(" in src
    assert "clinicalbert_runtime_v2.py" in src
    assert "ClinicalBertV2Runtime" in src
    assert "ARTIFACT_SHA256" in src
    assert "BASE_REVISION" in src

    assert "return_overflowing_tokens=True" in runtime_src
    assert "stride=self.stride" in runtime_src
    assert "sliding windows did not cover every report token" in runtime_src
    assert "averaged = sums / counts[:, None]" in runtime_src
