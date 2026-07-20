"""Regression: enforce v0.4.1 prod wiring for the ClinicalBERT report parser.

The v0.4.1 production plan requires:

  1. ``render.yaml`` publishes the environment variables the app.py NSCLC
     branch needs to route to the Modal deployment.
  2. ``src/oncology_arbiter/api/app.py`` imports the Modal client and
     actually instantiates it when ``CLINICALBERT_BACKEND=modal``.
  3. The NSCLC branch reads ``CLINICALBERT_MODAL_URL`` and calls the
     Modal client on the parsed report path.
  4. The Modal client module exists and exposes ``ClinicalBertModalClient``
     with a stdlib-only ``urllib`` import surface (Render free tier has no
     ``requests``; shipping a heavy dep would blow the 512 MB budget).

We assert on the source text and AST rather than importing app.py at test
time — importing app.py triggers full FastAPI + optional-backend wiring,
which is deliberately kept out of the ``regression`` marker's scope.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_YAML = REPO_ROOT / "render.yaml"
APP_PY = REPO_ROOT / "src" / "oncology_arbiter" / "api" / "app.py"
MODAL_CLIENT_PY = (
    REPO_ROOT / "src" / "oncology_arbiter" / "nlp" / "clinicalbert_modal_client.py"
)


@pytest.mark.regression
def test_render_yaml_declares_modal_env() -> None:
    """render.yaml must publish CLINICALBERT_BACKEND=modal + CLINICALBERT_MODAL_URL."""
    assert RENDER_YAML.exists(), f"render.yaml missing at {RENDER_YAML}"
    text = RENDER_YAML.read_text()
    # Order matters less than presence — the Render blueprint layout is
    # 'key:' / 'value:' pairs, so we scan for both.
    assert re.search(
        r"^\s*-\s*key:\s*CLINICALBERT_BACKEND\s*\n\s*value:\s*modal\s*$",
        text,
        re.MULTILINE,
    ), "render.yaml does not set CLINICALBERT_BACKEND=modal"
    assert re.search(
        r"^\s*-\s*key:\s*CLINICALBERT_MODAL_URL\s*\n\s*value:\s*"
        r"https://crispro-test--clinicalbert\s*$",
        text,
        re.MULTILINE,
    ), "render.yaml does not set CLINICALBERT_MODAL_URL to the crispro-test URL"
    # Co-Scientist MUST stay ON — otherwise elo_ranked_hypotheses is empty
    # even after the ClinicalBERT parse succeeds.
    assert re.search(
        r"^\s*-\s*key:\s*ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST\s*\n\s*value:\s*\"?1\"?\s*$",
        text,
        re.MULTILINE,
    ), "render.yaml must keep ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST=1"


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
def test_app_py_imports_and_uses_modal_client() -> None:
    """app.py NSCLC branch must import ClinicalBertModalClient and route on backend=modal."""
    assert APP_PY.exists(), f"missing {APP_PY}"
    src = APP_PY.read_text()

    # Must import the Modal client somewhere in app.py (top-level or inside
    # the /v1/case/full handler).
    assert "ClinicalBertModalClient" in src, (
        "app.py does not reference ClinicalBertModalClient"
    )

    # Must dispatch on CLINICALBERT_BACKEND env var and honor 'modal'.
    assert "CLINICALBERT_BACKEND" in src, (
        "app.py does not read CLINICALBERT_BACKEND"
    )
    # CLINICALBERT_MODAL_URL must be read either directly in app.py OR
    # inside the Modal client module app.py delegates to. Both count as
    # "wired" — but at least one of the two must reference it, otherwise
    # the env var in render.yaml is dead.
    modal_client_src = MODAL_CLIENT_PY.read_text()
    assert (
        "CLINICALBERT_MODAL_URL" in src
        or "CLINICALBERT_MODAL_URL" in modal_client_src
    ), (
        "Neither app.py nor clinicalbert_modal_client.py reads "
        "CLINICALBERT_MODAL_URL; render.yaml env var would be dead"
    )
    # And a modal-branch string literal in app.py, so the routing is not
    # just a shell of the env var check.
    assert (
        '"modal"' in src or "'modal'" in src
    ), "app.py does not branch on the 'modal' backend value"

    # AST-level sanity: parse succeeds — no syntax errors were introduced
    # by the ClinicalBERT wiring block.
    ast.parse(src)


@pytest.mark.regression
def test_nsclc_response_schema_carries_parsed_report() -> None:
    """The NsclcResponse envelope must expose parsed_report + parsed_report_provenance."""
    schemas_py = (
        REPO_ROOT / "src" / "oncology_arbiter" / "api" / "schemas.py"
    )
    assert schemas_py.exists(), f"missing {schemas_py}"
    src = schemas_py.read_text()

    tree = ast.parse(src)
    nsclc_cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "NsclcResponse":
            nsclc_cls = node
            break
    assert nsclc_cls is not None, "NsclcResponse class missing from schemas.py"

    field_names = {
        stmt.target.id
        for stmt in nsclc_cls.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    assert "parsed_report" in field_names, (
        f"NsclcResponse missing 'parsed_report' field; found {sorted(field_names)}"
    )
    assert "parsed_report_provenance" in field_names, (
        "NsclcResponse missing 'parsed_report_provenance' field; "
        f"found {sorted(field_names)}"
    )
