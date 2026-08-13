"""Focused production-integrity tests for remote specialist contracts."""
from __future__ import annotations

import ast
import base64
import importlib
import math

from oncology_arbiter.arbiter.manski import ManskiBoundsExceeded
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.agents.supervisor import StageResult, _record_llm
from oncology_arbiter.api.app import _score_explicit_therapy_triage, create_app
from oncology_arbiter.models.llm_client import GemmaClient, LlmUnavailable
from oncology_arbiter.models import specialist_clients as sc
from oncology_arbiter.nlp.clinicalbert_modal_client import (
    EXPECTED_APP_VERSION,
    EXPECTED_METRICS_SHA256,
    EXPECTED_MODEL_SHA256,
    EXPECTED_OVERLAP_TOKENS,
    EXPECTED_WINDOW_AGGREGATION,
    EXPECTED_WINDOW_TOKENS,
    ClinicalBertModalError,
    validate_clinicalbert_contract,
)


CASE_ID = "a" * 16
CASE_SHA = "a" * 64


def _clinicalbert_response() -> dict:
    return {
        "app_version": EXPECTED_APP_VERSION,
        "window_tokens": EXPECTED_WINDOW_TOKENS,
        "overlap_tokens": EXPECTED_OVERLAP_TOKENS,
        "window_aggregation": EXPECTED_WINDOW_AGGREGATION,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "metrics_sha256": EXPECTED_METRICS_SHA256,
        "n_windows": 2,
    }


def test_clinicalbert_contract_accepts_only_exact_release() -> None:
    validate_clinicalbert_contract(_clinicalbert_response())
    for key, bad_value in {
        "app_version": "clinicalbert-modal-v0.5.1",
        "window_tokens": 512,
        "overlap_tokens": 0,
        "window_aggregation": "first_window_only",
        "model_sha256": "0" * 64,
        "metrics_sha256": "1" * 64,
        "n_windows": 0,
    }.items():
        response = _clinicalbert_response()
        response[key] = bad_value
        with pytest.raises(ClinicalBertModalError, match="clinicalbert_contract_mismatch"):
            validate_clinicalbert_contract(response)


def test_biopsy_source_calls_only_clinicalbert_for_report_parsing() -> None:
    app_path = Path(importlib.import_module("oncology_arbiter.api.app").__file__)
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    biopsy = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "biopsy_analyze"
    )
    names = {node.id for node in ast.walk(biopsy) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(biopsy) if isinstance(node, ast.Attribute)}
    assert "_clinicalbert_biopsy_parse" in names
    assert not ({"parse_report", "parse_biopsy_report", "fuse_report_parse"} & (names | attributes))


def test_biopsy_clinicalbert_failure_has_no_regex_fallback(monkeypatch) -> None:
    cb = importlib.import_module("oncology_arbiter.nlp.clinicalbert_modal_client")
    old_parser = importlib.import_module("oncology_arbiter.nlp.report_parser_v2")
    monkeypatch.setenv("CLINICALBERT_MODAL_URL", "https://unit--clinicalbert")

    def fail_parse(self, report_text: str):
        raise ClinicalBertModalError("unit contract failure")

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError("legacy report parser must not be called")

    monkeypatch.setattr(cb.ClinicalBertModalClient, "parse", fail_parse)
    monkeypatch.setattr(old_parser, "parse_biopsy_report", forbidden_fallback, raising=False)
    monkeypatch.setattr(old_parser, "parse_report", forbidden_fallback, raising=False)

    response = TestClient(create_app()).post(
        "/v1/biopsy/analyze", json={"report_text": "ER positive. HER2 negative."}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pipeline_status"] == "failed_required_stage"
    assert body["report_parse"] is None
    receipt = next(r for r in body["stage_receipts"] if r["stage"] == "clinicalbert_pathology_parse")
    assert receipt["status"] == "failed_required"
    assert body["dss_prognosis"] is None


def _manifest(files: list[str] | None = None) -> dict:
    selected = files if files is not None else ["dicom_series/slice_0000.dcm"]
    return {
        "case_id": CASE_ID,
        "sha256_full": CASE_SHA,
        "files": selected,
        "byte_counts": {name: 3 for name in selected},
        "media_types": ["application/dicom-series"],
        "app_version": "case-storage-modal-v0.5.0-alpha",
        "disclaimer": "Research Use Only",
    }


def test_case_storage_manifest_and_file_identity(monkeypatch) -> None:
    payloads = [
        (_manifest(), 1.25),
        ({
            "case_id": CASE_ID,
            "file": "dicom_series/slice_0000.dcm",
            "bytes_b64": base64.b64encode(b"abc").decode("ascii"),
            "byte_count": 3,
            "app_version": "case-storage-modal-v0.5.0-alpha",
        }, 2.5),
    ]

    def fake_http(*args, **kwargs):
        return payloads.pop(0)

    monkeypatch.setattr(sc, "_http_json", fake_http)
    client = sc.CaseStorageClient("https://unit--case-storage")
    manifest = client.manifest(CASE_ID, request_id="req-1")
    retrieved = client.get_file(
        CASE_ID, "dicom_series/slice_0000.dcm", request_id="req-1"
    )
    assert manifest.output["manifest_sha256"] == manifest.receipt["artifact_sha256"]
    assert retrieved.output["bytes"] == b"abc"
    assert retrieved.output["byte_count"] == 3
    assert len(retrieved.output["sha256"]) == 64
    assert not payloads


@pytest.mark.parametrize(
    "unsafe",
    ["", ".", "..", "/absolute.dcm", "dicom_series/../x.dcm", "other/x.dcm", "a/b/c.dcm", "a\\b.dcm"],
)
def test_case_storage_rejects_unsafe_manifest_names(monkeypatch, unsafe: str) -> None:
    monkeypatch.setattr(sc, "_http_json", lambda *a, **k: (_manifest([unsafe]), 1.0))
    with pytest.raises(sc.SpecialistServiceError, match="unsafe"):
        sc.CaseStorageClient("https://unit--case-storage").manifest(CASE_ID, request_id="req")


def test_phikon_requires_exactly_768_numeric_values(monkeypatch) -> None:
    monkeypatch.setattr(
        sc,
        "_http_json",
        lambda *a, **k: ({"embeddings": [[0.25] * 768], "app_version": "phikon-live"}, 3.0),
    )
    call = sc.PhikonClient("https://unit--phikon.modal.run").embed(
        b"deidentified-image", request_id="req"
    )
    assert call.output["embedding_dim"] == 768
    assert len(call.output["vector_sha256"]) == 64
    assert call.receipt["warnings"] == ["embedding_only_no_subtype_or_clinical_interpretation"]

    monkeypatch.setattr(sc, "_http_json", lambda *a, **k: ({"embedding": [0.0] * 767}, 1.0))
    with pytest.raises(sc.SpecialistServiceError, match="768"):
        sc.PhikonClient("https://unit--phikon.modal.run").embed(b"x", request_id="req")


def test_luna16_requires_case_id_loaded_state_and_bundle(monkeypatch) -> None:
    valid = {
        "case_id": CASE_ID,
        "model_state": "loaded_luna16_retinanet",
        "bundle_version": "0.6.9",
        "detections": [{"score": 0.7}],
        "ingest_source": "case-storage",
    }
    monkeypatch.setattr(sc, "_http_json", lambda *a, **k: (valid, 4.0))
    call = sc.Luna16Client("https://unit--luna16.modal.run").detect(CASE_ID, request_id="req")
    assert call.output["detections"][0]["score"] == 0.7
    assert call.receipt["model_version"] == "0.6.9"

    for key, value in (("model_state", "proxy_lung_heuristic"), ("bundle_version", "0.6.8"), ("case_id", "b" * 16)):
        bad = dict(valid)
        bad[key] = value
        monkeypatch.setattr(sc, "_http_json", lambda *a, _bad=bad, **k: (_bad, 1.0))
        with pytest.raises(sc.SpecialistServiceError, match="mismatch|not loaded"):
            sc.Luna16Client("https://unit--luna16.modal.run").detect(CASE_ID, request_id="req")


class _Response:
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data
        self.text = "upstream failure" if status_code >= 400 else "ok"

    def json(self) -> dict:
        return self._data


def test_configured_medgemma_is_terminal_and_surfaces_provenance(monkeypatch) -> None:
    monkeypatch.setenv("MEDGEMMA_MODAL_URL", "https://unit--medgemma-27b")
    client = GemmaClient(google_key="must-not-be-used", timeout_s=1)
    calls: list[str] = []

    def success(url, **kwargs):
        calls.append(url)
        return _Response(200, {
            "text": "non-empty clinical research synthesis",
            "model": "google/medgemma-27b-it",
            "app_version": "medgemma-27b-modal-v0.5.0-production-ready",
            "request_id": "medgemma-request-1",
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "latency_s": 0.5,
            "honesty_warning": "research-use-only",
        })

    monkeypatch.setattr(client._session, "post", success)
    response = client.chat([{"role": "user", "content": "summarize"}])
    result = StageResult(stage="case_full", model_state="executed")
    _record_llm(result, response)
    serialized = result.as_dict()
    assert serialized["llm_provider"] == "medgemma_modal"
    assert serialized["llm_model"] == "google/medgemma-27b-it"
    assert serialized["llm_app_version"] == "medgemma-27b-modal-v0.5.0-production-ready"
    assert serialized["llm_request_ids"] == ["medgemma-request-1"]
    assert serialized["llm_prompt_tokens"] == 12
    assert serialized["llm_completion_tokens"] == 7
    assert calls == ["https://unit--medgemma-27b-chat.modal.run"]

    def fail(url, **kwargs):
        calls.append(url)
        return _Response(500, {})

    monkeypatch.setattr(client._session, "post", fail)
    with pytest.raises(LlmUnavailable, match="MedGemma Modal unavailable"):
        client.chat([{"role": "user", "content": "retry"}])
    assert calls[-1] == "https://unit--medgemma-27b-chat.modal.run"


def _triage_features(**updates) -> dict:
    features = {
        "histology": "invasive_ductal",
        "grade": 2,
        "er_positive": True,
        "pr_positive": True,
        "her2_positive": False,
        "ki67_pct": 20.0,
        "tumor_size_mm": 20.0,
        "lymph_nodes_pos": 0,
        "brca_pathogenic": False,
        "age_years": 50.0,
    }
    features.update(updates)
    return features


def test_therapy_triage_contrasts_are_mathematically_exact() -> None:
    zero = _score_explicit_therapy_triage(_triage_features(lymph_nodes_pos=0))
    one = _score_explicit_therapy_triage(_triage_features(lymph_nodes_pos=1))
    assert zero is not None and one is not None
    assert zero.term_contributions["node_status_positive"] == 0.0
    assert one.term_contributions["node_status_positive"] == 2.1
    assert math.isclose(one.logit - zero.logit, 2.1, rel_tol=0, abs_tol=1e-12)

    grade3 = _score_explicit_therapy_triage(_triage_features(grade=3))
    larger = _score_explicit_therapy_triage(_triage_features(tumor_size_mm=70.0))
    older = _score_explicit_therapy_triage(_triage_features(age_years=60.0))
    assert grade3 is not None and larger is not None and older is not None
    assert math.isclose(grade3.logit - zero.logit, 1.2, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(larger.logit - zero.logit, 1.4, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(older.logit - zero.logit, -0.05, rel_tol=0, abs_tol=1e-12)
    # MIGRATED: an explicitly-null node status used to return None. It now
    # raises through the Manski gate, and the exception carries the interval
    # that the silent None concealed. The arithmetic contrasts above are
    # unchanged -- only the missing-data branch moved.
    with pytest.raises(ManskiBoundsExceeded) as exc:
        _score_explicit_therapy_triage(_triage_features(lymph_nodes_pos=None))
    bounds = exc.value.bounds
    assert bounds.missing_features == ("node_status_positive",)
    assert (round(bounds.lower, 6), round(bounds.upper, 6)) == (0.153164, 0.596283)
    assert round(bounds.width, 6) == 0.443119
    # The width is exactly the sigmoid span of the +2.1 node coefficient over
    # the base logit, so the refusal is arithmetic, not a policy guess.
    assert round(bounds.upper - bounds.lower, 6) == round(bounds.width, 6)


def test_therapy_endpoint_uses_bridge_as_only_recommendation_source(monkeypatch) -> None:
    def fake_run(self, payload, *, request_id: str, required: bool = True):
        output = {
            "status": "succeeded",
            "bridge_version": "v3.1",
            "sl_indicated_drugs": [{"drug_name": "osimertinib", "sl_rationale": "EGFR actionability"}],
            "policy_overlays": [{
                "drug_name": "erlotinib",
                "policy_decision": "DISALLOW",
                "policy_rationale_short": "DOWNSTREAM_DRIVER_MISMATCH",
            }],
            "gap_summary": {},
            "provenance": {"source": "live-test-double"},
        }
        receipt = {
            "stage": "synthetic_lethality_therapy_bridge",
            "required": required,
            "status": "succeeded",
            "request_id": request_id,
            "service_name": "crispro-backend-v2",
            "app_version": "v3.1",
            "model_name": "SLTherapyBridge",
            "model_version": "v3.1",
            "warnings": ["research_actionability_hypotheses_not_treatment_benefit"],
            "error": None,
        }
        return sc.SpecialistCall(output=output, receipt=receipt)

    monkeypatch.setattr(sc.SLTherapyBridgeClient, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(sc.SLTherapyBridgeClient, "run", fake_run)
    response = TestClient(create_app()).post(
        "/v1/therapy/reason",
        json={
            "disease": "NSCLC",
            "cancer_type": "nsclc",
            "mutations": [{"gene": "EGFR", "hgvs_p": "p.L858R"}],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["regimen"] for item in body["recommended_options"]] == ["osimertinib"]
    assert [item["regimen"] for item in body["not_recommended"]] == ["erlotinib"]
    assert body["therapy_bridge"]["bridge_version"] == "v3.1"
    assert body["rules_sha256"] is None
    assert body["rules_model_id"] is None
    assert body["branch_id"] is None


def _dss_features() -> dict:
    return {
        "age": 58.0,
        "tumor_size_mm": 22.0,
        "nodes_positive": 1,
        "grade": 2,
        "er_positive": True,
        "pr_positive": True,
        "her2_positive": False,
    }


def test_dynamic_breast_dss_matches_biopsy_exactly() -> None:
    client = TestClient(create_app())
    biopsy = client.post("/v1/biopsy/analyze", json={"dss_features": _dss_features()})
    dynamic = client.post(
        "/v1/tumor_board/dynamic",
        json={
            "cancer": "breast",
            "dss_features": _dss_features(),
            "run_co_scientist": False,
        },
    )
    assert biopsy.status_code == dynamic.status_code == 200
    assert dynamic.json()["dss_prognosis"] == biopsy.json()["dss_prognosis"]
    assert dynamic.json()["ovarian_arbiter"] is None


def test_dynamic_hgsoc_exposes_retirement_without_patient_score() -> None:
    response = TestClient(create_app()).post(
        "/v1/tumor_board/dynamic",
        json={"cancer": "hgsoc", "run_co_scientist": False},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    retired = body["ovarian_arbiter"]
    assert retired["model_state"] == "retired"
    assert "p_positive" not in retired
    assert "probability" not in retired
    assert "risk_bucket" not in retired
    assert "driving_feature" not in retired
    receipt = next(r for r in body["stage_receipts"] if r["stage"] == "ovarian_prognostic_arbiter")
    assert receipt["status"] == "retired"
    assert body["dss_prognosis"] is None


def test_full_hgsoc_never_falls_through_to_breast() -> None:
    response = TestClient(create_app()).post(
        "/v1/case/full?cancer=hgsoc",
        json={"run_co_scientist": False},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cancer"] == "hgsoc"
    assert body["ovarian_arbiter"]["model_state"] == "retired"
    assert body["dss_prognosis"] is None
    assert body["biopsy"] is None
    assert body["therapy"] is None


def test_full_nsclc_requires_case_id_and_never_substitutes_heuristic() -> None:
    response = TestClient(create_app()).post(
        "/v1/case/full?cancer=nsclc",
        json={"run_co_scientist": False},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pipeline_status"] == "failed_required_stage"
    assert body["luna16_detection"] is None
    assert body["nsclc"]["model_state"] == "unavailable"
    assert body["nsclc"]["luna16"] is None
    assert body["nsclc"]["candidates"] == []
    assert body["nsclc"]["therapy_recommended"] == []
    assert body["nsclc"]["risk_score"] is None


def test_full_nsclc_uses_verified_case_manifest_then_luna16(monkeypatch) -> None:
    manifest_output = {
        **_manifest(),
        "manifest_sha256": "b" * 64,
    }
    luna_output = {
        "case_id": CASE_ID,
        "model_state": "loaded_luna16_retinanet",
        "model_name": "monai/lung_nodule_ct_detection@0.6.9",
        "bundle_version": "0.6.9",
        "app_version": "luna16-infer-v0.5.0-alpha",
        "ingest_source": f"modal-volume://oncology-arbiter-cases/{CASE_ID}/dicom_series/",
        "n_detections": 1,
        "top_score": 0.71,
        "detections": [{
            "center_z_mm": 1.0,
            "center_y_mm": 2.0,
            "center_x_mm": 3.0,
            "width_mm": 4.0,
            "height_mm": 5.0,
            "depth_mm": 6.0,
            "diameter_mm": 6.0,
            "score": 0.71,
        }],
        "inference_seconds": 4.4,
        "preprocessing_summary": {"hu_range": [-1024, 300]},
        "disclaimer": "Research Use Only",
    }

    def fake_manifest(self, case_id, *, request_id, required=True):
        return sc.SpecialistCall(
            manifest_output,
            {
                "stage": "case_storage", "required": required, "status": "succeeded",
                "request_id": request_id, "service_name": "case-storage",
                "app_version": "case-storage-modal-v0.5.0-alpha",
                "artifact_sha256": "b" * 64, "warnings": [], "error": None,
            },
        )

    def fake_detect(self, case_id, *, request_id, top_n=20, required=True):
        assert case_id == CASE_ID
        return sc.SpecialistCall(
            luna_output,
            {
                "stage": "luna16_detection", "required": required, "status": "succeeded",
                "request_id": request_id, "service_name": "luna16-infer",
                "app_version": "luna16-infer-v0.5.0-alpha",
                "model_name": "monai/lung_nodule_ct_detection@0.6.9",
                "model_version": "0.6.9", "warnings": [], "error": None,
            },
        )

    monkeypatch.setattr(sc.CaseStorageClient, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(sc.CaseStorageClient, "manifest", fake_manifest)
    monkeypatch.setattr(sc.Luna16Client, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(sc.Luna16Client, "detect", fake_detect)
    response = TestClient(create_app()).post(
        "/v1/case/full?cancer=nsclc",
        json={"case_id": CASE_ID, "run_co_scientist": False},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pipeline_status"] == "complete"
    assert body["case_manifest"]["manifest_sha256"] == "b" * 64
    assert body["luna16_detection"]["bundle_version"] == "0.6.9"
    assert body["nsclc"]["luna16"]["detections"][0]["score"] == 0.71
    luna_receipt = next(r for r in body["stage_receipts"] if r["stage"] == "luna16_detection")
    assert "manifest_sha256:" + "b" * 64 in luna_receipt["input_reference"]


def test_full_case_source_contains_no_local_ct_or_nccn_substitution() -> None:
    app_path = Path(importlib.import_module("oncology_arbiter.api.app").__file__)
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    full_case = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "case_full"
    )
    names = {node.id for node in ast.walk(full_case) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(full_case) if isinstance(node, ast.Attribute)}
    prohibited = {
        "series_dir", "read_ct_series", "run_lung_heuristic", "score_nsclc_therapy",
        "LungNoduleDetector",
    }
    assert not (prohibited & (names | attributes))


# --------------------------------------------------------------------------- #
# Retirement: the L2 template arbiter scored from free text / inferred inputs.
#
# Before this retirement, /v1/screening/analyze and /v1/biopsy/analyze both
# emitted an `arbiter_score` block sourced from an L2 logistic template whose
# feature dict was either empty or filled from a regex parse of free text. The
# templates carry n_training = 0, so with an empty feature dict the returned
# p_positive was nothing but sigma(intercept) -- a constant dressed as a
# patient-specific probability. The deleted test
# `test_medsiglip_score_present_alongside_arbiter_block` pinned exactly that
# behaviour ("the screening arbiter is a template with n_training=0 and empty
# features so its p_positive falls back to the intercept").
#
# The replacement invariant is that the single surviving arbiter call site is
# reachable only from an explicit, caller-supplied, complete feature vector,
# and is gated by Manski before release.
# --------------------------------------------------------------------------- #


def _app_tree() -> ast.Module:
    app_path = Path(importlib.import_module("oncology_arbiter.api.app").__file__)
    return ast.parse(app_path.read_text(encoding="utf-8"))


def _function_named(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def test_only_arbiter_call_site_requires_the_complete_explicit_vector() -> None:
    """There is exactly one `_score_arbiter` call, and free text cannot reach it."""
    tree = _app_tree()

    # Map every function node to its enclosing function so a call site can be
    # attributed. ast.walk() loses parentage, so build the link explicitly.
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    call_sites: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "_score_arbiter":
            enclosing: ast.AST | None = node
            while enclosing is not None and not isinstance(
                enclosing, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                enclosing = parent.get(enclosing)
            call_sites.append(
                enclosing.name if enclosing is not None else "<module>"
            )

    assert call_sites == ["_score_explicit_therapy_triage"], call_sites

    # ... and that one call site validates against the explicit wire schema and
    # keys off the declared required-field set rather than a parsed report.
    triage = _function_named(tree, "_score_explicit_therapy_triage")
    names = {n.id for n in ast.walk(triage) if isinstance(n, ast.Name)}
    attributes = {n.attr for n in ast.walk(triage) if isinstance(n, ast.Attribute)}
    assert "_TRIAGE_REQUIRED_FIELDS" in names
    assert "TherapyTriageFeatures" in names
    assert "model_validate" in attributes
    # No free-text-derived symbol may appear inside the only arbiter call site.
    free_text_symbols = {
        "report_text", "report_parse", "parse_report", "parse_biopsy_report",
        "_clinicalbert_biopsy_parse", "receptor_panel", "fuse_report_parse",
    }
    assert not (free_text_symbols & (names | attributes))


def test_screening_emits_no_template_arbiter_block() -> None:
    """`/v1/screening/analyze` hard-codes arbiter_score=None; no template escapes."""
    tree = _app_tree()
    screening = _function_named(tree, "screening_analyze")

    assigned: list[ast.expr] = [
        kw.value for node in ast.walk(screening) if isinstance(node, ast.Call)
        for kw in node.keywords if kw.arg == "arbiter_score"
    ]
    assert assigned, "screening_analyze no longer sets arbiter_score at all"
    for value in assigned:
        assert isinstance(value, ast.Constant) and value.value is None, ast.dump(value)

    names = {n.id for n in ast.walk(screening) if isinstance(n, ast.Name)}
    assert "_score_arbiter" not in names
    assert "load_arbiter" not in names


def test_biopsy_endpoint_emits_no_template_arbiter_score() -> None:
    """Both the failing and the succeeding biopsy paths return arbiter_score=None."""
    client = TestClient(create_app())

    # (a) free-text path: ClinicalBERT is not configured in unit context, so the
    #     required stage fails. The old behaviour attached a template arbiter
    #     score anyway; the new behaviour attaches nothing.
    free_text = client.post(
        "/v1/biopsy/analyze",
        json={"report_text": "Invasive ductal carcinoma, grade 2. ER positive."},
    )
    assert free_text.status_code == 200, free_text.text
    free_text_body = free_text.json()
    assert free_text_body["pipeline_status"] == "failed_required_stage"
    assert free_text_body["arbiter_score"] is None

    # (b) structured path: even when the DSS prognosis succeeds on a complete
    #     explicit vector, no L2 template arbiter rides along with it.
    structured = client.post("/v1/biopsy/analyze", json={"dss_features": _dss_features()})
    assert structured.status_code == 200, structured.text
    structured_body = structured.json()
    assert structured_body["dss_prognosis"] is not None
    assert structured_body["arbiter_score"] is None

    for body in (free_text_body, structured_body):
        assert "template" not in {
            receipt.get("model_state") for receipt in body["stage_receipts"]
        }
        assert not [
            receipt for receipt in body["stage_receipts"]
            if str(receipt.get("model_name") or "").endswith("_arbiter_template_v0")
        ]


def test_case_full_emits_no_template_arbiters_from_free_text() -> None:
    """A free-text case yields no template arbiter anywhere in the envelope."""
    response = TestClient(create_app()).post(
        "/v1/case/full?cancer=breast",
        json={
            "biopsy_input": {
                "report_text": "Invasive ductal carcinoma, grade 2. ER positive, HER2 negative."
            },
            "run_co_scientist": False,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    # Walk the whole envelope: no nested block may carry a template arbiter.
    def _walk(node: object) -> list[tuple[str, object]]:
        found: list[tuple[str, object]] = []
        if isinstance(node, dict):
            for key, value in node.items():
                found.append((key, value))
                found.extend(_walk(value))
        elif isinstance(node, list):
            for item in node:
                found.extend(_walk(item))
        return found

    pairs = _walk(body)
    template_states = [
        (key, value) for key, value in pairs
        if key == "model_state" and value == "template"
    ]
    assert template_states == [], template_states
    template_models = [
        (key, value) for key, value in pairs
        if key == "model_name" and str(value or "").endswith("_arbiter_template_v0")
    ]
    assert template_models == [], template_models

    # The therapy stage is the only arbiter surface, and free text does not
    # supply the explicit triage vector, so it must stay unscored.
    therapy = body.get("therapy")
    if therapy is not None:
        assert therapy.get("arbiter_score") is None
        assert therapy.get("therapy_triage") is None
