"""Unit tests for POST /v1/tumor_board/bundle (v0.4.0-alpha).

Contract under test:

* Round-trip: post the shipped AK bundle → 200 + envelope + bundle_sha256.
* Idempotent: identical bundle → identical sha256.
* Wrong contract_version → 422 (pydantic Literal + explicit route check).
* HIPAA_MODE=true flips provenance.model_state to LOADED_HIPAA_REDACTOR.
* HIPAA_MODE=false (or unset) keeps provenance.model_state = LOADED_AK_BUNDLE.
* The shipped demo sample /v1/demo/samples/ak_mbd4_lof_case is byte-for-byte
  equal to the payload accepted by /v1/tumor_board/bundle → the two paths
  agree on the same contract.
* /health.cancers now lists "hgsoc" alongside "breast" and "nsclc".

No network I/O — pure in-process TestClient. AUTH_MODE=off comes from
conftest.py; no key headers needed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app


AK_BUNDLE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "oncology_arbiter"
    / "api"
    / "static"
    / "demo_samples"
    / "ak_mbd4_lof_case.json"
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture(scope="module")
def bundle() -> dict:
    return json.loads(AK_BUNDLE_PATH.read_text())


# ------------------------------------------------------------------- happy


def test_happy_round_trip(client: TestClient, bundle: dict) -> None:
    r = client.post("/v1/tumor_board/bundle", json=bundle)
    assert r.status_code == 200, r.text
    body = r.json()

    # Envelope
    assert body["disclaimer"], "envelope disclaimer missing"
    assert body["caveat"], "envelope caveat missing"
    assert body["provenance"]["request_id"], "envelope request_id missing"
    assert body["provenance"]["model_state"] == "loaded_ak_bundle"
    assert body["provenance"]["model_name"] == "tumor_board_v3_multimodal"

    # New fields
    assert isinstance(body["bundle_sha256"], str) and len(body["bundle_sha256"]) == 64
    assert body["persisted_path"] is None  # no disk persistence yet

    # Bundle echoed back unchanged
    assert body["bundle"]["patient_id"] == "MBD4-LOF-DEMO-01"
    assert (
        body["bundle"]["contract_version"]
        == "tumor_board.v3.multimodal-with-manuscript-claims"
    )


def test_bundle_sha256_is_deterministic(client: TestClient, bundle: dict) -> None:
    r1 = client.post("/v1/tumor_board/bundle", json=bundle)
    r2 = client.post("/v1/tumor_board/bundle", json=bundle)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["bundle_sha256"] == r2.json()["bundle_sha256"]


# ------------------------------------------------------------------- rejects


def test_wrong_contract_version_rejected(client: TestClient, bundle: dict) -> None:
    bad = copy.deepcopy(bundle)
    bad["contract_version"] = "tumor_board.v2.legacy"
    r = client.post("/v1/tumor_board/bundle", json=bad)
    assert r.status_code == 422, r.text


def test_missing_patient_context_rejected(client: TestClient, bundle: dict) -> None:
    bad = copy.deepcopy(bundle)
    bad.pop("patient_context", None)
    r = client.post("/v1/tumor_board/bundle", json=bad)
    assert r.status_code == 422, r.text


def test_zero_recommended_drugs_rejected(client: TestClient, bundle: dict) -> None:
    bad = copy.deepcopy(bundle)
    bad["synthetic_lethality"]["recommended_drugs"] = []
    r = client.post("/v1/tumor_board/bundle", json=bad)
    # Pydantic min_length=1 on recommended_drugs → 422
    assert r.status_code == 422, r.text


# ------------------------------------------------------------------- hipaa


def test_hipaa_mode_flips_model_state(
    client: TestClient, bundle: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HIPAA_MODE", "true")
    r = client.post("/v1/tumor_board/bundle", json=bundle)
    assert r.status_code == 200, r.text
    assert r.json()["provenance"]["model_state"] == "loaded_hipaa_redactor"


def test_no_hipaa_mode_uses_ak_bundle_state(
    client: TestClient, bundle: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HIPAA_MODE", raising=False)
    r = client.post("/v1/tumor_board/bundle", json=bundle)
    assert r.status_code == 200
    assert r.json()["provenance"]["model_state"] == "loaded_ak_bundle"


# ------------------------------------------------------------------- surface parity


def test_demo_sample_matches_post_body(client: TestClient, bundle: dict) -> None:
    """GET /v1/demo/samples/ak_mbd4_lof_case must serve the same bundle a POST
    validates. The two surfaces must not drift or the SPA renders inconsistent
    evidence.
    """
    r = client.get("/v1/demo/samples/ak_mbd4_lof_case")
    assert r.status_code == 200, r.text
    served = r.json()
    assert served == bundle, "GET demo sample differs from bundle on disk"


def test_health_cancers_lists_hgsoc(client: TestClient) -> None:
    h = client.get("/health").json()
    assert "hgsoc" in h["cancers"]
    assert h["cancers"]["hgsoc"]["case_full"] is False
    # POST /v1/tumor_board/bundle must be advertised on the cancer entry
    assert "tumor_board/bundle" in h["cancers"]["hgsoc"]["endpoints"]


# ------------------------------------------------------------------- audit


def test_ak_bundle_audit_script_passes() -> None:
    """Sanity guard: the CI check_ak_bundle job must never diverge from the
    bundle shipped in this test suite. If someone edits the bundle without
    updating audit_ak_bundle.py, this test catches the drift locally.
    """
    import subprocess
    import sys

    # tests/unit/test_x.py -> repo root is parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "audit_ak_bundle.py"
    assert script.exists(), f"audit script missing at {script}"

    r = subprocess.run(
        [sys.executable, str(script), "-v"],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert r.returncode == 0, (
        f"audit script failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    assert "PASS" in r.stdout


# ------------------------------------------------------------------- SHA anchors


def test_manuscript_sha_is_pinned(bundle: dict) -> None:
    assert (
        bundle["synthetic_lethality"]["provenance"]["manuscript_repo_sha_at_audit"]
        == "d33f6403fb11b314c86fa74d9c56e07b7ac3d7b1"
    )


def test_backend_head_sha_is_pinned(bundle: dict) -> None:
    assert (
        bundle["synthetic_lethality"]["provenance"]["backend_head_sha"]
        == "bfd6d11fc872c11a13365b0682cea776a136c7f3"
    )


def test_key_tp53_anchor_matches_canonical(bundle: dict) -> None:
    """The single most-defense-critical anchor — must never drift."""
    atr_row = next(
        r for r in bundle["synthetic_lethality"]["provenance"]["evidence_matrix"]["rows"]
        if r["axis"] == "atr_wee1"
    )
    tp53 = next(
        a for a in atr_row["auxiliary_evidence"]
        if a["stratifier"] == "TP53_mutant_only"
    )
    assert tp53["p_value"] == 0.003002668797799231
    assert tp53["effect_size"] == -0.7404782024497254
