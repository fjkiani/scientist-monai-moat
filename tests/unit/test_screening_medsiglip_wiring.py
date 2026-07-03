"""Endpoint-level wiring tests for MedSigLIP ↔ /v1/screening/analyze.

Design rules enforced here (post-HAI-DEF-fix, 2026-07-02+):

  1. MedSigLIP is opt-in via `ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP=1`. Without
     it, the endpoint returns `model_state=placeholder` and `overall_score=None`.
  2. When MedSigLIP is enabled AND the gate report is ALLOWED, the endpoint
     returns `model_state=loaded_medsiglip` and `overall_score=probs[0]`
     (the malignant-mass zero-shot probability). NO proxy fallback runs.
  3. When MedSigLIP is enabled AND the gate report is FORBIDDEN or
     UNAUTHENTICATED, the endpoint MUST NOT silently fall back to the
     SigLIP proxy — the buggy pre-fix behavior. It returns
     `model_state=gated` unless the operator has ALSO set
     `ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY=1`, in which case the endpoint
     runs the proxy AND surfaces both the gate warning and the proxy
     warning in `warnings[]`.
  4. Warnings for gated denial follow the format
     `medsiglip_gated:{level}:{reason}` where level ∈
     {forbidden, unauthenticated, unknown}.

Tests are hermetic — no HuggingFace downloads and no real weights. They
monkey-patch `_run_medsiglip_on_preprocessed` and
`_run_siglip_proxy_on_preprocessed` at the app module level so we exercise
the endpoint's precedence logic, not the transformers stack.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api import app as app_module
from oncology_arbiter.api import create_app
from oncology_arbiter.api.schemas import ModelState
from oncology_arbiter.models.hai_def import (
    AccessLevel,
    GateReport,
    GatedAccessError,
)


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cbis_ddsm"
DICOM_PATH = FIXTURE_DIR / "Calc-Test_P_00038_LEFT_CC.dcm"


@pytest.fixture(scope="module")
def dicom_bytes_b64() -> str:
    if not DICOM_PATH.exists():
        pytest.skip(f"CBIS-DDSM fixture missing: {DICOM_PATH}")
    return base64.b64encode(DICOM_PATH.read_bytes()).decode("ascii")


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Reset the singletons between tests so env changes actually take effect.
    monkeypatch.setattr(app_module, "_MEDSIGLIP_SINGLETON", None, raising=False)
    monkeypatch.setattr(app_module, "_SIGLIP_PROXY_SINGLETON", None, raising=False)
    return TestClient(create_app())


# --------------------------------------------------------------------------- #
# Fake backend results — mirror the real dataclass shapes exactly


class _FakeMedSigLipResult:
    """Duck-typed drop-in for MedSigLipResult, no transformers needed."""

    def __init__(
        self,
        *,
        malignant_prob: float,
        without_prob: float,
        warnings: list[str] | None = None,
        gate_report: GateReport | None = None,
    ) -> None:
        self.source_path = "(preprocessed)"
        self.labels = [
            "a mammogram showing a malignant mass",
            "a mammogram showing no mass",
        ]
        self.probs = [malignant_prob, without_prob]
        top_idx = 0 if malignant_prob >= without_prob else 1
        self.top_label = self.labels[top_idx]
        self.top_prob = self.probs[top_idx]
        self.model_repo = "google/medsiglip-448"
        self.model_state = ModelState.LOADED_MEDSIGLIP
        self.input_resolution = 448
        self.logits_shape = (1, 2)
        self.warnings = list(warnings or [])
        self.gate_report = gate_report


class _FakeProxyResult:
    def __init__(self, *, malignant_prob: float, without_prob: float) -> None:
        self.source_path = "(preprocessed)"
        self.labels = [
            "a mammogram showing a malignant mass",
            "a mammogram showing no mass",
        ]
        self.probs = [malignant_prob, without_prob]
        top_idx = 0 if malignant_prob >= without_prob else 1
        self.top_label = self.labels[top_idx]
        self.top_prob = self.probs[top_idx]
        self.model_repo = "google/siglip-base-patch16-224"
        self.model_state = ModelState.PROXY_SIGLIP
        self.input_resolution = 224
        self.logits_shape = (1, 2)
        self.warnings = [
            "This score is from google/siglip-base-patch16-224 (proxy, NOT MedSigLIP)."
        ]


# --------------------------------------------------------------------------- #
# Helpers


def _post_screen(client: TestClient, b64: str) -> dict[str, Any]:
    resp = client.post(
        "/v1/screening/analyze",
        json={"dicom_bytes_b64": b64},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _install_medsiglip_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    malignant_prob: float,
    without_prob: float,
    gate_report: GateReport | None = None,
) -> _FakeMedSigLipResult:
    fake = _FakeMedSigLipResult(
        malignant_prob=malignant_prob,
        without_prob=without_prob,
        gate_report=gate_report
        or GateReport(
            repo_id="google/medsiglip-448",
            access_level=AccessLevel.ALLOWED,
            status_code=200,
            reason="preflight ok (HTTP 200)",
            has_token=True,
        ),
    )
    def _stub(preprocess_result: Any) -> _FakeMedSigLipResult:
        return fake
    monkeypatch.setattr(
        app_module, "_run_medsiglip_on_preprocessed", _stub, raising=True
    )
    return fake


def _install_medsiglip_gated(
    monkeypatch: pytest.MonkeyPatch,
    *,
    level: AccessLevel,
    status_code: int,
) -> None:
    err = GatedAccessError(
        repo_id="google/medsiglip-448",
        access_level=level,
        status_code=status_code,
        reason=f"stubbed gate denial (HTTP {status_code})",
    )
    def _stub(preprocess_result: Any) -> Any:
        raise err
    monkeypatch.setattr(
        app_module, "_run_medsiglip_on_preprocessed", _stub, raising=True
    )


def _install_proxy_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    malignant_prob: float,
    without_prob: float,
) -> _FakeProxyResult:
    fake = _FakeProxyResult(
        malignant_prob=malignant_prob, without_prob=without_prob
    )
    def _stub(preprocess_result: Any) -> _FakeProxyResult:
        return fake
    monkeypatch.setattr(
        app_module, "_run_siglip_proxy_on_preprocessed", _stub, raising=True
    )
    return fake


# --------------------------------------------------------------------------- #
# 1. MedSigLIP allowed → returns loaded_medsiglip, overall_score = probs[0]
# --------------------------------------------------------------------------- #


def test_medsiglip_allowed_takes_precedence_over_proxy(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    dicom_bytes_b64: str,
) -> None:
    """Happy path: gate ALLOWED, MedSigLIP result used, proxy never runs."""
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP", "1")
    # Set proxy=1 too so we can prove it does NOT run.
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY", "1")

    _install_medsiglip_success(
        monkeypatch, malignant_prob=7.976839697221294e-06, without_prob=1.390550164087963e-05
    )
    # Install a proxy stub that WOULD fail if called, to catch silent fallback.
    def _proxy_should_not_run(preprocess_result: Any) -> Any:
        raise AssertionError("proxy MUST NOT run when MedSigLIP succeeded")
    monkeypatch.setattr(
        app_module, "_run_siglip_proxy_on_preprocessed",
        _proxy_should_not_run, raising=True,
    )

    body = _post_screen(client, dicom_bytes_b64)
    assert body["provenance"]["model_state"] == "loaded_medsiglip"
    assert body["provenance"]["model_name"] == "google/medsiglip-448"
    assert body["overall_score"] == pytest.approx(7.976839697221294e-06)
    labels = [f["label"] for f in body["findings"]]
    assert labels == [
        "a mammogram showing a malignant mass",
        "a mammogram showing no mass",
    ]
    # Warnings list exists and is a list (may be empty if backend had none)
    assert isinstance(body["warnings"], list)


# --------------------------------------------------------------------------- #
# 2 + 3. MedSigLIP gate denied — NO silent proxy fallback
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "level,status_code",
    [
        (AccessLevel.FORBIDDEN, 403),
        (AccessLevel.UNAUTHENTICATED, 401),
    ],
)
def test_medsiglip_gate_denied_does_not_fall_back_to_proxy_by_default(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    dicom_bytes_b64: str,
    level: AccessLevel,
    status_code: int,
) -> None:
    """Regression guard for the silent-fallback bug.

    Pre-fix behaviour (WRONG): endpoint proceeded to the proxy and returned
    a proxy score under `model_state=loaded_medsiglip`, hiding the gate
    denial. Post-fix behaviour (RIGHT): with the proxy env NOT set, the
    endpoint returns `model_state=gated` with a `medsiglip_gated:...` warning.
    """
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP", "1")
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY", raising=False)
    _install_medsiglip_gated(monkeypatch, level=level, status_code=status_code)

    def _proxy_should_not_run(preprocess_result: Any) -> Any:
        raise AssertionError("proxy MUST NOT run when SIGLIP_PROXY flag is off")
    monkeypatch.setattr(
        app_module, "_run_siglip_proxy_on_preprocessed",
        _proxy_should_not_run, raising=True,
    )

    body = _post_screen(client, dicom_bytes_b64)
    assert body["provenance"]["model_state"] == "gated"
    assert body["overall_score"] is None
    assert body["findings"] == []
    assert any(
        w.startswith(f"medsiglip_gated:{level.value}:") for w in body["warnings"]
    ), body["warnings"]


# --------------------------------------------------------------------------- #
# 4. MedSigLIP gate denied + PROXY explicitly enabled → proxy runs
#    with BOTH warnings surfaced (gate reason AND proxy warning).
# --------------------------------------------------------------------------- #


def test_medsiglip_gate_denied_with_proxy_enabled_falls_back_with_both_warnings(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    dicom_bytes_b64: str,
) -> None:
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP", "1")
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY", "1")
    _install_medsiglip_gated(
        monkeypatch, level=AccessLevel.FORBIDDEN, status_code=403
    )
    _install_proxy_success(monkeypatch, malignant_prob=0.12, without_prob=0.88)

    body = _post_screen(client, dicom_bytes_b64)
    assert body["provenance"]["model_state"] == "proxy_siglip"
    assert body["provenance"]["model_name"] == "google/siglip-base-patch16-224"
    assert body["overall_score"] == pytest.approx(0.12)
    warnings = body["warnings"]
    # Must carry BOTH the gate denial AND the proxy honesty warning
    assert any(w.startswith("medsiglip_gated:forbidden:") for w in warnings), warnings
    assert any("proxy" in w.lower() or "siglip-base" in w for w in warnings), warnings


# --------------------------------------------------------------------------- #
# 5. MedSigLIP disabled + proxy enabled → proxy runs (no gate check).
# --------------------------------------------------------------------------- #


def test_medsiglip_disabled_proxy_still_works(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    dicom_bytes_b64: str,
) -> None:
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP", raising=False)
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY", "1")

    def _medsiglip_should_not_run(preprocess_result: Any) -> Any:
        raise AssertionError("MedSigLIP MUST NOT run when its flag is off")
    monkeypatch.setattr(
        app_module, "_run_medsiglip_on_preprocessed",
        _medsiglip_should_not_run, raising=True,
    )
    _install_proxy_success(monkeypatch, malignant_prob=0.34, without_prob=0.66)

    body = _post_screen(client, dicom_bytes_b64)
    assert body["provenance"]["model_state"] == "proxy_siglip"
    assert body["overall_score"] == pytest.approx(0.34)


# --------------------------------------------------------------------------- #
# 6. Both backends disabled → placeholder envelope.
# --------------------------------------------------------------------------- #


def test_all_backends_disabled_returns_placeholder(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    dicom_bytes_b64: str,
) -> None:
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP", raising=False)
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY", raising=False)

    def _should_not_run(preprocess_result: Any) -> Any:
        raise AssertionError("no backend should run when both env flags are off")
    monkeypatch.setattr(
        app_module, "_run_medsiglip_on_preprocessed", _should_not_run, raising=True
    )
    monkeypatch.setattr(
        app_module, "_run_siglip_proxy_on_preprocessed", _should_not_run, raising=True
    )

    body = _post_screen(client, dicom_bytes_b64)
    assert body["provenance"]["model_state"] == "placeholder"
    assert body["overall_score"] is None
    assert body["findings"] == []
    # Placeholder still emits the arbiter block per §4a
    assert body["arbiter_score"] is not None


# --------------------------------------------------------------------------- #
# 7. MedSigLIP score threads into the arbiter — same discretisation contract.
# --------------------------------------------------------------------------- #


def test_medsiglip_score_present_alongside_arbiter_block(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    dicom_bytes_b64: str,
) -> None:
    """Confirms overall_score and arbiter_score coexist without collision.

    The screening arbiter is a template with n_training=0 and empty features
    so its p_positive falls back to the intercept — decoupled from
    overall_score for now. This test locks in that both blocks are present
    and honest, so downstream refactors that feed MedSigLIP scores into the
    arbiter can be added deliberately.
    """
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP", "1")
    _install_medsiglip_success(
        monkeypatch, malignant_prob=0.42, without_prob=0.58
    )

    body = _post_screen(client, dicom_bytes_b64)
    assert body["overall_score"] == pytest.approx(0.42)
    ab = body["arbiter_score"]
    assert ab is not None
    assert ab["model_name"] == "screening_arbiter_template_v0"
    assert ab["model_state"] == "template"
    assert 0.0 <= ab["p_positive"] <= 1.0
    assert ab["risk_bucket"] in {"LOW", "MID", "HIGH"}


# --------------------------------------------------------------------------- #
# 8. ModelState.LOADED_MEDSIGLIP is distinct from PROXY_SIGLIP + PLACEHOLDER.
# --------------------------------------------------------------------------- #


def test_model_state_loaded_medsiglip_is_distinct() -> None:
    """Structural test — no HTTP call — that the enum entries collide-proof."""
    assert ModelState.LOADED_MEDSIGLIP.value == "loaded_medsiglip"
    assert ModelState.LOADED_MEDSIGLIP.value != ModelState.PROXY_SIGLIP.value
    assert ModelState.LOADED_MEDSIGLIP.value != ModelState.PLACEHOLDER.value
    assert ModelState.LOADED_MEDSIGLIP.value != ModelState.GATED.value


# --------------------------------------------------------------------------- #
# 9. Gate report content is logged (via response warnings prefix contract).
# --------------------------------------------------------------------------- #


def test_gated_response_warning_encodes_level_and_reason(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    dicom_bytes_b64: str,
) -> None:
    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP", "1")
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_SIGLIP_PROXY", raising=False)
    _install_medsiglip_gated(
        monkeypatch, level=AccessLevel.UNAUTHENTICATED, status_code=401
    )

    body = _post_screen(client, dicom_bytes_b64)
    warnings = body["warnings"]
    assert any(
        w.startswith("medsiglip_gated:unauthenticated:") and "HTTP 401" in w
        for w in warnings
    ), warnings
    # And the envelope's model_state is `gated`, not `unavailable`.
    assert body["provenance"]["model_state"] == "gated"
