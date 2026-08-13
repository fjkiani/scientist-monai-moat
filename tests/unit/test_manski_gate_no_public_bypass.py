"""The Manski gate has no public bypass, and blind inference is scope-locked.

These tests encode the audit finding that produced the gate. The arbiter used
to report ``p_lower_bound == p_upper_bound`` whenever no *boolean* was missing,
because ``L2LogisticArbiter.score`` tracked ``missing_bools`` and nothing else.
An absent one-hot silently encodes to the reference level and an absent
continuous silently encodes to ``0.0``; neither was recorded. On the therapy
template that made the interval fail-OPEN on 5 of 10 declared inputs — dropping
``grade`` reported a zero-width interval around 0.119834 when the assumption-free
set is actually [0.039392, 0.378952], and dropping ``ki67`` reported a zero-width
interval when the set is the whole of [0, 1].

Three properties are pinned here:

* a public route never emits a point estimate whose identified set is wider
  than ``MANSKI_MAX_WIDTH``;
* nothing in the request — header, query parameter, or body field — changes
  that, and the bypass machinery is *structurally* absent rather than
  defaulted off;
* blind inference exists only behind the ``research:blind_inference`` scope,
  including when ``ONCOLOGY_ARBITER_AUTH_MODE=off``.

Every expected width below was measured against the frozen artefact, not
guessed. Dropping ``histology`` costs 0.147815 and is *released*: a test that
demanded a 422 there would be pinning the wrong behaviour.
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app
from oncology_arbiter.arbiter import load_arbiter
from oncology_arbiter.arbiter.manski import (
    MANSKI_ERROR_CODE,
    MANSKI_MAX_WIDTH,
    RESEARCH_BLIND_INFERENCE_SCOPE,
    UNBOUNDED_ERROR_CODE,
    ManskiBounds,
    enforce_manski_gate,
)
from oncology_arbiter.auth import APIKeyDB


SRC = Path(__file__).resolve().parents[2] / "src" / "oncology_arbiter"

BASE_PANEL: dict[str, Any] = {
    "histology": "invasive_ductal",
    "grade": 2,  # int Literal[1,2,3] on the wire model, NOT the string "2"
    "er_positive": True,
    "pr_positive": True,
    "her2_positive": False,
    "ki67_pct": 20.0,
    "tumor_size_mm": 22.0,
    "lymph_nodes_pos": 0,
    "brca_pathogenic": False,
    "age_years": 58.0,
}

# Raw values under the `*_norm` keys: the artefact declares the divisor
# ("ki67_pct / 100.0") and applies it itself. See
# test_continuous_features_take_raw_values_not_pre_normalised.
BASE_FEATURES: dict[str, Any] = {
    "histology": "invasive_ductal",
    "grade": "2",
    "er_status_positive": True,
    "pr_status_positive": True,
    "her2_status_positive": False,
    "ki67_norm": 20.0,
    "tumor_size_norm": 22.0,
    "node_status_positive": False,
    "brca_status_known_pathogenic": False,
    "age_at_diagnosis_norm": 58.0,
}

# (case, dropped wire fields, expected gate code or None, measured width)
GATE_CASES: list[tuple[str, list[str], str | None, float]] = [
    ("complete", [], None, 0.000000),
    ("drop_pr", ["pr_positive"], None, 0.095615),
    ("drop_onehot_histology", ["histology"], None, 0.147815),
    ("drop_onehot_grade", ["grade"], MANSKI_ERROR_CODE, 0.339560),
    ("drop_er_her2", ["er_positive", "her2_positive"], MANSKI_ERROR_CODE, 0.468617),
    ("drop_all_bools",
     ["er_positive", "pr_positive", "her2_positive", "lymph_nodes_pos",
      "brca_pathogenic"],
     MANSKI_ERROR_CODE, 0.826868),
    ("drop_continuous_ki67", ["ki67_pct"], UNBOUNDED_ERROR_CODE, 1.000000),
    ("drop_continuous_size", ["tumor_size_mm"], UNBOUNDED_ERROR_CODE, 1.000000),
    ("drop_continuous_age", ["age_years"], UNBOUNDED_ERROR_CODE, 1.000000),
]

ROUTES: list[tuple[str, Any]] = [
    ("/v1/therapy/reason",
     lambda p: {"patient_context": {"cancer_type": "breast"}, "therapy_features": p}),
    ("/v1/tumor_board/dynamic", lambda p: {"cancer": "breast", "therapy_features": p}),
    ("/v1/case/full", lambda p: {"therapy_features": p}),
]


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    import os
    # 30/minute in production; this module fires far more than that and a 429
    # would silently masquerade as a passed bypass test.
    os.environ["ONCOLOGY_ARBITER_RATE_LIMIT"] = "100000/minute"
    return TestClient(create_app())


def _panel(drop: list[str]) -> dict[str, Any]:
    return {k: v for k, v in BASE_PANEL.items() if k not in drop}


def _gate_error(body: dict[str, Any]) -> str | None:
    """Gate 422s carry `error`; FastAPI schema 422s carry `detail`.

    Conflating the two is how a gate test passes against a broken gate.
    """
    err = body.get("error")
    if isinstance(err, str):
        return err
    if isinstance(err, dict):
        return err.get("code") or err.get("error")
    return None


def _find_score(body: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("arbiter_score", "therapy_triage"):
        v = body.get(key)
        if isinstance(v, dict):
            return v
    for v in body.values():
        if isinstance(v, dict):
            found = _find_score(v)
            if found is not None:
                return found
    return None


# --------------------------------------------------------------------------- #
# 1. The gate itself


@pytest.mark.parametrize("route,build", ROUTES, ids=[r[0] for r in ROUTES])
@pytest.mark.parametrize(
    "case,drop,expect,width", GATE_CASES, ids=[c[0] for c in GATE_CASES]
)
def test_public_route_gate(client, route, build, case, drop, expect, width):
    r = client.post(route, json=build(_panel(drop)))
    assert r.status_code != 429, "rate limited: the test is measuring the limiter"
    body = r.json()

    if expect is None:
        assert r.status_code == 200, (
            f"{case} has measured width {width} <= {MANSKI_MAX_WIDTH} and must be "
            f"released, got {r.status_code} {body}"
        )
        score = _find_score(body)
        assert score is not None and score.get("manski") is not None
        assert score["manski"]["identified"] is True
        assert score["manski"]["width"] == pytest.approx(width, abs=1e-5)
        assert score["manski"]["width"] <= MANSKI_MAX_WIDTH
        return

    assert r.status_code == 422, f"{case}: expected 422, got {r.status_code} {body}"
    code = _gate_error(body)
    assert code is not None, (
        f"{case}: 422 carried no `error` key, so it came from FastAPI schema "
        f"validation rather than the gate: {body}"
    )
    assert code == expect
    assert body["width"] == pytest.approx(width, abs=1e-5)
    assert body["public_bypass_available"] is False
    assert body["missing_features"], "a gated response must name what is missing"
    if expect == UNBOUNDED_ERROR_CODE:
        assert body["bounds"] == [0.0, 1.0]
        assert body["unobserved_unbounded"], (
            "an unbounded verdict must name the covariates with no declared support"
        )


def test_gate_never_emits_zero_width_for_missing_continuous():
    """The specific fail-open the audit found: absent continuous read as 0.0."""
    arb = load_arbiter("therapy")
    for feat in ("ki67_norm", "tumor_size_norm", "age_at_diagnosis_norm"):
        f = {k: v for k, v in BASE_FEATURES.items() if k != feat}
        result = arb.score(f)
        # The arbiter's own interval is still degenerate — that is the bug.
        assert result.p_lower_bound == result.p_upper_bound
        # The gate must not inherit it.
        b = ManskiBounds.from_arbiter(
            stage="therapy", arbiter=arb, features=f, result=result
        )
        assert b.width == 1.0
        assert (b.lower, b.upper) == (0.0, 1.0)
        assert feat in b.unobserved_unbounded
        with pytest.raises(Exception) as exc:
            enforce_manski_gate(b)
        assert exc.value.error_code == UNBOUNDED_ERROR_CODE


def test_unbounded_bounds_object_is_unconstructible_with_narrow_interval():
    """Make the forbidden state unrepresentable, not merely unreachable."""
    with pytest.raises(ValueError):
        ManskiBounds(
            stage="therapy", model_name="x", point_estimate=0.5,
            lower=0.4, upper=0.6, max_width=MANSKI_MAX_WIDTH,
            missing_features=("ki67_norm",),
            unobserved_unbounded=("ki67_norm",),
        )


# --------------------------------------------------------------------------- #
# 2. No bypass — structural, not defaulted


def test_enforce_manski_gate_accepts_no_bypass_argument():
    """A default-off flag is one careless keyword away from being on."""
    params = list(inspect.signature(enforce_manski_gate).parameters)
    assert params == ["bounds"], (
        f"enforce_manski_gate must take only the bounds; found {params}"
    )


BANNED_BYPASS_SYMBOLS = {
    "allow_blind_inference",
    "BLIND_INFERENCE_HEADER",
    "parse_blind_inference_header",
}
BANNED_BYPASS_HEADERS = {"x-allow-blind-inference", "x-manski-override"}


def _live_identifiers_and_strings(path: Path) -> tuple[set[str], set[str]]:
    """Identifiers and string *literals* that are not docstrings.

    A plain substring scan fails here: `manski.py` discusses
    ``allow_blind_inference`` in prose, explaining why the parameter was
    removed. Banning the word in comments would force the reason for the
    design to be deleted along with the code, which is the opposite of what
    an audit trail wants. So the ban applies to live code only.
    """
    import ast as _ast
    tree = _ast.parse(path.read_text())
    docstrings = set()
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.Module, _ast.FunctionDef, _ast.AsyncFunctionDef,
                             _ast.ClassDef)):
            ds = _ast.get_docstring(node, clean=False)
            if ds:
                docstrings.add(ds)
    names: set[str] = set()
    strings: set[str] = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name):
            names.add(node.id)
        elif isinstance(node, _ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, _ast.arg):
            names.add(node.arg)
        elif isinstance(node, _ast.keyword) and node.arg:
            names.add(node.arg)
        elif isinstance(node, _ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                strings.add(node.value)
    return names, strings


@pytest.mark.parametrize("module", ["api/app.py", "arbiter/manski.py"])
def test_bypass_symbols_absent_from_live_code(module):
    names, strings = _live_identifiers_and_strings(SRC / module)
    leaked = BANNED_BYPASS_SYMBOLS & (names | strings)
    assert not leaked, f"{module} still uses bypass symbols {sorted(leaked)}"
    lowered = {s.lower() for s in strings}
    leaked_headers = BANNED_BYPASS_HEADERS & lowered
    assert not leaked_headers, (
        f"{module} still carries bypass header literals {sorted(leaked_headers)}"
    )


GATED = {"patient_context": {"cancer_type": "breast"},
         "therapy_features": _panel(["er_positive", "her2_positive"])}


@pytest.mark.parametrize("header", [
    "X-Allow-Blind-Inference", "x-allow-blind-inference", "X-Manski-Override",
    "X-Debug-Bypass", "X-Research-Mode", "Allow-Blind-Inference",
    "X-Blind-Inference", "X-Manski-Max-Width", "X-Override-Gate",
])
@pytest.mark.parametrize("value", ["true", "TRUE", "1", " yes ", "on", "TrUe"])
def test_no_header_bypasses_the_gate(client, header, value):
    r = client.post("/v1/therapy/reason", json=GATED, headers={header: value})
    assert r.status_code == 422
    assert _gate_error(r.json()) == MANSKI_ERROR_CODE


@pytest.mark.parametrize("qs", [
    "?allow_blind_inference=true",
    "?allow_blind_inference=1&manski_max_width=1.0",
    "?manski_max_width=0.99",
    "?bypass=true&debug=1",
])
def test_no_query_parameter_bypasses_the_gate(client, qs):
    r = client.post(f"/v1/therapy/reason{qs}", json=GATED)
    assert r.status_code == 422
    assert _gate_error(r.json()) == MANSKI_ERROR_CODE


@pytest.mark.parametrize("extra", [
    {"allow_blind_inference": True},
    {"manski_max_width": 1.0},
    {"bypass_manski": True},
])
def test_no_body_field_bypasses_the_gate(client, extra):
    r = client.post("/v1/therapy/reason", json={**GATED, **extra})
    assert r.status_code != 200, f"body field {list(extra)} produced a 200"


# --------------------------------------------------------------------------- #
# 3. Research route: scope-locked, receipted


def _scoped_client(tmp_path, monkeypatch, scopes):
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_DB_PATH", str(tmp_path / "t.sqlite"))
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("ONCOLOGY_ARBITER_RATE_LIMIT", "100000/minute")
    # DEFAULT_DB is bound at import time; point the module at the temp file.
    import oncology_arbiter.auth.api_key as ak
    monkeypatch.setattr(ak, "DEFAULT_DB", tmp_path / "t.sqlite")
    db = APIKeyDB(tmp_path / "t.sqlite")
    raw, _ = db.issue("t", scopes=scopes)
    return TestClient(create_app()), raw


RR = "/v1/research/arbiter/identified_set"
PROBE = {
    "arbiter": "therapy",
    "features": {k: v for k, v in BASE_FEATURES.items()
                 if k not in ("er_status_positive", "her2_status_positive")},
    "acknowledge_not_for_clinical_use": True,
}


def test_research_route_closed_to_anonymous_even_with_auth_off(
    tmp_path, monkeypatch
):
    """AUTH_MODE=off must not hand out privileged scopes.

    This is the failure mode the switch exists to prevent: somebody disables
    auth for a demo and every privileged route opens with it.
    """
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "off")
    c, _ = _scoped_client(tmp_path, monkeypatch, [])
    r = c.post(RR, json=PROBE)
    assert r.status_code == 403
    assert r.json()["detail"]["error"] == "InsufficientScope"
    assert r.json()["detail"]["granted_scopes"] == []


def test_research_route_rejects_key_without_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "on")
    c, raw = _scoped_client(tmp_path, monkeypatch, [])
    r = c.post(RR, json=PROBE, headers={"X-API-Key": raw})
    assert r.status_code == 403
    assert r.json()["detail"]["required_scope"] == RESEARCH_BLIND_INFERENCE_SCOPE


def test_research_route_requires_explicit_acknowledgement(tmp_path, monkeypatch):
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "on")
    c, raw = _scoped_client(tmp_path, monkeypatch, [RESEARCH_BLIND_INFERENCE_SCOPE])
    r = c.post(RR, json={**PROBE, "acknowledge_not_for_clinical_use": False},
               headers={"X-API-Key": raw})
    assert r.status_code == 422


def test_research_route_releases_with_bounds_warnings_and_receipt(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "on")
    c, raw = _scoped_client(tmp_path, monkeypatch, [RESEARCH_BLIND_INFERENCE_SCOPE])
    r = c.post(RR, json=PROBE, headers={"X-API-Key": raw})
    assert r.status_code == 200
    b = r.json()

    # The number the public route refuses.
    assert b["point_estimate"] == pytest.approx(0.311310, abs=1e-6)
    assert b["bounds"]["lower"] == pytest.approx(0.155251, abs=1e-6)
    assert b["bounds"]["upper"] == pytest.approx(0.623868, abs=1e-6)
    assert b["bounds"]["identified"] is False
    assert b["would_have_been_rejected"] is True
    assert b["public_route_error_code"] == MANSKI_ERROR_CODE
    assert b["not_for_clinical_use"] is True

    # It cannot be returned naked: raw bounds, missing features, warnings.
    assert set(b["missing_features"]) == {
        "er_status_positive", "her2_status_positive"}
    assert b["warnings"], "an unidentified release must carry warnings"
    assert any("NOT_FOR_CLINICAL_USE" in w for w in b["warnings"])
    assert any("identified_set" in w for w in b["warnings"])

    # And it is receipted.
    rec = b["audit_receipt"]
    assert rec["required_scope"] == RESEARCH_BLIND_INFERENCE_SCOPE
    assert rec["route"] == RR
    assert rec["not_for_clinical_use"] is True
    assert rec["public_route_would_have_returned"] == {
        "status": 422, "error": MANSKI_ERROR_CODE}
    assert re.fullmatch(r"[0-9a-f]{64}", rec["request_features_sha256"])

    # The receipt reached the ledger, not just the response body.
    audit_root = Path(tmp_path / "audit")
    entries = [
        json.loads(line)
        for p in audit_root.rglob("*.jsonl")
        for line in p.read_text().splitlines()
    ]
    blind = [e for e in entries if e["endpoint"] == RR]
    assert len(blind) >= 1
    assert blind[0]["extra"]["blind_inference_receipt"]["not_for_clinical_use"] is True


def test_research_route_unbounded_case_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("ONCOLOGY_ARBITER_AUTH_MODE", "on")
    c, raw = _scoped_client(tmp_path, monkeypatch, [RESEARCH_BLIND_INFERENCE_SCOPE])
    feats = {k: v for k, v in BASE_FEATURES.items() if k != "ki67_norm"}
    r = c.post(RR, json={"arbiter": "therapy", "features": feats,
                         "acknowledge_not_for_clinical_use": True},
               headers={"X-API-Key": raw})
    assert r.status_code == 200
    b = r.json()
    assert b["public_route_error_code"] == UNBOUNDED_ERROR_CODE
    assert [b["bounds"]["lower"], b["bounds"]["upper"]] == [0.0, 1.0]
    assert b["unobserved_unbounded"] == ["ki67_norm"]
    assert any("zero_information" in w for w in b["warnings"])


# --------------------------------------------------------------------------- #
# 4. The unit contract the gate depends on


def test_continuous_features_take_raw_values_not_pre_normalised():
    """Pin the raw-input contract for `*_norm` keys.

    The key is named ``ki67_norm`` but ``feature_encodings["ki67_norm"]`` is
    the string ``"ki67_pct / 100.0"`` and ``_encode_continuous`` applies the
    divisor, so the caller passes the RAW percentage. Passing an already
    normalised 0.20 is silently accepted and attenuates the contribution
    100-fold. No exception, no warning, no range validation — the probability
    just moves 35% and stays plausible.

    This test exists so that nobody "fixes" the confusing name by making the
    artefact accept pre-normalised values, which would silently re-scale every
    stored coefficient and every Manski endpoint derived from them.
    """
    arb = load_arbiter("therapy")
    raw = arb.score(BASE_FEATURES)
    pre = arb.score(dict(BASE_FEATURES, ki67_norm=0.20, tumor_size_norm=0.44,
                         age_at_diagnosis_norm=0.58))

    assert raw.p_positive == pytest.approx(0.155251, abs=1e-6)
    assert pre.p_positive == pytest.approx(0.100763, abs=1e-6)
    # 0.9 * (20.0 / 100.0) == 0.18, not 0.9 * 0.20 == 0.18/100 * 100 ...
    assert raw.term_contributions["ki67_norm"] == pytest.approx(0.18, abs=1e-9)
    assert pre.term_contributions["ki67_norm"] == pytest.approx(0.0018, abs=1e-9)
    # Neither call raised or warned: the trap is silent.
    assert raw.missing_features == [] and pre.missing_features == []
