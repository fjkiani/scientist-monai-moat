"""Unit tests for POST /v1/elo/rank (v0.3.0-alpha).

Contract under test:

* Response envelope shape matches ``EloRankResponse`` (contract_version,
  baseline_ranking, enriched_ranking, matches, disease_context echo,
  applied_modifiers echo, n_candidates, warnings, provenance).
* Ranking is deterministic given the same seed + input.
* Duplicate drug_id in the request payload → HTTP 400 (validation refuses
  degenerate tournaments up front).
* Unknown modifier keys (drug_id not in the drugs list) surface as
  ``unknown_modifier_drug_id:<key>`` warnings, do NOT crash, do NOT get applied.
* Modifier reason strings for the canonical cases (HRD+PARP boost,
  PD-L1 CPS<10 penalty, GOG-0218 posture) are emitted on the corresponding
  match records so the SPA can render honest hover-cards.
* Confidence in the response is clamped to [0, 1] for display, but the
  underlying rank order reflects the raw uplift (regression guard against
  the tie-break bug that had olaparib_maintenance losing to niraparib
  after both hit the confidence ceiling).

No network I/O — pure in-process TestClient. AUTH_MODE=off comes from
conftest.py; no key headers needed.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app


HGSOC_DISEASE_CONTEXT = {
    "cancer": "hgsoc",
    "stage": "IV",
    "hrd_positive": True,
    "brca_mutated": True,
    "pd_l1_cps": 5,
    "prior_lines": 2,
}


def _drug(drug_id: str, regimen: str, line: int, conf: float, evidence=None):
    return {
        "drug_id": drug_id,
        "regimen": regimen,
        "line": line,
        "confidence": conf,
        "evidence": evidence or [],
        "honesty_markers": {},
    }


def _ev(url: str, quoted_text: str, source: str = "pubmed"):
    return {"url": url, "quoted_text": quoted_text, "source": source}


def _mini_hgsoc_bundle():
    """A trimmed 4-drug HGSOC bundle used in most tests. Full 10-drug
    bundle lives in the baked demo_samples fallback; the unit tests only
    need enough drugs to exercise every code branch (PARP + bev + PD-L1
    checkpoint + a plain penalty target)."""
    return [
        _drug(
            "olaparib_maintenance",
            "Olaparib 300mg BID maintenance",
            1,
            0.85,
            evidence=[
                _ev(
                    "https://pubmed.ncbi.nlm.nih.gov/30345884/",
                    "SOLO1: 60% reduction in progression risk with olaparib maintenance in BRCA-mutated advanced ovarian.",
                )
            ],
        ),
        _drug(
            "bevacizumab_gog0218",
            "Bevacizumab (GOG-0218 first-line)",
            1,
            0.70,
            evidence=[
                _ev(
                    "https://www.nejm.org/doi/full/10.1056/NEJMoa1104390",
                    "GOG-0218: bevacizumab addition improved PFS in advanced ovarian.",
                )
            ],
        ),
        _drug(
            "pembrolizumab_keynote100",
            "Pembrolizumab Q3W",
            2,
            0.55,
            evidence=[],
        ),
        _drug(
            "topotecan_monotherapy",
            "Topotecan monotherapy",
            4,
            0.30,
            evidence=[],
        ),
    ]


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_elo_rank_endpoint_listed_in_health(client):
    r = client.get("/health")
    assert r.status_code == 200, r.text
    endpoints = r.json()["endpoints"]
    assert "POST /v1/elo/rank" in endpoints, (
        f"/v1/elo/rank must be advertised in /health.endpoints, got: {endpoints}"
    )


def test_elo_rank_happy_path_envelope_shape(client):
    payload = {
        "drugs": _mini_hgsoc_bundle(),
        "modifiers": {
            "olaparib_maintenance": 0.30,
            "bevacizumab_gog0218": 0.15,
            "pembrolizumab_keynote100": -0.20,
        },
        "disease_context": HGSOC_DISEASE_CONTEXT,
        "k_factor": 16,
        "seed": 20260703,
    }
    r = client.post("/v1/elo/rank", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()

    # ApiEnvelope guarantees these
    assert body["contract_version"] == "v0.3.0-alpha"
    assert "provenance" in body and body["provenance"]["model_name"].startswith(
        "oa/co_scientist_elo"
    )
    assert body["provenance"]["model_state"] in ("proxy_co_scientist", "PROXY_CO_SCIENTIST")

    # Elo-specific shape
    assert body["n_candidates"] == 4
    assert len(body["baseline_ranking"]) == 4
    assert len(body["enriched_ranking"]) == 4
    assert len(body["matches"]) == 4  # one match record per drug

    # disease_context + applied_modifiers echoed back for the SPA
    assert body["disease_context"]["cancer"] == "hgsoc"
    assert body["applied_modifiers"]["olaparib_maintenance"] == 0.30

    # Every ranked entry has the expected fields
    for entry in body["enriched_ranking"]:
        for key in ("rank", "drug_id", "regimen", "line", "rating", "confidence", "wins", "losses", "draws"):
            assert key in entry, entry


def test_elo_rank_is_deterministic(client):
    payload = {
        "drugs": _mini_hgsoc_bundle(),
        "modifiers": {
            "olaparib_maintenance": 0.30,
            "pembrolizumab_keynote100": -0.20,
        },
        "disease_context": HGSOC_DISEASE_CONTEXT,
        "k_factor": 16,
        "seed": 20260703,
    }
    r1 = client.post("/v1/elo/rank", json=payload).json()
    r2 = client.post("/v1/elo/rank", json=payload).json()
    order1 = [e["drug_id"] for e in r1["enriched_ranking"]]
    order2 = [e["drug_id"] for e in r2["enriched_ranking"]]
    assert order1 == order2
    ratings1 = [round(e["rating"], 4) for e in r1["enriched_ranking"]]
    ratings2 = [round(e["rating"], 4) for e in r2["enriched_ranking"]]
    assert ratings1 == ratings2


def test_elo_rank_duplicate_drug_id_400(client):
    bundle = _mini_hgsoc_bundle()
    bundle.append(_drug("olaparib_maintenance", "Olaparib dup", 1, 0.80))
    r = client.post(
        "/v1/elo/rank",
        json={
            "drugs": bundle,
            "modifiers": {},
            "disease_context": HGSOC_DISEASE_CONTEXT,
        },
    )
    assert r.status_code == 400, r.text
    assert "duplicate" in r.text.lower() or "olaparib_maintenance" in r.text


def test_elo_rank_unknown_modifier_warns(client):
    r = client.post(
        "/v1/elo/rank",
        json={
            "drugs": _mini_hgsoc_bundle(),
            "modifiers": {"totally_made_up_drug": 0.5},
            "disease_context": HGSOC_DISEASE_CONTEXT,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    warnings = body.get("warnings") or []
    assert any(
        "unknown_modifier_drug_id" in w and "totally_made_up_drug" in w for w in warnings
    ), warnings
    # And the unknown key must NOT have been applied to any real drug
    for match in body["matches"]:
        assert match["applied_modifier"] == body["applied_modifiers"].get(
            match["drug_id"], 0.0
        )


def test_elo_rank_hrd_parp_boost_reason(client):
    r = client.post(
        "/v1/elo/rank",
        json={
            "drugs": _mini_hgsoc_bundle(),
            "modifiers": {"olaparib_maintenance": 0.30},
            "disease_context": HGSOC_DISEASE_CONTEXT,
        },
    )
    assert r.status_code == 200, r.text
    matches = {m["drug_id"]: m for m in r.json()["matches"]}
    reason = matches["olaparib_maintenance"]["reason"].lower()
    assert "hrd" in reason and "parp" in reason and "+0.30" in matches[
        "olaparib_maintenance"
    ]["reason"]


def test_elo_rank_pdl1_cps_low_penalty_reason(client):
    r = client.post(
        "/v1/elo/rank",
        json={
            "drugs": _mini_hgsoc_bundle(),
            "modifiers": {"pembrolizumab_keynote100": -0.20},
            "disease_context": HGSOC_DISEASE_CONTEXT,  # pd_l1_cps=5
        },
    )
    assert r.status_code == 200, r.text
    matches = {m["drug_id"]: m for m in r.json()["matches"]}
    reason = matches["pembrolizumab_keynote100"]["reason"]
    assert "PD-L1" in reason and "CPS" in reason and "-0.20" in reason


def test_elo_rank_bevacizumab_gog0218_reason(client):
    r = client.post(
        "/v1/elo/rank",
        json={
            "drugs": _mini_hgsoc_bundle(),
            "modifiers": {"bevacizumab_gog0218": 0.15},
            "disease_context": HGSOC_DISEASE_CONTEXT,
        },
    )
    assert r.status_code == 200, r.text
    matches = {m["drug_id"]: m for m in r.json()["matches"]}
    reason = matches["bevacizumab_gog0218"]["reason"]
    assert "GOG-0218" in reason or "Bevacizumab" in reason
    assert "+0.15" in reason


def test_elo_rank_confidence_clamped_but_order_preserved(client):
    """Regression guard: prior bug clamped confidence BEFORE ranking so two
    PARP inhibitors that both hit the [0,1] ceiling tie-broke by drug_id,
    incorrectly demoting olaparib_maintenance (higher baseline confidence)
    below niraparib_maintenance. Fix passes the raw score through Elo and
    clamps only at serialization time. This test asserts (a) displayed
    confidence stays in [0, 1], and (b) olaparib_maintenance still outranks
    niraparib_maintenance when both would over-saturate.
    """
    drugs = [
        _drug("olaparib_maintenance", "Olaparib 300mg BID", 1, 0.85),
        _drug("niraparib_maintenance", "Niraparib 200-300mg QD", 1, 0.75),
        _drug("topotecan_monotherapy", "Topotecan mono", 4, 0.30),
    ]
    modifiers = {
        "olaparib_maintenance": 0.30,  # 0.85 + 0.30 = 1.15 → clamped to 1.00 for display
        "niraparib_maintenance": 0.25,  # 0.75 + 0.25 = 1.00 → also at ceiling for display
    }
    r = client.post(
        "/v1/elo/rank",
        json={
            "drugs": drugs,
            "modifiers": modifiers,
            "disease_context": HGSOC_DISEASE_CONTEXT,
        },
    )
    assert r.status_code == 200, r.text
    ranking = r.json()["enriched_ranking"]
    order = [e["drug_id"] for e in ranking]
    for entry in ranking:
        assert 0.0 <= entry["confidence"] <= 1.0, entry
    assert order.index("olaparib_maintenance") < order.index(
        "niraparib_maintenance"
    ), f"olaparib_maintenance must outrank niraparib_maintenance even when both saturate; got {order}"


def test_elo_rank_demo_sample_fallback_is_baked(client):
    """The SPA demo fallback path — /v1/demo/samples/elo_rank — must resolve
    to the baked JSON when demo mode is enabled at build time. We don't
    force demo mode on in unit tests (conftest sets AUTH_MODE=off, not
    DEMO_MODE), so instead we assert the baked file exists on disk with
    the right contract_version. Live-service smoke of the endpoint is a
    Render-side check, not a pytest concern.
    """
    from pathlib import Path
    import json

    baked = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "oncology_arbiter"
        / "api"
        / "static"
        / "demo_samples"
        / "elo_rank.json"
    )
    assert baked.is_file(), f"baked demo sample missing: {baked}"
    body = json.loads(baked.read_text())
    assert body["contract_version"] == "v0.3.0-alpha"
    assert body["n_candidates"] >= 4
    assert len(body["enriched_ranking"]) == body["n_candidates"]
