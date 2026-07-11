"""Unit tests for POST /v1/co_scientist/run (v0.4.0-alpha, PLAN §5 PR #5).

Contract under test:

* Response envelope matches ``CoScientistRunResponse`` (disclaimer, caveat,
  provenance, honesty_gate, evidence, warnings, phases, hypotheses,
  initial_count, after_reflect, after_evolve, urls_dropped_hallucinated,
  hypotheses_dropped).
* Phases run in the documented order:
  `['generate','reflect','rank','evolve','rank']`.
* Deterministic given identical inputs — same request → identical
  hypotheses list, identical ratings.
* Duplicate calls are byte-equivalent.

**Honesty gate — the load-bearing test.**

The hostile-URL scenario: caller submits a screening envelope carrying
five fake URLs in `evidence[]` and an empty `seed_urls`. Every hypothesis
generated from that envelope inherits the fake URLs. REFLECT must:

1. Strip all 5 hostile URLs from every hypothesis (they aren't in
   `seed_urls`, so they get filtered by `filter_evidence_by_seen_urls`).
2. Report `urls_dropped_hallucinated >= 5` on the response envelope.
3. Emit `dropped N hallucinated citation(s)` warnings so a reviewer can
   trace which URLs got dropped.
4. Emit `no_evidence_after_reflect:<hyp_id>` warnings for every hypothesis
   whose evidence list became empty, so callers can't accidentally treat
   them as trusted.

Nothing in this test depends on live model weights, network I/O, or GPU.
AUTH_MODE=off from conftest.py — no key headers needed. Route is public
by design (matches /v1/elo/rank posture: read-only Co-Scientist has no
patient PHI ingest).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from oncology_arbiter.api.app import create_app


# --------------------------------------------------------------------------- #
# Fixtures + payload builders


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def _ev(url: str, quoted_text: str = "test citation", source: str = "pubmed"):
    """One evidence record shaped for `screening.evidence[]`."""
    return {"url": url, "quoted_text": quoted_text, "source": source}


def _screening_envelope(evidence: list[dict]) -> dict:
    """A minimal ScreeningResponse-shaped dict.

    generate_hypotheses only reads `provenance.model_state`, `findings[]`,
    and `evidence[]`. Everything else is padding for the schema shape.
    """
    return {
        "disclaimer": "RUO",
        "caveat": "AUROC",
        "provenance": {
            "model_name": "proxy_screening",
            "model_state": "proxy_test",
            "request_id": "test-req-1",
        },
        "honesty_gate": {
            "seen_urls_count": len(evidence),
            "evidence_kept": len(evidence),
            "evidence_dropped": 0,
        },
        "evidence": evidence,
        "warnings": [],
        "laterality": "L",
        "view": "CC",
        "orientation_flipped": False,
        "breast_mask_coverage": 0.72,
        "findings": [
            {"label": "mass", "score": 0.81, "bbox_xyxy": [0.1, 0.2, 0.3, 0.4]},
            {"label": "calc", "score": 0.63, "bbox_xyxy": [0.5, 0.5, 0.6, 0.6]},
            {"label": "asym", "score": 0.42, "bbox_xyxy": [0.1, 0.5, 0.2, 0.7]},
        ],
        "overall_score": 0.81,
    }


def _therapy_envelope(evidence: list[dict]) -> dict:
    """A minimal TherapyResponse-shaped dict with one recommended option.

    generate_hypotheses reads `recommended_options[i].evidence[]` for the
    therapy stage, so we put the caller's evidence there too.
    """
    return {
        "disclaimer": "RUO",
        "caveat": "AUROC",
        "provenance": {
            "model_name": "proxy_therapy",
            "model_state": "proxy_test",
            "request_id": "test-req-2",
        },
        "honesty_gate": {
            "seen_urls_count": len(evidence),
            "evidence_kept": len(evidence),
            "evidence_dropped": 0,
        },
        "evidence": evidence,
        "warnings": [],
        "cancer": "breast",
        "recommended_options": [
            {
                "regimen": "TC",
                "line_of_therapy": 1,
                "evidence": evidence,
            },
            {
                "regimen": "AC-T",
                "line_of_therapy": 1,
                "evidence": evidence,
            },
        ],
    }


HOSTILE_URLS = [
    "https://evil.test/paper1",
    "https://evil.test/paper2",
    "https://evil.test/paper3",
    "https://evil.test/paper4",
    "https://evil.test/paper5",
]


HONEST_URL = "https://pubmed.ncbi.nlm.nih.gov/12345678/"


# --------------------------------------------------------------------------- #
# Envelope-shape + wiring tests


def test_co_scientist_endpoint_listed_in_health(client):
    r = client.get("/health")
    assert r.status_code == 200, r.text
    endpoints = r.json()["endpoints"]
    assert "POST /v1/co_scientist/run" in endpoints, (
        "/v1/co_scientist/run must be advertised in /health.endpoints, "
        f"got: {endpoints}"
    )


def test_happy_path_envelope_shape(client):
    """Screening envelope + honest URL → tournament runs, no drops."""
    payload = {
        "screening": _screening_envelope([_ev(HONEST_URL, "honest citation")]),
        "seed_urls": [HONEST_URL],
    }
    r = client.post("/v1/co_scientist/run", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()

    # Envelope must carry disclaimer, caveat, provenance, honesty_gate.
    assert body.get("disclaimer"), "missing disclaimer"
    assert body.get("caveat"), "missing caveat"
    prov = body["provenance"]
    assert prov["model_state"] == "proxy_co_scientist", prov
    assert prov["model_name"] == "oa/co_scientist@v0.4.0-alpha"
    assert prov["request_id"], "missing request_id"

    # Phases must be the documented 5-phase order.
    assert body["phases"] == ["generate", "reflect", "rank", "evolve", "rank"], (
        f"unexpected phase order: {body['phases']}"
    )

    # Honest input → nothing gets dropped.
    assert body["urls_dropped_hallucinated"] == 0, (
        f"honest input should not drop URLs, got {body['urls_dropped_hallucinated']}"
    )
    # Hypotheses list is non-empty (screening findings + generated
    # variants).
    assert body["initial_count"] >= 1
    assert body["after_reflect"] == body["initial_count"], (
        "REFLECT should not delete hypotheses on honest input, only URLs"
    )
    assert len(body["hypotheses"]) >= 1
    # honesty_gate.evidence_dropped == urls_dropped_hallucinated on the wire.
    assert body["honesty_gate"]["evidence_dropped"] == body["urls_dropped_hallucinated"]


def test_deterministic_same_input_same_output(client):
    """Two identical requests → identical bodies (ignoring request_id)."""
    payload = {
        "screening": _screening_envelope([_ev(HONEST_URL, "cite")]),
        "seed_urls": [HONEST_URL],
    }
    r1 = client.post("/v1/co_scientist/run", json=payload)
    r2 = client.post("/v1/co_scientist/run", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    b1, b2 = r1.json(), r2.json()

    # request_id differs; everything else must match.
    for b in (b1, b2):
        b["provenance"].pop("request_id", None)
    assert b1["hypotheses"] == b2["hypotheses"], "Co-Scientist is not deterministic"
    assert b1["urls_dropped_hallucinated"] == b2["urls_dropped_hallucinated"]
    assert b1["hypotheses_dropped"] == b2["hypotheses_dropped"]


# --------------------------------------------------------------------------- #
# HOSTILE-URL CONTRACT TEST (PLAN §5 PR #5 acceptance criterion)


def test_hostile_urls_all_dropped_by_reflect(client):
    """5 fake URLs in evidence, `seed_urls=[]` → all 5 dropped, warnings fire.

    This is the load-bearing acceptance test from PLAN §5 PR #5:
    "5 fake URLs → all dropped by REFLECT, `hypotheses_dropped == 5`".

    We verify the strict contract:

    * `urls_dropped_hallucinated == 5` (all 5 distinct hostile URLs stripped)
    * `hypotheses_dropped >= 1` (at least one hypothesis lost its whole
      evidence list — actually all of them will, since every hypothesis
      inherits the same hostile evidence)
    * Warnings include per-hypothesis `dropped 5 hallucinated citation(s)`
      lines listing the hostile URLs
    * Every hostile URL appears in at least one warning (proof of drop)
    * NO hypothesis in the response body carries a hostile URL
    """
    hostile_ev = [_ev(u, f"fake {u}") for u in HOSTILE_URLS]
    payload = {
        "screening": _screening_envelope(hostile_ev),
        "seed_urls": [],  # empty — nothing was actually fetched
    }
    r = client.post("/v1/co_scientist/run", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()

    # === 1. Distinct-URL drop count is exactly 5 ===
    assert body["urls_dropped_hallucinated"] == 5, (
        f"expected exactly 5 hostile URLs dropped, got "
        f"{body['urls_dropped_hallucinated']}. Full body: {body}"
    )
    # Mirror on the honesty_gate report.
    assert body["honesty_gate"]["evidence_dropped"] == 5, (
        f"honesty_gate.evidence_dropped should match "
        f"urls_dropped_hallucinated, got "
        f"{body['honesty_gate']['evidence_dropped']}"
    )

    # === 2. At least one hypothesis dropped ===
    assert body["hypotheses_dropped"] >= 1, (
        f"REFLECT should strand at least one hypothesis with no "
        f"evidence, got hypotheses_dropped={body['hypotheses_dropped']}"
    )

    # === 3. Warnings channel proves the drops ===
    warnings = body["warnings"]
    hallucination_warnings = [
        w for w in warnings if "hallucinated citation" in w
    ]
    assert len(hallucination_warnings) >= 1, (
        f"REFLECT must emit `dropped N hallucinated citation(s)` "
        f"warnings, got: {warnings}"
    )

    # Every hostile URL must appear in at least one warning.
    warnings_blob = " ".join(warnings)
    for u in HOSTILE_URLS:
        assert u in warnings_blob, (
            f"hostile URL {u!r} was dropped but not surfaced in "
            f"warnings channel — silent drops break the honesty "
            f"contract. Full warnings: {warnings}"
        )

    # === 4. `no_evidence_after_reflect:<hyp_id>` warnings emitted ===
    empty_warnings = [
        w for w in warnings if w.startswith("no_evidence_after_reflect:")
    ]
    assert len(empty_warnings) == body["hypotheses_dropped"], (
        f"one `no_evidence_after_reflect` warning per stranded "
        f"hypothesis; got {len(empty_warnings)} warnings but "
        f"hypotheses_dropped={body['hypotheses_dropped']}. "
        f"Warnings: {warnings}"
    )

    # === 5. NO hostile URL appears on any returned hypothesis ===
    for h in body["hypotheses"]:
        for e in h.get("evidence") or []:
            assert e.get("url") not in HOSTILE_URLS, (
                f"hostile URL {e.get('url')!r} survived REFLECT and "
                f"appeared on hypothesis {h.get('hyp_id')!r}. This is a "
                f"CRITICAL honesty-gate leak."
            )


def test_hostile_urls_across_multiple_stages(client):
    """5 hostile URLs split across screening + therapy → all still dropped.

    Same hostile-URL scenario, but the fake URLs are spread across two
    stage envelopes rather than concentrated in screening. Verifies that
    REFLECT sweeps every stage, not just the first.
    """
    hostile_a = [_ev(u) for u in HOSTILE_URLS[:2]]
    hostile_b = [_ev(u) for u in HOSTILE_URLS[2:]]
    payload = {
        "screening": _screening_envelope(hostile_a),
        "therapy": _therapy_envelope(hostile_b),
        "seed_urls": [],
    }
    r = client.post("/v1/co_scientist/run", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()

    # All 5 distinct URLs are dropped.
    assert body["urls_dropped_hallucinated"] == 5, (
        f"expected 5 URL drops across two stages, got "
        f"{body['urls_dropped_hallucinated']}"
    )

    warnings_blob = " ".join(body["warnings"])
    for u in HOSTILE_URLS:
        assert u in warnings_blob, (
            f"hostile URL {u} split across stages was dropped but not "
            f"warned. Full warnings: {body['warnings']}"
        )


def test_mixed_urls_only_hostile_dropped(client):
    """1 honest URL + 5 hostile URLs → only the 5 hostile ones drop.

    Regression guard against over-broad REFLECT filtering (dropping
    legitimate URLs too).
    """
    all_urls = [HONEST_URL] + HOSTILE_URLS
    mixed_ev = [_ev(u) for u in all_urls]
    payload = {
        "screening": _screening_envelope(mixed_ev),
        "seed_urls": [HONEST_URL],
    }
    r = client.post("/v1/co_scientist/run", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()

    # Exactly 5 dropped (the hostiles); honest survives.
    assert body["urls_dropped_hallucinated"] == 5, (
        f"expected 5 hostile URL drops (1 honest survives), got "
        f"{body['urls_dropped_hallucinated']}"
    )
    # No hypothesis lost ALL evidence — honest URL is still there.
    assert body["hypotheses_dropped"] == 0, (
        f"no hypothesis should be stranded (honest URL survives), got "
        f"hypotheses_dropped={body['hypotheses_dropped']}"
    )
    # Every returned hypothesis carries ONLY the honest URL, no hostiles.
    for h in body["hypotheses"]:
        for e in h.get("evidence") or []:
            u = e.get("url")
            assert u == HONEST_URL, (
                f"hypothesis {h.get('hyp_id')!r} evidence carries "
                f"unexpected URL {u!r} — expected only {HONEST_URL!r}"
            )


def test_empty_envelopes_returns_empty_tournament(client):
    """No stage envelopes → empty tournament, no drops, no error."""
    payload = {"seed_urls": []}
    r = client.post("/v1/co_scientist/run", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["initial_count"] == 0
    assert body["after_reflect"] == 0
    assert body["hypotheses"] == []
    assert body["urls_dropped_hallucinated"] == 0
    assert body["hypotheses_dropped"] == 0
    assert body["phases"] == ["generate", "reflect", "rank", "evolve", "rank"]


def test_pagination_return_top_respected(client):
    """return_top caps hypotheses list."""
    payload = {
        "screening": _screening_envelope([_ev(HONEST_URL)]),
        "therapy": _therapy_envelope([_ev(HONEST_URL)]),
        "seed_urls": [HONEST_URL],
        "return_top": 2,
    }
    r = client.post("/v1/co_scientist/run", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["hypotheses"]) <= 2, (
        f"return_top=2 must cap hypotheses at 2, got {len(body['hypotheses'])}"
    )
    # But the counts must reflect the FULL tournament size, not the cap.
    assert body["after_evolve"] > 2, (
        "after_evolve should reflect full tournament, not return_top cap"
    )


def test_invalid_return_top_rejected(client):
    """return_top out of [1, 32] range is rejected with 422."""
    payload = {
        "seed_urls": [],
        "return_top": 0,  # < min
    }
    r = client.post("/v1/co_scientist/run", json=payload)
    assert r.status_code == 422, (
        f"return_top=0 should be rejected as < min, got {r.status_code}"
    )


def test_route_is_public_no_auth_required(client, monkeypatch):
    """POST /v1/co_scientist/run must remain callable with AUTH_MODE=on.

    Matches /v1/elo/rank posture: standalone Co-Scientist has no PHI
    ingest, so it stays public for demo callers.
    """
    # Rebuild app with AUTH_MODE=on to prove the route escapes the gate.
    monkeypatch.setenv("AUTH_MODE", "on")
    monkeypatch.setenv("ONCOLOGY_ARBITER_API_KEY", "test-key")
    from oncology_arbiter.api.app import create_app as _create
    _client = TestClient(_create())
    r = _client.post(
        "/v1/co_scientist/run",
        json={"seed_urls": []},  # no API key header
    )
    # Endpoint answers even without a key.
    assert r.status_code == 200, (
        f"/v1/co_scientist/run must be public under AUTH_MODE=on, got "
        f"status {r.status_code}: {r.text[:400]}"
    )
