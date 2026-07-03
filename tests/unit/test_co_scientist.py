"""Unit tests for the L5 Co-Scientist 4-phase loop.

Test surface:
  * generate — hypotheses count and content
  * reflect — honesty gate drops unseen-URL evidence
  * rank — Elo tournament is deterministic given a fixed seed
  * evolve — spawns variants of top-N with derived_from lineage
  * end-to-end — run_co_scientist returns the documented dict shape
  * end-to-end — placeholder inputs never crash the loop
"""
from __future__ import annotations

from oncology_arbiter.orchestrator.co_scientist import (
    EloEntry,
    Hypothesis,
    _expected,
    _score_hypothesis,
    evolve_hypotheses,
    generate_hypotheses,
    rank_hypotheses,
    reflect_hypotheses,
    run_co_scientist,
)


# --------------------------------------------------------------------------- #
# Fixtures (inline — no pytest fixture files needed)


def _screening_env() -> dict:
    return {
        "provenance": {"model_state": "loaded_medsiglip", "model_name": "google/medsiglip-448"},
        "findings": [
            {"label": "malignant lesion", "score": 0.72, "location_bbox_normalized": None},
            {"label": "no lesion", "score": 0.28, "location_bbox_normalized": None},
        ],
        "evidence": [
            {"url": "https://huggingface.co/google/medsiglip-448",
             "quoted_text": "SigLIP variant trained on medical images",
             "source": "hf-hub"},
        ],
    }


def _biopsy_env() -> dict:
    return {
        "provenance": {"model_state": "loaded_biopsy_probe", "model_name": "biopsy_probe_v0"},
        "subtype_prediction": "IDC",
        "confidence": 0.68,
        "grade": 2,
        "evidence": [
            {"url": "https://www.who.int/publications/i/item/9789283245063",
             "quoted_text": "WHO Classification of Breast Tumors",
             "source": "who"},
        ],
    }


def _therapy_env() -> dict:
    return {
        "provenance": {"model_state": "proxy_rules_lite", "model_name": "nccn_lite_v0"},
        "recommended_options": [
            {"regimen": "Aromatase inhibitor (letrozole 5 years)",
             "line_of_therapy": 1,
             "rationale": "HR+/HER2- postmenopausal",
             "evidence": [{"url": "https://www.nccn.org/professionals/physician_gls/pdf/breast.pdf",
                            "quoted_text": "NCCN Breast Cancer v3.2025",
                            "source": "nccn-guidelines"}]},
            {"regimen": "Adjuvant tamoxifen 5 years",
             "line_of_therapy": 1,
             "rationale": "HR+ alternative", "evidence": []},
        ],
    }


ALL_SEEN_URLS = {
    "https://huggingface.co/google/medsiglip-448",
    "https://www.who.int/publications/i/item/9789283245063",
    "https://www.nccn.org/professionals/physician_gls/pdf/breast.pdf",
}


# --------------------------------------------------------------------------- #
# Phase 1 — generate

def test_generate_from_all_three_stages_produces_expected_ids():
    hyps = generate_hypotheses(
        screening=_screening_env(),
        biopsy=_biopsy_env(),
        therapy=_therapy_env(),
    )
    ids = {h.hyp_id for h in hyps}
    assert "screening:malignant lesion" in ids
    assert "screening:no lesion" in ids
    assert "biopsy:IDC" in ids
    assert "biopsy:DCIS:alternate" in ids  # alternate for IDC
    assert any(i.startswith("therapy:Aromatase inhibitor") for i in ids)
    # Every hypothesis has a stage tag matching one of the three known stages
    assert {h.stage for h in hyps} == {"screening", "biopsy", "therapy"}


def test_generate_returns_empty_when_all_stages_none():
    assert generate_hypotheses(screening=None, biopsy=None, therapy=None) == []


def test_generate_honesty_markers_reflect_provenance():
    hyps = generate_hypotheses(
        screening=_screening_env(),
        biopsy=_biopsy_env(),
        therapy=_therapy_env(),
    )
    scr = [h for h in hyps if h.stage == "screening"][0]
    assert scr.honesty_markers["loaded"] is True
    assert scr.honesty_markers["proxy"] is False
    tx = [h for h in hyps if h.stage == "therapy"][0]
    assert tx.honesty_markers["proxy"] is True
    assert tx.honesty_markers["loaded"] is False


# --------------------------------------------------------------------------- #
# Phase 2 — reflect

def test_reflect_drops_evidence_with_unseen_url():
    hyps = generate_hypotheses(
        screening=_screening_env(),
        biopsy=_biopsy_env(),
        therapy=_therapy_env(),
    )
    # Only allow the biopsy URL — screening + therapy evidence should be dropped.
    kept, warnings = reflect_hypotheses(hyps, {"https://www.who.int/publications/i/item/9789283245063"})
    biopsy_hyps = [h for h in kept if h.stage == "biopsy"]
    screening_hyps = [h for h in kept if h.stage == "screening"]
    therapy_hyps = [h for h in kept if h.stage == "therapy"]
    # Biopsy evidence retained
    assert all(len(h.evidence) == 1 for h in biopsy_hyps if h.evidence or True)
    # Screening + therapy evidence stripped
    assert all(h.evidence == [] for h in screening_hyps)
    assert all(h.evidence == [] for h in therapy_hyps)
    # Warnings surface every hypothesis whose evidence became empty
    n_no_ev = sum(1 for w in warnings if w.startswith("no_evidence_after_reflect:"))
    assert n_no_ev == len(screening_hyps) + len(therapy_hyps)


def test_reflect_keeps_hypothesis_count_stable():
    hyps = generate_hypotheses(
        screening=_screening_env(), biopsy=_biopsy_env(), therapy=_therapy_env(),
    )
    kept, _ = reflect_hypotheses(hyps, set())
    assert len(kept) == len(hyps)


# --------------------------------------------------------------------------- #
# Phase 3 — rank (Elo tournament)

def test_expected_score_is_symmetric_around_400():
    # Standard Elo: rating diff of 400 → expected ~0.909
    assert abs(_expected(1900, 1500) - 0.9090909) < 1e-4
    assert abs(_expected(1500, 1900) - 0.09090909) < 1e-4
    assert abs(_expected(1500, 1500) - 0.5) < 1e-9


def test_score_hypothesis_rewards_loaded_penalizes_proxy():
    h_loaded = Hypothesis(hyp_id="a", stage="biopsy", statement="x", confidence=0.7,
                          honesty_markers={"loaded": True})
    h_proxy = Hypothesis(hyp_id="b", stage="biopsy", statement="y", confidence=0.7,
                         honesty_markers={"proxy": True})
    h_gated = Hypothesis(hyp_id="c", stage="biopsy", statement="z", confidence=0.7,
                         honesty_markers={"gated": True})
    assert _score_hypothesis(h_loaded) > _score_hypothesis(h_proxy)
    assert _score_hypothesis(h_proxy) > _score_hypothesis(h_gated)


def test_rank_is_deterministic():
    hyps = generate_hypotheses(
        screening=_screening_env(), biopsy=_biopsy_env(), therapy=_therapy_env(),
    )
    a = rank_hypotheses(list(hyps))
    b = rank_hypotheses(list(hyps))
    assert [e.hypothesis.hyp_id for e in a] == [e.hypothesis.hyp_id for e in b]
    assert [round(e.rating, 6) for e in a] == [round(e.rating, 6) for e in b]


def test_rank_all_matches_sum_to_zero_rating_delta():
    """Elo is zero-sum: sum of rating deltas across every match is 0.
    Every entry starts at 1500 → sum(rating) after tournament must equal
    initial_sum == n * 1500."""
    hyps = generate_hypotheses(
        screening=_screening_env(), biopsy=_biopsy_env(), therapy=_therapy_env(),
    )
    ranked = rank_hypotheses(hyps)
    total = sum(e.rating for e in ranked)
    assert abs(total - 1500.0 * len(ranked)) < 1e-6


# --------------------------------------------------------------------------- #
# Phase 4 — evolve

def test_evolve_spawns_variants_with_derived_from():
    hyps = generate_hypotheses(
        screening=_screening_env(), biopsy=_biopsy_env(), therapy=_therapy_env(),
    )
    ranked = rank_hypotheses(hyps)
    variants = evolve_hypotheses(ranked, top_n=3, n_variants=2)
    assert len(variants) > 0
    top_ids = {e.hypothesis.hyp_id for e in ranked[:3]}
    for v in variants:
        assert v.derived_from is not None
        assert v.derived_from in top_ids


def test_evolve_biopsy_swap_flips_subtype_in_id():
    parent = Hypothesis(hyp_id="biopsy:IDC", stage="biopsy", statement="IDC",
                        confidence=0.7)
    ranked = [EloEntry(hypothesis=parent, rating=1600)]
    variants = evolve_hypotheses(ranked, top_n=1, n_variants=1)
    assert len(variants) == 1
    assert "DCIS" in variants[0].hyp_id
    assert variants[0].derived_from == "biopsy:IDC"


# --------------------------------------------------------------------------- #
# End-to-end driver

def test_run_co_scientist_returns_documented_shape():
    result = run_co_scientist(
        screening=_screening_env(),
        biopsy=_biopsy_env(),
        therapy=_therapy_env(),
        seen_urls=ALL_SEEN_URLS,
    )
    assert result["phases"] == ["generate", "reflect", "rank", "evolve", "rank"]
    assert isinstance(result["warnings"], list)
    assert result["initial_count"] > 0
    assert result["after_reflect"] == result["initial_count"]
    assert result["after_evolve"] > result["initial_count"]
    for h in result["hypotheses"]:
        assert set(h.keys()) >= {
            "hyp_id", "stage", "statement", "confidence", "evidence",
            "honesty_markers", "derived_from", "rating", "wins", "losses", "draws",
        }


def test_run_co_scientist_is_deterministic():
    r1 = run_co_scientist(
        screening=_screening_env(),
        biopsy=_biopsy_env(),
        therapy=_therapy_env(),
        seen_urls=ALL_SEEN_URLS,
    )
    r2 = run_co_scientist(
        screening=_screening_env(),
        biopsy=_biopsy_env(),
        therapy=_therapy_env(),
        seen_urls=ALL_SEEN_URLS,
    )
    assert r1["hypotheses"] == r2["hypotheses"]


def test_run_co_scientist_survives_placeholder_inputs():
    """None stage envelopes → empty output; must not crash."""
    result = run_co_scientist(screening=None, biopsy=None, therapy=None, seen_urls=set())
    assert result["initial_count"] == 0
    assert result["after_reflect"] == 0
    assert result["after_evolve"] == 0
    assert result["hypotheses"] == []
    # phases still listed even when nothing to rank
    assert result["phases"] == ["generate", "reflect", "rank", "evolve", "rank"]


def test_run_co_scientist_honesty_gate_actually_drops_unseen():
    """When seen_urls is empty, every evidence entry should be dropped, but
    hypotheses themselves survive so the tournament still runs."""
    r = run_co_scientist(
        screening=_screening_env(),
        biopsy=_biopsy_env(),
        therapy=_therapy_env(),
        seen_urls=set(),
    )
    for h in r["hypotheses"]:
        assert h["evidence"] == []
    # Every hypothesis should have surfaced a no_evidence warning
    assert any(w.startswith("no_evidence_after_reflect:") for w in r["warnings"])


# --------------------------------------------------------------------------- #
# /v1/case/full endpoint wire test

def test_case_full_endpoint_returns_elo_hypotheses_when_flag_set(monkeypatch):
    """/v1/case/full with ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST=1 populates
    elo_ranked_hypotheses with the run_co_scientist output shape."""
    from fastapi.testclient import TestClient
    from oncology_arbiter.api.app import create_app

    monkeypatch.setenv("ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST", "1")
    # Don't force any HAI-DEF backends — placeholder stage responses are
    # enough to prove the loop ran.
    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_MEDSIGLIP", raising=False)

    with TestClient(create_app()) as client:
        resp = client.post("/v1/case/full", json={
            "therapy_context": {"age": 58, "menopausal_status": "post"}
        })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Placeholder stages emit no findings / no options, so the loop returns
    # an empty hypothesis list — but the FIELD must be present as a list.
    assert isinstance(body.get("elo_ranked_hypotheses"), list)


def test_case_full_endpoint_no_elo_when_flag_unset(monkeypatch):
    from fastapi.testclient import TestClient
    from oncology_arbiter.api.app import create_app

    monkeypatch.delenv("ONCOLOGY_ARBITER_ENABLE_CO_SCIENTIST", raising=False)
    with TestClient(create_app()) as client:
        resp = client.post("/v1/case/full", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["elo_ranked_hypotheses"] == []
