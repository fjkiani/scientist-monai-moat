"""Live integration test for the ``crispro--clinicalbert`` Modal app.

Skipped unless ``CLINICALBERT_MODAL_URL`` is set to the BASE url, e.g.
``https://crispro--clinicalbert`` (NOT a full ``...-healthz.modal.run``
form -- ``ClinicalBertModalEndpointConfig`` appends ``-healthz``/``-info``/
``-parse`` + ``.modal.run`` itself; see clinicalbert_modal_client.py).

Uses 8 real, de-identified TCGA pathology reports (breast_crc + nsclc;
tests/fixtures/clinicalbert_reports/real_reports_sample.json, provenance
in the sibling PROVENANCE.json).

DEPLOY-DRIFT FINDING (discovered while writing this file, confirmed live,
not from memory): the live deployment reports ``app_version =
"clinicalbert-modal-v0.5.1"``. Both the repo source
(``deploy/modal/clinicalbert_app.py``: ``APP_VERSION =
"clinicalbert-modal-v0.5.2-sliding-window"``) and the production client
(``clinicalbert_modal_client.py``: ``EXPECTED_APP_VERSION`` = the same
v0.5.2 string) agree with EACH OTHER but not with what is actually
deployed -- the same class of gap already found for medgemma-27b
(reviewed/committed code != deployed artifact). Unlike medgemma-27b this
is not just a version string: the live v0.5.1 response has no
``n_windows``/``window_tokens``/``overlap_tokens``/``window_aggregation``/
``model_sha256``/``metrics_sha256`` fields at all (confirmed via a raw,
non-validating call) -- v0.5.1 has no sliding-window code path, so
``ClinicalBertModalClient.parse()``'s own ``validate_clinicalbert_contract()``
rejects every real call. This is kept as a genuine, reproducible FAILING
test below, not mocked around.

This also RECONCILES a previously-flagged anomaly (quarantine_receipt.json
banned_product_claims[0]): two conflicting micro-F1 numbers existed for
"ClinicalBERT" -- 0.2535 (model-card, in-corpus, 296 reports, 2,084
sliding windows -- i.e. computed against v0.5.2-sliding-window weights
that were apparently never deployed) vs. 0.0809 (fleet-matrix). Hitting
the live ``/info`` endpoint in this file returns
``test_micro_f1 = 0.08089260808926081`` -- an exact match to the 0.0809
fleet-matrix figure. These are two different, real model versions, not
one model measured inconsistently: 0.0809 is what is actually live;
0.2535 is a newer, better, but never-deployed model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

LIVE = pytest.mark.skipif(
    not os.environ.get("CLINICALBERT_MODAL_URL"),
    reason="CLINICALBERT_MODAL_URL not set (skipping live Modal test)",
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "clinicalbert_reports"
    / "real_reports_sample.json"
)

LIVE_DEPLOYED_APP_VERSION = "clinicalbert-modal-v0.5.1"
LIVE_DEPLOYED_COMBINED_MICRO_F1 = 0.08089260808926081


def _load_fixture() -> list[dict]:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"fixture missing: {FIXTURE_PATH}")
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def _client():
    from oncology_arbiter.nlp.clinicalbert_modal_client import (
        ClinicalBertModalClient,
        ClinicalBertModalEndpointConfig,
    )

    return ClinicalBertModalClient(endpoints=ClinicalBertModalEndpointConfig.from_env())


def _raw_endpoints():
    from oncology_arbiter.nlp.clinicalbert_modal_client import ClinicalBertModalEndpointConfig

    return ClinicalBertModalEndpointConfig.from_env()


@LIVE
def test_healthz_reports_ok() -> None:
    d = _client().healthz()
    assert d["status"] == "ok"
    assert d["app"] == "clinicalbert"
    assert isinstance(d.get("version"), str) and d["version"]
    assert "disclaimer" in d


@LIVE
def test_info_reports_real_v051_provenance_not_synthetic_default() -> None:
    """Decisive test flagged in /mnt/results/quarantine_receipt.json:
    the live model must report REAL-v0.5.1-snorkel-openrouter-llm, not the
    hardcoded SYNTHETIC-v0.3.1 fallback that fires when a deployed
    metrics.json lacks a top-level 'provenance' key. This passes: the
    live deployment is genuinely on real-corpus provenance, just an
    older (v0.5.1, pre-sliding-window) real model than the repo expects
    -- see the deploy-drift test below for that separate finding.
    """
    d = _client().info()
    provenance = d.get("provenance")
    assert provenance == "REAL-v0.5.1-snorkel-openrouter-llm", (
        f"clinicalbert live /info reports provenance={provenance!r}. Expected "
        "'REAL-v0.5.1-snorkel-openrouter-llm'. If this is 'SYNTHETIC-v0.3.1', "
        "the deployed metrics.json is missing its 'provenance' key and the "
        "app silently fell back to the banned-claim default -- a genuine "
        "production defect, not a test bug -- do not silence this assertion."
    )
    assert d.get("base_model") == "emilyalsentzer/Bio_ClinicalBERT"
    assert isinstance(d.get("test_micro_f1"), float)


@LIVE
def test_info_app_version_and_sliding_window_contract_matches_client_expectation() -> None:
    """HONEST FAILING TEST -- genuine deploy-drift, not mocked around.

    Repo source (clinicalbert_app.py) and the production client
    (clinicalbert_modal_client.py) both declare
    'clinicalbert-modal-v0.5.2-sliding-window' as the expected app_version,
    and agree with each other. The live deployment answers with
    'clinicalbert-modal-v0.5.1' and its response has NO window_tokens /
    overlap_tokens / window_aggregation / model_sha256 / metrics_sha256
    fields at all (verified via a raw, non-validating call -- not merely
    a version-string mismatch, an entirely different, older code path).
    """
    from oncology_arbiter.nlp.clinicalbert_modal_client import EXPECTED_APP_VERSION

    d = _client().info()
    live_app_version = d.get("app_version")
    assert live_app_version == EXPECTED_APP_VERSION, (
        f"DEPLOY-DRIFT: live clinicalbert app_version={live_app_version!r} "
        f"but repo source + production client both expect "
        f"{EXPECTED_APP_VERSION!r}. The reviewed/committed "
        "v0.5.2-sliding-window code (with model_sha256/metrics_sha256 "
        "provenance pinning) was never `modal deploy`'d -- same class of "
        "gap already found for medgemma-27b. Live /info is missing these "
        f"keys entirely: "
        f"{[k for k in ('window_tokens','overlap_tokens','window_aggregation','model_sha256','metrics_sha256') if d.get(k) is None]}."
    )


@LIVE
@pytest.mark.parametrize("idx", [0, 2, 6])
def test_parse_via_production_client_succeeds_on_real_reports(idx: int) -> None:
    """HONEST FAILING TEST (downstream consequence of the deploy-drift
    above, exercised via a few representative real reports rather than
    all 8 -- the root cause and failure mode are identical for all of
    them, see test_parse_entity_type_recall_via_raw_endpoint below for
    the full-sample raw measurement). The *intended* production contract
    is that ClinicalBertModalClient.parse() succeeds end-to-end on real
    report text; on the currently-deployed v0.5.1 it cannot, because
    validate_clinicalbert_contract() correctly refuses a response that
    lacks the v0.5.2 sliding-window fields it was written to pin.
    """
    from oncology_arbiter.nlp.clinicalbert_modal_client import ClinicalBertModalError

    rec = _load_fixture()[idx]
    client = _client()
    try:
        client.parse(rec["text"])
    except ClinicalBertModalError as exc:
        pytest.fail(
            f"parse() raised on real report #{idx} ({rec['cancer']}, "
            f"{len(rec['entities'])} gold entities) -- expected root cause: "
            f"deploy-drift (live=v0.5.1, client expects v0.5.2-sliding-window). "
            f"Raw error: {exc}"
        )


@LIVE
def test_parse_entity_type_recall_via_raw_endpoint_bypassing_strict_contract() -> None:
    """Mathematically interrogate the *currently-deployed* v0.5.1 model's
    real behavior on real text, independent of the (correctly failing)
    strict v0.5.2 contract check above. Calls the raw endpoint directly
    (bypassing ClinicalBertModalClient.parse()'s validate_*() gate, same
    technique used for the phikon raw-vector test) across all 8 real
    reports and computes entity-TYPE presence/recall.

    This is a live, direct, mechanistic corroboration of why the fleet
    micro-F1 is low: this v0.5.1 deployment has no sliding-window
    aggregation, so Bio_ClinicalBERT's underlying max-sequence-length
    truncates long reports -- entities appearing after the truncation
    point are structurally unreachable, not just imprecisely predicted.
    """
    from oncology_arbiter.nlp.clinicalbert_modal_client import _post_json

    endpoints = _raw_endpoints()
    records = _load_fixture()
    tp = fp = fn = 0
    per_type_counts: dict[str, dict[str, int]] = {}
    n_tokens_by_record = []
    for rec in records:
        d = _post_json(endpoints.parse, {"report_text": rec["text"]}, timeout=30)
        assert "error" not in d, f"raw parse() returned an error body: {d.get('error')}"
        assert d["app_version"] == LIVE_DEPLOYED_APP_VERSION
        n_tokens_by_record.append(d["n_tokens"])
        pred_types = set(d["parsed"].keys())
        gold_types = {e["entity_type"] for e in rec["entities"]}
        for et in gold_types | pred_types:
            per_type_counts.setdefault(et, {"tp": 0, "fp": 0, "fn": 0})
        for et in gold_types & pred_types:
            tp += 1
            per_type_counts[et]["tp"] += 1
        for et in pred_types - gold_types:
            fp += 1
            per_type_counts[et]["fp"] += 1
        for et in gold_types - pred_types:
            fn += 1
            per_type_counts[et]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )

    print(
        f"\n[clinicalbert LIVE v0.5.1, raw endpoint, n=8 real reports] "
        f"tp={tp} fp={fp} fn={fn} precision={precision:.4f} recall={recall:.4f} "
        f"entity_type_micro_f1={micro_f1:.4f}"
    )
    print(f"n_tokens per report (no sliding window -> hard truncation ceiling): {n_tokens_by_record}")
    print(f"per-entity-type breakdown: {json.dumps(per_type_counts, indent=2)}")
    print(
        f"Live /info self-reported combined test_micro_f1 = "
        f"{LIVE_DEPLOYED_COMBINED_MICRO_F1} (BIO-span level, full 296-report "
        "held-out set) -- this test's entity-type-presence micro-F1 on 8 "
        "reports is a different methodology/sample, reported for direction "
        "only, not as a reproduction of that number."
    )
    assert tp > 0, (
        "live v0.5.1 endpoint found ZERO true-positive entity types across "
        "all 8 real reports -- that would indicate a fully broken "
        "deployment, not merely low F1."
    )


@LIVE
def test_parse_rejects_empty_report_text_client_side() -> None:
    """Client-side guard (no network round trip): parse() must refuse an
    empty report_text rather than silently sending it upstream."""
    from oncology_arbiter.nlp.clinicalbert_modal_client import ClinicalBertModalError

    with pytest.raises(ClinicalBertModalError):
        _client().parse("")
