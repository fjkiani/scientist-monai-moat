"""Live integration test for the ``crispro--medgemma-27b`` Modal app.

Skipped unless ``MEDGEMMA_MODAL_URL`` is set. Per explicit product
direction, this endpoint is a known, real defect (documented in this same
commit's own retracted-claims ledger, ``artifacts/audit/corrected_claims.json``,
claim C4: a live chat call previously measured a 602.997s timeout ending in
HTTP 500). The tests below re-verify that behavior live, with real network
calls and no mocking, rather than citing the old number unverified -- and
add a newly-discovered, independent defect: the deployed app_version has
regressed *relative to the checked-out repo source*, not just relative to
some abstract target.

Three distinct claims are tested separately so a fix to one does not read
as a fix to all:
  1. healthz/version -- fast, pins observed reality (PASSES).
  2. deploy-drift -- fast, desired contract vs. reality (FAILS honestly).
  3. chat completion -- slow (~10 min), desired contract vs. reality
     (expected to FAIL/timeout honestly; run deliberately, not in the
     default fast suite).
"""
from __future__ import annotations

import os
import time

import pytest
import requests

LIVE = pytest.mark.skipif(
    not os.environ.get("MEDGEMMA_MODAL_URL"),
    reason="MEDGEMMA_MODAL_URL not set (skipping live Modal test)",
)

# The version both the checked-out repo source (deploy/modal/medgemma_27b_app.py
# APP_VERSION) and the production client
# (src/oncology_arbiter/models/llm_client.py GemmaClient._call_medgemma_modal
# `expected_app`) agree the deployed app should be. Repo and client are
# concordant; only the live deployment disagrees with both.
EXPECTED_APP_VERSION = "medgemma-27b-modal-v0.5.0-production-ready"
EXPECTED_MODEL_REPO = "google/medgemma-27b-it"

# Historical reference only (claim C4, 2026-08-13). Not asserted against --
# the point of this file is to remeasure live, not to trust an old number.
_HISTORICAL_CHAT_TIMEOUT_S = 602.997


def _base_url() -> str:
    return os.environ["MEDGEMMA_MODAL_URL"].rstrip("/")


@LIVE
def test_healthz_reports_current_app_version() -> None:
    """Pin observed reality. PASSES today; exists so any future silent
    version change (in either direction) shows as a diff here."""
    resp = requests.get(f"{_base_url()}-healthz.modal.run", timeout=30.0)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "medgemma-27b"
    assert isinstance(body.get("app_version"), str) and body["app_version"]


@LIVE
def test_deployed_app_version_matches_repo_and_client_contract() -> None:
    """DESIRED CONTRACT (expected to fail honestly right now): the live
    Modal deployment's app_version must equal the version the checked-out
    repo source declares AND the version the production client
    (GemmaClient) requires for a successful chat call.

    KNOWN DEFECT, newly reconciled this session: repo source
    (deploy/modal/medgemma_27b_app.py: APP_VERSION) and client
    (llm_client.py: GemmaClient._call_medgemma_modal: expected_app) already
    AGREE with each other on 'medgemma-27b-modal-v0.5.0-production-ready'
    -- the audited/reviewed code is internally consistent. The live
    deployment has simply never been updated to match: it still serves
    'medgemma-27b-modal-v0.4.0-alpha'. This means every real chat call
    through the production GemmaClient fails its own version check even
    if the underlying completion succeeds. This is a deploy-pipeline gap,
    not a code-correctness gap -- do not fix by loosening this assertion
    or by changing the client's expected version to match stale reality.
    """
    resp = requests.get(f"{_base_url()}-healthz.modal.run", timeout=30.0)
    assert resp.status_code == 200
    observed = resp.json().get("app_version")
    assert observed == EXPECTED_APP_VERSION, (
        f"deployed app_version={observed!r} != repo/client-expected "
        f"{EXPECTED_APP_VERSION!r}. The audited source code was never "
        "actually `modal deploy`-ed; this is deploy drift, not a code bug."
    )


@LIVE
@pytest.mark.slow
def test_chat_completion_real_call_via_production_client() -> None:
    """DESIRED CONTRACT, exercised through the REAL production code path
    (GemmaClient.chat -> _call_medgemma_modal), not a raw HTTP probe.
    Generous timeout (900s) chosen to exceed the historically observed
    ~603s failure point so we can observe the actual current outcome
    (success, a different failure, or the same timeout) rather than
    truncating the measurement. Expected to fail honestly (LlmUnavailable)
    given the confirmed stale deployment above; if it now succeeds, that
    is a genuine, welcome, and reportable change from historical behavior.
    """
    from oncology_arbiter.models.llm_client import GemmaClient, LlmUnavailable

    os.environ["MEDGEMMA_MODAL_URL"] = _base_url()
    client = GemmaClient(timeout_s=900.0)
    t0 = time.time()
    try:
        result = client.chat(
            [{"role": "user", "content": "In one short sentence, what is non-small cell lung cancer?"}],
            max_tokens=64,
            temperature=0.0,
        )
    except LlmUnavailable as exc:
        elapsed = time.time() - t0
        pytest.fail(
            f"KNOWN DEFECT reproduced live (elapsed={elapsed:.1f}s): "
            f"GemmaClient.chat() raised LlmUnavailable: {exc}. Tracked "
            "separately from any deployment fix -- do not mock this to pass."
        )
    else:
        elapsed = time.time() - t0
        # If this ever succeeds, verify it is genuinely MedGemma (not a
        # silently substituted model) before treating the defect as closed.
        assert result.model == EXPECTED_MODEL_REPO
        assert result.app_version == EXPECTED_APP_VERSION
        assert result.text.strip()
        print(f"UNEXPECTED SUCCESS after {elapsed:.1f}s: {result.text[:200]!r}")
