"""Live integration test for the ``crispro--gemma-fallback`` Modal app.

Skipped unless ``GEMMA_FALLBACK_MODAL_URL`` is set. There is no client class
for this app anywhere in the repository (grep confirms zero references to
"gemma-fallback"/"gemma_fallback" outside this test) -- it is an
out-of-band Modal deployment not tracked by any in-repo Python, unlike
Phikon which at least has ``PhikonClient``. ``GEMMA_FALLBACK_MODAL_URL`` is
introduced here as a new, self-documenting convention matching the other
``*_MODAL_URL`` env vars in ``specialist_clients.py``/``llm_client.py``; it
does not replace or alias any existing variable.

Per explicit product direction: this endpoint is a known, real defect
(it serves ``Qwen/Qwen2.5-7B-Instruct`` under a "gemma-fallback" label, not
any Gemma-family model). The model-identity test below is written against
the *desired* contract and is expected to fail honestly until the
deployment is fixed -- it must never be weakened or mocked to pass. It is
kept in its own file, separate from the endpoints that are actually
healthy, so a red result here does not obscure genuinely green evidence
elsewhere and does not get silently "fixed" by loosening the assertion.
"""
from __future__ import annotations

import os
import time

import pytest
import requests

LIVE = pytest.mark.skipif(
    not os.environ.get("GEMMA_FALLBACK_MODAL_URL"),
    reason="GEMMA_FALLBACK_MODAL_URL not set (skipping live Modal test)",
)


def _base_url() -> str:
    return os.environ["GEMMA_FALLBACK_MODAL_URL"].rstrip("/")


def _chat(payload: dict, timeout: float = 60.0) -> tuple[int, dict]:
    resp = requests.post(
        f"{_base_url()}-chat.modal.run",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    try:
        body = resp.json()
    except ValueError:
        body = {"_raw_text": resp.text[:500]}
    return resp.status_code, body


@LIVE
def test_healthz_reports_qwen_not_gemma() -> None:
    """Pin the currently observed reality: healthz self-reports the model
    it actually serves. This test PASSES because it documents observed
    fact, not a desired contract -- it exists so a future silent model
    swap (Qwen -> something else, still not Gemma) is caught as a diff,
    not just re-confirming today's known defect forever."""
    resp = requests.get(f"{_base_url()}-healthz.modal.run", timeout=30.0)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "gemma-fallback"
    assert body["model"] == "Qwen/Qwen2.5-7B-Instruct"


@LIVE
def test_chat_completion_infra_is_healthy() -> None:
    """The serving infrastructure itself works: a real prompt gets a real,
    coherent, fast completion. This isolates "is the container/runtime
    broken" (no -- it is not) from "is the correct model deployed" (no --
    tested separately below). Collapsing both into one assertion would
    hide that the underlying defect is purely a model-identity/labeling
    problem, not an infra outage.
    """
    t0 = time.time()
    status, body = _chat(
        {
            "messages": [{"role": "user", "content": "Reply with exactly the two words: hello world"}],
            "max_tokens": 20,
            "temperature": 0.0,
        },
        timeout=60.0,
    )
    elapsed = time.time() - t0
    assert status == 200, body
    assert "choices" in body and len(body["choices"]) > 0, body
    content = body["choices"][0]["message"]["content"].strip().lower()
    assert "hello world" in content, f"expected a coherent completion, got {content!r}"
    assert elapsed < 30.0, f"completion took {elapsed:.1f}s, expected well under 30s for a 20-token reply"


@LIVE
def test_chat_completion_model_identity_must_be_gemma_family() -> None:
    """DESIRED CONTRACT (expected to fail honestly right now): an app
    deployed under the label "gemma-fallback" and called as the fallback
    for Gemma-family completions must report a Gemma-family model in its
    OpenAI-compatible response, not Qwen.

    Current observed reality (real network call, no mock):
    ``model: "qwen2.5-7b-instruct"``. This is a genuine, tracked product
    defect -- fixing it means redeploying the correct base model, not
    editing this assertion. Do not mock this endpoint to pass.
    """
    status, body = _chat(
        {
            "messages": [{"role": "user", "content": "What model are you?"}],
            "max_tokens": 10,
            "temperature": 0.0,
        },
        timeout=60.0,
    )
    assert status == 200, body
    reported_model = str(body.get("model", ""))
    assert "gemma" in reported_model.lower(), (
        f"gemma-fallback app reported model={reported_model!r}; expected a "
        "Gemma-family model identity. KNOWN DEFECT: this app currently "
        "serves Qwen/Qwen2.5-7B-Instruct under the gemma-fallback label. "
        "Tracked separately from any deployment fix -- do not silence."
    )
