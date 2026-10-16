---
name: real-elo-co-scientist-no-retreat
description: Prevents proxy substitution, scope retreat, and path-of-least-resistance completion claims for real Elo, Co-Scientist, MedGemma, tumor-board synthesis, and clinical hypothesis ranking work in scientist-monai-moat. Use whenever Elo, Co-Scientist, hypothesis tournaments, or model-backed clinical synthesis are requested.
---

# Real Elo + Co-Scientist: no retreat

Repo: `/Users/fahadkiani/Desktop/development/scientist-monai-moat`

## Hard distinction

The requested endpoint is the model-backed loop in
`src/oncology_arbiter/agents/supervisor.py::execute_stage`:

`GENERATE → EVIDENCE → REFLECT → MedGemma pairwise TOURNAMENT/Elo → META_REVIEW`.

These do **not** satisfy that request:

- `/v1/offline_ranker/run`
- `/v1/elo/rank`
- caller-supplied modifiers or confidence bumps
- `run_placeholder()`
- rules-lite, static sorting, or deterministic `orchestrator/co_scientist.py`
- green tests that mock the supervisor itself

Keep offline routes honestly named, but never present them as the real endpoint.

Also load personal `audit-reject-no-hedge` and `no-path-of-least-resistance`.
**AUDIT-REJECT ≠ DONE:** if a REJECT audit listed P1s, every P1 must show
this-turn `Grep`/`Read` proof of a fix before any DONE / live-PASS language.
A prior live receipt does not waive new fail-closed holes.

## Start gate

Run before editing:

```bash
cd /Users/fahadkiani/Desktop/development/scientist-monai-moat
git branch --show-current
git rev-parse HEAD
rg -n "execute_stage|_run_tournament|offline_ranker|elo_rank" \
  src/oncology_arbiter/agents/supervisor.py \
  src/oncology_arbiter/api/routers \
  src/oncology_arbiter/api/schemas
```

Write the requested terminal condition in chat:

`DONE requires identity-locked MedGemma requests, fetched evidence, valid pairwise judgments, Elo updates, and a grounded meta-review. Proxy/offline output fails.`

## Implementation loop

1. Read:
   - `src/oncology_arbiter/agents/supervisor.py`
   - `src/oncology_arbiter/models/llm_client.py`
   - `src/oncology_arbiter/models/medgemma_1_5_4b_wiring.py`
   - `src/oncology_arbiter/api/routers/co_scientist.py`
   - `src/oncology_arbiter/api/schemas/co_scientist.py`
2. Preserve `/v1/offline_ranker/run` as explicitly offline.
3. Wire the real endpoint only to `execute_stage`.
4. Require `MEDGEMMA_MODAL_URL`; alternate LLMs and deterministic rankers are not fallback.
5. Fail closed unless:
   - `model_state == "executed"`
   - provider is `medgemma_modal`
   - `MEDGEMMA_MODAL_URL` host is in `PINNED_CHAT_HOSTS` (`assert_pinned_chat_url`)
   - model/app/revision match `verify_artifact_identity()`
   - `/v1/co-scientist/run` uses `require_verified_api_key` (AUTH_MODE=off is not anon access)
   - every generated hypothesis completes evidence-backed REFLECT before Elo
   - every tournament pair has winner, margin, nonempty reason, and request ID
   - at least one fetched evidence URL survives the honesty gate
   - the meta-review is non-empty
6. Record pairwise winner, reason, pre/post Elo, and model request ID.
7. Never derive a clinical modifier from keywords alone or accept a caller modifier as clinical evidence.
8. Never mark `executed` when `evidence[]` is empty. That state is
   `llm_evidence_empty` and HTTP 503. Pairwise Elo without PubMed/EuropePMC
   hits is the same retreat as restoring `/v1/offline_ranker/run`.
9. `/v1/case/full` `elo_ranked_hypotheses` is `TumorBoardAssembler.elo_ranked_from_medgemma(co_scientist)`.
   A hardcoded `[]` on a successful MedGemma run is a hedge.

Captured live miss: `artifacts/medgemma_co_scientist/live_biopsy_20261016T042510Z.json`
had `executed=true` and `evidence_retained=false`. That receipt is not DONE.

## Test gate

Tests must use a realistic deidentified oncology context and a scripted
identity-bound MedGemma transport. Mock HTTP/model transport and evidence tools;
do not mock `execute_stage`, `_run_tournament`, or Elo arithmetic.

Required assertions:

- GENERATE, REFLECT, TOURNAMENT, and META_REVIEW all consume model responses.
- Retrieved URLs are the only URLs retained.
- Every expected pair has a completed judgment and request ID.
- Elo moves according to those judgments.
- Missing MedGemma configuration returns 503.
- Incomplete pairwise judgments never return a successful real response.
- Offline endpoints remain labelled `deterministic_offline_ranker`.

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_agents_supervisor.py \
  tests/unit/test_real_co_scientist_endpoint.py -q --tb=short

uvx ruff check --select E9,F63,F7,F82,F401,E402 \
  src/oncology_arbiter/agents/supervisor.py \
  src/oncology_arbiter/api/routers/co_scientist.py \
  src/oncology_arbiter/api/schemas/co_scientist.py \
  tests/unit/test_real_co_scientist_endpoint.py \
  scripts/run_live_co_scientist_receipt.py
```

## Live gate

Unit green is not live DONE. With `MEDGEMMA_MODAL_URL` configured, save a
machine-readable receipt under `artifacts/medgemma_co_scientist/` containing:

- model, app version, revision SHA
- model request IDs
- fetched evidence URLs
- expected/completed tournament-pair counts
- pairwise Elo receipts
- meta-review SHA-256

Receipt must include `"verdict": "PASS"` and `gates.verdict_pass=true`.
`PASS = all(gates)` in `scripts/run_live_co_scientist_receipt.py`; the process
exits nonzero otherwise. The command is:

```bash
PYTHONPATH=src MEDGEMMA_MODAL_URL=<pinned chat URL> \
  .venv/bin/python scripts/run_live_co_scientist_receipt.py \
  --stage biopsy --n-hypotheses 2 --n-evidence-top-k 2
```

No receipt path this turn means `IMPLEMENTED_NOT_LIVE_VERIFIED`, never DONE.

## Anti-retreat audit

Before commit/push, post:

| Ask | Implemented path | Easier substitute found | Substitute rejected? | Evidence | Confidence |
|---|---|---|---|---|---|

Any row where the easier substitute became the deliverable is REJECT. Fix it;
do not explain it away.
