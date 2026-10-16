---
name: no-path-of-least-resistance
description: >
  Blocks Zo from finishing Alpha's hard ask via the easier public route, proxy,
  offline ranker, mock, empty-evidence "executed" path, or by declaring DONE
  while an audit still has open P1s. Use whenever Alpha asks for the real/live
  /identity-locked path, says Zo hedged/retreated/drifted, attaches
  anti-sandbagging, or a subagent/zeta-enforcer returns REJECT/escalated.
---

# No path of least resistance

## Captured failure (2026-10-16 — Elo / Co-Scientist)

Alpha asked for **real** Elo + Co-Scientist. Zo hedged twice:

1. Restored public `POST /v1/offline_ranker/run` / treated `/v1/elo/rank` as the product.
2. After a later live MedGemma receipt, claimed DONE while a REJECT audit still
   listed open P1s (auth-off anon, unpinned Modal URL, unreflected hyps in Elo,
   incomplete pair JSON, ruff/token field holes).

**That is the path of least resistance:** ship the softer story, keep tests green,
leave the hard fail-closed holes for "follow-up." Forbidden.

Repo: `/Users/fahadkiani/Desktop/development/scientist-monai-moat`

Companions:

- `.cursor/skills/real-elo-co-scientist-no-retreat/SKILL.md`
- personal `~/.cursor/skills/no-path-of-least-resistance/SKILL.md`
- personal `~/.cursor/skills/audit-reject-no-hedge/SKILL.md`
- personal `anti-sandbagging`

## Hard laws

1. **The easier substitute is not the deliverable.**
2. **AUDIT-REJECT ≠ DONE** — fix every P1 with this-turn Grep proof before DONE.
3. **UNIT-GREEN ≠ LIVE-DONE.**
4. **LIVE-PARTIAL ≠ DONE** — empty evidence / incomplete REFLECT / incomplete pairs = FAIL.
5. **SKILL-ON-DISK ≠ SKILL-RUN** — post depth checklist or label PARTIAL.

## Forbidden substitutes

- `POST /v1/offline_ranker/run`
- `POST /v1/elo/rank`
- `orchestrator.co_scientist.run_co_scientist`
- `run_placeholder()`
- mocking `execute_stage` / `_run_tournament`
- `executed` with empty `evidence[]`
- DONE while audit P1s open

Required path: `execute_stage` via `POST /v1/co-scientist/run` with
`require_verified_api_key` + `assert_pinned_chat_url`.

## Start gate

```bash
cd /Users/fahadkiani/Desktop/development/scientist-monai-moat
git branch --show-current
git rev-parse HEAD
rg -n "offline_ranker|require_verified_api_key|assert_pinned_chat_url|elo_ranked_hypotheses=\[\]|llm_evidence_empty" \
  src/oncology_arbiter/agents/supervisor.py \
  src/oncology_arbiter/api/routers \
  src/oncology_arbiter/api/services/tumor_board.py \
  src/oncology_arbiter/models
```

Terminal condition (write in chat before coding):

`DONE requires identity-locked MedGemma, pinned chat URL, verified API key, fetched evidence, every-hyp REFLECT, pairwise Elo with margin+reason, meta-review, zero open audit P1s.`

## Depth checklist

| # | Gate | Status |
|---|---|---|
| 1 | Branch + HEAD this turn | |
| 2 | Retreat Grep this turn | |
| 3 | `/v1/co-scientist/run` → `execute_stage` | |
| 4 | Offline routes not the deliverable | |
| 5 | `require_verified_api_key` | |
| 6 | `assert_pinned_chat_url` | |
| 7 | Empty evidence fail-closed | |
| 8 | Every hyp REFLECT before Elo | |
| 9 | Pair margin+reason+request_id | |
| 10 | Focused pytest this turn | |
| 11 | Ruff E9,F63,F7,F82,F401,E402 | |
| 12 | Live receipt or honest NOT_LIVE | |
| 13 | Audit P1 count = 0 | |
| 14 | Anti-retreat table | |

## Test + live

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

PYTHONPATH=src MEDGEMMA_MODAL_URL=<pinned chat URL> \
  .venv/bin/python scripts/run_live_co_scientist_receipt.py \
  --stage biopsy --n-hypotheses 2 --n-evidence-top-k 2
```

## Anti-retreat audit

| Ask | Implemented path | Easier substitute found | Substitute rejected? | Evidence | Confidence |
|---|---|---|---|---|---|
