---
name: oncology-moat-medgemma
description: >-
  Forces correct MedGemma/Gemma weights and working therapy chat tonight.
  Aligns wave4 604s timeout + seven_endpoint qwen-as-gemma. Bans healthz theater.
---

# Oncology Moat — MedGemma / Gemma (TONIGHT)

**RUO.**

## Canon

- Wave4: MedGemma chat timeout ~605s
- `seven_endpoint_live_io_test_evidence.json`: gemma-fallback = qwen2.5-7b

## Tonight bar — capability must work

1. Acquire/verify the intended MedGemma/Gemma weights and record model/revision
   SHA. Missing access triggers credential resolution and continued download,
   never a blocked/pending exit.
2. Fix image/runtime/prewarm so chat loads those weights and completes under
   a documented SLA. Qwen cannot satisfy a Gemma capability.
3. Make `/info` report the exact loaded model/revision SHA.
4. Wire therapy/co-scientist callers to the working endpoint.
5. Run identity + real chat + caller integration tests.
6. Commit deploy/source changes and push.

Forbidden exits: fail-closed substitution, removal/rename to dodge the
capability, health-only PASS, pending/blocked/deferred, or timeout log as work.

## DONE

| Gate | Evidence |
|------|----------|
| Intended model/revision SHA loaded | |
| Chat SLA met on real prompt | |
| Therapy caller integration PASS | |
| `src/`/`deploy/` in stat | |
| tip moved | |

Log force-add without code = **FAIL**.
