# Model card: txgemma-9b-chat (L4c primary — currently gated)

**Repo**: `google/txgemma-9b-chat` (HAI-DEF gated)
**Backup repo**: `google/txgemma-9b-predict`
**Status under current session token (2026-07-03)**: **HTTP 403 FORBIDDEN**.
The token `HF_TOKEN=hf_SfwLOG…` has NOT accepted the HAI-DEF terms for
TxGemma. Preflight refuses to load weights; endpoint falls through to the
NCCN-lite rules proxy (see `therapy_nccn_lite.md`) when that flag is on.

**Status**: **RESEARCH USE ONLY** — NOT FDA cleared. Even if terms are
accepted and weights load, TxGemma is a Google research LLM whose outputs
are language-model recommendations, NOT verified clinical decisions.

## Intended use (when reachable)

Given a structured feature set (receptor status, grade, stage, age,
menopausal status, biopsy subtype), generate a therapy plan citing
NCCN sections. Output flows into `TherapyResponse.recommended_options`
tagged `ModelState.LOADED_TXGEMMA`.

## Wire contract

* Env flag: `ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA=1`
* Endpoint: `POST /v1/therapy/reason`
* Precedence: TxGemma first → rules-lite fallback if enabled → placeholder.
* NEVER silently falls back — a `txgemma_gated:<level>:<reason>` warning
  MUST appear on every response where preflight was denied.

## What this model is NOT

* **Not a certified clinical decision system**. LLM-generated
  recommendations must be reviewed by a certified breast oncologist.
* **Not a citation oracle**. TxGemma can hallucinate NCCN section
  numbers; the frontend MUST cross-check any cited section against the
  live NCCN PDF.
* **Not verified on any breast cancer benchmark by us**. Google
  publishes TxGemma primarily as a research LLM for translational
  therapeutics use cases; specific breast-cancer performance is not
  claimed here.

## Failure modes

| Preflight outcome | Response |
|-------------------|----------|
| ALLOWED (never reached here) | `loaded_txgemma` + recommendations |
| FORBIDDEN (403, current)     | `gated` → falls through to rules-lite (if flag on) or placeholder |
| UNAUTHENTICATED (401)        | `gated` → same fallthrough |

## Honesty warning (verbatim, emitted when loaded)

> TxGemma is a Google research LLM (HAI-DEF gated). Its outputs are
> recommendations from a generative language model, NOT verified
> clinical decisions. It MUST NOT be used to make treatment choices.
> Real clinical use requires a certified breast oncologist and a full
> guideline consultation. RESEARCH USE ONLY.

## Preflight verification (2026-07-03, worker-0)

```
GET https://huggingface.co/api/models/google/txgemma-9b-chat
Authorization: Bearer hf_SfwLOG…
→ HTTP 403
```

Same result for `google/txgemma-9b-predict`.

RESEARCH USE ONLY. See `AUROC_CAVEAT` and `RUO_DISCLAIMER` in
`src/oncology_arbiter/__init__.py`.
