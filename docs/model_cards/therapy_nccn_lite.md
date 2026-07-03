# Model card: therapy-nccn-lite (L4c fallback)

**Version**: `nccn-lite-v0` (2026-07-03)
**Type**: **Deterministic rules-based lookup** — NOT a machine learning model.
**Source guideline**: NCCN Guidelines Version 2.2025 — Invasive Breast
Cancer / DCIS, `https://www.nccn.org/professionals/physician_gls/pdf/breast.pdf`.
**Status**: **RESEARCH USE ONLY** — NOT FDA cleared, NOT for clinical
decision-making.

## Intended use

Serve as a **transparent, cited fallback** when the HAI-DEF-gated TxGemma
recommender is not reachable (which is the case under the current session
token — 403 forbidden). Given receptor status, grade, stage, and optionally
menopausal status + biopsy subtype, return a small set of therapy options
each stamped with the exact NCCN section that authorized it.

## What this model is NOT

* **Not a full NCCN parser**. It encodes a small subset of the
  guideline branches (DCIS, metastatic, HER2+, TNBC, HR+/HER2-). It
  does not encode contraindications, drug interactions, prior therapy
  history, tumor board consensus, or trial enrollment.
* **Not a substitute for a breast oncologist**. Every recommendation
  MUST be reviewed by a certified breast oncologist and cross-checked
  against the live NCCN guideline PDF.
* **Not a trained model**. `n_training` is null; there is no
  training set. All rules are hand-encoded from published NCCN sections.

## Rule branches

| Branch | Trigger | NCCN section |
|--------|---------|--------------|
| DCIS | `subtype == "DCIS"` | DCIS-1 / DCIS-2 / DCIS-3 |
| Metastatic | `stage contains "M1"` | BINV-Q / BINV-R / BINV-P |
| HER2 positive | `HER2 == true && !M1` | BINV-J / BINV-K / BINV-L |
| Triple negative | `!ER && !PR && !HER2 && !M1` | BINV-M / BINV-J |
| HR+/HER2- (default) | `(ER || PR) && !HER2 && !M1` | BINV-J / BINV-N |

Full rule table lives in `src/oncology_arbiter/arbiter/models/therapy_rules_v0.json`.

## Wire contract

* Env flag (opt-in fallback): `ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY=1`
* Env flag (primary): `ONCOLOGY_ARBITER_ENABLE_THERAPY_TXGEMMA=1` — attempts
  TxGemma first, falls through to rules on FORBIDDEN/UNAUTHENTICATED.
* Endpoint: `POST /v1/therapy/reason`
* Output: `TherapyResponse` with
  * `provenance.model_state ∈ {placeholder, loaded_txgemma, proxy_rules_lite}`
  * Each recommendation carries an `EvidenceRecord` with `url =
    https://www.nccn.org/professionals/physician_gls/pdf/breast.pdf`
    and `quoted_text = "NCCN <section>"`
  * `warnings` always includes the rules-lite honesty warning.

## Honesty warning (verbatim, always emitted)

> This therapy recommendation is from a rules-lite lookup, NOT from a
> trained ML model or a live TxGemma agent. It maps receptor/stage/grade
> to a small fixed table of published NCCN Guideline sections. It does
> NOT reason about individual comorbidities, drug interactions, prior
> therapy history, tumor board consensus, or trial enrollment. It MUST
> NOT be used for treatment decisions. Real clinical use requires a
> certified breast oncologist and a full guideline consultation.

## Failure modes

| Scenario | Response |
|----------|----------|
| Both flags off | `placeholder` + empty recommendations |
| TxGemma on + gated + rules-lite on | `proxy_rules_lite` + `txgemma_gated:forbidden:...` warning |
| TxGemma on + gated + rules-lite off | `placeholder` + `txgemma_gated:forbidden:...` warning (NO silent fabrication) |
| Rules-lite on only | `proxy_rules_lite` with cited recommendations |
| Input matches no branch | `proxy_rules_lite` with empty options + warning |

## Reproducibility

* Rule table: `src/oncology_arbiter/arbiter/models/therapy_rules_v0.json`
* Deterministic: same input → identical output.

RESEARCH USE ONLY. See `AUROC_CAVEAT` and `RUO_DISCLAIMER` in
`src/oncology_arbiter/__init__.py`.
