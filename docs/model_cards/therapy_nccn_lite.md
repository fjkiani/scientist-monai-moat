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

## v0.2 hardening (2026-07-04)

### On-disk rules fingerprint

The exact JSON file the engine is compiled against is pinned by SHA-256 at
module import; every `TherapyRulesResult` surfaces the digest on
`rules_sha256`.

| Field | Value |
|-------|-------|
| Rules file | `src/oncology_arbiter/arbiter/models/therapy_rules_v0.json` |
| SHA-256 (v0.2 pin) | `6c3106470f8276e042109c61d04e4d2c95dc250682658c51de0ac827ed0ff316` |
| `rules_model_id` | `nccn-lite-v0` |
| `_EXPECTED_RULES_URL` | `https://www.nccn.org/professionals/physician_gls/pdf/breast.pdf` |

If the file is edited on disk without lifting `model_id` to `nccn-lite-v1`,
the module still loads and clients see the new SHA in `rules_sha256`. Auditors
comparing two runs can therefore detect any silent guideline drift.

### JSON schema drift guard

`_load_rules()` is now strict at import time. It fails LOUDLY with
`RulesetIntegrityError` on any of:

* `model_id != "nccn-lite-v0"`
* `source_document_url != https://www.nccn.org/professionals/physician_gls/pdf/breast.pdf`
* any branch missing `branch_id` / `nccn_section` / `recommended`
* any branch with an empty `recommended` list
* duplicate `branch_id`
* a `branch_id` in the JSON that is not present in the code coverage set
  `_COVERED_BRANCH_IDS` (i.e. someone added a branch to JSON without wiring
  Python)

### Input contract (`strict=True`)

`apply_nccn_lite_rules(..., strict=True)` raises `InvalidInputError` (a
`ValueError`) on:

| Check | Rejects |
|-------|---------|
| `receptor_status` schema | dict missing ER / PR / HER2, non-`bool` values |
| `grade` | anything not `int in [1, 3]`; explicitly rejects `bool` |
| `stage` | non-`str`, empty, or does not match `T[0-4][a-c]?N[0-3][a-c]?M[01]` and does not contain `M1` |
| `menopausal_status` | anything outside `{"premenopausal", "postmenopausal", "unknown", None}` |

`strict=False` remains the default for back-compatibility with the existing
`/v1/therapy/reason` payload.

### Explicit `menopausal_status="unknown"` branch

For HR+/HER2- tumors with unknown menopause status the engine now returns a
safe-default combination and refuses to guess:

* **Tamoxifen (5–10 years)** — safe across peri/post/pre states (BINV-J)
* **Menopause status evaluation (LH/FSH panel)** as a `category="workup"`
  option (BINV-J)
* Warning: `"menopausal_status=unknown: cannot pick AI vs tamoxifen
  deterministically..."` explaining why AI was NOT selected.

The engine will NOT recommend an aromatase inhibitor without a confirmed
postmenopausal status.

### New result envelope fields

```
TherapyRulesResult(
    ...
    rules_sha256: Optional[str] = None,
    rules_model_id: Optional[str] = None,
    branch_id: Optional[str] = None,  # dcis | metastatic | her2_positive |
                                       # triple_negative | hr_positive_her2_negative |
                                       # fallthrough
)
```

### Test coverage snapshot (v0.2, worker-4)

| Suite | Count |
|-------|-------|
| Baseline branch tests (`test_therapy_rules_lite.py`) | 8 |
| Hardening tests (`test_therapy_rules_lite_hardening.py`) | 22 |
| Full repo regression | 495 passed, 86 skipped, 0 failed |
