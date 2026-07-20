# APEX end-to-end audit — v0.4.1 ClinicalBERT parse on prod

**Date**: 2026-07-20
**Prod endpoint**: `POST https://oncology-arbiter.onrender.com/v1/case/full?cancer=nsclc`
**Local endpoint**: `POST http://127.0.0.1:8131/v1/case/full?cancer=nsclc`
**Deployed commit**: `a9947e8` (Render deploy `dep-d9epk9btqb8s73avbhh0`, status=live)
**Fixture**: 955-char NSCLC pathology report, no CT (`nsclc_ct_input` omitted).

## What we fired

```bash
python /mnt/shared-workspace/shared/apex_case/fire_prod_v041.py     # prod
python /mnt/shared-workspace/shared/apex_case/fire_local_no_ct.py   # local
```

Same 955-char report body on both.

## Result matrix

| field                                            | local (`clinicalbert_local`) | prod (`clinicalbert_modal`) |
|--------------------------------------------------|------------------------------|-----------------------------|
| HTTP status                                      | 200                          | 200                         |
| response bytes                                   | 3107                         | 1841                        |
| `nsclc.model_state`                              | `placeholder`                | `placeholder`               |
| `nsclc.model_name`                               | `nsclc_placeholder_v0`       | `nsclc_placeholder_v0`      |
| `nsclc.parsed_report` (key present in schema)    | yes                          | yes                         |
| `nsclc.parsed_report` (value)                    | 14-entity dict               | `None`                      |
| `nsclc.parsed_report_provenance` (key present)   | yes                          | yes                         |
| `nsclc.parsed_report_provenance.source`          | `clinicalbert_local`         | `clinicalbert_modal`        |
| `nsclc.parsed_report_provenance.training_seed`   | 42                           | (n/a — error branch)        |
| `nsclc.parsed_report_provenance.test_micro_f1`   | 0.971672                     | (n/a — error branch)        |
| `nsclc.parsed_report_provenance.error`           | absent                       | `HTTP 404 ...invalid function call` |
| `elo_ranked_hypotheses.n`                        | 0                            | 0                           |
| RUO disclaimer verbatim                          | present                      | present                     |

`elo n=0` on both is expected: the therapy stub that seeds Elo only runs
inside the real-pipeline branch (which requires CT + `ALLOW_SERIES_DIR=1`
gate). On the placeholder branch we correctly return `elo_ranked_hypotheses=[]`
with the placeholder warning. This is documented behavior — an
`elo_n_hypotheses>0` requires the real CT pipeline; there is no path to
Elo on report-only requests on Render (nor should there be — the therapy
tiers depend on the arbiter-scored risk features from CT).

## Prod provenance detail

```json
{
  "source": "clinicalbert_modal",
  "error": "ClinicalBertModalError: HTTP 404 calling https://crispro-test--clinicalbert-parse.modal.run: b'modal-http: invalid function call\\n'",
  "wall_seconds": 0.272
}
```

This is the honest failure shape. The Modal app `clinicalbert` in workspace
`crispro-test` is **not yet deployed** because the workspace's spend cap
is still tripped. Contract preserved: no fabricated parse, no crash, no
silent None; the error is stamped so downstream clients can distinguish
"parser unreachable" from "parser said nothing."

## Local provenance detail

```json
{
  "provenance": "SYNTHETIC-v0.3.1",
  "base_model": "emilyalsentzer/Bio_ClinicalBERT",
  "training_seed": 42,
  "test_micro_f1": 0.971672395401209,
  "app_version": "clinicalbert-local-v0.4.1-alpha",
  "n_tokens": 173,
  "seconds": 2.248,
  "n_entity_types": 14,
  "wall_seconds": 8.489,
  "source": "clinicalbert_local"
}
```

Fourteen entity types extracted from the 173-token report:
`TMB, KI67_PCT, GRADE, HER2_VALUE, ROS1, PD_L1_TPS, ALK, EGFR, KRAS, MET,
HER2_AMP, MSI, T_STAGE, N_STAGE`.

## Two-bug root-cause chain

Prod initially returned `parsed_report_provenance: None` (no `source` key
at all), which is neither "success" nor "modal_error". The chain that
produced the silent None:

**Bug A — parse nested in CT-gated branch.** The v0.4.1 dispatch block
lived inside the real-pipeline branch (`app.py` L2216, post-heuristic).
That branch runs only when both `nsclc_ct_input.series_dir` is set AND
`ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1` is set on the server. On Render,
the series_dir gate cannot safely open (client-controlled filesystem
paths), so every report-only request bailed out at the placeholder
branch above (L2140) before touching the parse block. Fix: extracted
`_run_clinicalbert_parse` helper (module scope) and called it from both
branches. Local smoke on the placeholder branch now returns 14 entities.

**Bug B — Render did not sync new env vars from render.yaml on update.**
Render's `autoDeploy: true` triggers a rebuild on `main` push, but it
does NOT re-apply `envVars` from `render.yaml` on updates — only on
service creation. Even after Bug A was fixed, prod still returned `None`
because `CLINICALBERT_BACKEND` was unset on the running dyno. Verified
via Render API: `GET /v1/services/srv-d943p68js32c73dhqreg/env-vars`
returned no `CLINICALBERT_*` keys. Fix: `PUT` the missing keys via API,
then trigger a fresh deploy. After redeploy, provenance surfaces the
`clinicalbert_modal_error` shape (as it should — Modal is not yet up).

## Live-deploy record

| deploy id                        | status | commit    | note                                    |
|----------------------------------|--------|-----------|-----------------------------------------|
| dep-d9epk9btqb8s73avbhh0         | live   | a9947e8   | manual API redeploy after env-var PUT   |
| dep-d9epe6km0tmc738s18g0         | prior  | a9947e8   | auto from push; env vars unset          |
| dep-d9eotdm1a83c73blcgdg         | prior  | cfd1f62   | first v0.4.1 push (parse in CT branch)  |

## Bundle contract preservation

- Bundle sha256: `0121fc84fc798af57aad78f2c9274506eac769313d4b21329ad9e3775c5b3a4c` (unchanged)
- 7 anchor p-values (unchanged): `0.021484496737088882, 0.015328932966132268, 0.003002668797799231, 0.045165724128583974, 0.02533035508952329, 0.07445705343975263, 0.6047878879741422`
- Manuscript sha: `d33f6403fb11b314c86fa74d9c56e07b7ac3d7b1` (unchanged)
- Backend sha: `bfd6d11fc872c11a13365b0682cea776a136c7f3` (unchanged)
- Elo seed=20260619, k=16 (unchanged)

None of these were touched by v0.4.1.

## What ships when Modal deploys

Currently blocked on `crispro-test` workspace spend cap. Once raised:

```bash
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... MODAL_PROFILE=crispro-test
modal deploy deploy/modal/clinicalbert_app.py
```

That publishes the three URLs:
- `https://crispro-test--clinicalbert-healthz.modal.run`
- `https://crispro-test--clinicalbert-info.modal.run`
- `https://crispro-test--clinicalbert-parse.modal.run`

Render's `CLINICALBERT_MODAL_URL` is already `https://crispro-test--clinicalbert`
(the base — the stdlib client suffixes `-parse.modal.run`). No further
Render changes required.

Expected prod post-modal-live response shape:

```json
{
  "nsclc": {
    "model_state": "placeholder",
    "model_name": "nsclc_placeholder_v0",
    "parsed_report": { "KRAS": "mutated", "EGFR": "wild_type", ... },
    "parsed_report_provenance": {
      "source": "clinicalbert_modal",
      "training_seed": 42,
      "test_micro_f1": 0.971672,
      ...
    }
  },
  "elo_ranked_hypotheses": [],  // still empty without CT
  ...
}
```
