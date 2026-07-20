# v0.4.1 ClinicalBERT — production flip checklist

Living document tracking the v0.4.1 flip. Feature branch and main are
now at `a9947e8` on GitHub, Render is deployed and honestly reporting
its state, and Modal is the only remaining external gate.

## Status as of 2026-07-20 03:47 UTC

| gate                                             | state                       |
|--------------------------------------------------|-----------------------------|
| ClinicalBERT fine-tune (5 seeds, F1 ≥ 0.85)      | GREEN (mean 0.9716 ± 0.0001)|
| Local uvicorn smoke (report-only)                | GREEN (14 entities)          |
| Local uvicorn smoke (real pipeline w/ CT)        | GREEN (13 entities, elo n=8)|
| Regression suite (`tests/regression/`)           | GREEN (10/10)               |
| Branch pushed to GitHub main                     | GREEN (`a9947e8`)           |
| Render service redeployed on `a9947e8`           | GREEN (`dep-d9epk9btqb8s73avbhh0`) |
| Render env vars applied (CLINICALBERT_BACKEND)   | GREEN (via Render API PUT)  |
| Prod parsed_report_provenance surfaces honest state | GREEN (`clinicalbert_modal_error`, HTTP 404) |
| **Modal `clinicalbert` app deployed**            | **BLOCKED — spend cap on `crispro-test`** |
| Prod parsed_report populated end-to-end          | **BLOCKED by Modal**        |

## What's already committed on `a9947e8`

- Fine-tuned Bio_ClinicalBERT weights (best seed 42, test micro-F1 = 0.9717).
  On worker-agg at `/workspace/clinicalbert_best/`; staged for Modal bake
  at `/mnt/shared-workspace/shared/clinicalbert_stage/`.
- `deploy/modal/clinicalbert_app.py` — Modal app, `min_containers=1` warm,
  weights mounted into `/model/`.
- `src/oncology_arbiter/nlp/clinicalbert_modal_client.py` — stdlib-only
  urllib client (fits Render 512 MB budget).
- `src/oncology_arbiter/nlp/clinicalbert_local_client.py` — in-process
  torch fallback for local uvicorn dev smoke.
- `src/oncology_arbiter/api/app.py::_run_clinicalbert_parse()` —
  module-scope helper that dispatches on `CLINICALBERT_BACKEND=modal|local`
  and stamps `parsed_report + parsed_report_provenance`. Called from BOTH
  the placeholder branch (report-only, no CT) and the real-pipeline
  branch (CT + `ALLOW_SERIES_DIR=1`). Parse is CT-independent.
- `src/oncology_arbiter/api/schemas.py::NsclcResponse` — declares
  `parsed_report: dict | None` + `parsed_report_provenance: dict | None`.
- `render.yaml` — publishes `CLINICALBERT_BACKEND=modal` +
  `CLINICALBERT_MODAL_URL=https://crispro-test--clinicalbert`.
- `tests/regression/test_clinicalbert_f1_floor.py` (F1 >= 0.85 per seed).
- `tests/regression/test_clinicalbert_modal_wire.py` (render.yaml env,
  stdlib-only client, app.py dispatch, schema fields).
- `docs/model_cards/report_parser_clinicalbert_v1.md` — v0.4.1 rewrite
  with RUO disclaimer + real 5-seed metrics + two known failure modes.
- `docs/PROGRESS_LEDGER.json` — new `L2-report-parser` LIVE.

## Two-bug patch record

- **`cfd1f62`** first v0.4.1 push. Parse dispatch nested inside the
  CT-gated real-pipeline branch. All report-only prod requests fell
  through the placeholder branch, which returned before the parse block.
  Local smokes ran through the real branch (had CT) and looked green,
  masking the shape gap.
- **`a9947e8`** hotfix. Extracted `_run_clinicalbert_parse` helper to
  module scope; both NSCLC branches call it. Report-only prod requests
  now fire the parser and get a truthful `parsed_report_provenance`
  regardless of CT.

## Prod verification receipts

- Local (report-only) receipt: `apex_end_to_end_local_no_ct_v1.json`
  (14 entities, `source: clinicalbert_local`, seed=42, F1=0.9716).
- Local (real pipeline w/ CT) receipt: `apex_end_to_end_local_v1.json`
  (13 entities, `elo n=8`).
- Prod receipt: `apex_end_to_end_prod_v1.json`. Body:
  ```json
  {
    "nsclc": {
      "model_state": "placeholder",
      "model_name": "nsclc_placeholder_v0",
      "parsed_report": null,
      "parsed_report_provenance": {
        "source": "clinicalbert_modal",
        "error": "ClinicalBertModalError: HTTP 404 calling https://crispro-test--clinicalbert-parse.modal.run: b'modal-http: invalid function call\\n'",
        "wall_seconds": 0.272
      }
    }
  }
  ```
  Provenance is honest: parse block fired, Modal endpoint 404s because
  the Modal app is not yet deployed.

## Remaining external flip — Modal deploy

**Blocker**: workspace `crispro-test` returned "workspace billing cycle
spend limit reached" on `modal deploy deploy/modal/clinicalbert_app.py`.

**Path A — raise cap on crispro-test**:

1. Modal dashboard -> workspace `crispro-test` -> Billing -> raise spend
   limit (~$5 headroom is plenty for `min_containers=1` warm on CPU).
2. From this repo root, with `MODAL_TOKEN_ID/SECRET/PROFILE` for
   `crispro-test` in env (already in
   `/mnt/shared-workspace/secrets/env-secrets.sh`):

   ```bash
   source /mnt/shared-workspace/secrets/env-secrets.sh
   modal deploy deploy/modal/clinicalbert_app.py
   ```

3. Smoke:

   ```bash
   curl -sS https://crispro-test--clinicalbert-healthz.modal.run | jq
   curl -sS https://crispro-test--clinicalbert-info.modal.run | jq
   curl -sS -X POST https://crispro-test--clinicalbert-parse.modal.run \
     -H "content-type: application/json" \
     -d '{"text": "EGFR: no activating mutation. KRAS G12C: DETECTED. TMB: 9.2 mut/Mb."}' | jq
   ```

   Expect HTTP 200, `provenance=SYNTHETIC-v0.3.1`, `training_seed=42`,
   `test_micro_f1=0.9716...`, and the RUO disclaimer verbatim.

4. **No Render redeploy required.** `CLINICALBERT_BACKEND=modal` and
   `CLINICALBERT_MODAL_URL=https://crispro-test--clinicalbert` are
   already set on the dyno; the next request will hit the new Modal
   endpoint automatically.

5. Refire prod: `python /mnt/shared-workspace/shared/apex_case/fire_prod_v041.py`.
   Expect `parsed_report_provenance.source=clinicalbert_modal` with
   `training_seed=42` (no `error` field), and `parsed_report` populated.

**Path B — new workspace**:

1. Create workspace with budget; grab `MODAL_TOKEN_ID/SECRET/PROFILE`.
2. Update `render.yaml::CLINICALBERT_MODAL_URL` and the two env vars on
   Render (via API or dashboard) to the new URL prefix
   (`https://<new-ws>--clinicalbert`).
3. Update `tests/regression/test_clinicalbert_modal_wire.py::
   test_render_yaml_declares_modal_env` to match the new prefix.
4. Retry `modal deploy` as in Path A step 2.

## Render env-var management gotcha

Render's `autoDeploy: true` deploys code changes on push to main, but it
does **not** re-apply `envVars` from `render.yaml` on updates — only on
service creation. When `render.yaml` gained `CLINICALBERT_BACKEND` /
`CLINICALBERT_MODAL_URL`, the running service did not pick them up
automatically. Fix: `PUT /v1/services/<sid>/env-vars` with the merged
list (Render replaces the full set on PUT), then `POST /deploys`.

Recorded so we do this reflexively on the next env-var addition. Docs
that reference "just push and Render will pick it up" apply to code
only.

## Recovery of the fine-tuned weights

If `/workspace/clinicalbert_best/` on `worker-agg` is wiped:

- Per-seed durable copies at `/mnt/shared-workspace/shared/clinicalbert_runs/seed<N>/`
  contain `metrics.json`, `train_summary.json`, `corpus_manifest.json`,
  `train.log`. **The safetensors are not in shared-workspace** (S3 does
  not support the safetensors random-access write pattern); they exist
  only on `worker-agg`.
- If lost, re-train with `bash /mnt/shared-workspace/shared/clinicalbert_stage/run_seed.sh 42`.
  Pinned config: SYNTHETIC-v0.3.1, epochs=3, lr=5e-5, batch=8, max_len=192.
  Wall-clock: ~1h on a 32 GB worker.
