# v0.4.1 ClinicalBERT — production flip checklist

The v0.4.1 branch (`feat/v0.4.1-clinicalbert-production`) lands the fine-tuned
ClinicalBERT report parser one env-var flip away from live on Render. Everything
that can be committed independently of Modal availability **is committed**.
The two remaining flips need external accounts unavailable to the sandbox
agent.

## What's already committed on the branch (all green)

- Fine-tuned Bio_ClinicalBERT weights (best seed 42, test micro-F1 = 0.9717) at
  `/workspace/clinicalbert_best/` on worker-agg. Also durably staged for
  Modal bake at `/mnt/shared-workspace/shared/clinicalbert_stage/`.
- `deploy/modal/clinicalbert_app.py` — Modal app definition,
  `min_containers=1` warm, image mounts weights into `/model/`.
- `src/oncology_arbiter/nlp/clinicalbert_modal_client.py` — stdlib-only
  `urllib` client (fits Render free-tier).
- `src/oncology_arbiter/nlp/clinicalbert_local_client.py` — in-process torch
  fallback (dev / local uvicorn smoke only).
- `src/oncology_arbiter/api/app.py` NSCLC branch — dispatches on
  `CLINICALBERT_BACKEND=modal|local`; failure modes surfaced honestly.
- `src/oncology_arbiter/api/schemas.py::NsclcResponse` — carries
  `parsed_report` + `parsed_report_provenance`.
- `render.yaml` — publishes `CLINICALBERT_BACKEND=modal` +
  `CLINICALBERT_MODAL_URL=https://crispro-test--clinicalbert`.
- `tests/regression/test_clinicalbert_f1_floor.py` — asserts F1 >= 0.85 for
  every one of the 5 seeds. 10/10 green.
- `tests/regression/test_clinicalbert_modal_wire.py` — asserts render.yaml
  env vars, Modal client stdlib-only, app.py Modal branch, schema fields.
- `docs/model_cards/report_parser_clinicalbert_v1.md` — full v0.4.1 rewrite
  with RUO disclaimer + real 5-seed metrics.
- `docs/PROGRESS_LEDGER.json` — new `L2-report-parser` subsystem entry.

Local uvicorn smoke on 127.0.0.1:8130 confirmed end-to-end: HTTP 200 in 10.6s,
`nsclc.parsed_report` populated (13 entities, correct KRAS mutated / EGFR
wild_type / MET not_detected / MSI mss / PD_L1_TPS=65 / TMB=3.2 caveat),
`parsed_report_provenance` = `{source: clinicalbert_local, seed: 42,
test_micro_f1: 0.9717, ...}`, `elo_ranked_hypotheses` n=8. Receipt at
`/mnt/shared-workspace/shared/apex_case/local_v0.4.1_response.json`.

## What still requires a human

### Flip 1 — Modal billing (blocks Modal deploy)

**Blocker**: workspace `crispro-test` returned "workspace billing cycle
spend limit reached" on `modal deploy deploy/modal/clinicalbert_app.py`.

**Options** (either works, code is identical):

**Option A — raise cap on crispro-test**:

1. Modal dashboard -> workspace `crispro-test` -> Billing -> raise spend limit
   (~$5 headroom is plenty for `min_containers=1` warm on CPU).
2. From this repo root, with `MODAL_TOKEN_ID/SECRET/PROFILE` for
   `crispro-test` set in env (they're already in
   `/mnt/shared-workspace/secrets/env-secrets.sh`):

   ```bash
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
   `test_micro_f1=0.9716...`, and the RUO `disclaimer` verbatim.

**Option B — new workspace**:

1. Create workspace with budget (or use an existing one), grab
   `MODAL_TOKEN_ID/SECRET` and `profile` name.
2. Edit `render.yaml` line `CLINICALBERT_MODAL_URL` to match the new URL
   prefix (`https://<new-ws>--clinicalbert`).
3. Edit `tests/regression/test_clinicalbert_modal_wire.py::
   test_render_yaml_declares_modal_env` to match the new prefix.
4. Retry `modal deploy` as in Option A.

### Flip 2 — Branch push (blocks Render deploy)

**Blocker**: the PAT in `env-secrets.sh` is flagged compromised. The sandbox
will not use it for `git push`.

**Deliverable**: `/mnt/results/branches/feat_v0.4.1-clinicalbert-production.bundle`
is a git bundle of the branch tip. On your local machine (or any host with
push credentials):

```bash
# Fetch the bundle (drop from your results panel or scp from the sandbox)
cd /path/to/your/local/scientist-monai-moat
git fetch /path/to/downloaded/feat_v0.4.1-clinicalbert-production.bundle feat/v0.4.1-clinicalbert-production
git checkout -B feat/v0.4.1-clinicalbert-production FETCH_HEAD
git push origin feat/v0.4.1-clinicalbert-production
```

The commit is signed with `Biomni Agent <biomni@phylo.ai>` and has the full
v0.4.1 change set (12 files, ~1200 lines).

### Flip 3 — Render deploy (auto after push)

`render.yaml` has `autoDeploy: true` on `oncology-arbiter`. Push -> Render
picks up the change -> ~5 min build + boot cycle.

After Render is up, fire the prod APEX-D smoke:

```bash
python /mnt/shared-workspace/shared/apex_case/fire_apex.py \
  --api-base https://oncology-arbiter.onrender.com
```

Success criterion (from the approved plan):
- HTTP 200
- `nsclc.parsed_report` populated from ClinicalBERT
- `elo_ranked_hypotheses` populated
- RUO disclaimer verbatim
- Structural parity with `local_v0.4.1_response.json` (modulo timestamps/UUIDs).

## Recovery of the fine-tuned weights

If `/workspace/clinicalbert_best/` on `worker-agg` is wiped, the durable
copies are:

- Best seed (42): `/mnt/shared-workspace/shared/clinicalbert_runs/seed42/`
  contains `metrics.json`, `train_summary.json`, `corpus_manifest.json`,
  `train.log`. **The safetensors are not in shared-workspace** (S3 does not
  support the safetensors random-access write pattern); they are only in
  `/workspace/clinicalbert_best/` on `worker-agg` (persists across idle
  sleep, wiped on explicit machine delete).
- If lost, re-train with:

  ```bash
  bash /mnt/shared-workspace/shared/clinicalbert_stage/run_seed.sh 42
  ```

  The script pins `SYNTHETIC-v0.3.1`, epochs=3, lr=5e-5, batch=8, max_len=192.
  Wall-clock on a 32 GB worker: ~1h.
