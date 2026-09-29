---
name: oncology-moat-platform
description: >-
  Master hardwire for scientist-monai-moat oncology platform. Aligns Alpha's
  arbiter reports under .cursor/rules/arbriter with live branch truth
  (templates n_training=0). Forces tonight delivery of weights/code — bans
  multi-month evidence theater. Use for oncology arbiter, sob remediations,
  tumor-board backend, or any moat ship claim.
---

# Oncology Moat — Platform Master (TONIGHT)

**RUO.** Deadline: **tonight (same calendar day as Alpha's order)** — not Wave-N in five months.

## Canon (re-read with tools — chat memory untrusted)

| Source | Role |
|--------|------|
| `.cursor/rules/arbriter/` | Alpha's packed sob reports (index, waves, audit items) |
| `artifacts/audit/gate_violation_bf5e54f8.json` | Binding Findings A–L — **RED_NOT_MERGEABLE** |
| `artifacts/audit/quarantine_receipt.json` | Synthetic biopsy + banned ClinicalBERT headlines |
| `docs/regulatory/WEIGHT_REDISTRIBUTION.md` | Named weight floors (CBIS/Mammo/LUNA refine) |
| Tip `a81e9a9` | Measurement only — **not** training credit |

## Branch truth vs report theater (spot-check every boot)

```bash
cd /Users/fahadkiani/Desktop/development/scientist-monai-moat   # or /workspace/smm
python3 - <<'PY'
import json, pathlib
for p in pathlib.Path('src/oncology_arbiter/arbiter/models').glob('*.json'):
    d=json.loads(p.read_text())
    print(p.name, 'n_training=', d.get('n_training'), 'synthetic=', d.get('n_training_synthetic'))
PY
```

**Expected on current tip (fail if you claim otherwise without SHA proof):**

| File | Truth |
|------|-------|
| `screening_arbiter_template_v0.json` | `n_training=0` — hand coefficients |
| `biopsy_arbiter_template_v0.json` | `n_training=0` |
| `therapy_arbiter_template_v0.json` | `n_training=0` |
| `biopsy_probe_v0.json` | synthetic, `embed_dim=768` |
| Wave reports' `*_arbiter_v1_*` / `*_v2_*` | **NOT on this branch** — do not cite as shipped |

Reports under `arbriter/` that headline Wave-3 AUROC 0.85+/0.92 **without matching files on HEAD** = **sandbag if used as DONE**.

## Always load

1. `anti-sandbagging`
2. `oncology-moat-anti-fortress`
3. **One** capability skill for the active lane
4. This platform skill

## Capability router → tonight artifact

| Lane | Skill | Tonight MUST land (git tip moves) |
|------|-------|-----------------------------------|
| fortress / push / PAT | `oncology-moat-anti-fortress` | Commits on origin; tip ≠ prior |
| stage arbiters (screen/biopsy/therapy logistic) | `oncology-moat-stage-arbiters` | **All 3** templates replaced by real `n_training>0` weights + metrics |
| MedSigLIP slide biopsy head | `oncology-moat-biopsy` | Real `biopsy_probe_v1`, 1152-d, trained and wired |
| CBIS / screening | `oncology-moat-cbis-medsiglip` | New trained joblib + product wiring + held-out metrics |
| ClinicalBERT | `oncology-moat-clinicalbert` | New real-corpus weights **and** matching live deployment |
| LUNA16 | `oncology-moat-luna16` | Resample/crash code **and** trained refinement weights |
| Phikon | `oncology-moat-phikon` | Probe joblib + embeddings sha pinned |
| MedGemma/Gemma | `oncology-moat-medgemma` | Correct model weights served; identity and chat live |
| Mammo RetinaNet | `oncology-moat-weight-floors` | Real trained checkpoint + AUROC ≥0.85 + product-loaded SHA |
| Weight floors / NOT_TRAINED | `oncology-moat-weight-floors` | Train every named checkpoint; no null/pending status |

**Default order tonight if Alpha says "platform":**
`stage-arbiters` → `biopsy` → `clinicalbert` → `luna16` → `cbis-medsiglip` → `phikon` → `medgemma`.

## No-exit law

These labels and substitutions are forbidden in any reply, receipt, commit, or
status field: `PENDING`, `BLOCKED`, `DEFERRED`, `NOT_TRAINED`, `KILL`,
`TOMBSTONE`, capability removal, claim deletion, fail-closed substitution, or
"download started" as completion. A missing dataset triggers acquisition in
the same run; missing credentials trigger anti-fortress; a failed train triggers
debug/retry. The terminal condition is a trained artifact wired into the
capability and pushed to origin.

## Global bans (instant abort)

- Using `report_*.md` headlines as substitute for files on `HEAD`
- "Items 1–5" root-cause / inventory / md5 digs as training DONE
- Proof-only pushes (`docs/proofs/**` without `models/` or `src/`/`deploy/`)
- Fortress (MCP contents:write, bundle-only, plan-forbids-PAT)
- Promising five-month Wave-2 as answer to tonight order
- Claiming Wave-6 trained arbiters present when only `*_template_v0.json` exists
- Converting any capability to pending/blocked/deferred, deleting it, or
  substituting a warning/error path for working weights

## Boot checklist (post in reply)

```text
[ ] Read arbriter/report_wave_reports_index.md §4–§5 (ClinicalBERT number table)
[ ] Shell: n_training dump for arbiter/models/*.json
[ ] Shell: git ls-remote tip
[ ] Declared tonight lane + skill name
[ ] Forbidden list: 3 bullets
```

## Platform DONE tonight

Every capability DONE table must be green with weight + wiring + held-out
metrics + origin SHA. Metric floors may honestly fail, but the model must be
trained, persisted, loaded, exercised, and pushed. **No capability may be
removed or parked.**

Before any DONE/PASS/push-ready claim, create the machine-readable
`artifacts/tonight_delivery.json` from the real outputs and run:

```bash
python .cursor/skills/oncology-moat-platform/validate_tonight_delivery.py \
  --repo /path/to/scientist-monai-moat
```

The validator requires all ten capability entries, unique non-LFS artifacts,
non-synthetic `n_training>0`, patient-disjoint schema-v2 train manifests,
tracked dataset + split manifests whose SHA256 values are recomputed, disjoint
sample/patient IDs, exact held-out-row coverage bound to target hashes,
recomputable metrics, capability floors, unique artifact-bound wiring/tests,
and exact SHA linkage. It rejects symlinks, duplicate/reused artifacts or
evidence paths, aggregate-only metrics, 768-d biopsy heads, Qwen labeled as
Gemma, missing Mammo weights, and untracked files. Neural checkpoints
(ClinicalBERT, LUNA16, Mammo RetinaNet) must be exported as structurally valid
Safetensors; pickle/ZIP/ONNX stand-ins are rejected. Biopsy's floor is AUROC
only, and LUNA's `delta_froc_at_2` is recomputed as candidate minus baseline.
**A failing gate requires continued execution, not a status-label escape.**
