# AK MBD4-LOF Tumor Board Integration (v0.4.0-alpha)

This document describes how the oncology-arbiter surface consumes the
`tumor_board.v3.multimodal-with-manuscript-claims` bundle emitted by the
`crispro-backend-v2` fix branch `fix/mbd4-atr-strong-tier` at HEAD
`bfd6d11fc872c11a13365b0682cea776a136c7f3`.

## 1. Contract

The single ground-truth artifact is the pydantic model
`oncology_arbiter.api.schemas.TumorBoardBundle`. It has a
`contract_version: Literal["tumor_board.v3.multimodal-with-manuscript-claims"]`
field — the pinning is enforced at parse time by pydantic. The route
`POST /v1/tumor_board/bundle` additionally re-checks the literal so the
error message is more actionable than a raw pydantic ValidationError.

## 2. Provenance chain

Two SHAs travel with every bundle:

| Field                                              | Value                                       | What it locks |
|----------------------------------------------------|---------------------------------------------|---------------|
| `synthetic_lethality.provenance.manuscript_repo_sha_at_audit` | `d33f6403fb11b314c86fa74d9c56e07b7ac3d7b1` | The MBD4-LOF manuscript repo revision that produced the underlying statistics |
| `synthetic_lethality.provenance.backend_head_sha`  | `bfd6d11fc872c11a13365b0682cea776a136c7f3` | The `crispro-backend-v2` build that emitted the bundle |

The CI job `check_ak_bundle` (see §5) fails the pull request if either
value drifts from the shipped bundle without a corresponding change to
this document.

## 3. Anchor values

Eight anchor values come from the 2-minute clinical defense script. Six
are directly present in the bundle at full float precision; two come
from `tumor_board_evidence_chain.json` on the backend side and are
proved-in only by the manuscript SHA:

### In the bundle (verified by `scripts/audit_ak_bundle.py`)

| Modality / stratifier            | Statistic | Value                    | Effect size (Cohen's d) |
|----------------------------------|-----------|--------------------------|-------------------------|
| TP53_mutant_only ln_IC50         | p-value   | `0.003002668797799231`   | `-0.7404782024497254`   |
| MSI_purge ln_IC50                | p-value   | `0.015328932966132268`   | `-0.6227047947387747`   |
| leave_one_out_LOF worst          | p-value   | `0.045165724128583974`   | (not reported)          |
| non_bowel_lineage ln_IC50        | p-value   | `0.02533035508952329`    | `-0.5988100454283156`   |
| MBD4_LOF_vs_WT (axis partner)    | p-value   | `0.07445705343975263`    | `-0.3594348912682576`   |
| PARP1 expression LOF-vs-comparator (falsification) | p-value | `0.6047878879741422` | (not reported) — n_mut=19, n_wt=1498 |

### Manuscript-side only (locked by SHA)

| Endpoint                        | Statistic | Value      |
|---------------------------------|-----------|------------|
| TP53_mutant_only AUC            | p-value   | `0.000873` |
| TP53_mutant_only AUC            | Cohen's d | `-0.889`   |

These two live in `tumor_board_evidence_chain.json` on `crispro-backend-v2`
at manuscript SHA `d33f6403`. They are NOT surfaced by the frontend; only
the six ln_IC50 anchors above render in the SL Evidence Moat.

## 4. Six-row evidence matrix

The evidence matrix must contain exactly six rows in this order — the
`axis` field is the enum key the frontend keys off (see FE-CRIT-001):

| Axis (enum)         | tier                                     | manuscript claim type            |
|---------------------|------------------------------------------|----------------------------------|
| `cytidine_analogs`  | Validated SL therapeutic lever           | `validated_benchmark`            |
| `parp_inhibitors`   | Mechanistic candidate only               | `falsified_mechanism`            |
| `atr_wee1`          | Strong candidate dependency axis         | `primary_new_candidate_axis`     |
| `wrn`               | Not supported / negative                 | *null*                           |
| `immunotherapy`     | Mechanistic candidate only               | *null*                           |
| `pkmyt1`            | Not supported / negative                 | *null*                           |

The `atr_wee1` row carries six auxiliary evidence entries:

- 4 × `modality=stress_test`: `MSI_purge`, `TP53_mutant_only`,
  `leave_one_out_LOF`, `non_bowel_lineage`
- 1 × `modality=axis_partner` (adavosertib vs MBD4-LOF)
- 1 × `modality=falsification_arm` (PARP1 expression LOF-vs-comparator)

The `parp_inhibitors` row carries a `falsification_narrative` of
575 characters that begins with the exact phrase:

> Falsified mechanism — PARP inhibitors (e.g., olaparib, talazoparib) are NOT recommended...

## 5. CI gates

Two mandatory jobs on every PR:

1. **`check_ak_bundle`** — runs `python scripts/audit_ak_bundle.py -v` against
   `src/oncology_arbiter/api/static/demo_samples/ak_mbd4_lof_case.json`.
   Any assertion failure (missing anchor, wrong SHA, wrong claim tier,
   wrong effect size) blocks merge.
2. **`no_hf_token_grep`** — greps the diff for `hf_[A-Za-z0-9]{30,}` and
   fails if any match is found. Prevents accidental HuggingFace token
   leaks in cassettes or test fixtures.

## 6. HIPAA posture

`POST /v1/tumor_board/bundle` reads the `HIPAA_MODE` env var at request
time. When `HIPAA_MODE=true`:

- The response `provenance.model_state` is `LOADED_HIPAA_REDACTOR`
  (instead of `LOADED_AK_BUNDLE`).
- The bundle is echoed verbatim in the response body — the actual
  redaction stub lives in `crispro-backend-v2` at
  `api/middleware/hipaa_pii.py` (class `HIPAAPIIMiddleware`, 8 pattern
  types: email, phone, SSN, MRN, DOB, genomic coordinate, patient id,
  names). The oncology-arbiter surface trusts that upstream redaction
  has already happened.
- The demo sample at `GET /v1/demo/samples/ak_mbd4_lof_case` ships with
  `patient_id="MBD4-LOF-DEMO-01"`, which is the redacted alias of the
  real AK patient in the backend dossier system. The unredacted patient
  id **never** appears in this repository.

Live redaction wiring (including the audit-ledger cross-check with
crispro-backend-v2) lands in PR #6.

## 7. Files added in v0.4.0-alpha

```
src/oncology_arbiter/api/schemas.py                                   # +15 classes, +8 ModelStates
src/oncology_arbiter/api/app.py                                       # +hgsoc cancer, +POST /v1/tumor_board/bundle
src/oncology_arbiter/api/static/demo_samples/ak_mbd4_lof_case.json    # 63,498 bytes, live bundle
scripts/audit_ak_bundle.py                                            # 36-check CI enforcer
docs/audit/AK_MBD4_INTEGRATION.md                                     # this file
docs/regulatory/WEIGHT_REDISTRIBUTION.md                              # weights floor policy
.github/workflows/ci.yml                                              # ak_bundle + no_hf_token + weights_meet_floor
```

## 8. Endpoint quick reference

| Method | Path                                          | Auth      | Purpose |
|--------|-----------------------------------------------|-----------|---------|
| GET    | `/v1/demo/samples`                            | none      | List demo samples (returns `ak_mbd4_lof_case`) |
| GET    | `/v1/demo/samples/ak_mbd4_lof_case`           | none      | Read the AK bundle as static JSON |
| POST   | `/v1/tumor_board/bundle`                      | required  | Validate + echo a caller-supplied bundle; returns bundle_sha256 |
| GET    | `/health`                                     | none      | Lists `hgsoc` under `cancers` |
