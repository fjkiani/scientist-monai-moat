# Model Card — NSCLC Lung Heuristic + NCCN-lite (Oncology Arbiter v0.2)

**Model name (wire):** `nsclc_lung_heuristic_v0+nccn_nsclc_lite_v0`
**Model state (wire):** `proxy_lung_heuristic`
**Intended use:** Research / demonstration only.
**Not for use:** Diagnosis, screening, therapy selection, or any clinical decision. **Not FDA-cleared.**

## 1. What this actually is

This card describes the NSCLC track that ships behind
`POST /v1/case/full?cancer=nsclc`. It is a two-stage **classical** pipeline
plus a **rules-lite lookup**:

1. **Lung heuristic** (`oncology_arbiter.lung.pipeline.run_lung_heuristic`)
   — Hounsfield-unit thresholding + connected-components on a CT volume.
2. **Arbiter** (`oncology_arbiter.lung.arbiter.score_nsclc`) — piecewise
   linear map from max-nodule-diameter to logit, plus a small count bonus.
3. **NCCN-lite therapy** (`oncology_arbiter.models.nccn_nsclc_rules.score_nsclc_therapy`)
   — hardcoded dict keyed on risk bucket + mass-addendum flag; every
   option carries an NCCN URL and section marker.

**There is no trained model in this track.** No neural network is
invoked; no weights are loaded; no dataset was used to fit anything.
Threshold values were pinned from radiology domain literature and the
diameter→logit anchors were chosen to interpolate through common
Fleischner-2017 nodule-size boundaries. If you flip
`ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1` and point `nsclc_ct_input.series_dir`
at a CT series, you get real numbers from the heuristic; if you don't,
the endpoint returns a shape-only placeholder with the same JSON schema.

## 2. Inputs / outputs

**Input** — one CT series directory on the server. LIDC-IDRI layout is
assumed: `<root>/LIDC-IDRI-*/<StudyUID>/CT_<SeriesUID>/*.dcm`. A single
`CT_<SeriesUID>` directory is the granularity the API expects; the
walker in `oncology_arbiter.data.lidc_idri` can resolve the first CT
series for a given patient id.

**Output** — an `NsclcResponse` block on `FullCaseResponse.nsclc`:

| Field | Meaning |
| --- | --- |
| `lung_voxel_fraction` | fraction of voxels in the dilated lung mask |
| `n_candidates_total` | total connected components in the intersect |
| `n_candidates_kept` | components surviving `[min_voxels, max_voxels]` |
| `max_diameter_mm` | isotropic-equivalent diameter of the largest kept blob |
| `candidates[]` | up to `top_n` blob records with (label, voxel_count, diameter_mm, mean_hu, centroid_zyx_vox) |
| `risk_score` | sigmoid(logit) — **not** a calibrated probability |
| `risk_bucket` | one of `NEGATIVE / LOW / MID / HIGH` |
| `driving_feature` | `max_diameter_mm` or `mass_diameter_gt_30mm` or `multiple_candidates` |
| `logit` | raw arbiter logit before sigmoid |
| `therapy_recommended[]` | NCCN-lite options for the bucket, plus mass addendum if HIGH+>30 mm |
| `therapy_not_recommended[]` | anti-recommendations (e.g. no chemo for LOW) |
| `read_seconds`, `heuristic_seconds`, `n_slices`, `series_dir` | provenance / timing |

Every therapy option carries `citation_url`, `nccn_section`, and
`rationale` fields, and every response includes an `NSCLC_RULES_PROXY_WARNING`
in `warnings[]` that names what this pipeline is not.

## 3. Algorithm details

### 3.1 Hounsfield-unit thresholds (`oncology_arbiter.lung.pipeline`)

```
LUNG_HU_MAX      = -500.0   # HU < this ⇒ aerated
BODY_HU_MIN      = -900.0   # HU > this ⇒ soft-tissue silhouette
NODULE_HU_MIN    = -300.0   # HU >= this AND
NODULE_HU_MAX    = +200.0   # HU <= this ⇒ candidate voxel
DEFAULT_MIN_VOXELS  = 8
DEFAULT_MAX_VOXELS  = 20_000
DEFAULT_DILATE_ITER = 3
```

Per-slice body silhouette: largest connected component of `HU > BODY_HU_MIN`,
then `binary_fill_holes`. Lung mask: `body & (HU < LUNG_HU_MAX)`.
Dilated lung mask (`iterations=3`) then intersected with the nodule HU
window; connected components on the intersect are the candidates.

### 3.2 Diameter → risk (`oncology_arbiter.lung.arbiter`)

Voxel-count → isotropic-equivalent diameter (mm) uses the sphere
formula `d = 2 * (3 * n * dz * dy * dx / (4π))^(1/3)`. The logit is a
piecewise-linear interpolation through anchor points:

```
DIAMETER_LOGIT_ANCHORS = [
    (0.0, -4.0),
    (4.0, -1.5),   # LOW / NEGATIVE hinge
    (8.0,  0.0),   # MID hinge
    (15.0, 1.5),
    (30.0, 3.0),   # mass addendum kicks in
    (60.0, 4.5),
]
BUCKET_LOW_MAX_MM = 4.0   # < 4 mm ⇒ LOW
BUCKET_MID_MAX_MM = 8.0   # < 8 mm ⇒ MID, else HIGH
COUNT_BONUS_GT_2 = 0.25
COUNT_BONUS_GT_5 = 0.5
```

`driving_feature` defaults to `max_diameter_mm`, becomes
`mass_diameter_gt_30mm` on HIGH bucket with diameter > 30 mm, and becomes
`multiple_candidates` on LOW/MID with more than 5 kept blobs.

### 3.3 NCCN-lite therapy (`oncology_arbiter.models.nccn_nsclc_rules`)

An inline Python dict keyed on `risk_bucket`; each bucket has
`recommended` and `not_recommended` lists. Every option carries
`name / category / citation_url / rationale / nccn_section`. HIGH bucket
with `driving_feature == "mass_diameter_gt_30mm"` OR `max_diameter_mm >
30` appends a mass addendum recommending urgent thoracic-oncology
referral. `NsclcTherapyRulesResult.model_state = "proxy_rules_lite"`.

## 4. Data provenance

**LIDC-IDRI** is the reference cohort assumed by the LIDC walker.

- Collection: `lidc_idri`
- TCIA DOI: **10.7937/K9/TCIA.2015.LO9QL9SX**
- Reference paper DOI (Armato *et al.*, Medical Physics 2011):
  **10.1118/1.3528204**
- License: **CC-BY-3.0**
- Patient count: **1,010**
- Reference image count: **244,527**

The pipeline itself never ingests the LIDC XML annotations; only the
DICOM series are used to build the HU volume. `citation_url`s in the
therapy block point at NCCN and Fleischner:

- **NCCN NSCLC v5.2026:** <https://www.nccn.org/professionals/physician_gls/pdf/nscl.pdf>
- **Fleischner 2017 (MacMahon *et al.*, Radiology 284(1):228-243):**
  DOI **10.1148/radiol.2017161659**

## 5. Serving contract

Two-path branch inside `POST /v1/case/full?cancer=nsclc`:

1. **Placeholder / shape-only** — used when `nsclc_ct_input` is absent
   OR `ONCOLOGY_ARBITER_ALLOW_SERIES_DIR` is not truthy. Returns the
   same JSON shape with every numeric field `None`, plus a warning that
   explains what to flip.
2. **Real pipeline** — used when both `nsclc_ct_input.series_dir` is
   set AND `ONCOLOGY_ARBITER_ALLOW_SERIES_DIR=1` on the server. The
   env gate exists to prevent client-controlled filesystem paths from
   being trusted in shared / public deployments.

The API key middleware (`require_api_key`) is the standard tenant
guard. Responses always carry the `RUO_DISCLAIMER` and `AUROC_CAVEAT`
strings on the envelope.

## 6. Evaluation

**None.** No AUROC, no sensitivity / specificity, no per-nodule PR
curve is claimed. The three-patient regression against LIDC-IDRI-0001
/ 0002 / 0003 that ships with the tests is a sanity check that the
same top blob is found across runs (matching `max_diameter_mm` to
within display precision, matching `n_candidates_kept`), not an
accuracy claim.

## 7. Limitations

- Uses classical CV only; anything a trained detector would improve
  (small ground-glass nodules, part-solid lesions, subpleural nodules
  near the diaphragm) is silently missed by HU thresholding.
- The synthetic-cube unit test picks up a soft-tissue body-shell blob
  larger than the planted nodule — the pipeline is behaving as designed;
  the test locates the planted nodule by centroid, not by "biggest blob".
- LIDC-IDRI is a curated screening / thoracic-CT cohort; other CT
  protocols (contrast, spacing, kernel) may shift the HU histograms.
- The NCCN-lite rules do **not** reason about histology, PD-L1,
  driver mutations, staging (T/N/M), comorbidities, prior therapy,
  tumor-board input, or clinical-trial eligibility. Every response
  includes the `NSCLC_RULES_PROXY_WARNING` string that says so.

## 8. Reproducibility

- Unit tests: `tests/unit/test_nsclc_pipeline.py` (42 tests, pinned
  HU thresholds, pinned diameter→logit anchors, sphere-formula
  sanity, synthetic-cube end-to-end with an in-memory volume).
- API tests: `tests/unit/test_case_full_nsclc_api.py` (4 tests,
  synthetic pydicom CT series written at test time — no cohort
  download required for CI).
- Manual three-patient regression: `LIDC-IDRI-0001` → 35.45 mm HIGH,
  `LIDC-IDRI-0002` → 26.75 mm HIGH, `LIDC-IDRI-0003` → 30.22 mm HIGH,
  matching direct-pipeline invocation.

## 9. Contact / license

The pipeline itself is MIT-licensed under the parent repository. LIDC
CT slices are CC-BY-3.0 and are **not** redistributed with this
repository; you must download them yourself from TCIA.
