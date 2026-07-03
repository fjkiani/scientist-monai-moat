# IRB Protocol Template — Oncology-Arbiter Investigational Study

**Sponsor**: `[[SPONSOR_INSTITUTION]]`
**Principal Investigator**: `[[PI_NAME]], [[PI_DEGREES]], [[PI_INSTITUTION]]`
**Co-Investigators**: `[[CO_INVESTIGATORS]]`
**IRB Number**: `[[IRB_NUMBER]]` (to be assigned by `[[IRB_NAME]]`)
**Protocol Version**: `[[PROTOCOL_VERSION]]` (this document is a template — bracketed fields must be filled by the sponsoring institution)
**Protocol Date**: `[[PROTOCOL_DATE]]`
**Software Version Under Study**: `oncology-arbiter [[GIT_SHA]]` (must match a tagged release)

---

## Regulatory statement

This protocol governs a **retrospective, non-interventional evaluation** of the oncology-arbiter software system. The system is `RESEARCH USE ONLY` and has **not** been cleared by the FDA, marked by CE, or authorized by any national medical device authority. Its outputs will **not** be used to change any patient's clinical management during this study. See `src/oncology_arbiter/__init__.py::RUO_DISCLAIMER`.

Performance metrics reported by the training pipeline are subject to the `AUROC_CAVEAT` (see `src/oncology_arbiter/__init__.py::AUROC_CAVEAT`) — literature-derived training labels may inflate apparent discrimination relative to prospective performance. Expected prospective AUROC based on independent validation is 0.70–0.85.

---

## 1. Background & Significance

Screening mammography has a well-documented false-positive burden. Over 10 years of annual screening starting at age 40, the cumulative false-positive **recall** probability is **61.3%**, and the cumulative false-positive **biopsy** probability is **7.0%** (biennial: 41.6% and 4.8%, respectively) [Hubbard et al. 2011, *Annals of Internal Medicine* 155(8):481-492, DOI 10.7326/0003-4819-155-8-201110180-00004, PMID 22007042, PMC3209800].

Vision-language foundation models have demonstrated strong zero-shot performance on medical imaging tasks. Specifically, MedSigLIP (`google/medsiglip-448`, 400M+400M vision+text encoder, 448×448, 64 text tokens; arXiv 2507.05201) reports an invasive breast cancer AUC of 0.933 zero-shot and 0.930 linear-probe on 5,000 cases (3-way classification, biopsy histopathology). Note: MedSigLIP was not trained on screening mammograms; its breast coverage is biopsy histopathology only. Screening mammography classification in oncology-arbiter uses a domain-specific detector trained on EMBED and CBIS-DDSM, arbitrated post-hoc by an L2 logistic scorer combining detector output with subgroup and clinical metadata.

The scientific question this protocol addresses is whether an integrated multi-stage reasoning system that combines a domain-specific mammography detector, a biopsy-stage vision-language model, and a therapy-reasoning language model — arbitrated by a calibrated logistic scorer with source-cited evidence — can reduce the false-positive burden of screening mammography without loss of sensitivity, when used as a second reader.

## 2. Specific Aims

**Aim 1 (Analytical validity).** Characterize the discriminative performance of each stage of the oncology-arbiter system on institution-held-out CBIS-DDSM and EMBED subsets, stratified by breast density, patient race, and patient age. Institution-level splits are used rather than patient-level splits; patient-level splits within a single institution over-estimate external generalizability.

**Aim 2 (Reader-augmentation).** In a retrospective reader study, quantify the change in recall rate at fixed sensitivity when radiologists interpret cases with vs. without oncology-arbiter output visible. Primary outcome: recall rate. Prespecified minimally-detectable-difference: Δ recall = 5 percentage points, α = 0.05 (two-sided), power = 0.80.

**Aim 3 (Fairness).** Test the null hypothesis that oncology-arbiter's arbiter score has equivalent calibration (Brier score, Hosmer-Lemeshow p-value) across strata of race, age, and breast density.

## 3. Study Population

**Inclusion**:
- Screening mammography studies performed at `[[STUDY_SITE]]` between `[[START_DATE]]` and `[[END_DATE]]`.
- Patients ≥ `[[MIN_AGE]]` years old.
- Studies with a documented follow-up outcome (subsequent imaging, biopsy result, or ≥ 12-month clinical follow-up).

**Exclusion**:
- Patients with known genetic risk syndromes (BRCA1/2, TP53, PALB2) — separate study.
- Patients with breast implants at the time of imaging.
- Studies flagged with poor image quality by the acquiring institution.
- Patients who have opted out of AI-based research per site policy.

**Sample size**: `[[N_CASES]]` cases, of which `[[N_MALIGNANT]]` are pathology-confirmed malignant. Sample size computed for Aim 2 primary outcome (Δ recall = 5pp, α=0.05, power=0.80) assuming baseline recall of `[[BASELINE_RECALL]]` and correlated readings (see Statistical Analysis Plan §6).

## 4. Data Sources & Data Use Agreements

**Training data (not enrolling new subjects; already in force)**:
- **CBIS-DDSM** — public, DOI 10.7937/K9/TCIA.2016.7O02S9CY, license CC-BY-3.0. No DUA required. See `tests/fixtures/cbis_ddsm/README.md`.
- **EMBED** — Emory Breast Imaging Dataset, restricted access. DUA `[[EMBED_DUA_ID]]` executed on `[[EMBED_DUA_DATE]]`.
- **TCGA-BRCA** — for tumor characteristics; NIH TCGA public access with citation.

**Evaluation data (enrolling under this protocol)**:
- Retrospective screening mammograms from `[[STUDY_SITE]]` under a data use agreement `[[SITE_DUA_ID]]` executed with `[[STUDY_SITE_INSTITUTION]]`.

**Data transfer**: PHI-limited datasets transferred via `[[TRANSFER_MECHANISM]]` (e.g., site-controlled SFTP, DUA-covered). No PHI leaves the study site's HIPAA-covered environment.

## 5. Model Description

The oncology-arbiter system comprises four models arbitrated by an L2 logistic scorer:

- **L4a — Screening detector**: MONAI-based lesion detector (EfficientDet or Mask R-CNN backbone) trained on CBIS-DDSM + EMBED. Model card: `artifacts/reports/model_card_l4a_screening.md`.
- **L4b — Biopsy classifier**: MedSigLIP zero-shot + linear-probe on biopsy histopathology. Model card: `artifacts/reports/model_card_l4b_biopsy.md`.
- **L4c — Therapy reasoner**: TxGemma; language-only stage that ingests biopsy results and patient context. Model card: `artifacts/reports/model_card_l4c_therapy.md`.
- **L3 — Arbiter**: L2-regularized logistic regression over L4 outputs, feature scores, and metadata. Model card: `artifacts/reports/model_card_l3_arbiter.md`.

Each model's weights, config, and training data manifest are pinned by SHA-256. Model versions are recorded per case in the AI Prediction Ledger (`artifacts/reports/ai_prediction_ledger_schema.sql`). Silent model updates are prohibited during the study window.

## 6. Statistical Analysis Plan

**Primary outcome (Aim 2)**: Recall rate (proportion of studies with BI-RADS ≥ 0 recall or biopsy recommendation) in the AI-augmented arm minus recall rate in the unaided arm.

**Test**: Paired-reader analysis (each radiologist reads both arms in randomized order with a ≥ 4-week washout). Two-sided McNemar test with sample size chosen to detect Δ recall = 5pp at α = 0.05, power = 0.80. Under the assumption of correlation ρ = 0.6 between paired reads and baseline recall of `[[BASELINE_RECALL]]`, the required paired sample size is `[[SAMPLE_SIZE]]` studies.

**Secondary outcomes**:
- Sensitivity at fixed specificity, per-reader and pooled.
- Time-to-decision (median seconds per case in the AI-augmented arm vs unaided arm).
- Subgroup analyses by race, age tertile, and BI-RADS breast density category.

**Interim analysis**: One planned interim at 50% enrollment. O'Brien-Fleming boundary; futility check if conditional power < 20%.

**Handling missing data**: Cases with incomplete follow-up (< 12 months without biopsy or next screen) are censored from the primary outcome and reported separately.

## 7. Risk / Benefit Assessment

**Risks**:
- **Loss of PHI confidentiality** — mitigated by DUA-covered transfer, no cloud egress of PHI, HIPAA §164.312 technical safeguards (see §9).
- **Model-driven decision drift** — mitigated by the study design: AI output does not change clinical management during this study. Radiologists issue the final BI-RADS.
- **Automation bias** in the AI-augmented reader arm — mitigated by paired-crossover design and washout period.

**Benefits**:
- No direct benefit to individual subjects (retrospective).
- Societal benefit if the system demonstrates recall-rate reduction: potential reduction in follow-up imaging and biopsies for future patients.

**Data breach plan**: `[[SITE_INCIDENT_RESPONSE_POLICY]]`.

## 8. Informed Consent Process

Because this is a **retrospective study of previously acquired imaging** with a limited-PHI dataset, we request a **waiver of informed consent** under 45 CFR 46.116(f) on the grounds that:
- The research involves no more than minimal risk.
- The waiver will not adversely affect the rights and welfare of the subjects.
- The research could not practicably be carried out without the waiver.
- Subjects will be provided with additional pertinent information after participation, where appropriate.

For prospective subjects (Aim 2 crossover reader study, if participating radiologists are enrolled as human subjects), the informed consent template at `artifacts/reports/informed_consent_template.md` will be used and modified per HIPAA §164.508.

## 9. Data Security & HIPAA §164.312

- **Access control (§164.312(a))**: Study data resides in a `[[SITE_ENCLAVE_NAME]]` HIPAA-compliant enclave with role-based access. AI predictions logged with `case_id` (opaque, non-PHI mapping maintained separately by the site).
- **Audit control (§164.312(b))**: All API calls to oncology-arbiter are audit-logged to `artifacts/audit/audit-YYYY-MM-DD.jsonl` with `request_id`, code SHA, model SHAs, and outcome. See `src/oncology_arbiter/api/audit.py`.
- **Integrity (§164.312(c))**: Model artifacts pinned by SHA-256. Ledger entries write-once + updated-only via signed diff.
- **Person or entity authentication (§164.312(d))**: API-key-per-user with rotation. No shared credentials.
- **Transmission security (§164.312(e))**: TLS 1.3 minimum. Certificate pinning for the site enclave.

## 10. Adverse Event & Deviation Reporting

Because no patient management changes during this study, "adverse events" in the FDA-device sense are not applicable. However, the following events are reportable within `[[REPORTABLE_WINDOW_DAYS]]` days:

- Any breach or suspected breach of PHI.
- Discovery of a model behavior that, if the system had been used clinically, could have caused patient harm (e.g., systematic failure on a protected subgroup).
- Deviation from the prespecified statistical analysis plan.

Reports go to `[[IRB_NAME]]` and `[[SPONSOR_INSTITUTION_HRPP]]` per `[[REPORTABLE_EVENT_POLICY]]`.

## 11. References

1. Hubbard RA, Kerlikowske K, Flowers CI, Yankaskas BC, Zhu W, Miglioretti DL. Cumulative probability of false-positive recall or biopsy recommendation after 10 years of screening mammography: a cohort study. *Ann Intern Med.* 2011;155(8):481-492. DOI: 10.7326/0003-4819-155-8-201110180-00004. PMID: 22007042. PMC3209800.
2. Sellergren A, et al. MedGemma Technical Report. arXiv 2507.05201.
3. 45 CFR 46 — Protection of Human Subjects. HHS Common Rule.
4. 45 CFR 164 — HIPAA Security & Privacy Rules.
5. Google Health AI Developer Foundations. MedSigLIP model card. https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card

---

*This is a template. All bracketed placeholders must be replaced with institution-specific values before IRB submission. Institution review of the RUO_DISCLAIMER and AUROC_CAVEAT text (from `src/oncology_arbiter/__init__.py`) is required — the language conveys real limitations of the system and must be presented to human subjects in Aim 2 verbatim.*
