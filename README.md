# oncology-arbiter

**RESEARCH USE ONLY — not validated for clinical decision-making.**

Open-architecture breast oncology reasoning platform spanning **screening → biopsy → therapy**, with calibrated per-stage arbiters, cited evidence, and honest performance disclosure.

Built on:
- **MONAI + EfficientDet** for screening mammography lesion detection (domain-trained on EMBED / CBIS-DDSM)
- **MedSigLIP** (`google/medsiglip-448`, 400M+400M vision+text, 448×448) for biopsy histopathology embedding
- **TxGemma 9B** (`google/txgemma-9b`, Gemma-2 based, TDC-trained) for therapy reasoning
- **Co-Scientist** multi-agent orchestrator (open-source re-implementation of Gottweis et al., *Nature* 2026)
- **L2-regularized calibrated logistic arbiters** at each stage, replicating the pattern proven in `progression_arbiter`

## Regulatory posture

**Investigational / IRB path.** This system is built as an IRB-ready validation platform. It carries a Research Use Only disclaimer on every response body and every published artifact. It is designed to be dropped into an academic radiology research partnership for prospective validation study.

**Not FDA-cleared. Not CE-marked. Not intended for clinical use.**

Every stage output includes:
- `arbiter_score` — calibrated probability
- `risk_bucket` — LOW / MID / HIGH
- `recommendation` — SHORT_INTERVAL_RESCAN / ADDITIONAL_WORKUP_REQUIRED / IMMEDIATE_WORKUP (or stage-specific variant)
- `term_contributions` — per-feature score contribution (interpretable arbiter internals)
- `driving_feature` — argmax|contribution| feature name
- `evidence[]` — seen-URL-filtered citations backing the reasoning
- `provenance` — `{ code_sha, model_sha256_l4a, model_sha256_l4b, model_sha256_l4c, arbiter_json_sha, dataset_id_used_for_training }`
- `disclaimer` — RUO language, verbatim

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│ L5 Co-Scientist Orchestrator                              │
│    Generation → Reflection → Ranking → Evolution          │
│    → Meta-review, Elo tournament, SQLite queue            │
└───────────────────────────────────────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
   ┌───────────┐  ┌───────────┐  ┌───────────┐
   │ L4a       │  │ L4b       │  │ L4c       │
   │ Screening │  │ Biopsy    │  │ Therapy   │
   │ Mammo     │  │ Histopath │  │ Reasoning │
   │ (MONAI)   │  │(MedSigLIP)│  │(TxGemma)  │
   └───────────┘  └───────────┘  └───────────┘
         │              │              │
         └──────────────┼──────────────┘
                        ▼
   ┌───────────────────────────────────────────┐
   │ L3 Calibrated Arbiter                     │
   │    L2 logistic, term contributions,       │
   │    driving_feature, AUROC_CAVEAT preserved│
   └───────────────────────────────────────────┘
                        │
                        ▼
   ┌───────────────────────────────────────────┐
   │ L2 Evidence Layer                         │
   │    PubMed / arXiv / Europe PMC            │
   │    Reflection seen_urls honesty filter    │
   └───────────────────────────────────────────┘
                        │
                        ▼
   ┌───────────────────────────────────────────┐
   │ L1 Data Layer                             │
   │    DICOM ingest, EMBED / CBIS-DDSM        │
   │    De-ID (DICOM PS3.15 basic profile)     │
   └───────────────────────────────────────────┘
```

## API surface

```
POST /v1/screening/analyze     { dicom_url } → screening_arbiter output
POST /v1/biopsy/analyze        { wsi_url, report_text } → biopsy_arbiter output
POST /v1/therapy/reason        { biopsy_output, patient_context } → therapy_arbiter output
POST /v1/case/full             { dicom_url, wsi_url?, report_text?, patient_context? }
                               → full Co-Scientist orchestrated pipeline
GET  /v1/artifacts/{cat}/{fn}  path-traversal-safe artifact streaming
GET  /v1/model-cards           per-stage model cards as JSON
GET  /v1/health
```

## Status

**Phase 1 — Foundation** (in progress)

- [x] Repo scaffolded (this commit)
- [ ] FastAPI router with placeholder endpoints — in progress
- [ ] Arbiter template ported from progression_arbiter pattern — in progress
- [ ] Co-Scientist essentials ported — in progress
- [ ] Model card + IRB artifact templates published — in progress
- [ ] MedSigLIP + TxGemma HAI-DEF acceptance — human loop, not yet started
- [ ] EMBED DUA submitted — human loop (~4-8 weeks processing lead time), not yet started
- [ ] CBIS-DDSM ingested — not yet started

**Phase 2 — Screening reader** (blocked on EMBED / CBIS-DDSM ingestion)
**Phase 3-4 — Biopsy + Therapy stages** (blocked on HAI-DEF acceptance)
**Phase 5 — Orchestrator wiring** (unblocks after Phases 2-4)
**Phase 6 — Demo + jedi-v2 integration** (parallel to Phase 5)
**Phase 7 — IRB partner outreach** (parallel to Phase 5-6)

## References

See `PLAN.md` at `/mnt/results/execution_trace/PLAN.md` for the full architecture rationale, verified citations, and honest performance caveats.

## License

Apache 2.0 for our code. Third-party models retain their original licenses:
- MedSigLIP / MedGemma / TxGemma: Health AI Developer Foundations Terms of Use
- Co-Scientist: MIT (per upstream)
- MONAI: Apache 2.0
