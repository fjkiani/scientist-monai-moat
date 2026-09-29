---
name: oncology-moat-phikon
description: >-
  Forces Phikon tonight to pin embeddings sha + commit fitted probe joblib.
  Aligns arbriter Finding L / population validation — bans 5-PNG + metrics-only.
---

# Oncology Moat — Phikon (TONIGHT)

**RUO.** Not a screening claim. Still must ship artifacts.

## Forbidden

- Metrics JSON without joblib
- `/workspace/pathology_crc/...` only, no sha pin
- 5 fixture PNGs as train set

## Tonight bar

1. Locate embeddings. If absent, acquire NCT-CRC-HE-100K +
   CRC-VAL-HE-7K and run `scripts/embed_phikon_nct_crc.py` immediately;
   continue until embeddings complete. A PID is progress, not delivery.
2. Verify complete `done_mask`, finite 768-d rows, patient-disjoint official
   split, and embedding/manifest SHA256.
3. Fit and `joblib.dump` → `models/phikon_nct_crc_logreg_v1.joblib`; commit.
4. Wire the product to load the joblib and expose its SHA.
5. Update `docs/proofs/phikon_nct_crc_population_validation.json` with weight
   SHA, embedding SHA, train/held-out metrics, and exact reproduce command.
6. Run integration smoke, commit, and push.

Forbidden exits: missing-data status, download/embedding PID as completion,
metrics-only, tombstone, capability removal, pending/blocked/deferred.

## DONE

| Gate | Evidence |
|------|----------|
| complete embeddings + manifest SHA | |
| joblib on HEAD + SHA256 | |
| held-out metrics + product loaded SHA | |
| tip moved | |
