-- ai_prediction_ledger_schema.sql
-- Canonical case-level prediction ledger for the oncology-arbiter system.
--
-- Design goals:
--   * Every case that flows through the API leaves exactly one row here.
--   * Every prediction is pinned to code SHA + per-model SHA + input hash so a
--     future reviewer can determine which artifact produced which prediction.
--   * Radiologist ground truth and biopsy outcome are nullable; they are filled
--     in retrospectively when the follow-up becomes available.
--   * Site data managers can export as CSV for external validation studies
--     using `.mode csv` and `.output` in sqlite3.
--
-- Validated with: sqlite3 :memory: < ai_prediction_ledger_schema.sql
--
-- This schema is versioned; changes require a migration file
-- `artifacts/reports/migrations/YYYY-MM-DD_<slug>.sql` and are reviewed by the
-- IRB/HRPP office before deployment.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS predictions (
    -- Primary key: opaque, non-PHI. The site maintains a separate mapping
    -- between case_id and internal MRN/accession per HIPAA §164.312(a).
    case_id                      TEXT    PRIMARY KEY,

    -- Input provenance: hash of the raw DICOM byte stream received by the API.
    -- SHA-256, 64 hex characters. Reproducibility contract: same bytes ⇒ same
    -- prediction (modulo non-deterministic model paths, which must be
    -- documented in model_versions_json).
    input_dicom_sha256           TEXT    NOT NULL CHECK (length(input_dicom_sha256) = 64),

    -- Code provenance: SHA of the oncology-arbiter git commit that produced
    -- the prediction. Full 40-char git SHA.
    code_sha                     TEXT    NOT NULL CHECK (length(code_sha) = 40),

    -- Model provenance: JSON blob of per-model SHAs.
    -- Schema:
    --   {
    --     "l4a_screening": "<sha256>",
    --     "l4b_biopsy":    "<sha256>",
    --     "l4c_therapy":   "<sha256>",
    --     "l3_arbiter":    "<sha256 of arbiter.json>",
    --     "l3_arbiter_version": "<semver>",
    --     "code_version":  "<oncology_arbiter.__version__>"
    --   }
    model_versions_json          TEXT    NOT NULL,

    -- Full prediction envelope, exactly as returned by /v1/case/full.
    -- Includes arbiter_score, term_contributions, driving_feature,
    -- evidence[], honesty_gate_report, orchestrator_trace.
    prediction_json              TEXT    NOT NULL,

    -- Radiologist ground truth (filled in after the reader interprets the case).
    -- Nullable during the imaging→reading gap.
    -- Values: NULL | 'BI_RADS_0' | 'BI_RADS_1' | 'BI_RADS_2' | 'BI_RADS_3' |
    --                'BI_RADS_4A' | 'BI_RADS_4B' | 'BI_RADS_4C' | 'BI_RADS_5' | 'BI_RADS_6'
    radiologist_final_call       TEXT,

    -- Which reader read this case (opaque reader ID; the site maps to name).
    radiologist_id               TEXT,

    -- Whether the radiologist was shown oncology-arbiter output while reading.
    -- Aim 2 (reader study) crossover flag. Values: NULL | 'unaided' | 'aided'.
    arm                          TEXT
                                 CHECK (arm IN ('unaided', 'aided') OR arm IS NULL),

    -- Biopsy outcome (only populated for cases that went to biopsy).
    -- Values: NULL | 'malignant' | 'benign' | 'benign_without_callback' | 'not_biopsied'
    biopsy_outcome               TEXT
                                 CHECK (biopsy_outcome IN
                                        ('malignant', 'benign',
                                         'benign_without_callback',
                                         'not_biopsied')
                                        OR biopsy_outcome IS NULL),

    -- Days from imaging to biopsy resolution (nullable).
    days_to_outcome              INTEGER
                                 CHECK (days_to_outcome IS NULL OR days_to_outcome >= 0),

    -- Adjudicated ground truth pathology (from tumor board; may differ from
    -- the raw biopsy_outcome after multidisciplinary review).
    adjudicated_pathology        TEXT,

    -- Timestamps in strict ISO-8601 UTC, e.g. '2026-07-01T00:00:00Z'.
    -- We do NOT use SQLite's DATETIME because it is timezone-ambiguous.
    created_at                   TEXT    NOT NULL
                                 CHECK (created_at LIKE '____-__-__T__:__:__%Z'),
    updated_at                   TEXT    NOT NULL
                                 CHECK (updated_at LIKE '____-__-__T__:__:__%Z')
);

-- Indices for retrospective queries.
CREATE INDEX IF NOT EXISTS idx_predictions_code_sha       ON predictions (code_sha);
CREATE INDEX IF NOT EXISTS idx_predictions_arm            ON predictions (arm);
CREATE INDEX IF NOT EXISTS idx_predictions_biopsy_outcome ON predictions (biopsy_outcome);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at     ON predictions (created_at);

-- Audit trail: every mutation to a row is captured as an append-only log.
-- Radiologist calls / biopsy outcomes get filled in later; this table records
-- the diff so a reviewer can reconstruct history without trusting `updated_at`.
CREATE TABLE IF NOT EXISTS prediction_updates (
    update_id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id                      TEXT    NOT NULL REFERENCES predictions (case_id),

    -- Which field(s) changed. JSON: {"radiologist_final_call": {"old": null, "new": "BI_RADS_2"}, ...}
    diff_json                    TEXT    NOT NULL,

    -- Who made the change (opaque user_id). Optional (system updates set NULL).
    updated_by                   TEXT,

    updated_at                   TEXT    NOT NULL
                                 CHECK (updated_at LIKE '____-__-__T__:__:__%Z'),

    -- Signature over `case_id || diff_json || updated_at` using the site's
    -- signing key. Deferred for v1.0 — kept nullable for now, becomes NOT NULL
    -- once signing infrastructure is in place. Tracked in errata_2026_signature.md.
    signature                    TEXT
);

CREATE INDEX IF NOT EXISTS idx_prediction_updates_case_id ON prediction_updates (case_id);

-- Institutional metadata table: names the sites participating in the study.
-- Kept minimal — full site info lives outside this ledger to reduce PHI surface.
CREATE TABLE IF NOT EXISTS sites (
    site_id                      TEXT    PRIMARY KEY,
    site_name                    TEXT    NOT NULL,
    dua_id                       TEXT    NOT NULL,
    dua_effective_from           TEXT    NOT NULL,
    dua_effective_to             TEXT
);

-- Convenience view: cases with resolved outcomes only, for primary analysis.
CREATE VIEW IF NOT EXISTS predictions_resolved AS
    SELECT *
    FROM predictions
    WHERE biopsy_outcome IS NOT NULL
       OR radiologist_final_call IN ('BI_RADS_1', 'BI_RADS_2');

-- Convenience view: cases still awaiting outcomes, for follow-up scheduling.
CREATE VIEW IF NOT EXISTS predictions_pending AS
    SELECT case_id, created_at,
           CAST( (julianday('now') - julianday(substr(created_at, 1, 10))) AS INTEGER ) AS days_since_prediction
    FROM predictions
    WHERE biopsy_outcome IS NULL
      AND (radiologist_final_call IS NULL
           OR radiologist_final_call NOT IN ('BI_RADS_1', 'BI_RADS_2'));
