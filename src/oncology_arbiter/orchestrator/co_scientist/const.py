"""Constants for the L5 Co-Scientist clinical reasoning swarm."""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Engine identity

ENGINE_LIVE = "medgemma_clinical_reasoning_swarm"
ENGINE_OFFLINE = "offline_scalar_ranker"
PHASES = ("generate", "reflect", "rank", "evolve", "rank")

# --------------------------------------------------------------------------- #
# Elo / tournament

ELO_K_FACTOR = 16
ELO_INITIAL_RATING = 1500.0
ELO_SCALE = 400.0
TOURNAMENT_SEED = 20260703
LIVE_MAX_PAIRS = 3

# --------------------------------------------------------------------------- #
# Hypothesis generation bounds

MAX_SCREENING_FINDINGS = 4
MAX_THERAPY_OPTIONS = 3
DEFAULT_THERAPY_CONFIDENCE = 0.5
DEFAULT_TOP_N_EVOLVE = 3
DEFAULT_N_VARIANTS = 2
DEFAULT_RETURN_TOP = 8

# --------------------------------------------------------------------------- #
# Offline scalar markers (quarantined path only)

OFFLINE_EVIDENCE_CAP = 10.0
OFFLINE_LOADED_BONUS = 0.5
OFFLINE_PROXY_PENALTY = 0.3
OFFLINE_GATED_PENALTY = 0.5
OFFLINE_SCREENING_CONF_DELTAS = (0.1, -0.1)
OFFLINE_THERAPY_LINE_DELTAS = (1, -1)

# Biopsy subtype alternates for offline tournament contrast / evolve swaps.
BIOPSY_ALTERNATE_MAP = {
    "benign_or_normal": "invasive_carcinoma",
    "in_situ_carcinoma": "invasive_carcinoma",
    "invasive_carcinoma": "in_situ_carcinoma",
    "IDC": "DCIS",
    "DCIS": "IDC",
    "benign": "IDC",
}

BIOPSY_EVOLVE_SWAP_MAP = {
    "invasive_carcinoma": "in_situ_carcinoma",
    "in_situ_carcinoma": "invasive_carcinoma",
    "benign_or_normal": "invasive_carcinoma",
    "IDC": "DCIS",
    "DCIS": "IDC",
    "benign": "IDC",
}

# --------------------------------------------------------------------------- #
# MedGemma JSON chat

MEDGEMMA_JUDGE_MAX_TOKENS = 1400
MEDGEMMA_EVOLVE_MAX_TOKENS = 1400
MEDGEMMA_JUDGE_TEMPERATURE = 0.2
MEDGEMMA_EVOLVE_TEMPERATURE = 0.4
MEDGEMMA_RETRY_TEMPERATURE = 0.0
PATIENT_CONTEXT_JSON_CHARS = 2500
EVIDENCE_JSON_CHARS = 800
EVIDENCE_ITEMS_IN_PROMPT = 2
EVOLVE_CONTEXT_JSON_CHARS = 3500
EVOLVE_ALLOWED_URLS_CAP = 20
RATIONALE_MAX_CHARS = 1200
EVOLVE_RATIONALE_CLIP = 240

MARGIN_LABEL_TO_FLOAT = {
    "clear": 0.9,
    "moderate": 0.6,
    "marginal": 0.35,
    "tie": 0.0,
}

WINNER_ALIASES = {
    "A": "A",
    "B": "B",
    "HYPOTHESIS A": "A",
    "HYPOTHESIS B": "B",
    "DRAW": "DRAW",
    "TIE": "DRAW",
}

JSON_RETRY_SUFFIX = (
    "\n\nReturn ONLY one JSON object. No thought block. "
    "No markdown. No prose. First character must be '{'."
)

JUDGE_SYSTEM = (
    "You are an expert clinical oncologist on a molecular tumor board. "
    "Compare Option A vs Option B for this patient based on:\n"
    "1. Biological mechanism vs receptor profile\n"
    "2. Toxicity burden and contraindications\n"
    "3. Clinical trial evidence\n"
    "Return JSON: "
    '{"winner": "A" | "B" | "DRAW", "margin": float between 0.0 and 1.0, '
    '"clinical_rationale": "..."}\n'
    "Start with '{'. RESEARCH USE ONLY."
)

EVOLVE_SYSTEM = (
    "You are a research-use-only oncology co-scientist. Given the winning "
    "regimen and patient biomarker profile, propose exactly N clinical "
    "adaptations: (1) a toxicity de-escalation variant, (2) a combinatorial "
    "resistance-interception partner (when N>=2). Return ONLY JSON:\n"
    '{"variants":[{"statement":"...","rationale":"...","confidence":0.0}]}\n'
    "confidence in [0,1]. Cite only URLs listed under ALLOWED_URLS. "
    "No markdown. First character '{'. RESEARCH USE ONLY."
)

FALLBACK_RATIONALE = (
    "MedGemma judge returned no clinical_rationale; treated as DRAW."
)
OFFLINE_RATIONALE = "offline_scalar_score"
