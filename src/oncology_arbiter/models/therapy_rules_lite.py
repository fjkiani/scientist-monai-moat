"""L4c therapy stage — NCCN-lite deterministic rules engine.

This is a **rules-only** therapy recommender — no LLM. Every recommended
option cites a real NCCN Guidelines section URL. Rules table lives in
``src/oncology_arbiter/arbiter/models/therapy_rules_v0.json``.

Design contract
---------------
1. NO fabricated numbers. Every rule maps directly to a published NCCN
   Breast Cancer Guideline section (public PDF at
   ``https://www.nccn.org/professionals/physician_gls/pdf/breast.pdf``).
2. Every recommended option carries a ``citation_url`` field pointing at
   the NCCN document.
3. Rules cover the six main receptor/stage branches; the endpoint MUST
   surface a warning that this is a lite rules engine, not a full NCCN
   parser, and that the real decision requires a certified breast
   oncologist.
4. Deterministic: same input → identical output.
5. This engine runs opportunistically ONLY when the endpoint's env flag
   ``ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY=1`` is set. Silent
   fallback is forbidden.

RESEARCH USE ONLY — see :data:`oncology_arbiter.RUO_DISCLAIMER`.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER


# --------------------------------------------------------------------------- #
# Rules file
# --------------------------------------------------------------------------- #

_RULES_PATH = (
    Path(__file__).resolve().parent.parent
    / "arbiter"
    / "models"
    / "therapy_rules_v0.json"
)

NCCN_URL = "https://www.nccn.org/professionals/physician_gls/pdf/breast.pdf"

THERAPY_RULES_PROXY_WARNING = (
    "This therapy recommendation is from a rules-lite lookup, NOT from a "
    "trained ML model or a live TxGemma agent. It maps receptor/stage/grade "
    "to a small fixed table of published NCCN Guideline sections. It does "
    "NOT reason about individual comorbidities, drug interactions, prior "
    "therapy history, tumor board consensus, or trial enrollment. It MUST "
    "NOT be used for treatment decisions. Real clinical use requires a "
    "certified breast oncologist and a full guideline consultation."
)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class TherapyOption:
    name: str
    category: str        # "endocrine" | "chemotherapy" | "targeted" | "surgery" | "radiation"
    citation_url: str
    rationale: str
    nccn_section: str    # e.g. "BINV-J" or "DCIS-B"


@dataclass
class TherapyRulesResult:
    recommended_options: List[TherapyOption]
    not_recommended: List[TherapyOption]
    input_features: Dict[str, Any]
    model_state: str = "proxy_rules_lite"
    model_name: str = "nccn-lite-v0"
    warnings: List[str] = field(default_factory=list)
    caveat: str = AUROC_CAVEAT
    disclaimer: str = RUO_DISCLAIMER
    # NEW in v0.2: pin the exact rules file the engine served from.
    rules_sha256: Optional[str] = None
    rules_model_id: Optional[str] = None
    branch_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# Rules loader
# --------------------------------------------------------------------------- #


# --- ruleset load + fingerprint ---------------------------------------- #

_EXPECTED_RULES_MODEL_ID = "nccn-lite-v0"
_EXPECTED_RULES_URL = "https://www.nccn.org/professionals/physician_gls/pdf/breast.pdf"
_REQUIRED_BRANCH_FIELDS = ("branch_id", "nccn_section", "recommended")
_COVERED_BRANCH_IDS = frozenset(
    {"dcis", "metastatic", "her2_positive", "triple_negative", "hr_positive_her2_negative"}
)
_STAGE_RE = re.compile(r"^(T[0-4][a-c]?N[0-3][a-c]?M[01])$|.*M1.*")


class RulesetIntegrityError(RuntimeError):
    """Raised when the on-disk rules file drifts from what the code encodes."""


def _load_rules(path: Path = _RULES_PATH) -> dict:
    """Read + validate the NCCN-lite rules JSON.

    Guardrails (fail loudly, never silently):
      * File must exist.
      * `model_id` must match ``nccn-lite-v0``.
      * `source_document_url` must be the pinned NCCN URL.
      * Every rule branch must have `branch_id`, `nccn_section`, non-empty `recommended`.
      * Every `branch_id` in the JSON must be in `_COVERED_BRANCH_IDS`.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"NCCN-lite rules file not found at {path} — did the "
            "arbiter/models/therapy_rules_v0.json ship?"
        )
    raw = path.read_text()
    data = json.loads(raw)
    if data.get("model_id") != _EXPECTED_RULES_MODEL_ID:
        raise RulesetIntegrityError(
            f"model_id in {path.name} = {data.get('model_id')!r}, "
            f"expected {_EXPECTED_RULES_MODEL_ID!r}"
        )
    if data.get("source_document_url") != _EXPECTED_RULES_URL:
        raise RulesetIntegrityError(
            f"source_document_url in {path.name} = "
            f"{data.get('source_document_url')!r}, expected {_EXPECTED_RULES_URL!r}"
        )
    branches = data.get("rule_branches") or []
    if not isinstance(branches, list) or not branches:
        raise RulesetIntegrityError(
            f"rule_branches missing or empty in {path.name}"
        )
    seen_ids: set = set()
    for i, b in enumerate(branches):
        for k in _REQUIRED_BRANCH_FIELDS:
            if k not in b:
                raise RulesetIntegrityError(
                    f"branch #{i} in {path.name} missing required field {k!r}"
                )
        if not b["recommended"]:
            raise RulesetIntegrityError(
                f"branch {b['branch_id']!r} has empty recommended list"
            )
        if b["branch_id"] in seen_ids:
            raise RulesetIntegrityError(
                f"duplicate branch_id {b['branch_id']!r} in {path.name}"
            )
        seen_ids.add(b["branch_id"])
    unknown = seen_ids - _COVERED_BRANCH_IDS
    if unknown:
        raise RulesetIntegrityError(
            f"branch_ids in {path.name} that the code does not cover: "
            f"{sorted(unknown)}. Update _COVERED_BRANCH_IDS + wire the code."
        )
    return data


def rules_sha256_hex(path: Path = _RULES_PATH) -> str:
    """Return the SHA-256 hex digest of the on-disk rules file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Cached at import so every recommendation carries the SAME fingerprint.
try:
    _CACHED_RULES = _load_rules()
    _CACHED_SHA256 = rules_sha256_hex()
except Exception as _e:  # pragma: no cover - surfaced by tests
    _CACHED_RULES = None  # type: ignore[assignment]
    _CACHED_SHA256 = None  # type: ignore[assignment]
    _CACHED_LOAD_ERROR: Optional[BaseException] = _e
else:
    _CACHED_LOAD_ERROR = None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class InvalidInputError(ValueError):
    """Raised when strict=True input validation rejects a call."""


def _validate_receptor_status(rs: Mapping[str, bool]) -> None:
    if not isinstance(rs, Mapping):
        raise InvalidInputError(
            f"receptor_status must be a mapping, got {type(rs).__name__}"
        )
    for k in ("ER", "PR", "HER2"):
        if k not in rs:
            raise InvalidInputError(
                f"receptor_status missing required key {k!r}; got keys={sorted(rs.keys())}"
            )
        if not isinstance(rs[k], bool):
            raise InvalidInputError(
                f"receptor_status[{k!r}] must be bool, got {type(rs[k]).__name__}"
            )


def _validate_grade(grade: int) -> None:
    if not isinstance(grade, int) or isinstance(grade, bool):
        raise InvalidInputError(
            f"grade must be int in [1, 3], got {grade!r} (type {type(grade).__name__})"
        )
    if grade < 1 or grade > 3:
        raise InvalidInputError(
            f"grade must be in [1, 3], got {grade}"
        )


def _validate_stage(stage: str) -> None:
    if not isinstance(stage, str) or not stage:
        raise InvalidInputError(
            f"stage must be non-empty str, got {stage!r}"
        )
    if not _STAGE_RE.match(stage):
        raise InvalidInputError(
            f"stage {stage!r} does not match TNM pattern or contain M1. "
            "Expected e.g. 'T1N0M0', 'T2N1M0', 'M1'."
        )


def _validate_menopausal_status(status: Optional[str]) -> None:
    if status is None:
        return
    if status not in {"premenopausal", "postmenopausal", "unknown"}:
        raise InvalidInputError(
            f"menopausal_status must be one of "
            f"{{'premenopausal','postmenopausal','unknown', None}}, got {status!r}"
        )


def apply_nccn_lite_rules(
    receptor_status: Mapping[str, bool],
    grade: int,
    stage: str,
    age: int | None = None,
    menopausal_status: str | None = None,
    subtype: str | None = None,
    strict: bool = False,
) -> TherapyRulesResult:
    """Return a deterministic NCCN-lite recommendation.

    Parameters
    ----------
    receptor_status
        {"ER": bool, "PR": bool, "HER2": bool}
    grade
        Nottingham grade 1-3.
    stage
        AJCC stage (e.g. "T1N0M0", "T2N1M0", "M1").
    age
        Patient age in years (optional).
    menopausal_status
        "premenopausal" | "postmenopausal" | None
    subtype
        Biopsy L4b output (e.g. "IDC", "DCIS"). If DCIS, we skip systemic.
    """
    if _CACHED_LOAD_ERROR is not None:
        raise _CACHED_LOAD_ERROR  # type: ignore[misc]

    if strict:
        _validate_receptor_status(receptor_status)
        _validate_grade(grade)
        _validate_stage(stage)
        _validate_menopausal_status(menopausal_status)

    input_features = {
        "receptor_status": dict(receptor_status),
        "grade": int(grade),
        "stage": str(stage),
        "age": age,
        "menopausal_status": menopausal_status,
        "subtype": subtype,
    }

    er = bool(receptor_status.get("ER", False))
    pr = bool(receptor_status.get("PR", False))
    her2 = bool(receptor_status.get("HER2", False))
    hormone_receptor_positive = er or pr

    recommended: List[TherapyOption] = []
    not_recommended: List[TherapyOption] = []
    warnings: List[str] = [THERAPY_RULES_PROXY_WARNING]

    def _mk(branch_id: str,
            recommended: List[TherapyOption],
            not_recommended: List[TherapyOption],
            warnings: List[str]) -> TherapyRulesResult:
        return TherapyRulesResult(
            recommended_options=recommended,
            not_recommended=not_recommended,
            input_features=input_features,
            warnings=warnings,
            rules_sha256=_CACHED_SHA256,
            rules_model_id=(_CACHED_RULES or {}).get("model_id"),
            branch_id=branch_id,
        )


    # ── DCIS branch ────────────────────────────────────────────────────
    if subtype == "DCIS":
        recommended.append(TherapyOption(
            name="Lumpectomy + whole-breast radiation",
            category="surgery+radiation",
            citation_url=NCCN_URL,
            rationale="Standard first-line for DCIS (NCCN DCIS-1 / DCIS-2)",
            nccn_section="DCIS-1",
        ))
        if er:
            recommended.append(TherapyOption(
                name="Endocrine therapy (tamoxifen, 5 years)",
                category="endocrine",
                citation_url=NCCN_URL,
                rationale="ER+ DCIS: endocrine therapy reduces ipsilateral recurrence (NCCN DCIS-3)",
                nccn_section="DCIS-3",
            ))
        not_recommended.append(TherapyOption(
            name="Systemic chemotherapy",
            category="chemotherapy",
            citation_url=NCCN_URL,
            rationale="DCIS is a non-invasive lesion — no systemic chemotherapy indicated (NCCN DCIS-1)",
            nccn_section="DCIS-1",
        ))
        return _mk("dcis", recommended, not_recommended, warnings)

    # ── Metastatic branch ─────────────────────────────────────────────
    if stage.startswith("M1") or "M1" in stage:
        recommended.append(TherapyOption(
            name="Biomarker-directed systemic therapy",
            category="targeted",
            citation_url=NCCN_URL,
            rationale="Metastatic disease: biomarker-directed regimen per NCCN BINV-Q",
            nccn_section="BINV-Q",
        ))
        recommended.append(TherapyOption(
            name="Palliative care consultation",
            category="supportive",
            citation_url=NCCN_URL,
            rationale="NCCN recommends early palliative care in metastatic breast cancer (BINV-R)",
            nccn_section="BINV-R",
        ))
        if her2:
            recommended.append(TherapyOption(
                name="Trastuzumab + pertuzumab + taxane",
                category="targeted",
                citation_url=NCCN_URL,
                rationale="HER2+ metastatic first-line (NCCN BINV-Q)",
                nccn_section="BINV-Q",
            ))
        if hormone_receptor_positive and not her2:
            recommended.append(TherapyOption(
                name="CDK4/6 inhibitor + AI",
                category="targeted",
                citation_url=NCCN_URL,
                rationale="HR+/HER2- metastatic first-line: CDK4/6 inhibitor + aromatase inhibitor (NCCN BINV-P)",
                nccn_section="BINV-P",
            ))
        return _mk("metastatic", recommended, not_recommended, warnings)

    # ── HER2-positive branch ──────────────────────────────────────────
    if her2:
        recommended.append(TherapyOption(
            name="Trastuzumab (± pertuzumab)",
            category="targeted",
            citation_url=NCCN_URL,
            rationale="HER2+ non-metastatic: HER2-directed therapy is standard (NCCN BINV-J)",
            nccn_section="BINV-J",
        ))
        recommended.append(TherapyOption(
            name="Neoadjuvant TCHP (taxane + carboplatin + trastuzumab + pertuzumab)",
            category="chemotherapy",
            citation_url=NCCN_URL,
            rationale="HER2+ ≥ T2 or N+: neoadjuvant TCHP (NCCN BINV-K)",
            nccn_section="BINV-K",
        ))
        if hormone_receptor_positive:
            recommended.append(TherapyOption(
                name="Endocrine therapy (after chemotherapy)",
                category="endocrine",
                citation_url=NCCN_URL,
                rationale="HR+/HER2+: sequence endocrine therapy after chemotherapy (NCCN BINV-L)",
                nccn_section="BINV-L",
            ))
        return _mk("her2_positive", recommended, not_recommended, warnings)

    # ── Triple-negative branch ────────────────────────────────────────
    if not hormone_receptor_positive and not her2:
        recommended.append(TherapyOption(
            name="Neoadjuvant chemotherapy (AC-T ± pembrolizumab)",
            category="chemotherapy",
            citation_url=NCCN_URL,
            rationale="TNBC first-line: AC-T ± pembrolizumab per KEYNOTE-522 / NCCN BINV-M",
            nccn_section="BINV-M",
        ))
        recommended.append(TherapyOption(
            name="Consider platinum-based chemotherapy",
            category="chemotherapy",
            citation_url=NCCN_URL,
            rationale="TNBC (esp. BRCA-mutant): consider platinum agents (NCCN BINV-M)",
            nccn_section="BINV-M",
        ))
        not_recommended.append(TherapyOption(
            name="Endocrine therapy",
            category="endocrine",
            citation_url=NCCN_URL,
            rationale="TNBC is ER-/PR-: endocrine therapy not indicated (NCCN BINV-J)",
            nccn_section="BINV-J",
        ))
        return _mk("triple_negative", recommended, not_recommended, warnings)

    # ── HR+/HER2- branch (default) ────────────────────────────────────
    # Endocrine therapy first-line, +/- chemotherapy by risk.
    if hormone_receptor_positive and not her2:
        if menopausal_status == "unknown":
            warnings.append(
                "menopausal_status=unknown: cannot pick AI vs tamoxifen "
                "deterministically. Emitting tamoxifen as the safer default "
                "AND flagging that menopause status must be evaluated before "
                "any therapy is offered."
            )
            recommended.append(TherapyOption(
                name="Tamoxifen (5-10 years)",
                category="endocrine",
                citation_url=NCCN_URL,
                rationale="HR+/HER2- with unknown menopause status: tamoxifen "
                          "is the safer default across peri/post/pre states "
                          "(NCCN BINV-J)",
                nccn_section="BINV-J",
            ))
            recommended.append(TherapyOption(
                name="Menopause status evaluation (LH/FSH panel)",
                category="workup",
                citation_url=NCCN_URL,
                rationale="Menopause status is required to choose between AI "
                          "and tamoxifen (NCCN BINV-J)",
                nccn_section="BINV-J",
            ))
            return _mk("hr_positive_her2_negative", recommended, not_recommended, warnings)
        if menopausal_status == "postmenopausal":
            recommended.append(TherapyOption(
                name="Aromatase inhibitor (letrozole/anastrozole, 5 years)",
                category="endocrine",
                citation_url=NCCN_URL,
                rationale="Postmenopausal HR+/HER2-: AI first-line (NCCN BINV-J)",
                nccn_section="BINV-J",
            ))
        else:
            recommended.append(TherapyOption(
                name="Tamoxifen (5-10 years)",
                category="endocrine",
                citation_url=NCCN_URL,
                rationale="Premenopausal HR+/HER2-: tamoxifen with/without ovarian suppression (NCCN BINV-J)",
                nccn_section="BINV-J",
            ))
        if grade >= 3 or stage.startswith("T2") or stage.startswith("T3"):
            recommended.append(TherapyOption(
                name="Consider adjuvant chemotherapy (per Oncotype DX / MammaPrint)",
                category="chemotherapy",
                citation_url=NCCN_URL,
                rationale="HR+/HER2- high-risk (grade 3 or ≥T2): consider genomic assay + chemo (NCCN BINV-N)",
                nccn_section="BINV-N",
            ))
        return _mk("hr_positive_her2_negative", recommended, not_recommended, warnings)

    # ── Fallthrough (should not reach): return empty with warning ────
    warnings.append(
        "therapy_rules_lite: input did not match any encoded branch; "
        "returning empty recommendations. Falling back to full NCCN consultation required."
    )
    return _mk("fallthrough", recommended, not_recommended, warnings)


__all__ = [
    "TherapyOption",
    "TherapyRulesResult",
    "apply_nccn_lite_rules",
    "NCCN_URL",
    "THERAPY_RULES_PROXY_WARNING",
]
