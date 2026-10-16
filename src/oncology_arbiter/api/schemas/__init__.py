"""Pydantic schemas for the oncology-arbiter API.

Every response envelope carries the RUO disclaimer and AUROC caveat inline
so downstream consumers cannot strip them without noticing. Every response
also carries a `provenance` block indicating where the results came from —
placeholder, cached, live model — so a consumer can distinguish a stub from
a real inference at wire level.

This package is the modularized form of the former monolithic schemas.py.
Import from `oncology_arbiter.api.schemas` remains the public surface.
"""
from __future__ import annotations

from .common import (
    THERAPY_PROGNOSIS_MODEL_NAME,
    LEGACY_THERAPY_BENEFIT_MODEL_NAME,
    THERAPY_PROGNOSIS_UNCERTAINTY,
    RETIRED_MODEL_STATE_VALUES,
    ModelState,
    EvidenceRecord,
    StageError,
    StageReceipt,
    HonestyGateReport,
    ManskiBoundsBlock,
    ArbiterScore,
    GateReport,
    Provenance,
    ApiEnvelope,
)

from .screening import (
    ScreeningRequest,
    ScreeningFusionMetadata,
    ScreeningFusionRequest,
    ScreeningFinding,
    ScreeningResponse,
)

from .biopsy import (
    BreastDssFeatures,
    BreastDssPrognosis,
    BiopsyRequest,
    BiopsyReceptorPanel,
    ExtendedReceptorField,
    ReportParseBlock,
    BiopsyResponse,
    StageBiopsyScoreRequest,
    StageBiopsyScoreResponse,
)

from .therapy import (
    TherapyPatientContext,
    TherapyRequest,
    TherapyOption,
    TherapyResponse,
    TherapyTriageFeatures,
    PatientMutation,
    RankedDrug,
)

from .case import (
    NsclcCandidate,
    NsclcTherapyOption,
    Luna16Detection,
    Luna16DetectionBlock,
    NsclcResponse,
    FullCaseRequest,
    FullCaseResponse,
)

from .tumor_board import (
    DynamicTumorBoardRequest,
    DynamicTumorBoardResponse,
    ModalityEvidence,
    EvidenceMatrixRow,
    EvidenceMatrix,
    RecommendedDrug,
    SlProvenance,
    SyntheticLethalityBundle,
    PrimaryAlteration,
    Diagnosis,
    LineagePriors,
    PatientContext,
    TumorBoardBundle,
    TumorBoardBundleResponse,
)

from .meta import (
    ModelCardSummary,
    ModelCardsIndex,
    ArtifactCategory,
    WeightsProvenance,
    AuditExportResponse,
    HealthResponse,
    DemoCaseResponse,
)

from .elo import (
    EloDrugCandidate,
    EloRankRequest,
    EloMatchRecord,
    EloRankedEntry,
    EloRankResponse,
)

from .co_scientist import (
    CoScientistRunRequest,
    CoScientistRunResponse,
)

from .research import (
    ResearchIdentifiedSetRequest,
    ResearchIdentifiedSetResponse,
)

__all__ = [
    "THERAPY_PROGNOSIS_MODEL_NAME",
    "LEGACY_THERAPY_BENEFIT_MODEL_NAME",
    "THERAPY_PROGNOSIS_UNCERTAINTY",
    "RETIRED_MODEL_STATE_VALUES",
    "ModelState",
    "EvidenceRecord",
    "StageError",
    "StageReceipt",
    "HonestyGateReport",
    "ManskiBoundsBlock",
    "ArbiterScore",
    "GateReport",
    "Provenance",
    "ApiEnvelope",
    "ScreeningRequest",
    "ScreeningFusionMetadata",
    "ScreeningFusionRequest",
    "ScreeningFinding",
    "ScreeningResponse",
    "BreastDssFeatures",
    "BreastDssPrognosis",
    "BiopsyRequest",
    "BiopsyReceptorPanel",
    "ExtendedReceptorField",
    "ReportParseBlock",
    "BiopsyResponse",
    "StageBiopsyScoreRequest",
    "StageBiopsyScoreResponse",
    "TherapyPatientContext",
    "TherapyRequest",
    "TherapyOption",
    "TherapyResponse",
    "TherapyTriageFeatures",
    "PatientMutation",
    "RankedDrug",
    "NsclcCandidate",
    "NsclcTherapyOption",
    "Luna16Detection",
    "Luna16DetectionBlock",
    "NsclcResponse",
    "FullCaseRequest",
    "FullCaseResponse",
    "DynamicTumorBoardRequest",
    "DynamicTumorBoardResponse",
    "ModalityEvidence",
    "EvidenceMatrixRow",
    "EvidenceMatrix",
    "RecommendedDrug",
    "SlProvenance",
    "SyntheticLethalityBundle",
    "PrimaryAlteration",
    "Diagnosis",
    "LineagePriors",
    "PatientContext",
    "TumorBoardBundle",
    "TumorBoardBundleResponse",
    "ModelCardSummary",
    "ModelCardsIndex",
    "ArtifactCategory",
    "WeightsProvenance",
    "AuditExportResponse",
    "HealthResponse",
    "DemoCaseResponse",
    "EloDrugCandidate",
    "EloRankRequest",
    "EloMatchRecord",
    "EloRankedEntry",
    "EloRankResponse",
    "CoScientistRunRequest",
    "CoScientistRunResponse",
    "ResearchIdentifiedSetRequest",
    "ResearchIdentifiedSetResponse",
]
