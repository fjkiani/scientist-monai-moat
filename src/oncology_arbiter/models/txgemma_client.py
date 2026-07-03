"""L4c therapy stage — TxGemma client (gated).

Design contract (mirrors MedSigLip client shape)
------------------------------------------------
1. Preflight HAI-DEF access first via ``check_hai_def_access``.
2. If preflight returns anything other than ALLOWED, raise
   ``GatedAccessError`` — NEVER silently fall through to a proxy.
   The endpoint decides whether to fall back to
   ``therapy_rules_lite.apply_nccn_lite_rules`` — this client itself
   does not.
3. If preflight passes (which under the current token it does NOT),
   we would load the model with ``transformers.AutoModelForCausalLM``
   and run a therapy-planning prompt. That code path is present but
   deferred behind the preflight — no attempt to load weights happens
   without a green preflight.
4. Honesty warning is always attached: TxGemma is a research LLM and
   its recommendations are NOT clinical advice.

Notes on repo IDs
-----------------
Google publishes TxGemma at ``google/txgemma-9b-chat`` (chat) and
``google/txgemma-9b-predict`` (structured). Both are HAI-DEF gated and
return HTTP 403 for accounts that have not accepted the terms. We use
``google/txgemma-9b-chat`` as the primary target because the therapy
recommendation task is conversational; if a caller opts into the
predict variant explicitly we honor that via the ``repo_id`` kwarg.

Under the current session token (`HF_TOKEN=hf_SfwLOG…`), preflight
returns FORBIDDEN (403) for BOTH TxGemma repos. This client's public
contract therefore reduces to:

    raise GatedAccessError(access_level=FORBIDDEN, reason="txgemma_gated:forbidden:...")

with the honesty header intact. Callers MUST catch this and fall back
to the rules-lite proxy under
``ONCOLOGY_ARBITER_ENABLE_THERAPY_RULES_PROXY=1``.

RESEARCH USE ONLY — see :data:`oncology_arbiter.RUO_DISCLAIMER`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER
from oncology_arbiter.models.hai_def import (
    AccessLevel,
    GateReport,
    GatedAccessError,
    check_hai_def_access,
)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TXGEMMA_CHAT_REPO = "google/txgemma-9b-chat"
TXGEMMA_PREDICT_REPO = "google/txgemma-9b-predict"

TXGEMMA_HONESTY_WARNING = (
    "TxGemma is a Google research LLM (HAI-DEF gated). Its outputs are "
    "recommendations from a generative language model, NOT verified "
    "clinical decisions. It MUST NOT be used to make treatment choices. "
    "Real clinical use requires a certified breast oncologist and a full "
    "guideline consultation. RESEARCH USE ONLY."
)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class TxGemmaTherapyResult:
    """Structured output when preflight ALLOWS and weights load successfully.

    Under the current token/gates this is unreachable; the client raises
    ``GatedAccessError`` first. Kept for documentation + future re-wiring
    once HAI-DEF acceptance for TxGemma is completed.
    """
    recommendations: List[str]
    reasoning: str
    input_features: Dict[str, Any]
    model_state: str = "loaded_txgemma"
    model_name: str = TXGEMMA_CHAT_REPO
    warnings: List[str] = field(default_factory=list)
    caveat: str = AUROC_CAVEAT
    disclaimer: str = RUO_DISCLAIMER
    gate_report: Optional[GateReport] = None


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class TxGemmaClient:
    """HAI-DEF gated TxGemma client — preflight-first, no silent fallback."""

    def __init__(
        self,
        repo_id: str = TXGEMMA_CHAT_REPO,
        *,
        preflight_fn: Callable[[str], GateReport] = check_hai_def_access,
    ):
        self.repo_id = repo_id
        self._preflight_fn = preflight_fn
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------ #
    # Preflight (public — used by unit tests)
    # ------------------------------------------------------------------ #

    def preflight(self) -> GateReport:
        report = self._preflight_fn(self.repo_id)
        if not report.allowed:
            raise GatedAccessError(
                repo_id=self.repo_id,
                access_level=report.access_level,
                status_code=report.status_code,
                reason=f"txgemma_gated:{report.access_level.value}:{report.reason}",
            )
        return report

    # ------------------------------------------------------------------ #
    # Weight-load (deferred — preflight must pass first)
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        report = self.preflight()  # raises unless ALLOWED
        # Deferred import to avoid loading transformers weights unless we get here
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers not installed — install oncology-arbiter[ml]"
            ) from exc
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        if self._model is None:  # pragma: no cover — never reached without token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.repo_id, torch_dtype="auto"
            )

    # ------------------------------------------------------------------ #
    # Therapy recommendation
    # ------------------------------------------------------------------ #

    def recommend_therapy(
        self,
        receptor_status: Mapping[str, bool],
        grade: int,
        stage: str,
        age: int | None = None,
        menopausal_status: str | None = None,
        subtype: str | None = None,
    ) -> TxGemmaTherapyResult:
        """Return TxGemma-generated therapy recommendations.

        Under current token: preflight raises FORBIDDEN before any weight
        loading. This method is included for structural symmetry with
        MedSigLip — endpoint code MUST wrap it in try/except.
        """
        # This will raise GatedAccessError under the current token.
        self._load()

        # ── Unreachable under current token, but documented ──
        input_features = {  # pragma: no cover
            "receptor_status": dict(receptor_status),
            "grade": int(grade),
            "stage": str(stage),
            "age": age,
            "menopausal_status": menopausal_status,
            "subtype": subtype,
        }
        prompt = self._build_prompt(input_features)  # pragma: no cover
        rec_text = self._generate(prompt)  # pragma: no cover
        recs = self._parse_recommendations(rec_text)  # pragma: no cover
        return TxGemmaTherapyResult(  # pragma: no cover
            recommendations=recs,
            reasoning=rec_text,
            input_features=input_features,
            warnings=[TXGEMMA_HONESTY_WARNING],
            gate_report=self._preflight_fn(self.repo_id),
        )

    # ------------------------------------------------------------------ #
    # Helpers (unreachable under current token; kept for future use)
    # ------------------------------------------------------------------ #

    def _build_prompt(self, features: Dict[str, Any]) -> str:  # pragma: no cover
        rs = features["receptor_status"]
        return (
            "You are a breast oncology treatment planner. Given the following "
            "features, recommend a therapy plan citing NCCN sections:\n"
            f"- Receptor status: ER={rs.get('ER')} PR={rs.get('PR')} HER2={rs.get('HER2')}\n"
            f"- Grade: {features['grade']}\n"
            f"- Stage: {features['stage']}\n"
            f"- Age: {features['age']}\n"
            f"- Menopausal status: {features['menopausal_status']}\n"
            f"- Biopsy subtype: {features['subtype']}\n"
            "Output a numbered list of concrete therapy options with NCCN section citations."
        )

    def _generate(self, prompt: str) -> str:  # pragma: no cover
        assert self._tokenizer is not None and self._model is not None
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=512)
        return self._tokenizer.decode(out[0], skip_special_tokens=True)

    def _parse_recommendations(self, text: str) -> List[str]:  # pragma: no cover
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        recs = [ln for ln in lines if ln[:2].strip().rstrip(".").isdigit()]
        return recs or lines[:5]


__all__ = [
    "TxGemmaClient",
    "TxGemmaTherapyResult",
    "TXGEMMA_CHAT_REPO",
    "TXGEMMA_PREDICT_REPO",
    "TXGEMMA_HONESTY_WARNING",
]
