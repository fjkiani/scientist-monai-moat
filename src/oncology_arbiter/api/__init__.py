"""FastAPI application for oncology-arbiter.

Exposes the 4 endpoints named in PLAN §4a:

    POST /v1/screening/analyze  — mammogram screening (MONAI + MedSigLIP)
    POST /v1/biopsy/analyze     — WSI + report reasoning (MedSigLIP + TxGemma)
    POST /v1/therapy/reason     — therapy recommendation (TxGemma)
    POST /v1/case/full          — end-to-end orchestrated case (Co-Scientist)

All endpoints today return honest RUO placeholder responses. They will be
wired to real models in Phase 2+ but the schemas, provenance envelope, and
disclaimer wiring are locked in NOW so downstream code can be built against
them.

Every response includes:
    * `disclaimer` — the RUO_DISCLAIMER constant (research use only)
    * `caveat`     — the AUROC_CAVEAT constant (interpret AUROC with care)
    * `model_state` — "placeholder" | "loaded" | "loading" | "unavailable"
    * `evidence`   — Co-Scientist-style {url, quoted_text} list, empty by default
    * `honesty_gate` — {seen_urls_count, kept, dropped} from reflection.py

Nothing here calls a neural network. That's Phase 2+.
"""
from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
