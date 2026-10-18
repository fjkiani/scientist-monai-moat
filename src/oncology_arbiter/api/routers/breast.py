"""Breast screening, biopsy, therapy, and stage-score routes."""
from __future__ import annotations

from fastapi import APIRouter

from . import biopsy, screening, stage, therapy

router = APIRouter()
router.include_router(screening.router)
router.include_router(biopsy.router)
router.include_router(therapy.router)
router.include_router(stage.router)
