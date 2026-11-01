"""Health, model cards, demo samples, research release, and Elo utilities."""
from __future__ import annotations

from fastapi import APIRouter

from . import catalog, co_scientist, elo, health, research

router = APIRouter()
router.include_router(health.router)
router.include_router(catalog.router)
router.include_router(research.router)
router.include_router(elo.router)
router.include_router(co_scientist.router)
