"""Pydantic schema for MedGemma structured pathology extraction (RUO)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PathologyExtraction(BaseModel):
    """Strict biomarker panel extracted from a pathology report."""

    er_status: Literal["POSITIVE", "NEGATIVE", "EQUIVOCAL", "UNKNOWN"] = "UNKNOWN"
    er_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    pr_status: Literal["POSITIVE", "NEGATIVE", "EQUIVOCAL", "UNKNOWN"] = "UNKNOWN"
    her2_status: Literal[
        "0",
        "1+",
        "2+",
        "3+",
        "AMPLIFIED",
        "NON_AMPLIFIED",
        "POSITIVE",
        "NEGATIVE",
        "EQUIVOCAL",
        "UNKNOWN",
    ] = "UNKNOWN"
    nottingham_grade: Optional[Literal[1, 2, 3]] = None
    tumor_size_mm: Optional[float] = Field(default=None, ge=0.0, le=500.0)
    disclaimer: str = (
        "RESEARCH USE ONLY — MedGemma structured extraction; not FDA-cleared; "
        "not for clinical decision-making."
    )


EXTRACTION_SYSTEM_PROMPT = """You are a research-use-only oncology report parser.
Extract biomarkers from the pathology report into ONE JSON object with EXACTLY these keys:
{
  "er_status": "POSITIVE" | "NEGATIVE" | "EQUIVOCAL" | "UNKNOWN",
  "er_percent": number or null,
  "pr_status": "POSITIVE" | "NEGATIVE" | "EQUIVOCAL" | "UNKNOWN",
  "her2_status": "0" | "1+" | "2+" | "3+" | "AMPLIFIED" | "NON_AMPLIFIED" | "POSITIVE" | "NEGATIVE" | "EQUIVOCAL" | "UNKNOWN",
  "nottingham_grade": 1 | 2 | 3 | null,
  "tumor_size_mm": number or null
}
Rules:
- Use UNKNOWN or null when not stated. Do not invent values.
- If ER is described as positive with a percent, set er_percent.
- tumor_size_mm must be millimeters (convert cm × 10).
- Output JSON only. No markdown fences. No commentary.
"""
