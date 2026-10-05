"""Shared helper utilities for API endpoints."""
from typing import Any


def failed_stage_receipt(
    stage: str,
    required: bool,
    request_id: str,
    code: str,
    message: str,
    *,
    service_name: str | None = None,
    input_reference: str | None = None,
) -> dict[str, Any]:
    """Build a failed stage receipt for pipeline tracking."""
    return {
        "stage": stage,
        "required": required,
        "status": "failed_required" if required else "failed_optional",
        "request_id": request_id,
        "service_name": service_name,
        "input_reference": input_reference,
        "warnings": [],
        "error": {"code": code, "message": message[:500]},
    }


def skipped_stage_receipt(
    stage: str,
    request_id: str,
    message: str,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Build a skipped stage receipt for pipeline tracking."""
    return {
        "stage": stage,
        "required": required,
        "status": "skipped_not_applicable",
        "request_id": request_id,
        "warnings": [message],
        "error": None,
    }


def pipeline_status(receipts: list[dict[str, Any]]) -> str:
    """Determine overall pipeline status from stage receipts."""
    if any(receipt.get("status") == "failed_required" for receipt in receipts):
        return "failed_required_stage"
    if any(receipt.get("status") == "failed_optional" for receipt in receipts):
        return "partial_failure"
    return "complete"
