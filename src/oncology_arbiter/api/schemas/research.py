"""Schemas for privileged research identified-set routes."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import ManskiBoundsBlock



# --------------------------------------------------------------------------- #
# Research-only blind inference (authenticated, privileged scope)
#
# These models back `/v1/research/arbiter/identified_set`, the ONLY route in
# the API that may return a point estimate whose Manski interval is wider than
# `MANSKI_MAX_WIDTH`. It is not a bypass of the public gate: it is a different
# contract. The caller must hold the `research:blind_inference` scope, the
# response carries the raw bounds and the full unobserved-feature list rather
# than hiding them, and every release writes an audit receipt naming the
# tenant. `not_for_clinical_use` is a literal True and cannot be negotiated.


class ResearchIdentifiedSetRequest(BaseModel):
    """Ask for the identified set of a stage probability under an incomplete panel."""
    model_config = ConfigDict(extra="forbid")
    arbiter: Literal["screening", "biopsy", "therapy"] = Field(
        ...,
        description="Which frozen L2 arbiter to interrogate.",
    )
    features: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Partial feature panel. Any declared feature omitted here is "
            "ranged over its entire admissible support to build the bounds."
        ),
    )
    acknowledge_not_for_clinical_use: Literal[True] = Field(
        ...,
        description=(
            "Must be literal true. A research release is only meaningful if "
            "the caller has stated in the request that it is not clinical."
        ),
    )



class ResearchIdentifiedSetResponse(BaseModel):
    """Raw bounds plus the point estimate the public gate would have refused."""
    model_config = ConfigDict(extra="forbid")
    request_id: str
    arbiter: str
    model_name: str
    point_estimate: float = Field(
        ...,
        ge=0.0, le=1.0,
        description=(
            "One coordinate inside `bounds`. When `identified` is False this "
            "number is NOT an estimate of anything; it is the value the "
            "encoder happens to produce when unobserved covariates are pinned "
            "at their reference level."
        ),
    )
    bounds: ManskiBoundsBlock
    would_have_been_rejected: bool = Field(
        ...,
        description="True when a public route would have answered HTTP 422 for this exact panel.",
    )
    public_route_error_code: str | None = Field(
        None,
        description="`ManskiBoundsExceeded`, `IDENTIFIED_SET_UNBOUNDED`, or null when the panel is identified.",
    )
    missing_features: list[str] = Field(default_factory=list)
    unobserved_unbounded: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description="Mandatory provenance warnings. Never empty when `identified` is False.",
    )
    audit_receipt: dict[str, Any] = Field(
        ...,
        description="Receipt written to the tenant audit ledger for this release.",
    )
    not_for_clinical_use: Literal[True] = True
