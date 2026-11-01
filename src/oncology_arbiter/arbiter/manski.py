"""Fail-closed identification gate on the assumption-free (Manski) bounds.

An L2 arbiter that is missing part of its input panel is not "slightly less
sure" - it is *partially identified*.  Encoding the unobserved features at
their reference level yields a number, but that number is one point inside an
interval whose endpoints are obtained by ranging every unobserved feature over
its entire admissible support.  When that interval is wide the point estimate
carries almost no information about where the truth lies, and handing it to a
clinician is worse than handing them nothing, because a number implies a claim
to knowledge that the model does not have.

Measured on the committed templates with the whole boolean panel unobserved:

    screening  [0.11920292, 0.57444252]  width 0.45523959
    biopsy     [0.00739154, 0.78583498]  width 0.77844344
    therapy    [0.03557119, 0.91682730]  width 0.88125611

A therapy interval of width 0.881 spans almost the entire unit interval: the
model is blind, and 0.1857 is not an estimate, it is a coordinate.

Why the gate does not read ``ArbiterResult.missing_features``
-------------------------------------------------------------
That field lists only unobserved *booleans*.  ``L2LogisticArbiter.score``
substitutes the reference level for an absent one-hot feature and 0.0 for an
absent continuous feature, records neither, and then reports
``p_lower_bound == p_upper_bound``.  Omitting ``grade`` from the therapy panel
moves the point estimate from 0.155251 to 0.119834 while the arbiter still
certifies width 0.000000; the true identified set is [0.039392, 0.378952],
width 0.3396.  Omitting a continuous covariate such as ``ki67_norm`` is worse:
the artefact declares a divisor but no admissible range, so the assumption-free
interval is the whole of [0, 1] and the arbiter still reports width zero.
Gating on ``missing_features`` would therefore have left the gate open on
exactly the inputs it cannot bound.  :func:`schema_missing_features` reads the
declared schema instead.

The two failure modes are distinct and are reported distinctly
--------------------------------------------------------------
``ManskiBoundsExceeded``
    The identified set is finite but too wide.  The bounds are real numbers and
    are returned to the caller, who can close the interval by supplying the
    listed features.

``IdentifiedSetUnbounded``
    At least one unobserved covariate has *no declared support at all*.  There
    is no finite interval to report and no amount of extra assumption-free
    reasoning will produce one; the identified set is [0, 1] by construction.
    Collapsing this into "a wide interval" would imply the artefact contains
    information it does not contain, so it carries its own error code,
    ``IDENTIFIED_SET_UNBOUNDED``.  A zero-width interval must never be emitted
    for this case.

There is no public bypass
-------------------------
:func:`enforce_manski_gate` takes no argument that can open it.  Not a header,
not an environment variable, not a config key, not a keyword default.  That is
deliberate and it is structural: the failure mode being guarded against is a
deployment where somebody quietly turned the gate off, and a bypass parameter
that merely defaults to ``False`` is one call site away from being ``True``.

The blind-inference override exists only as a *different function*
(:func:`research_release`) reachable only from an authenticated research route
that requires the :data:`RESEARCH_BLIND_INFERENCE_SCOPE` privileged scope, and
it emits an audit receipt naming the tenant that asked for it.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "MANSKI_MAX_WIDTH",
    "MANSKI_ERROR_CODE",
    "UNBOUNDED_ERROR_CODE",
    "RESEARCH_BLIND_INFERENCE_SCOPE",
    "PROVISIONAL_UNOBSERVED_WARNING",
    "INTERVAL_SEMANTICS",
    "UNBOUNDED_SEMANTICS",
    "ManskiBounds",
    "ManskiGateError",
    "ManskiBoundsExceeded",
    "IdentifiedSetUnbounded",
    "enforce_manski_gate",
    "research_release",
    "blind_inference_receipt",
    "feature_contribution_range",
    "schema_missing_features",
    "widest",
]


#: Maximum admissible width of the identified set before a point estimate is
#: withheld.  0.25 means "the truth is pinned to at most a quarter of the
#: probability scale".  Anything wider is reported as an interval, not a point.
MANSKI_MAX_WIDTH: float = 0.25

MANSKI_ERROR_CODE: str = "ManskiBoundsExceeded"

#: Distinct code for "no finite identified set exists", which is a different
#: statement from "the identified set is wide".
UNBOUNDED_ERROR_CODE: str = "IDENTIFIED_SET_UNBOUNDED"

#: Privileged scope required to reach the research-only blind-inference route.
#: Deliberately namespaced so it cannot be confused with an ordinary tenant
#: capability, and deliberately NOT granted to the anonymous principal that
#: ``ONCOLOGY_ARBITER_AUTH_MODE=off`` mints for local development.
RESEARCH_BLIND_INFERENCE_SCOPE: str = "research:blind_inference"

#: Provenance marker attached to any estimate released through the research
#: route.  Its presence in a payload means the number is not gated.
PROVISIONAL_UNOBSERVED_WARNING: str = "PROVISIONAL_UNOBSERVED_WARNING"

#: What the interval means, quoted verbatim into every 422 body so a client
#: cannot mistake it for a confidence interval.
INTERVAL_SEMANTICS: str = (
    "assumption-free (Manski) bounds obtained by ranging every unobserved "
    "feature over its entire admissible support; not a confidence interval, "
    "and carrying no sampling or distributional assumption"
)

UNBOUNDED_SEMANTICS: str = (
    "at least one unobserved covariate has no declared admissible support in "
    "the model artefact, so no finite identified set exists; the bounds are "
    "[0, 1] by construction and the point estimate carries zero information "
    "about this patient"
)

#: The arbiter clips the logit at +/-30 before the sigmoid; reuse the same
#: constant so a bound computed here is byte-comparable with one computed in
#: ``logistic.py``.
_LOGIT_CLIP: float = 30.0

#: Float slack for the boundary comparison.  ``p_lower_bound`` and
#: ``p_upper_bound`` are rounded to 6 dp by the arbiter, so an interval that is
#: exactly ``MANSKI_MAX_WIDTH`` wide must not be rejected by representation
#: error alone.
_WIDTH_TOL: float = 1e-9

#: Slack for the "point estimate lies inside its own bounds" invariant, which
#: is checked against the same 6 dp rounding.
_CONTAINMENT_TOL: float = 1e-6


@dataclass(frozen=True)
class ManskiBounds:
    """The identified set for one stage's probability, plus its point estimate.

    ``lower``/``upper`` are a *logical* consequence of the fitted coefficients
    and the missingness pattern, not a confidence interval, and they carry no
    sampling assumption whatsoever.
    """

    stage: str
    model_name: str
    lower: float
    upper: float
    point_estimate: float
    missing_features: tuple[str, ...] = ()
    unobserved_unbounded: tuple[str, ...] = ()
    max_width: float = MANSKI_MAX_WIDTH

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("stage must be a non-empty label")
        if self.max_width <= 0.0:
            raise ValueError(f"max_width must be positive, got {self.max_width!r}")
        for name, value in (
            ("lower", self.lower),
            ("upper", self.upper),
            ("point_estimate", self.point_estimate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value!r} is outside the unit interval")
        if self.lower > self.upper:
            raise ValueError(
                f"inverted interval for {self.stage}: "
                f"lower={self.lower!r} > upper={self.upper!r}"
            )
        if not (
            self.lower - _CONTAINMENT_TOL
            <= self.point_estimate
            <= self.upper + _CONTAINMENT_TOL
        ):
            # A point estimate outside its own identified set means the
            # encoding used to produce it disagrees with the encoding used to
            # produce the endpoints. That is an arbiter bug, not a wide
            # interval, and it must not be silently gated.
            raise ValueError(
                f"point_estimate={self.point_estimate!r} lies outside its own "
                f"Manski bounds [{self.lower!r}, {self.upper!r}] for "
                f"stage={self.stage!r}: the point and the endpoints were not "
                "produced by the same encoding"
            )
        if not self.missing_features and self.width > _CONTAINMENT_TOL:
            raise ValueError(
                f"stage={self.stage!r} reports no missing features but a "
                f"non-degenerate interval of width {self.width!r}: with a "
                "complete panel the bounds must collapse onto the point"
            )
        if self.unobserved_unbounded and (self.lower, self.upper) != (0.0, 1.0):
            # The whole point of the unbounded branch is that a zero-width -
            # or indeed any narrow - interval is a false claim of knowledge.
            # Enforce the only admissible answer as a type invariant so no
            # future caller can construct the forbidden object at all.
            raise ValueError(
                f"stage={self.stage!r} declares unbounded covariates "
                f"{list(self.unobserved_unbounded)} but reports the finite "
                f"interval [{self.lower!r}, {self.upper!r}]; with no declared "
                "support the identified set is exactly [0.0, 1.0]"
            )

    @property
    def width(self) -> float:
        """Width of the identified set. 0.0 means point-identified."""
        return self.upper - self.lower

    @property
    def unbounded(self) -> bool:
        """True when no finite identified set exists."""
        return bool(self.unobserved_unbounded)

    @property
    def identified(self) -> bool:
        """True when the interval is tight enough to release a point estimate."""
        if self.unbounded:
            return False
        return self.width <= self.max_width + _WIDTH_TOL

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model_name": self.model_name,
            "lower": self.lower,
            "upper": self.upper,
            "width": self.width,
            "max_width": self.max_width,
            "identified": self.identified,
            "unbounded": self.unbounded,
            "missing_features": list(self.missing_features),
            "unobserved_unbounded": list(self.unobserved_unbounded),
            "interval_semantics": (
                UNBOUNDED_SEMANTICS if self.unbounded else INTERVAL_SEMANTICS
            ),
        }

    @classmethod
    def from_result(
        cls,
        *,
        stage: str,
        model_name: str,
        result: Any,
        max_width: float = MANSKI_MAX_WIDTH,
    ) -> "ManskiBounds":
        """Build bounds from an :class:`~.logistic.ArbiterResult`.

        Read structurally rather than by import so this module stays free of a
        dependency on the arbiter implementation and can gate any scorer that
        exposes the same four attributes.

        Note this constructor reproduces the arbiter's *boolean-only* view and
        is therefore NOT sufficient for gating; use :meth:`from_arbiter`.
        """
        missing = tuple(getattr(result, "missing_features", ()) or ())
        return cls(
            stage=stage,
            model_name=model_name,
            lower=float(getattr(result, "p_lower_bound")),
            upper=float(getattr(result, "p_upper_bound")),
            point_estimate=float(getattr(result, "p_positive")),
            missing_features=missing,
            max_width=max_width,
        )

    @classmethod
    def from_arbiter(
        cls,
        *,
        stage: str,
        arbiter: Any,
        features: Mapping[str, Any],
        result: Any,
        max_width: float = MANSKI_MAX_WIDTH,
    ) -> "ManskiBounds":
        """Schema-complete identified set for ``result``.

        This is the constructor the gate uses.  It ranges over *every* declared
        feature the caller omitted - booleans, one-hot levels and continuous
        covariates - not only the booleans the arbiter chose to report, and it
        verifies as a post-condition that the arbiter's own boolean-only
        interval is contained in this one.  A containment failure means the two
        encodings disagree and is raised rather than gated.
        """
        coefficients: Mapping[str, float] = getattr(arbiter, "coefficients", {}) or {}
        declared: Mapping[str, Any] = getattr(arbiter, "feature_encodings", {}) or {}
        missing = schema_missing_features(arbiter, features)

        delta_down = 0.0
        delta_up = 0.0
        unbounded: list[str] = []
        for name in missing:
            lo, hi, bounded = feature_contribution_range(
                name, declared[name], coefficients
            )
            if bounded:
                delta_down += lo
                delta_up += hi
            else:
                unbounded.append(name)

        logit = float(getattr(result, "logit"))
        point = float(getattr(result, "p_positive"))
        if unbounded:
            # An unobserved covariate with no declared support admits any
            # log-odds. The identified set is the entire unit interval: the
            # model has told us nothing at all about this patient.
            p_lo, p_hi = 0.0, 1.0
        else:
            p_lo = round(_sigmoid(logit + delta_down), 6)
            p_hi = round(_sigmoid(logit + delta_up), 6)

        bounds = cls(
            stage=stage,
            model_name=str(getattr(arbiter, "model_name", stage)),
            lower=p_lo,
            upper=p_hi,
            point_estimate=point,
            missing_features=missing,
            unobserved_unbounded=tuple(unbounded),
            max_width=max_width,
        )

        # Post-condition: the arbiter's boolean-only interval must sit inside
        # the schema-complete one. If it does not, one of the two encodings is
        # wrong and neither interval can be trusted.
        arb_lo = float(getattr(result, "p_lower_bound", p_lo))
        arb_hi = float(getattr(result, "p_upper_bound", p_hi))
        if arb_lo < p_lo - _CONTAINMENT_TOL or arb_hi > p_hi + _CONTAINMENT_TOL:
            raise ValueError(
                f"stage={stage!r}: the arbiter's interval [{arb_lo!r}, {arb_hi!r}] is "
                f"not contained in the schema-complete interval [{p_lo!r}, {p_hi!r}]; "
                "logistic.py and manski.py disagree about the admissible support "
                "of at least one feature"
            )
        return bounds


def _sigmoid(logit: float) -> float:
    clipped = max(-_LOGIT_CLIP, min(_LOGIT_CLIP, logit))
    return 1.0 / (1.0 + math.exp(-clipped))


def feature_contribution_range(
    feature: str,
    spec: Any,
    coefficients: Mapping[str, float],
) -> tuple[float, float, bool]:
    """Range of log-odds an *unobserved* ``feature`` could contribute.

    Returns ``(lower, upper, bounded)``.  The classification order mirrors
    :meth:`L2LogisticArbiter.score` exactly, because a range derived from a
    different reading of the same spec would not bound the same quantity.

    * **boolean** - admissible values ``{0.0, 1.0}``, so the contribution is
      one of ``{0, c}`` and the range is ``[min(0, c), max(0, c)]``.
    * **one-hot** - admissible values are the declared levels plus the
      reference level, which contributes exactly 0.  The range is therefore
      ``[min(0, c_1..c_k), max(0, c_1..c_k)]``.
    * **continuous** - the artefact declares only a divisor (e.g.
      ``"ki67_pct / 100.0"``).  There is *no declared support*, so the
      assumption-free range is the whole real line and ``bounded`` is False.
      A zero coefficient is the one exception: it contributes 0 whatever the
      value, so the feature is irrelevant rather than unidentified.

    Note that every bounded branch contains 0, which is what the encoder
    substitutes for an absent value.  That is why the point estimate always
    lies inside the interval: it is the coordinate at which every unobserved
    feature is pinned to its reference level.
    """
    if isinstance(spec, Mapping) and "ONE_HOT" in spec:
        level_coefs = [
            float(coefficients.get(f"{feature}_{lvl}", 0.0)) for lvl in spec["ONE_HOT"]
        ]
        candidates = [0.0, *level_coefs]  # 0.0 is the REFERENCE level
        return min(candidates), max(candidates), True
    if isinstance(spec, Mapping) and set(spec.keys()) <= {"true", "false", "unknown"}:
        coef = float(coefficients.get(feature, 0.0))
        return min(0.0, coef), max(0.0, coef), True
    if isinstance(spec, (str, Mapping)):
        coef = float(coefficients.get(feature, 0.0))
        if coef == 0.0:
            return 0.0, 0.0, True
        return -math.inf, math.inf, False
    raise ValueError(f"Unsupported encoding spec for feature {feature!r}: {spec!r}")


def schema_missing_features(
    arbiter: Any, features: Mapping[str, Any]
) -> tuple[str, ...]:
    """Every declared feature the caller did not supply a value for.

    ``ArbiterResult.missing_features`` lists only unobserved *booleans*: the
    encoder substitutes the reference level for absent one-hot and continuous
    features and never records that it did so.  Gating on that field alone
    would leave the gate open on exactly the inputs it cannot bound.
    """
    declared = getattr(arbiter, "feature_encodings", {}) or {}
    return tuple(sorted(name for name in declared if features.get(name) is None))


class ManskiGateError(Exception):
    """Base class for every identification failure. Always HTTP 422.

    A single base lets the API register one exception handler while keeping
    the two failure modes distinguishable by ``error_code`` in the body.
    """

    status_code: int = 422
    error_code: str = "IdentificationFailure"
    semantics: str = INTERVAL_SEMANTICS

    def __init__(self, bounds: ManskiBounds, detail: str) -> None:
        self.bounds = bounds
        super().__init__(detail)

    def payload(self) -> dict[str, Any]:
        """Structured 422 body. The caller gets the interval, never the point."""
        b = self.bounds
        return {
            "error": self.error_code,
            "detail": str(self),
            "stage": b.stage,
            "model_name": b.model_name,
            "point_estimate_withheld": True,
            "bounds": [b.lower, b.upper],
            "width": b.width,
            "max_width": b.max_width,
            "missing_features": list(b.missing_features),
            "unobserved_unbounded": list(b.unobserved_unbounded),
            "interval_semantics": self.semantics,
            "required_action": self.required_action(),
            # Stated explicitly so a client cannot infer that some header,
            # query parameter or retry will produce a number here.
            "public_bypass_available": False,
        }

    def required_action(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class ManskiBoundsExceeded(ManskiGateError):
    """The identified set is finite but wider than the admissible maximum."""

    error_code = MANSKI_ERROR_CODE
    semantics = INTERVAL_SEMANTICS

    def __init__(self, bounds: ManskiBounds) -> None:
        super().__init__(
            bounds,
            f"{bounds.stage} is only partially identified: bounds "
            f"[{bounds.lower:.6f}, {bounds.upper:.6f}] have width "
            f"{bounds.width:.6f} > {bounds.max_width:.2f}",
        )

    def required_action(self) -> str:
        return (
            "supply explicit values for the listed missing_features; the "
            "interval closes deterministically as features are observed and "
            "collapses to a point when the panel is complete"
        )


class IdentifiedSetUnbounded(ManskiGateError):
    """No finite identified set exists for this input."""

    error_code = UNBOUNDED_ERROR_CODE
    semantics = UNBOUNDED_SEMANTICS

    def __init__(self, bounds: ManskiBounds) -> None:
        super().__init__(
            bounds,
            f"{bounds.stage} is unidentified: covariates "
            f"{list(bounds.unobserved_unbounded)} are unobserved and the model "
            "artefact declares no admissible support for them, so the "
            "identified set is the whole of [0, 1]",
        )

    def required_action(self) -> str:
        return (
            "supply explicit values for "
            f"{list(self.bounds.unobserved_unbounded)}. No interval can be "
            "reported without them: the artefact declares a normalisation "
            "divisor but no admissible range, so ranging the covariate over "
            "its support ranges the log-odds over the entire real line. "
            "Declaring a finite support in the model artefact would also "
            "resolve this, and is the correct long-term fix."
        )


def enforce_manski_gate(bounds: ManskiBounds) -> tuple[str, ...]:
    """Apply the public gate. There is no argument that can open it.

    Returns the provenance warnings to attach to a released response, which is
    the empty tuple in the identified case.  Raises :class:`ManskiGateError`
    otherwise.

    The signature is intentionally bypass-free.  An ``allow_blind_inference``
    keyword defaulting to ``False`` would be one call site, one header parse or
    one environment variable away from ``True``, and the whole purpose of this
    gate is to make that impossible to express.  Research access lives in
    :func:`research_release`.
    """
    if bounds.unbounded:
        raise IdentifiedSetUnbounded(bounds)
    if not bounds.identified:
        raise ManskiBoundsExceeded(bounds)
    return ()


def research_release(bounds: ManskiBounds) -> tuple[str, ...]:
    """Warnings that MUST accompany a point estimate released un-gated.

    Callable only from the authenticated research route.  It never raises: the
    research contract is that the caller receives the number *together with*
    an unambiguous statement of how little it means.
    """
    if bounds.identified:
        return ()
    warnings = [PROVISIONAL_UNOBSERVED_WARNING]
    if bounds.unbounded:
        warnings.append(
            f"{UNBOUNDED_ERROR_CODE}:covariates_"
            f"{','.join(bounds.unobserved_unbounded)}_have_no_declared_support"
            "_so_the_identified_set_is_[0,1]_and_this_point_estimate_carries"
            "_zero_information"
        )
    else:
        warnings.append(
            f"manski_width_{bounds.width:.6f}_exceeds_max_{bounds.max_width:.2f}"
        )
    warnings.append(
        f"identified_set:[{bounds.lower:.6f},{bounds.upper:.6f}]"
        ":the_point_estimate_is_one_coordinate_inside_this_interval"
    )
    warnings.append(
        f"unobserved_features:{','.join(bounds.missing_features)}"
        if bounds.missing_features
        else "unobserved_features:none_declared"
    )
    warnings.append(
        "released_via_authenticated_research_route:"
        "NOT_FOR_CLINICAL_USE:this_response_would_be_HTTP_422_on_any_public_route"
    )
    return tuple(warnings)


def blind_inference_receipt(
    bounds: ManskiBounds,
    *,
    tenant_id: str,
    key_prefix: str,
    route: str,
    request_features: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit receipt for one un-gated research release.

    Records who asked, for what, and what the gate would have done, so a
    release that later turns up in a clinical context can be traced back to the
    tenant that requested it.  The feature payload is fingerprinted rather than
    copied so the receipt is safe to log next to PHI-bearing requests.
    """
    canonical = json.dumps(
        {k: request_features[k] for k in sorted(request_features)},
        sort_keys=True,
        default=str,
    )
    would_have_been = (
        UNBOUNDED_ERROR_CODE
        if bounds.unbounded
        else (MANSKI_ERROR_CODE if not bounds.identified else None)
    )
    return {
        "receipt_type": "blind_inference_release",
        "issued_at": time.time(),
        "tenant_id": tenant_id,
        "key_prefix": key_prefix,
        "required_scope": RESEARCH_BLIND_INFERENCE_SCOPE,
        "route": route,
        "stage": bounds.stage,
        "model_name": bounds.model_name,
        "request_features_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "public_route_would_have_returned": (
            {"status": 422, "error": would_have_been} if would_have_been else None
        ),
        "raw_bounds": [bounds.lower, bounds.upper],
        "width": bounds.width,
        "point_estimate": bounds.point_estimate,
        "missing_features": list(bounds.missing_features),
        "unobserved_unbounded": list(bounds.unobserved_unbounded),
        "interval_semantics": (
            UNBOUNDED_SEMANTICS if bounds.unbounded else INTERVAL_SEMANTICS
        ),
        "not_for_clinical_use": True,
    }


def widest(bounds: Iterable[ManskiBounds]) -> ManskiBounds | None:
    """Return the least-identified stage, used when gating a composite.

    Unbounded stages sort ahead of every finite one regardless of width, since
    "no interval exists" dominates "the interval is wide".
    """
    ordered: Sequence[ManskiBounds] = sorted(
        bounds, key=lambda b: (b.unbounded, b.width), reverse=True
    )
    return ordered[0] if ordered else None
