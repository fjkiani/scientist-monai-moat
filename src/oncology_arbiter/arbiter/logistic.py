"""L2-regularised screening, biopsy, and therapy-receipt arbiters.

Production factories load identity-locked v1 JSON artifacts trained on
patient-disjoint public CBIS-DDSM or METABRIC cohorts. The paired
``features``/``coefficients`` representation matches the canonical delivery
validator, while mapping-form coefficients remain readable for historical test
fixtures. Any artifact with ``n_training == 0`` is rejected from production.

The therapy target is observed METABRIC chemotherapy receipt—not response,
benefit, efficacy, or a causal treatment recommendation. Every score carries
its retrospective-validation caveat and research-use-only disclaimer.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from oncology_arbiter import AUROC_CAVEAT, RUO_DISCLAIMER

# ── Constants ──────────────────────────────────────────────────────────

# Risk buckets — same thresholds as progression_arbiter, deliberately.
# Keeps the L5 UI colour bands consistent across all four arbiters in the
# platform (progression / screening / biopsy / therapy).
RISK_BUCKETS: Dict[str, tuple[float, float]] = {
    "LOW": (0.0, 0.3),
    "MID": (0.3, 0.7),
    "HIGH": (0.7, 1.01),
}

# Numerical guardrails for the sigmoid, to keep unit tests deterministic even
# when a caller passes huge logits from a malformed coefficient file.
_LOGIT_CLIP = 30.0

# Absolute tolerance used by the sum-of-terms invariant test in
# ``tests/unit/test_arbiter_l2_logistic.py``.
SUM_OF_TERMS_TOL = 1e-9

# ── Missingness handling ───────────────────────────────────────────────
#
# A previous revision encoded an unsupplied boolean as the midpoint 0.5.
# That is not a neutral choice: for a linear logit the midpoint injects
# ``0.5 * sum(coef)`` of unearned log-odds. On the shipped screening
# template (coefficients +0.6 prior_biopsy_history, +0.5
# family_history_first_degree, +1.2 brca_status_known_pathogenic) an
# all-unknown history therefore added exactly +1.15 to the logit and moved
# the intercept-only probability from sigmoid(-2.0) = 0.11920292 to
# sigmoid(-0.85) = 0.29938558 — a 2.51x inflation produced entirely by
# absent data. The therapy template was worse at +1.35.
#
# Two admissible repairs exist, and they are NOT interchangeable:
#
#   REFERENCE_LEVEL      Encode a missing boolean at the reference level 0.0
#                        so it contributes exactly 0.0 log-odds. This is the
#                        unique constant encoding under which an all-unknown
#                        patient scores the artefact's own intercept-only
#                        base rate, because sigmoid(intercept + 0) ==
#                        sigmoid(intercept). Encoding the *feature* at a
#                        prevalence value p instead still injects
#                        ``p * sum(coef)``; e.g. p = 0.1192 on the screening
#                        template yields logit -1.72584 -> 0.15111, which
#                        overshoots the 0.11920292 base rate by +0.0319
#                        absolute (+26.8% relative). Feature-space mean
#                        imputation and probability-space base rates are
#                        different objects, and sigmoid(E[x]) != E[sigmoid(x)]
#                        by Jensen's inequality.
#
#   DECLARED_INDICATOR   Use a dedicated missingness indicator, but ONLY when
#                        the frozen artefact declares a fitted coefficient
#                        for it. An indicator coefficient cannot be invented:
#                        these templates carry n_training = 0, so there is no
#                        data from which to estimate one.
#
# Whichever repair applies, ``score()`` also returns assumption-free bounds
# obtained by driving every missing boolean to both admissible extremes, so
# the caller can see how much of the probability is unconstrained by data.
MISSINGNESS_REFERENCE_LEVEL = "reference_level_zero_logit_contribution"
MISSINGNESS_DECLARED_INDICATOR = "declared_missingness_indicator"

# Encoding value for an unsupplied boolean. Immutable: 0.5 is prohibited.
MISSING_BOOL_ENCODING = 0.0

# Suffix used for a declared missingness-indicator coefficient.
MISSING_INDICATOR_SUFFIX = "__missing"


# ── Result dataclass ───────────────────────────────────────────────────


@dataclass
class ArbiterResult:
    """Structured output of every L2 arbiter score() call.

    Kept as a dataclass so the JSON serialisation is deterministic and the
    test suite can rely on ``asdict()`` for the sum-of-terms invariant.
    """

    p_positive: float
    logit: float
    risk_bucket: str
    recommendation: str
    term_contributions: Dict[str, float]
    driving_feature: str
    driving_feature_contribution: float
    # ── missingness accounting (see MISSINGNESS_POLICY) ───────────────
    # Boolean features the caller did not supply. These contribute exactly
    # 0.0 to the logit unless the frozen artefact declares a fitted
    # missingness-indicator coefficient.
    missing_features: List[str] = field(default_factory=list)
    missingness_policy: str = ""
    # Assumption-free (Manski) bounds on p_positive obtained by setting every
    # missing boolean to its two admissible extremes (False=0.0 / True=1.0).
    # When nothing is missing, both equal p_positive.
    p_lower_bound: float = 0.0
    p_upper_bound: float = 1.0
    disclaimer: str = RUO_DISCLAIMER
    caveat: str = AUROC_CAVEAT
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain-dict view (used by the L5 API layer)."""
        out = asdict(self)
        return out


# ── Core arbiter class ─────────────────────────────────────────────────


class L2LogisticArbiter:
    """L2-regularised logistic scorer loading a frozen JSON coefficient file.

    Schema (matches ``progression_arbiter_model_v1.json`` verbatim, minus the
    domain-specific feature names):

        {
          "model_name":        str,
          "model_type":        "L2_regularized_logistic_regression",
          "lambda":            float,
          "n_training":        int,
          "intercept":         float,
          "features":          [feature_name, ...],
          "coefficients":      [float, ...],
          "feature_encodings": {
              "<feature>": {"ONE_HOT": [...], "REFERENCE": "..."} | dict[str,float] | str,
              ...
          },
          "recommendations":   {"LOW": str, "MID": str, "HIGH": str},
          "positive_class":    str,        # human-readable name of P(y=1)
          "performance": {
              "cv_auroc_mean": float,
              "cv_auroc_std":  float,
              "brier_score":   float,
              "AUROC_CAVEAT":  str
          },
          "disclaimer":        str
        }

    The class deliberately mirrors the reference ``ProgressionArbiter``
    docstring pragmas: encoded booleans use 1.0 / 0.0 / 0.5 for
    True / False / None (unknown), and continuous features are normalised
    by dividing by the divisor declared in ``feature_encodings``.
    """

    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        with model_path.open() as f:
            self._model: Dict[str, Any] = json.load(f)

        # Required top-level keys
        for required in ("intercept", "coefficients", "feature_encodings", "recommendations"):
            if required not in self._model:
                raise ValueError(f"Frozen model at {model_path} missing required key '{required}'")

        self.model_path: Path = model_path
        self.intercept: float = float(self._model["intercept"])
        raw_coefficients = self._model["coefficients"]
        if isinstance(raw_coefficients, Mapping):
            # Backward-compatible reader for historical/template fixtures. The
            # production v1 artifacts use the validator-bound paired-list form.
            self.coefficients = {str(k): float(v) for k, v in raw_coefficients.items()}
        elif isinstance(raw_coefficients, list):
            features = self._model.get("features")
            if (
                not isinstance(features, list)
                or not features
                or len(features) != len(raw_coefficients)
                or not all(isinstance(value, str) and value for value in features)
            ):
                raise ValueError(
                    f"Frozen model at {model_path} has invalid paired features/coefficients"
                )
            # Length already validated above; avoid zip(..., strict=True) for Py3.9 hosts.
            self.coefficients = {
                str(feature): float(coefficient)
                for feature, coefficient in zip(features, raw_coefficients)
            }
        else:
            raise ValueError(
                f"Frozen model at {model_path} coefficients must be a mapping or list"
            )
        self.feature_encodings: Dict[str, Any] = dict(self._model["feature_encodings"])
        self.recommendations: Dict[str, str] = dict(self._model["recommendations"])
        self.model_name: str = str(self._model.get("model_name", "unknown"))
        self.model_type: str = str(self._model.get("model_type", "L2_regularized_logistic_regression"))
        self.n_training: int = int(self._model.get("n_training", 0))
        self.positive_class: str = str(self._model.get("positive_class", "positive"))
        self.disclaimer: str = str(self._model.get("disclaimer", RUO_DISCLAIMER))
        self.performance: Dict[str, Any] = dict(self._model.get("performance", {}))

        # Honesty invariant: every frozen model MUST carry AUROC_CAVEAT.
        if "AUROC_CAVEAT" not in self.performance:
            raise ValueError(
                f"Frozen model at {model_path} missing performance.AUROC_CAVEAT — "
                "honesty gate requires every arbiter to declare its AUROC caveat."
            )

        # Missingness invariant: an artefact may not declare a non-zero
        # encoding for an unobserved boolean. A non-zero "unknown" level
        # multiplies straight into the logit, so it manufactures risk from the
        # absence of data. Reject it at load time rather than silently
        # overriding it, so the artefact and the code cannot disagree.
        for name, spec in self.feature_encodings.items():
            if not (isinstance(spec, Mapping) and set(spec.keys()) <= {"true", "false", "unknown"}):
                continue
            unknown_level = spec.get("unknown")
            if unknown_level is None:
                continue
            injected = float(unknown_level) * float(self.coefficients.get(name, 0.0))
            # Allow non-zero unknown encodings only when the paired coefficient is
            # exactly 0 (zero log-odds contribution). Otherwise fail closed.
            if abs(injected) > 0.0:
                raise ValueError(
                    f"Frozen model at {model_path} declares feature {name!r} with "
                    f"unknown={unknown_level!r}. A missing boolean must encode to 0.0 so it "
                    f"contributes exactly zero log-odds; unknown={unknown_level!r} would inject "
                    f"{injected:+.6f} of unearned "
                    "log-odds. Declare unknown=null (or 0.0), and supply a fitted "
                    f"'{name}{MISSING_INDICATOR_SUFFIX}' coefficient if missingness is to be modelled."
                )

    # -- feature-value encoding ----------------------------------------

    def _encode_one_hot(self, feature: str, value: Optional[str]) -> Dict[str, float]:
        """Encode a categorical feature using the ONE_HOT schema in the JSON.

        Reference class contributes 0 to the logit, matching the reference
        implementation. Unknown values raise ValueError to fail loudly (we
        do not silently fall back to reference — that would mask bugs).
        """
        enc = self.feature_encodings.get(feature)
        if not isinstance(enc, Mapping) or "ONE_HOT" not in enc:
            raise ValueError(f"Feature {feature!r} does not have a ONE_HOT encoding")
        one_hot_levels: Sequence[str] = enc["ONE_HOT"]
        reference: str = enc.get("REFERENCE", "OTHER")
        terms: Dict[str, float] = {}
        # None or the reference class → all one-hot levels contribute 0.
        if value is None or value == reference:
            for lvl in one_hot_levels:
                key = f"{feature}_{lvl}"
                terms[key] = 0.0
            return terms
        if value not in one_hot_levels:
            allowed = list(one_hot_levels) + [reference]
            raise ValueError(
                f"Feature {feature!r} got value {value!r}; allowed = {allowed}"
            )
        for lvl in one_hot_levels:
            key = f"{feature}_{lvl}"
            coef = self.coefficients.get(key, 0.0)
            terms[key] = coef * (1.0 if lvl == value else 0.0)
        return terms

    def _encode_bool(self, feature: str, value: Optional[bool]) -> float:
        """Encode a boolean feature. ``True`` → 1.0, ``False`` → 0.0.

        A missing value (``None``) encodes to :data:`MISSING_BOOL_ENCODING`
        (0.0, the reference level) so that absent data contributes *exactly*
        zero log-odds. The former midpoint 0.5 is prohibited: it fabricated
        risk from missingness (see :data:`MISSINGNESS_REFERENCE_LEVEL`).

        Callers that need to know a value was absent must read
        ``ArbiterResult.missing_features`` — the encoding itself is
        indistinguishable from an observed ``False`` by construction, which is
        precisely why the missingness is reported out of band rather than
        smuggled into the logit.
        """
        if value is True:
            return 1.0
        if value is False:
            return 0.0
        if value is None:
            return MISSING_BOOL_ENCODING
        raise ValueError(f"Feature {feature!r} expected bool|None, got {value!r}")

    def _bool_features(self) -> List[str]:
        """Names of every feature declared with a boolean encoding."""
        out: List[str] = []
        for name, spec in self.feature_encodings.items():
            if isinstance(spec, Mapping) and set(spec.keys()) <= {"true", "false", "unknown"}:
                out.append(name)
        return out

    def _missing_indicator_coef(self, feature: str) -> Optional[float]:
        """Return a declared missingness-indicator coefficient, or ``None``.

        We never synthesise this value. If the frozen artefact does not
        declare ``"<feature>__missing"`` in ``coefficients``, the reference
        level applies instead.
        """
        key = f"{feature}{MISSING_INDICATOR_SUFFIX}"
        if key in self.coefficients:
            return float(self.coefficients[key])
        return None

    def _encode_continuous(self, feature: str, value: float) -> float:
        """Divide by the divisor declared in feature_encodings for this feature.

        The reference JSON stores divisors as strings like
        ``"raw_weeks / 52.0"`` — we only care about the trailing number.
        """
        enc = self.feature_encodings.get(feature)
        if enc is None:
            raise ValueError(f"Feature {feature!r} not declared in feature_encodings")
        divisor = self._extract_divisor(enc)
        if divisor == 0:
            raise ValueError(f"Feature {feature!r} declared divisor 0")
        return float(value) / divisor

    @staticmethod
    def _extract_divisor(spec: Any) -> float:
        """Pull the divisor out of a continuous-feature encoding spec.

        Accepts either the reference string form (``"raw / 52.0"``) or an
        explicit ``{"divisor": 52.0}`` dict for readability. Defaults to 1.0
        if none is declared.
        """
        if isinstance(spec, Mapping):
            if "divisor" in spec:
                return float(spec["divisor"])
            return 1.0
        if isinstance(spec, str):
            # Look for the trailing float / int in the string.
            import re
            match = re.search(r"/\s*([0-9]+(?:\.[0-9]+)?)", spec)
            if match:
                return float(match.group(1))
            return 1.0
        return 1.0

    # -- public scoring API --------------------------------------------

    def score(self, features: Mapping[str, Any]) -> ArbiterResult:
        """Score a feature dict and return an :class:`ArbiterResult`.

        Feature keys must match the ``feature_encodings`` block of the
        frozen JSON. Anything unrecognised raises ``ValueError`` — we do
        not silently drop features because that would break the sum-of-terms
        invariant tests rely on.
        """
        # Verify no unrecognised features
        allowed_features = set(self.feature_encodings.keys())
        for k in features:
            if k not in allowed_features:
                raise ValueError(
                    f"Unrecognised feature {k!r}; allowed = {sorted(allowed_features)}"
                )

        terms: Dict[str, float] = {"intercept": self.intercept}
        missing_bools: List[str] = []
        used_declared_indicator = False

        for feat_name, spec in self.feature_encodings.items():
            value = features.get(feat_name)
            if isinstance(spec, Mapping) and "ONE_HOT" in spec:
                one_hot_terms = self._encode_one_hot(feat_name, value)
                terms.update(one_hot_terms)
            elif isinstance(spec, Mapping) and set(spec.keys()) <= {"true", "false", "unknown"}:
                coef = self.coefficients.get(feat_name, 0.0)
                terms[feat_name] = coef * self._encode_bool(feat_name, value)
                if value is None:
                    missing_bools.append(feat_name)
                    # Option (a): a dedicated indicator, used only when the
                    # artefact declares a fitted coefficient for it.
                    indicator_coef = self._missing_indicator_coef(feat_name)
                    if indicator_coef is not None:
                        terms[f"{feat_name}{MISSING_INDICATOR_SUFFIX}"] = indicator_coef
                        used_declared_indicator = True
            elif isinstance(spec, (str, Mapping)):
                # Treat as continuous
                if value is None:
                    # Continuous features default to 0.0 when unspecified,
                    # matching the reference implementation.
                    numeric_value = 0.0
                else:
                    numeric_value = self._encode_continuous(feat_name, value)
                coef = self.coefficients.get(feat_name, 0.0)
                terms[feat_name] = coef * numeric_value
            else:
                raise ValueError(f"Unsupported encoding spec for feature {feat_name!r}: {spec!r}")

        # Compute logit + probability (with clipping to keep sigmoid finite)
        logit_raw = sum(terms.values())
        logit_clipped = max(-_LOGIT_CLIP, min(_LOGIT_CLIP, logit_raw))
        p = 1.0 / (1.0 + math.exp(-logit_clipped))

        # Assumption-free bounds over the missing booleans. Each missing
        # feature can only have been True or False, so driving the negative
        # coefficients to their worst case and the positive ones to theirs
        # brackets the probability without assuming any prevalence. The
        # reference-level point estimate always lies inside this interval.
        delta_down = sum(min(0.0, self.coefficients.get(f, 0.0)) for f in missing_bools)
        delta_up = sum(max(0.0, self.coefficients.get(f, 0.0)) for f in missing_bools)
        p_lo = 1.0 / (1.0 + math.exp(-max(-_LOGIT_CLIP, min(_LOGIT_CLIP, logit_raw + delta_down))))
        p_hi = 1.0 / (1.0 + math.exp(-max(-_LOGIT_CLIP, min(_LOGIT_CLIP, logit_raw + delta_up))))

        # Risk bucket
        bucket = "MID"
        for name, (lo, hi) in RISK_BUCKETS.items():
            if lo <= p < hi:
                bucket = name
                break

        # Driving feature (largest |contribution|, excluding intercept)
        non_intercept = {k: v for k, v in terms.items() if k != "intercept"}
        if non_intercept:
            driving = max(non_intercept, key=lambda k: abs(non_intercept[k]))
            driving_val = non_intercept[driving]
        else:
            driving = "intercept"
            driving_val = self.intercept

        # Keep the full term dict; the test suite verifies the invariant
        #     sum(term_contributions.values()) == logit
        # so filtering here would break that.
        full_terms = {k: round(v, 6) for k, v in terms.items()}

        return ArbiterResult(
            p_positive=round(p, 6),
            logit=round(logit_raw, 6),
            risk_bucket=bucket,
            recommendation=self.recommendations[bucket],
            term_contributions=full_terms,
            driving_feature=driving,
            driving_feature_contribution=round(driving_val, 6),
            missing_features=sorted(missing_bools),
            missingness_policy=(
                MISSINGNESS_DECLARED_INDICATOR if used_declared_indicator
                else MISSINGNESS_REFERENCE_LEVEL
            ),
            p_lower_bound=round(p_lo, 6),
            p_upper_bound=round(p_hi, 6),
            disclaimer=self.disclaimer,
            caveat=self.performance.get("AUROC_CAVEAT", AUROC_CAVEAT),
            metadata={
                "model_name": self.model_name,
                "model_type": self.model_type,
                "n_training": self.n_training,
                "positive_class": self.positive_class,
                "model_state": "template" if self.n_training == 0 else "frozen",
            },
        )

    def score_batch(self, batch: Sequence[Mapping[str, Any]]) -> List[ArbiterResult]:
        return [self.score(item) for item in batch]

    def explain(self, result: ArbiterResult) -> str:
        """Human-readable explanation string — mirrors ProgressionArbiter.explain."""
        template_prefix = "[TEMPLATE — coefficients illustrative] " if self.n_training == 0 else ""
        lines = [
            f"{template_prefix}P({self.positive_class}) = {result.p_positive:.1%}  |  Risk bucket: {result.risk_bucket}",
            f"Recommendation: {result.recommendation.replace('_', ' ').lower()}",
            f"Driving feature: {result.driving_feature} ({result.driving_feature_contribution:+.4f})",
            "",
            "Active contributions (|value| > 1e-4):",
        ]
        for feat, val in sorted(
            result.term_contributions.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        ):
            if abs(val) < 1e-4:
                continue
            lines.append(f"  {feat:44s}  {val:+.4f}")
        lines.append("")
        lines.append(result.disclaimer)
        lines.append(result.caveat)
        return "\n".join(lines)


# ── Model factory ──────────────────────────────────────────────────────

def load_arbiter(name: str) -> L2LogisticArbiter:
    """Load and identity-verify a trained v1 arbiter by short name.

    Production routing is fail-closed: every route passes through its unique
    artifact-identity wiring module, and an ``n_training == 0`` template can
    never enter the product path.
    """
    if name == "screening":
        from .stage_screening_wiring import load_stage_screening_arbiter

        model = load_stage_screening_arbiter()
    elif name == "biopsy":
        from .stage_biopsy_wiring import load_stage_biopsy_arbiter

        model = load_stage_biopsy_arbiter()
    elif name == "therapy":
        from .stage_therapy_wiring import load_stage_therapy_arbiter

        model = load_stage_therapy_arbiter()
    else:
        raise ValueError(
            f"Unknown arbiter {name!r}; allowed = ['biopsy', 'screening', 'therapy']"
        )
    if model.n_training <= 0:
        raise RuntimeError(
            f"Refusing untrained production arbiter {model.model_name!r}: n_training={model.n_training}"
        )
    return model


def screening_arbiter() -> L2LogisticArbiter:
    """Return the screening arbiter (mammogram → recall / diagnostic workup)."""
    return load_arbiter("screening")


def biopsy_arbiter() -> L2LogisticArbiter:
    """Return the biopsy arbiter (equivocal lesion → core-needle biopsy)."""
    return load_arbiter("biopsy")


def therapy_arbiter() -> L2LogisticArbiter:
    """Return the therapy arbiter (biopsy result → therapy intensity)."""
    return load_arbiter("therapy")
