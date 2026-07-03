"""L2-regularised logistic arbiter — screening / biopsy / therapy templates.

Design intent
-------------
This module deliberately mirrors ``ProgressionArbiter`` from
``fjkiani/org.backend/capabilities/progression_arbiter/arbiter.py`` almost
line-for-line (the reference model has an exact ``AUROC_CAVEAT`` string and
frozen JSON coefficient schema — we preserve that contract). Three concrete
adaptations plug into the oncology-arbiter L3 layer:

    screening_arbiter → decides whether a mammogram warrants recall / diagnostic workup
    biopsy_arbiter    → decides whether an equivocal lesion warrants core-needle biopsy
    therapy_arbiter   → decides therapy intensity given a positive biopsy

Frozen model artefacts live under ``models/``. They ship with:

* ``n_training = 0`` and an ``AUROC_CAVEAT`` that flags them as a TEMPLATE,
  not a trained model. Prospective AUROC is *not* claimed until real EMBED /
  CBIS-DDSM training runs land in Phase 3.
* Illustrative but coherent coefficient signs (e.g. BI-RADS ≥ 4 pushes
  toward biopsy, PPV_baseline pushes toward recall). These are *not* fit
  values; they exist so the sum-of-terms determinism, honesty wiring, and
  API/plumbing can be locked in immediately.

Honesty gates
-------------
Every ``score()`` call returns ``AUROC_CAVEAT`` alongside the probability.
The L5 API surface must not strip these fields — the ``ModelState.PLACEHOLDER``
enum value is used at the response envelope layer while the arbiter itself
returns ``model_state="template"`` in the ``metadata`` block.

RESEARCH USE ONLY — see :data:`oncology_arbiter.RUO_DISCLAIMER`.
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
          "coefficients":      {feature_name: float, ...},
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
        self.coefficients: Dict[str, float] = {k: float(v) for k, v in self._model["coefficients"].items()}
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
        """True → 1.0, False → 0.0, None → 0.5 (unknown)."""
        if value is True:
            return 1.0
        if value is False:
            return 0.0
        if value is None:
            return 0.5
        raise ValueError(f"Feature {feature!r} expected bool|None, got {value!r}")

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

        for feat_name, spec in self.feature_encodings.items():
            value = features.get(feat_name)
            if isinstance(spec, Mapping) and "ONE_HOT" in spec:
                one_hot_terms = self._encode_one_hot(feat_name, value)
                terms.update(one_hot_terms)
            elif isinstance(spec, Mapping) and set(spec.keys()) <= {"true", "false", "unknown"}:
                coef = self.coefficients.get(feat_name, 0.0)
                terms[feat_name] = coef * self._encode_bool(feat_name, value)
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

_MODELS_DIR = Path(__file__).parent / "models"


def load_arbiter(name: str) -> L2LogisticArbiter:
    """Load a frozen arbiter by short name.

    ``name`` must be one of ``"screening"``, ``"biopsy"``, ``"therapy"``.
    """
    slug_map = {
        "screening": "screening_arbiter_template_v0.json",
        "biopsy":    "biopsy_arbiter_template_v0.json",
        "therapy":   "therapy_arbiter_template_v0.json",
    }
    if name not in slug_map:
        raise ValueError(f"Unknown arbiter {name!r}; allowed = {sorted(slug_map)}")
    model_path = _MODELS_DIR / slug_map[name]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Frozen arbiter model not found at {model_path}. "
            "Did the templated JSON get shipped in this package?"
        )
    return L2LogisticArbiter(model_path)


def screening_arbiter() -> L2LogisticArbiter:
    """Return the screening arbiter (mammogram → recall / diagnostic workup)."""
    return load_arbiter("screening")


def biopsy_arbiter() -> L2LogisticArbiter:
    """Return the biopsy arbiter (equivocal lesion → core-needle biopsy)."""
    return load_arbiter("biopsy")


def therapy_arbiter() -> L2LogisticArbiter:
    """Return the therapy arbiter (biopsy result → therapy intensity)."""
    return load_arbiter("therapy")
