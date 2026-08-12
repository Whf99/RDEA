"""Dependency-free evidential transformations stated explicitly in the paper.

This module implements only the public output parameterization.  It does not
contain the segmentation network, SPPC translation, ESIL/DEMH objectives, or
any training configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Tuple


class EvidentialValueError(ValueError):
    """Raised when a public evidential transform receives invalid values."""


@dataclass(frozen=True)
class PixelEvidence:
    """One pixel's evidence-derived Dirichlet prediction."""

    evidence: Tuple[float, ...]
    alpha: Tuple[float, ...]
    probability: Tuple[float, ...]
    uncertainty: float
    predicted_class: int


def _finite_tuple(values: Iterable[float], name: str) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) < 2:
        raise EvidentialValueError(f"{name} must contain at least two classes")
    if not all(isfinite(value) for value in result):
        raise EvidentialValueError(f"{name} must contain only finite values")
    return result


def evidence_to_dirichlet(evidence: Iterable[float]) -> Tuple[float, ...]:
    """Apply the paper's public relation ``alpha = evidence + 1``."""

    values = _finite_tuple(evidence, "evidence")
    if any(value < 0.0 for value in values):
        raise EvidentialValueError("evidence must be nonnegative")
    return tuple(value + 1.0 for value in values)


def dirichlet_mean(alpha: Iterable[float]) -> Tuple[float, ...]:
    """Return the categorical mean of positive Dirichlet parameters."""

    values = _finite_tuple(alpha, "alpha")
    if any(value <= 0.0 for value in values):
        raise EvidentialValueError("Dirichlet parameters must be positive")
    strength = sum(values)
    return tuple(value / strength for value in values)


def clip_dirichlet(
    alpha: Iterable[float], lower: float, upper: float
) -> Tuple[float, ...]:
    """Clip positive Dirichlet parameters to caller-supplied bounds.

    The paper states compact-interval clipping for DEMH stability. Bounds are
    supplied by the caller so the transform remains explicit.
    """

    values = _finite_tuple(alpha, "alpha")
    if any(value <= 0.0 for value in values):
        raise EvidentialValueError("Dirichlet parameters must be positive")
    if not isfinite(lower) or not isfinite(upper):
        raise EvidentialValueError("clipping bounds must be finite")
    if lower <= 0.0 or lower >= upper:
        raise EvidentialValueError("bounds must satisfy 0 < lower < upper")
    return tuple(min(upper, max(lower, value)) for value in values)


def inverse_strength_uncertainty(
    alpha: Iterable[float], epsilon: float
) -> float:
    """Compute ``C / (sum(alpha) + epsilon)`` for one pixel."""

    values = _finite_tuple(alpha, "alpha")
    if any(value <= 0.0 for value in values):
        raise EvidentialValueError("Dirichlet parameters must be positive")
    if not isfinite(epsilon) or epsilon <= 0.0:
        raise EvidentialValueError("epsilon must be positive and finite")
    return len(values) / (sum(values) + epsilon)


def project_pixel_evidence(
    evidence: Iterable[float], epsilon: float
) -> PixelEvidence:
    """Build a complete public prediction record for one pixel."""

    evidence_values = _finite_tuple(evidence, "evidence")
    alpha = evidence_to_dirichlet(evidence_values)
    probability = dirichlet_mean(alpha)
    predicted_class = max(range(len(probability)), key=probability.__getitem__)
    return PixelEvidence(
        evidence=evidence_values,
        alpha=alpha,
        probability=probability,
        uncertainty=inverse_strength_uncertainty(alpha, epsilon),
        predicted_class=predicted_class,
    )
