"""Small, dependency-free evaluation helpers for the public release."""

from __future__ import annotations

from math import dist, isfinite
from typing import Iterable, Sequence


class MetricValueError(ValueError):
    """Raised when metric inputs violate the public contract."""


def dice_score(
    prediction: Iterable[bool], reference: Iterable[bool]
) -> float:
    """Compute binary Dice from equally sized flattened masks."""

    predicted = tuple(bool(value) for value in prediction)
    expected = tuple(bool(value) for value in reference)
    if not predicted or len(predicted) != len(expected):
        raise MetricValueError("prediction and reference must be nonempty and equal")
    intersection = sum(p and r for p, r in zip(predicted, expected))
    denominator = sum(predicted) + sum(expected)
    if denominator == 0:
        return 1.0
    return 2.0 * intersection / denominator


def expected_calibration_error(
    confidences: Sequence[float], correctness: Sequence[bool], bins: int
) -> float:
    """Compute equal-width ECE for flattened confidence/correctness values."""

    if not confidences or len(confidences) != len(correctness):
        raise MetricValueError("confidence and correctness arrays must align")
    if bins < 1:
        raise MetricValueError("bins must be positive")
    values = tuple(float(value) for value in confidences)
    if any(not isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise MetricValueError("confidence values must lie in [0, 1]")

    total = len(values)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            item
            for item, confidence in enumerate(values)
            if lower <= confidence <= upper
            and (index == bins - 1 or confidence < upper)
        ]
        if not members:
            continue
        average_confidence = sum(values[item] for item in members) / len(members)
        accuracy = sum(bool(correctness[item]) for item in members) / len(members)
        error += len(members) / total * abs(accuracy - average_confidence)
    return error


def average_symmetric_surface_distance(
    first_surface: Sequence[Sequence[float]],
    second_surface: Sequence[Sequence[float]],
    spacing: Sequence[float],
) -> float:
    """Compute ASSD from two nonempty sets of surface coordinates.

    Surface coordinates and voxel spacing must use the same axes.
    """

    first = tuple(tuple(float(value) for value in point) for point in first_surface)
    second = tuple(tuple(float(value) for value in point) for point in second_surface)
    scale = tuple(float(value) for value in spacing)
    if not first or not second or not scale:
        raise MetricValueError("surfaces and spacing must be nonempty")
    dimensions = len(scale)
    if any(len(point) != dimensions for point in first + second):
        raise MetricValueError("surface points must match spacing dimensions")
    flattened = (*scale, *(value for point in first + second for value in point))
    if any(not isfinite(value) for value in flattened) or any(
        value <= 0.0 for value in scale
    ):
        raise MetricValueError("coordinates must be finite and spacing positive")

    scaled_first = tuple(
        tuple(value * scale[axis] for axis, value in enumerate(point))
        for point in first
    )
    scaled_second = tuple(
        tuple(value * scale[axis] for axis, value in enumerate(point))
        for point in second
    )

    def directed_sum(
        source: Sequence[Sequence[float]], target: Sequence[Sequence[float]]
    ) -> float:
        return sum(min(dist(point, other) for other in target) for point in source)

    total_distance = directed_sum(scaled_first, scaled_second) + directed_sum(
        scaled_second, scaled_first
    )
    return total_distance / (len(scaled_first) + len(scaled_second))
