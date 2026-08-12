"""Paper-aligned public blueprints for SPPC, ESIL, and DEMH.

The blueprints make data flow, optimization roles, and stop-gradient direction
explicit through small dependency-free data structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable, Tuple

from .interfaces import PairedViewSpec, TensorShape, validate_paired_view


class ConsistencyPath(str, Enum):
    """The two ESIL paths stated in the paper."""

    TEACHER_STUDENT = "teacher_student"
    ORIGINAL_TRANSLATED = "original_translated"


class ViewDirection(str, Enum):
    """The bidirectional paired-view directions used by SPPC."""

    SOURCE_TO_TARGET = "source_to_target"
    TARGET_TO_SOURCE = "target_to_source"


@dataclass(frozen=True)
class SPPCBlueprint:
    """Public SPPC topology without frequency-aware implementation details."""

    directions: Tuple[ViewDirection, ...] = (
        ViewDirection.SOURCE_TO_TARGET,
        ViewDirection.TARGET_TO_SOURCE,
    )
    preserves_spatial_shape: bool = True
    translators_frozen_during_adaptation: bool = True


@dataclass(frozen=True)
class ESILBlueprint:
    """Public ESIL paths and reliability semantics."""

    paths: Tuple[ConsistencyPath, ...] = (
        ConsistencyPath.TEACHER_STUDENT,
        ConsistencyPath.ORIGINAL_TRANSLATED,
    )
    reliability_signal: str = "inverse_dirichlet_strength"
    reliability_is_pixelwise: bool = True
    reliability_control_detached: bool = True
    teacher_target_detached: bool = True


@dataclass(frozen=True)
class DEMHBlueprint:
    """Public DEMH anchor direction and stability semantics."""

    parameter_space: str = "dirichlet"
    discrepancy: str = "symmetric_kl"
    original_view_role: str = "detached_anchor"
    translated_view_role: str = "optimized_view"
    reliability_is_pixelwise: bool = True
    parameters_clipped_for_stability: bool = True


@dataclass(frozen=True)
class ESILPlan:
    """Shape and gradient-flow plan for one ESIL comparison."""

    path: ConsistencyPath
    probability_shape: TensorShape
    reliability_shape: TensorShape
    compared_target_detached: bool
    reliability_control_detached: bool = True


@dataclass(frozen=True)
class DEMHPlan:
    """Shape and gradient-flow plan for one DEMH paired-view comparison."""

    anchor_shape: TensorShape
    optimized_shape: TensorShape
    reliability_shape: TensorShape
    original_anchor_detached: bool = True
    clipping_required: bool = True


@dataclass(frozen=True)
class RDEAComponentBlueprint:
    """Public composition of the three RDEA components."""

    sppc: SPPCBlueprint = SPPCBlueprint()
    esil: ESILBlueprint = ESILBlueprint()
    demh: DEMHBlueprint = DEMHBlueprint()


@dataclass(frozen=True)
class ObjectiveTerms:
    """Public assembly ``L_total = L_sup + L_ESIL + L_DEMH``."""

    supervised: float
    esil: float
    demh: float

    def __post_init__(self) -> None:
        values = (self.supervised, self.esil, self.demh)
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("objective terms must be finite and nonnegative")

    @property
    def total(self) -> float:
        return self.supervised + self.esil + self.demh


def build_component_blueprint() -> RDEAComponentBlueprint:
    """Return the immutable component description."""

    return RDEAComponentBlueprint()


def _dense_shape(shape: Iterable[int], name: str) -> TensorShape:
    values = tuple(int(value) for value in shape)
    if len(values) != 4 or any(value <= 0 for value in values):
        raise ValueError(f"{name} must follow [batch, classes, height, width]")
    if values[1] < 2:
        raise ValueError(f"{name} must contain at least two classes")
    return values


def make_sppc_pair_spec(
    image_shape: Iterable[int], original_domain: str, translated_domain: str
) -> PairedViewSpec:
    """Build and validate the public geometry of an SPPC pair.

    This helper validates paired-view geometry and domain direction.
    """

    shape = tuple(int(value) for value in image_shape)
    pair = PairedViewSpec(
        original_shape=shape,
        translated_shape=shape,
        original_domain=original_domain,
        translated_domain=translated_domain,
    )
    validate_paired_view(pair)
    return pair


def build_esil_plan(
    probability_shape: Iterable[int], path: ConsistencyPath
) -> ESILPlan:
    """Describe one ESIL path without implementing reliability or divergence."""

    shape = _dense_shape(probability_shape, "probability_shape")
    batch, _, height, width = shape
    return ESILPlan(
        path=path,
        probability_shape=shape,
        reliability_shape=(batch, 1, height, width),
        compared_target_detached=path is ConsistencyPath.TEACHER_STUDENT,
    )


def build_demh_plan(alpha_shape: Iterable[int]) -> DEMHPlan:
    """Describe DEMH's original-anchor direction without computing its loss."""

    shape = _dense_shape(alpha_shape, "alpha_shape")
    batch, _, height, width = shape
    return DEMHPlan(
        anchor_shape=shape,
        optimized_shape=shape,
        reliability_shape=(batch, 1, height, width),
    )
