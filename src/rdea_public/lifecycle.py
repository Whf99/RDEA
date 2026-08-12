"""Auditable state machine for the two-stage RDEA lifecycle.

The state machine checks stage ordering and component roles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from typing import Iterable, Tuple

from .interfaces import OptimizationStage


class LifecycleError(RuntimeError):
    """Raised when the paper's stage ordering is violated."""


class ModelRole(str, Enum):
    STUDENT = "student"
    EMA_TEACHER = "ema_teacher"


@dataclass(frozen=True)
class RDEAState:
    """Observable, non-numerical state of an RDEA run."""

    stage: OptimizationStage = OptimizationStage.PAIRED_VIEW_CONSTRUCTION
    translators_trained: bool = False
    translators_frozen: bool = False
    teacher_initialized: bool = False
    student_updates: int = 0
    teacher_updates: int = 0
    inference_role: ModelRole | None = None


def ema_update_values(
    teacher: Iterable[float], student: Iterable[float], momentum: float
) -> Tuple[float, ...]:
    """Apply the paper's generic EMA relation to equally sized value vectors.

    This utility operates on value vectors; model traversal and buffer updates
    are responsibilities of the training integration.
    """

    teacher_values = tuple(float(value) for value in teacher)
    student_values = tuple(float(value) for value in student)
    if not teacher_values or len(teacher_values) != len(student_values):
        raise LifecycleError("teacher and student vectors must align")
    if any(not isfinite(value) for value in teacher_values + student_values):
        raise LifecycleError("EMA values must be finite")
    if not isfinite(momentum) or not 0.0 <= momentum < 1.0:
        raise LifecycleError("momentum must lie in [0, 1)")
    return tuple(
        momentum * teacher_value + (1.0 - momentum) * student_value
        for teacher_value, student_value in zip(teacher_values, student_values)
    )


def complete_sppc(state: RDEAState) -> RDEAState:
    """Mark Stage I complete without exposing the translator implementation."""

    if state.stage is not OptimizationStage.PAIRED_VIEW_CONSTRUCTION:
        raise LifecycleError("SPPC can only be completed during Stage I")
    return replace(state, translators_trained=True)


def begin_adaptation(state: RDEAState) -> RDEAState:
    """Freeze SPPC and initialize the EMA teacher from the student."""

    if not state.translators_trained:
        raise LifecycleError("Stage II requires a completed SPPC stage")
    return replace(
        state,
        stage=OptimizationStage.EVIDENTIAL_ADAPTATION,
        translators_frozen=True,
        teacher_initialized=True,
    )


def record_adaptation_step(state: RDEAState) -> RDEAState:
    """Record one student update followed by one EMA teacher update."""

    if state.stage is not OptimizationStage.EVIDENTIAL_ADAPTATION:
        raise LifecycleError("adaptation steps are only valid during Stage II")
    if not state.translators_frozen or not state.teacher_initialized:
        raise LifecycleError("Stage II requires frozen translators and a teacher")
    return replace(
        state,
        student_updates=state.student_updates + 1,
        teacher_updates=state.teacher_updates + 1,
    )


def select_inference_model(state: RDEAState) -> RDEAState:
    """Select the EMA teacher, the only inference role stated in the paper."""

    if state.teacher_updates < 1:
        raise LifecycleError("inference requires at least one EMA teacher update")
    return replace(state, inference_role=ModelRole.EMA_TEACHER)
