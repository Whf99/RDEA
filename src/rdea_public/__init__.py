"""Public interface contracts for the reviewer-facing RDEA release.

The research implementation is intentionally not part of this package.
"""

from .interfaces import (
    DatasetInterface,
    EvaluatorInterface,
    InterfaceError,
    ModelInterface,
    PipelineInterface,
    SegmentationRequest,
    SegmentationResponse,
    describe_contract,
    validate_request,
)
from .stubs import PrivateCoreAdapter

__all__ = [
    "DatasetInterface",
    "EvaluatorInterface",
    "InterfaceError",
    "ModelInterface",
    "PipelineInterface",
    "PrivateCoreAdapter",
    "SegmentationRequest",
    "SegmentationResponse",
    "describe_contract",
    "validate_request",
]
