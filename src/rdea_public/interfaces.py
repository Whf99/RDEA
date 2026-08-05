"""Public contracts only.

This module deliberately contains no model, training, loss, preprocessing, or
post-processing implementation. It defines the boundary used by callers of
the private research core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple


TensorShape = Tuple[int, ...]
DomainName = str


class InterfaceError(ValueError):
    """Raised when a caller violates the public data contract."""


@dataclass(frozen=True)
class SegmentationRequest:
    """Caller-facing description of one inference request.

    The actual image tensor is intentionally not accepted by this public
    contract package. Only the shape and non-sensitive request metadata are
    represented here.
    """

    image_shape: TensorShape
    source_domain: DomainName
    target_domain: Optional[DomainName] = None
    class_count: Optional[int] = None
    sample_id: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentationResponse:
    """Shape-level description of the private core output."""

    mask_shape: TensorShape
    uncertainty_shape: Optional[TensorShape] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


class ModelInterface(Protocol):
    """Minimal lifecycle contract for the private segmentation model."""

    def load_weights(self, artifact_id: str) -> None:
        """Load a private artifact identified outside this repository."""

    def set_domain(self, domain: DomainName) -> None:
        """Select the source or target domain context."""

    def predict(self, request: SegmentationRequest) -> SegmentationResponse:
        """Return a shape-compatible segmentation response."""


class DatasetInterface(Protocol):
    """Caller-facing dataset contract without dataset implementation."""

    def __len__(self) -> int:
        """Return the number of samples visible to the caller."""

    def get_request(self, index: int) -> SegmentationRequest:
        """Describe one sample without exposing the underlying data."""


class EvaluatorInterface(Protocol):
    """Metric boundary used by the private evaluation implementation."""

    def evaluate(
        self,
        prediction: SegmentationResponse,
        reference: Mapping[str, Any],
    ) -> Mapping[str, float]:
        """Return named metrics for a private prediction/reference pair."""


class PipelineInterface(Protocol):
    """High-level orchestration boundary."""

    def run(self, request: SegmentationRequest) -> SegmentationResponse:
        """Execute the private pipeline for one request."""


def validate_request(request: SegmentationRequest) -> None:
    """Validate only public shape and metadata constraints.

    This helper deliberately does not normalize data, load files, infer
    labels, or perform any model computation.
    """

    if len(request.image_shape) != 4:
        raise InterfaceError(
            "image_shape must follow [batch, channels, height, width]"
        )
    if any(int(dimension) <= 0 for dimension in request.image_shape):
        raise InterfaceError("all image dimensions must be positive")
    if not request.source_domain:
        raise InterfaceError("source_domain is required")
    if request.class_count is not None and request.class_count < 2:
        raise InterfaceError("class_count must be at least 2 when provided")


def describe_contract() -> Mapping[str, Sequence[str]]:
    """Return a non-sensitive summary for documentation and smoke tests."""

    return {
        "request": [
            "image_shape: [batch, channels, height, width]",
            "source_domain: caller-defined domain label",
            "target_domain: optional caller-defined domain label",
            "class_count: optional positive class count",
        ],
        "response": [
            "mask_shape: shape of the segmentation output",
            "uncertainty_shape: optional shape of an uncertainty output",
            "metadata: non-sensitive response metadata",
        ],
        "lifecycle": [
            "load_weights(artifact_id)",
            "set_domain(domain)",
            "predict(request)",
        ],
    }

