"""Non-reconstructive placeholders for the private research core."""

from __future__ import annotations

from .interfaces import (
    DomainName,
    SegmentationRequest,
    SegmentationResponse,
)


class PrivateCoreAdapter:
    """Public boundary for an implementation that is not released yet.

    The methods intentionally fail instead of returning fake predictions. This
    keeps the interface demonstrable without implying that the public package
    can reproduce the paper results.
    """

    def __init__(self, implementation_id: str = "private-core") -> None:
        self.implementation_id = implementation_id

    def load_weights(self, artifact_id: str) -> None:
        raise NotImplementedError(
            "The research implementation and weights are withheld until "
            "the paper is accepted."
        )

    def set_domain(self, domain: DomainName) -> None:
        raise NotImplementedError(
            "Domain-specific behavior belongs to the withheld research core."
        )

    def predict(self, request: SegmentationRequest) -> SegmentationResponse:
        raise NotImplementedError(
            "Prediction is intentionally unavailable in this interface-only release."
        )

