"""Non-reconstructive placeholders for the private RDEA core."""

from __future__ import annotations

from .interfaces import (
    EvidentialPredictionSpec,
    PairedViewSpec,
    SegmentationRequest,
    SegmentationResponse,
)


_WITHHELD = (
    "The RDEA implementation is withheld during peer review and will be "
    "released after paper acceptance."
)


class PrivateCoreAdapter:
    """Paper-aligned lifecycle boundary with no executable research logic."""

    def __init__(self, implementation_id: str = "rdea-private-core") -> None:
        self.implementation_id = implementation_id

    def load_weights(self, artifact_id: str) -> None:
        raise NotImplementedError(_WITHHELD)

    def construct_paired_view(
        self, request: SegmentationRequest
    ) -> PairedViewSpec:
        """SPPC boundary."""
        raise NotImplementedError(_WITHHELD)

    def predict_evidence(
        self, request: SegmentationRequest
    ) -> EvidentialPredictionSpec:
        """Student/teacher evidential-output boundary."""
        raise NotImplementedError(_WITHHELD)

    def compute_esil(self, pair: PairedViewSpec) -> float:
        """ESIL boundary."""
        raise NotImplementedError(_WITHHELD)

    def compute_demh(self, pair: PairedViewSpec) -> float:
        """DEMH boundary."""
        raise NotImplementedError(_WITHHELD)

    def update_teacher(self) -> None:
        """EMA teacher-update boundary."""
        raise NotImplementedError(_WITHHELD)

    def predict(self, request: SegmentationRequest) -> SegmentationResponse:
        """EMA-teacher inference boundary retained for compatibility."""
        raise NotImplementedError(_WITHHELD)

