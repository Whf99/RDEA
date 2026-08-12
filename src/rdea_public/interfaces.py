"""Public contracts for RDEA.

The names, stages, and tensor semantics in this module follow the paper
"Reliability-Driven Evidential Adaptation for Cross-Domain Medical Image
Segmentation".  The executable parts are limited to public contracts and
generic transformations already stated in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple


TensorShape = Tuple[int, ...]
SliceOffsets = Tuple[int, ...]
DomainName = str


class InterfaceError(ValueError):
    """Raised when a caller violates the public RDEA contract."""


class MethodComponent(str, Enum):
    """Component names used in the paper."""

    SPPC = "SPPC"
    ESIL = "ESIL"
    DEMH = "DEMH"


class OptimizationStage(str, Enum):
    """The two optimization stages defined by the paper."""

    PAIRED_VIEW_CONSTRUCTION = "stage_i_sppc"
    EVIDENTIAL_ADAPTATION = "stage_ii_rdea"


@dataclass(frozen=True)
class SegmentationRequest:
    """Shape-level request for the 2.5D evidential segmentation model.

    ``image_shape`` follows ``[batch, stacked_slices, height, width]``.
    ``slice_offsets`` records which adjacent slices form the channel stack.
    The package never receives the medical-image payload itself.
    """

    image_shape: TensorShape
    slice_offsets: SliceOffsets
    source_domain: DomainName
    target_domain: Optional[DomainName] = None
    class_count: Optional[int] = None
    sample_id: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedViewSpec:
    """Public description of an SPPC original/translated pair."""

    original_shape: TensorShape
    translated_shape: TensorShape
    original_domain: DomainName
    translated_domain: DomainName
    structure_preserving: bool = True


@dataclass(frozen=True)
class EvidentialPredictionSpec:
    """Tensor contract for one evidential segmentation prediction."""

    evidence_shape: TensorShape
    dirichlet_shape: TensorShape
    probability_shape: TensorShape
    uncertainty_shape: TensorShape
    mask_shape: TensorShape


@dataclass(frozen=True)
class ReliabilitySpec:
    """Shape-level contract for the reliability-induced pixel measure."""

    hard_mask_shape: TensorShape
    soft_weight_shape: TensorShape
    detached_from_gradient: bool = True


@dataclass(frozen=True)
class SegmentationResponse:
    """Public response contract returned by the EMA teacher at inference."""

    prediction: EvidentialPredictionSpec
    model_role: str = "ema_teacher"
    metadata: Mapping[str, str] = field(default_factory=dict)


class PairedViewConstructorInterface(Protocol):
    """SPPC paired-view construction interface."""

    def construct(self, request: SegmentationRequest) -> PairedViewSpec:
        """Construct an original/translated pair with matching geometry."""


class EvidentialModelInterface(Protocol):
    """Student/EMA-teacher evidential model boundary."""

    def load_weights(self, artifact_id: str) -> None:
        """Load a model artifact by identifier."""

    def predict_evidence(
        self, request: SegmentationRequest
    ) -> EvidentialPredictionSpec:
        """Describe evidence, Dirichlet mean, uncertainty, and mask outputs."""


class ReliabilityMeasureInterface(Protocol):
    """ESIL reliability-measure boundary."""

    def describe(
        self,
        first: EvidentialPredictionSpec,
        second: EvidentialPredictionSpec,
    ) -> ReliabilitySpec:
        """Describe hard selection and soft reliability weight tensors."""


class AdaptationObjectiveInterface(Protocol):
    """Boundary for the paper-aligned Stage-II objective."""

    def supervised(self, pair: PairedViewSpec) -> float:
        """Return the source and translated-source objective."""

    def esil(self, pair: PairedViewSpec) -> float:
        """Return the ESIL categorical-invariance objective."""

    def demh(self, pair: PairedViewSpec) -> float:
        """Return the DEMH evidential-invariance objective."""


class TeacherUpdaterInterface(Protocol):
    """EMA update boundary between student and teacher."""

    def update_teacher(self) -> None:
        """Apply one EMA update after a student optimization step."""


class DatasetInterface(Protocol):
    """Caller-facing dataset contract without data-loading implementation."""

    def __len__(self) -> int:
        """Return the number of samples visible to the caller."""

    def get_request(self, index: int) -> SegmentationRequest:
        """Describe one 2.5D sample without exposing the image payload."""


class EvaluatorInterface(Protocol):
    """Paper metric boundary: DSC, ASSD, and ECE."""

    def evaluate(
        self,
        prediction: SegmentationResponse,
        reference: Mapping[str, Any],
    ) -> Mapping[str, float]:
        """Return named evaluation metrics."""


class RDEAPipelineInterface(Protocol):
    """Two-stage RDEA lifecycle boundary."""

    def prepare_paired_views(self) -> None:
        """Stage I: train SPPC and materialize paired views."""

    def adapt(self) -> None:
        """Stage II: freeze SPPC and optimize student with RDEA losses."""

    def infer(self, request: SegmentationRequest) -> SegmentationResponse:
        """Run inference with the retained EMA teacher only."""


# Backward-compatible aliases for the first public-interface release.
ModelInterface = EvidentialModelInterface
PipelineInterface = RDEAPipelineInterface


def validate_request(request: SegmentationRequest) -> None:
    """Validate public 2.5D shape semantics only."""

    if len(request.image_shape) != 4:
        raise InterfaceError(
            "image_shape must follow [batch, stacked_slices, height, width]"
        )
    if any(int(dimension) <= 0 for dimension in request.image_shape):
        raise InterfaceError("all image dimensions must be positive")
    if len(request.slice_offsets) != request.image_shape[1]:
        raise InterfaceError(
            "stacked_slices must equal the number of slice_offsets"
        )
    if len(request.slice_offsets) < 2:
        raise InterfaceError("2.5D input requires multiple adjacent slices")
    if 0 not in request.slice_offsets:
        raise InterfaceError("slice_offsets must include the center slice (0)")
    if tuple(sorted(request.slice_offsets)) != request.slice_offsets:
        raise InterfaceError("slice_offsets must be strictly ordered")
    if len(set(request.slice_offsets)) != len(request.slice_offsets):
        raise InterfaceError("slice_offsets must not contain duplicates")
    expected_offsets = tuple(
        range(request.slice_offsets[0], request.slice_offsets[-1] + 1)
    )
    if request.slice_offsets != expected_offsets:
        raise InterfaceError("slice_offsets must describe consecutive slices")
    if not request.source_domain:
        raise InterfaceError("source_domain is required")
    if request.class_count is not None and request.class_count < 2:
        raise InterfaceError("class_count must be at least 2 when provided")


def describe_contract() -> Mapping[str, Sequence[str]]:
    """Return a non-sensitive paper-to-code contract summary."""

    return {
        "method": [
            "RDEA: Reliability-Driven Evidential Adaptation",
            "SPPC: Structure-Preserving Paired-View Construction",
            "ESIL: Evidence-Induced Selective Invariance Learning",
            "DEMH: Dirichlet Evidential Matching and Harmonization",
        ],
        "input": [
            "2.5D adjacent-slice stack",
            "image_shape: [batch, stacked_slices, height, width]",
            "source and optional target domain labels",
        ],
        "evidential_output": [
            "nonnegative evidence e",
            "Dirichlet parameters alpha = e + 1",
            "categorical mean p",
            "inverse-strength uncertainty u",
        ],
        "optimization": [
            "Stage I: train SPPC paired-view construction",
            "Stage II: freeze translators and optimize supervised + ESIL + DEMH",
            "update teacher from student by EMA",
            "retain EMA teacher only for inference",
        ],
        "evaluation": ["DSC", "ASSD", "ECE"],
        "public_implementation": [
            "evidence-to-Dirichlet projection",
            "two-stage lifecycle validation",
            "generic DSC, ASSD, and ECE helpers",
        ],
    }


def expected_prediction_spec(
    request: SegmentationRequest,
) -> EvidentialPredictionSpec:
    """Derive paper-aligned output shapes from a validated request."""

    validate_request(request)
    if request.class_count is None:
        raise InterfaceError("class_count is required to derive output shapes")
    batch, _, height, width = request.image_shape
    dense_shape = (batch, request.class_count, height, width)
    return EvidentialPredictionSpec(
        evidence_shape=dense_shape,
        dirichlet_shape=dense_shape,
        probability_shape=dense_shape,
        uncertainty_shape=(batch, 1, height, width),
        mask_shape=(batch, height, width),
    )


def validate_paired_view(pair: PairedViewSpec) -> None:
    """Validate SPPC's public geometry and cross-domain semantics."""

    if pair.original_shape != pair.translated_shape:
        raise InterfaceError("SPPC paired views must preserve spatial shape")
    if len(pair.original_shape) != 4:
        raise InterfaceError("paired views must use [batch, channels, H, W]")
    if any(dimension <= 0 for dimension in pair.original_shape):
        raise InterfaceError("paired-view dimensions must be positive")
    if not pair.original_domain or not pair.translated_domain:
        raise InterfaceError("paired-view domains are required")
    if pair.original_domain == pair.translated_domain:
        raise InterfaceError("SPPC translation must cross domain labels")

