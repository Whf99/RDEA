from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdea_public import (  # noqa: E402
    InterfaceError,
    MethodComponent,
    OptimizationStage,
    PairedViewSpec,
    SegmentationRequest,
    describe_contract,
    expected_prediction_spec,
    validate_paired_view,
    validate_request,
)


class PublicContractTests(unittest.TestCase):
    def make_request(self) -> SegmentationRequest:
        return SegmentationRequest(
            image_shape=(2, 3, 128, 128),
            slice_offsets=(-1, 0, 1),
            source_domain="source",
            target_domain="target",
            class_count=2,
        )

    def test_valid_2p5d_request_is_accepted(self) -> None:
        validate_request(self.make_request())

    def test_single_slice_request_is_rejected(self) -> None:
        request = SegmentationRequest(
            image_shape=(1, 1, 128, 128),
            slice_offsets=(0,),
            source_domain="source",
        )
        with self.assertRaises(InterfaceError):
            validate_request(request)

    def test_channel_and_offset_count_must_match(self) -> None:
        request = SegmentationRequest(
            image_shape=(1, 3, 128, 128),
            slice_offsets=(-1, 0, 1, 2),
            source_domain="source",
        )
        with self.assertRaises(InterfaceError):
            validate_request(request)

    def test_unspecified_stack_width_is_not_assumed(self) -> None:
        request = SegmentationRequest(
            image_shape=(1, 5, 128, 128),
            slice_offsets=(-2, -1, 0, 1, 2),
            source_domain="source",
            class_count=2,
        )
        validate_request(request)

    def test_offsets_must_be_adjacent(self) -> None:
        request = SegmentationRequest(
            image_shape=(1, 3, 128, 128),
            slice_offsets=(-2, 0, 2),
            source_domain="source",
        )
        with self.assertRaises(InterfaceError):
            validate_request(request)

    def test_output_shapes_are_derived_without_model_code(self) -> None:
        prediction = expected_prediction_spec(self.make_request())
        self.assertEqual(prediction.evidence_shape, (2, 2, 128, 128))
        self.assertEqual(prediction.uncertainty_shape, (2, 1, 128, 128))
        self.assertEqual(prediction.mask_shape, (2, 128, 128))

    def test_sppc_pair_preserves_geometry_and_crosses_domains(self) -> None:
        validate_paired_view(
            PairedViewSpec(
                original_shape=(1, 3, 128, 128),
                translated_shape=(1, 3, 128, 128),
                original_domain="source",
                translated_domain="target",
            )
        )

    def test_paper_component_names_are_exposed(self) -> None:
        self.assertEqual(
            {component.value for component in MethodComponent},
            {"SPPC", "ESIL", "DEMH"},
        )

    def test_two_paper_optimization_stages_are_exposed(self) -> None:
        self.assertEqual(len(OptimizationStage), 2)

    def test_contract_declares_ema_teacher_inference(self) -> None:
        lifecycle = " ".join(describe_contract()["optimization"])
        self.assertIn("EMA teacher only for inference", lifecycle)

if __name__ == "__main__":
    unittest.main()

