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
    PrivateCoreAdapter,
    SegmentationRequest,
    describe_contract,
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

    def test_private_core_does_not_predict(self) -> None:
        with self.assertRaises(NotImplementedError):
            PrivateCoreAdapter().predict(self.make_request())


if __name__ == "__main__":
    unittest.main()

