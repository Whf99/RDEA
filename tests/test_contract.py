from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdea_public import (  # noqa: E402
    InterfaceError,
    PrivateCoreAdapter,
    SegmentationRequest,
    validate_request,
)


class PublicContractTests(unittest.TestCase):
    def test_valid_request_is_accepted(self) -> None:
        request = SegmentationRequest(
            image_shape=(2, 1, 128, 128),
            source_domain="source",
            target_domain="target",
            class_count=2,
        )
        validate_request(request)

    def test_invalid_shape_is_rejected(self) -> None:
        request = SegmentationRequest(
            image_shape=(1, 128, 128),
            source_domain="source",
        )
        with self.assertRaises(InterfaceError):
            validate_request(request)

    def test_private_core_does_not_predict(self) -> None:
        request = SegmentationRequest(
            image_shape=(1, 1, 32, 32),
            source_domain="source",
        )
        with self.assertRaises(NotImplementedError):
            PrivateCoreAdapter().predict(request)


if __name__ == "__main__":
    unittest.main()

