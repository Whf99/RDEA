"""Demonstrate the public contract without running the private model."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdea_public import (  # noqa: E402
    PrivateCoreAdapter,
    SegmentationRequest,
    describe_contract,
    validate_request,
)


def main() -> None:
    request = SegmentationRequest(
        image_shape=(1, 1, 256, 256),
        source_domain="source",
        target_domain="target",
        class_count=2,
        metadata={"mode": "contract-only"},
    )
    validate_request(request)

    print("Public interface contract:")
    print(json.dumps(describe_contract(), indent=2))
    print("\nValidated request:")
    print(json.dumps(asdict(request), indent=2))

    try:
        PrivateCoreAdapter().predict(request)
    except NotImplementedError as error:
        print("\nPrivate core boundary reached as expected:")
        print(error)


if __name__ == "__main__":
    main()

