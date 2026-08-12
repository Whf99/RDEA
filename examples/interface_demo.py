"""Demonstrate the paper-aligned contract without running private RDEA code."""

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
    # Illustrative shape contract only; this is not an experimental setting.
    request = SegmentationRequest(
        image_shape=(1, 3, 256, 256),
        slice_offsets=(-1, 0, 1),
        source_domain="source",
        target_domain="target",
        class_count=2,
        metadata={"mode": "contract-only", "input": "2.5D"},
    )
    validate_request(request)

    print("Paper-aligned RDEA interface contract:")
    print(json.dumps(describe_contract(), indent=2))
    print("\nValidated 2.5D request:")
    print(json.dumps(asdict(request), indent=2))

    try:
        PrivateCoreAdapter().predict(request)
    except NotImplementedError as error:
        print("\nPrivate core boundary reached as expected:")
        print(error)


if __name__ == "__main__":
    main()

