"""Demonstrate the paper-aligned public RDEA components."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdea_public import (  # noqa: E402
    RDEAState,
    SegmentationRequest,
    describe_contract,
    begin_adaptation,
    build_component_blueprint,
    complete_sppc,
    project_pixel_evidence,
    record_adaptation_step,
    select_inference_model,
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

    # Demonstration epsilon only; not an experimental configuration.
    prediction = project_pixel_evidence((0.0, 2.0), epsilon=1e-12)
    print("\nPublic evidential projection example:")
    print(json.dumps(asdict(prediction), indent=2))

    print("\nPublic component blueprint:")
    print(json.dumps(asdict(build_component_blueprint()), indent=2))

    lifecycle = complete_sppc(RDEAState())
    lifecycle = begin_adaptation(lifecycle)
    lifecycle = record_adaptation_step(lifecycle)
    lifecycle = select_inference_model(lifecycle)
    print("\nValidated lifecycle state:")
    print(json.dumps(asdict(lifecycle), indent=2))

if __name__ == "__main__":
    main()

