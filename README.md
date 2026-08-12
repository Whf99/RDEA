# RDEA - Public Interface Release

**Reliability-Driven Evidential Adaptation for Cross-Domain Medical Image Segmentation**

> **Reviewer-facing interface package.

## Release status

This is a limited public release prepared for peer review. It documents the interfaces used by the segmentation pipeline and provides a dependency-free contract test. The core implementation is withheld prior to acceptance.

**Full source code coming soon:** the complete source code will be released after the paper is accepted.

The public package is **not an executable reproduction of the paper results**. The interface demo validates the expected call structure and data contract only; it does not perform segmentation or recreate the reported metrics.

## What is included

- Typed 2.5D request and evidential-response contracts.
- Paper-aligned SPPC, ESIL, DEMH, student/EMA-teacher, pipeline, and evaluator interfaces.
- The two-stage optimization lifecycle and teacher-only inference role.
- Domain-selection and artifact-loading entry points at the interface level.
- A private-core adapter stub that deliberately raises `NotImplementedError`.
- A dependency-free interface demo and contract tests.
- Selected qualitative comparison figures from the paper.

## What is intentionally withheld

- The 2.5D U-Net architecture and custom modules.
- SPPC frequency pathways and translator implementation.
- ESIL reliability computation and categorical discrepancy implementation.
- DEMH Dirichlet discrepancy and numerical-stability implementation.
- Training, validation, supervised-loss, uncertainty, and post-processing implementations.
- Private datasets, annotations, raw predictions, checkpoints, and weights.
- Exact internal preprocessing details and sensitive hyperparameter combinations.
- Machine-specific paths, logs, private dependencies, and experiment scripts.

The omitted components are not reconstructible from this package alone. The public files expose only the boundary between the private implementation and its callers.

## Public interface

The public contract follows the paper's two-stage lifecycle:

```text
Stage I: SPPC paired-view construction
        -> freeze translators
Stage II: supervised source learning + ESIL + DEMH
        -> student update -> EMA teacher update
Inference: retained EMA teacher -> evidence, probabilities, uncertainty, mask
```

The contract is defined in [`src/rdea_public/interfaces.py`](src/rdea_public/interfaces.py), with a direct paper-to-code mapping in [`docs/PAPER_CONTRACT.md`](docs/PAPER_CONTRACT.md). It specifies the 2.5D input semantics, paired-view geometry, evidential outputs, reliability tensors, optimization stages, and evaluation metrics without exposing the internal computation.

### Method components

- **SPPC - Structure-Preserving Paired-View Construction:** constructs original/translated cross-domain views while preserving geometry.
- **ESIL - Evidence-Induced Selective Invariance Learning:** constrains categorical inconsistency under an evidence-induced reliability measure.
- **DEMH - Dirichlet Evidential Matching and Harmonization:** aligns paired Dirichlet predictions to reduce class-preference and evidence-strength drift.

The segmentation boundary uses a 2.5D adjacent-slice stack with shape `[batch, stacked_slices, height, width]`. The output contract includes nonnegative evidence, Dirichlet parameters, categorical mean, inverse-strength uncertainty, and a segmentation mask. The exact stack width and all executable computations remain private until acceptance.

## Interface-only demonstration

The demonstration intentionally stops at the private implementation boundary:

```text
python examples/interface_demo.py
```

Run the contract tests with:

```text
python -m unittest discover -s tests -v
```

No model weights, medical images, or private runtime dependencies are required for these checks.

## Repository layout

```text
src/rdea_public/interfaces.py       # public data and protocol contracts
src/rdea_public/stubs.py            # intentionally unavailable core adapter
examples/interface_demo.py          # schema/lifecycle demonstration only
tests/test_contract.py              # interface tests only
configs/public_interface.json       # redacted schema example
docs/PAPER_CONTRACT.md              # paper-to-interface mapping
docs/figures/                       # selected qualitative comparison figures
```

## Qualitative comparisons

### Brain qualitative comparison

![Brain qualitative comparison](docs/figures/figure_brain_qualitative.png)

### Heart qualitative comparison

![Heart qualitative comparison](docs/figures/figure_heart_qualitative.png)

### Nasopharyngeal carcinoma qualitative comparison

![Nasopharyngeal carcinoma qualitative comparison](docs/figures/figure_npc_qualitative.png)

## Disclosure statement

This repository defines the public software boundary available during peer review. The interface package, qualitative comparison figures, and contract tests are released separately from the private implementation. The complete implementation, training details, and reproducibility materials will be released after acceptance.
