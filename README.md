# RDEA

**Reliability-Driven Evidential Adaptation for Cross-Domain Medical Image Segmentation**

**Full source code coming soon:** the complete source code and end-to-end training pipeline will be released after the paper is accepted.

## Included components

- Typed 2.5D input and evidential-output contracts.
- SPPC, ESIL, and DEMH component blueprints.
- Student/EMA-teacher lifecycle and two-stage optimization checks.
- A concise `train.py` with command-line arguments, two-stage orchestration, component calls, and structured logging.
- Evidence-to-Dirichlet projection, Dirichlet mean, inverse-strength uncertainty, and bounded clipping.
- Generic EMA, DSC, ASSD, and ECE utilities.
- Dependency-free examples and contract tests.
- Selected qualitative comparison figures from the paper.

## Method overview

```text
Stage I: SPPC paired-view construction
        -> freeze translators
Stage II: supervised source learning + ESIL + DEMH
        -> student update -> EMA teacher update
Inference: EMA teacher -> evidence, probabilities, uncertainty, mask
```

### SPPC

Structure-Preserving Paired-View Construction defines bidirectional source-to-target and target-to-source view pairs while preserving spatial geometry. The translators are fixed during evidential adaptation.

### ESIL

Evidence-Induced Selective Invariance Learning describes two consistency paths:

- teacher-student consistency;
- original-translated consistency.

The public contracts record pixel-wise reliability shapes and the corresponding stop-gradient roles.

### DEMH

Dirichlet Evidential Matching and Harmonization aligns paired predictions in Dirichlet parameter space. The original view is represented as the stable anchor, while the translated view is the optimized branch. Parameter clipping is represented through caller-supplied bounds.

## Input and output contracts

The segmentation input is a 2.5D adjacent-slice stack with shape:

```text
[batch, stacked_slices, height, width]
```

The example uses three adjacent slices. The contract accepts consecutive adjacent-slice stacks and derives the following output shapes from the request:

- evidence: `[batch, classes, height, width]`;
- Dirichlet parameters: `[batch, classes, height, width]`;
- categorical mean: `[batch, classes, height, width]`;
- uncertainty: `[batch, 1, height, width]`;
- segmentation mask: `[batch, height, width]`.

## Public modules

- `train.py`: command-line configuration, two-stage training orchestration, module integration, and JSONL logging.
- `interfaces.py`: request, output, paired-view, pipeline, and evaluator contracts.
- `components.py`: SPPC topology, ESIL path plans, DEMH anchor plan, and objective assembly.
- `evidential.py`: evidence projection, Dirichlet mean, uncertainty, and clipping.
- `lifecycle.py`: two-stage ordering, generic EMA relation, and inference-role validation.
- `metrics.py`: dependency-free DSC, coordinate-based ASSD, and equal-width ECE.

The direct paper-to-code mapping is documented in [`docs/PAPER_CONTRACT.md`](docs/PAPER_CONTRACT.md).

## Quick check

Inspect the paper-aligned training plan:

```text
python train.py --check --dataset brats --source-domain FLAIR --target-domain T2
```

Run the example:

```text
python examples/interface_demo.py
```

Run the tests:

```text
python -m unittest discover -s tests -v
```

Only the Python standard library is required.

## Repository layout

```text
train.py                            # two-stage training entry point
src/rdea_public/interfaces.py       # public data and protocol contracts
src/rdea_public/components.py       # component blueprints
src/rdea_public/evidential.py       # Dirichlet transformations
src/rdea_public/lifecycle.py        # two-stage lifecycle checks
src/rdea_public/metrics.py          # DSC, ASSD, and ECE utilities
examples/interface_demo.py          # component demonstration
tests/test_contract.py              # interface contract tests
tests/test_public_components.py     # utility and lifecycle tests
tests/test_training_entrypoint.py   # training order and integration tests
configs/public_interface.json       # public schema summary
docs/PAPER_CONTRACT.md              # paper-to-code mapping
docs/figures/                       # qualitative comparison figures
```

## Qualitative comparisons

### Brain qualitative comparison

![Brain qualitative comparison](docs/figures/figure_brain_qualitative.png)

### Heart qualitative comparison

![Heart qualitative comparison](docs/figures/figure_heart_qualitative.png)

### Nasopharyngeal carcinoma qualitative comparison

![Nasopharyngeal carcinoma qualitative comparison](docs/figures/figure_npc_qualitative.png)
