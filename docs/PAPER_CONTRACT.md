# Paper-to-code contract

This package follows the terminology and lifecycle in **Reliability-Driven Evidential Adaptation for Cross-Domain Medical Image Segmentation**.

| Paper concept | Public code | Coverage |
|---|---|---|
| Algorithm 1 | `train.py`, `run_training` | Command line, two-stage orchestration, component calls, and logging |
| RDEA | `RDEAPipelineInterface`, `RDEAState`, `ObjectiveTerms` | Lifecycle and objective assembly |
| SPPC | `SPPCBlueprint`, `make_sppc_pair_spec` | Bidirectional topology and paired-view geometry |
| ESIL | `ESILBlueprint`, `ESILPlan` | Consistency paths, reliability shapes, and gradient roles |
| DEMH | `DEMHBlueprint`, `DEMHPlan` | Anchor direction, parameter space, and clipping contract |
| Student/EMA teacher | `RDEAState`, `ema_update_values` | Update ordering and generic EMA relation |
| Dirichlet evidence | `evidential.py`, `EvidentialPredictionSpec` | Evidence projection and output shapes |
| 2.5D input | `SegmentationRequest` | Consecutive adjacent-slice semantics |
| Two-stage optimization | `OptimizationStage`, `lifecycle.py` | Stage ordering and transition checks |
| Teacher inference | `select_inference_model` | EMA-teacher inference role |
| DSC, ASSD, ECE | `metrics.py`, `EvaluatorInterface` | Dependency-free evaluation helpers |

## Lifecycle

1. Stage I constructs structure-preserving original/translated view pairs with SPPC.
2. Stage II freezes the translators and initializes the teacher from the student.
3. The student is optimized with supervised, ESIL, and DEMH objectives.
4. The teacher is updated by exponential moving average after each student update.
5. The EMA teacher provides evidential predictions during inference.

## Component relationships

- ESIL records teacher-student and original-translated consistency paths.
- DEMH records the original view as the detached evidential anchor and the translated view as the optimized branch.
- Evidential utilities implement `alpha = evidence + 1`, the Dirichlet mean, inverse-strength uncertainty, and bounded clipping.
- Objective assembly follows `L_total = L_sup + L_ESIL + L_DEMH`.
