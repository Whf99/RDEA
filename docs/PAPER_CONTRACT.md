# Paper-to-code contract

This reviewer-facing package follows the terminology and lifecycle in the paper **Reliability-Driven Evidential Adaptation for Cross-Domain Medical Image Segmentation**.

| Paper concept | Public code contract | Disclosure level |
|---|---|---|
| RDEA | `RDEAPipelineInterface` | Lifecycle only |
| SPPC | `PairedViewConstructorInterface` and `PairedViewSpec` | Input/output geometry only |
| ESIL | `ReliabilityMeasureInterface` and `AdaptationObjectiveInterface.esil` | Shape and call boundary only |
| DEMH | `AdaptationObjectiveInterface.demh` | Call boundary only |
| Student/EMA teacher | `EvidentialModelInterface` and `TeacherUpdaterInterface` | Roles and lifecycle only |
| Dirichlet evidence | `EvidentialPredictionSpec` | Output tensor shapes only |
| 2.5D U-Net input | `SegmentationRequest` | Adjacent-slice stack semantics only |
| Two-stage optimization | `OptimizationStage` | Stage order only |
| Teacher-only inference | `RDEAPipelineInterface.infer` | Deployment role only |
| DSC, ASSD, ECE | `EvaluatorInterface` | Metric names only |

## Lifecycle represented by the interfaces

1. Stage I trains SPPC to construct structure-preserving original/translated pairs.
2. Stage II freezes the translators, initializes a teacher from the student, and optimizes the student with supervised, ESIL, and DEMH objectives.
3. The teacher is updated by exponential moving average after student updates.
4. Only the EMA teacher is retained for inference.

The public code deliberately does not implement the frequency pathway, evidential activation, reliability mask/weight, categorical or Dirichlet discrepancies, supervised loss, optimizer, schedules, or checkpoint handling. Consequently, the paper results cannot be reproduced from this repository.
