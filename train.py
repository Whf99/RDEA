"""Paper-aligned training entry point for RDEA.

This file exposes the command line, two-stage control flow, component calls,
and structured logging used by the training pipeline.  Numerical model,
dataset, translation, and objective implementations are supplied through a
backend factory so that this entry point remains independent of a specific
medical-image framework.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdea_public import (  # noqa: E402
    ObjectiveTerms,
    RDEAState,
    begin_adaptation,
    complete_sppc,
    record_adaptation_step,
    select_inference_model,
)


class TrainingStage(str, Enum):
    """Public choices for the paper's two-stage training schedule."""

    ALL = "all"
    SPPC = "sppc"
    ADAPT = "adapt"


PAPER_IMAGE_SIZE = {"mmwhs": 256, "brats": 240, "npc": 256}


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration fields stated in the paper or needed for integration."""

    dataset: str
    source_domain: str
    target_domain: str
    stage: TrainingStage = TrainingStage.ALL
    source_manifest: Optional[Path] = None
    target_manifest: Optional[Path] = None
    output_dir: Path = Path("runs/rdea")
    sppc_epochs: Optional[int] = None
    segmentation_epochs: int = 300
    batch_size: int = 16
    learning_rate: float = 1e-4
    ema_decay: float = 0.99
    optimizer: str = "adamw"
    slice_offsets: tuple[int, ...] = (-1, 0, 1)
    log_every: int = 20

    @property
    def image_size(self) -> int:
        return PAPER_IMAGE_SIZE[self.dataset]

    def validate(self, execution: bool = False) -> None:
        if self.dataset not in PAPER_IMAGE_SIZE:
            raise ValueError("dataset must be one of: mmwhs, brats, npc")
        if not self.source_domain or not self.target_domain:
            raise ValueError("source and target domains are required")
        if self.source_domain == self.target_domain:
            raise ValueError("source and target domains must differ")
        if self.segmentation_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive and finite")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1)")
        if self.optimizer != "adamw":
            raise ValueError("the paper specifies AdamW for both stages")
        if len(self.slice_offsets) < 2 or 0 not in self.slice_offsets:
            raise ValueError("2.5D input requires adjacent slices including 0")
        if tuple(sorted(self.slice_offsets)) != self.slice_offsets:
            raise ValueError("slice offsets must be ordered")
        expected = tuple(range(self.slice_offsets[0], self.slice_offsets[-1] + 1))
        if self.slice_offsets != expected:
            raise ValueError("slice offsets must be consecutive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.sppc_epochs is not None and self.sppc_epochs <= 0:
            raise ValueError("sppc_epochs must be positive when provided")
        if execution:
            if self.source_manifest is None or self.target_manifest is None:
                raise ValueError("both data manifests are required for execution")
            if self.stage in (TrainingStage.ALL, TrainingStage.SPPC):
                if self.sppc_epochs is None:
                    raise ValueError(
                        "--sppc-epochs is required when Stage I is executed"
                    )


@dataclass(frozen=True)
class LossTerm:
    """One differentiable backend loss and its scalar logging value."""

    value: float
    handle: Any = field(repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value < 0.0:
            raise ValueError("loss values must be finite and nonnegative")


class TrainingLogger(Protocol):
    def event(self, name: str, **values: Any) -> None:
        """Record one named training event."""


class DataModule(Protocol):
    def sppc_batches(self, epoch: int) -> Iterable[Any]:
        """Yield aligned source/target batches for Stage I."""

    def adaptation_batches(self, epoch: int) -> Iterable[Any]:
        """Yield labeled-source and unlabeled-target batches for Stage II."""


class SPPCModule(Protocol):
    def configure(self, optimizer: str, learning_rate: float) -> None:
        """Configure the Stage-I translator optimizer."""

    def train_step(self, batch: Any) -> Mapping[str, float]:
        """Update bidirectional translators with SPPC objectives."""

    def save(self, output_dir: Path) -> str:
        """Save the trained Stage-I translators and return an artifact id."""

    def load(self) -> None:
        """Load trained translators for an adaptation-only run."""

    def freeze(self) -> None:
        """Freeze both translation directions before Stage II."""

    def construct_pairs(self, batch: Any) -> Any:
        """Construct source/translated-source and target/translated-target pairs."""


class EvidentialModelModule(Protocol):
    def configure(self, optimizer: str, learning_rate: float) -> None:
        """Configure the student segmentation optimizer."""

    def initialize_teacher_from_student(self) -> None:
        """Copy student parameters to the EMA teacher."""

    def predict_student(self, paired_batch: Any) -> Any:
        """Return student evidence, Dirichlet means, and uncertainties."""

    def predict_teacher(self, paired_batch: Any) -> Any:
        """Return detached EMA-teacher targets for the ESIL path."""

    def update_student(self, losses: Sequence[Any]) -> None:
        """Minimize the assembled supervised, ESIL, and DEMH objectives."""

    def update_teacher(self, decay: float) -> None:
        """Apply one EMA update after the student update."""


class RDEAObjectiveModule(Protocol):
    def reliability(
        self, paired_batch: Any, student_predictions: Any, teacher_predictions: Any
    ) -> Any:
        """Build detached pixel-wise reliability controls for ESIL and DEMH."""

    def supervised(self, paired_batch: Any, student_predictions: Any) -> LossTerm:
        """Compute labeled source and translated-source segmentation terms."""

    def esil(
        self,
        paired_batch: Any,
        student_predictions: Any,
        teacher_predictions: Any,
        reliability: Any,
    ) -> LossTerm:
        """Compute teacher-student and original-translated categorical terms."""

    def demh(
        self, paired_batch: Any, student_predictions: Any, reliability: Any
    ) -> LossTerm:
        """Compute original-anchor paired Dirichlet harmonization terms."""


class EvaluationModule(Protocol):
    def evaluate_teacher(self) -> Mapping[str, float]:
        """Evaluate the EMA teacher with DSC, ASSD, and ECE."""

    def save_teacher(self, output_dir: Path) -> str:
        """Save the retained EMA teacher and return an artifact id."""


@dataclass(frozen=True)
class TrainingBackend:
    """Framework adapter consumed by the public orchestration code."""

    data: DataModule
    sppc: SPPCModule
    model: EvidentialModelModule
    objectives: RDEAObjectiveModule
    evaluation: EvaluationModule


class JsonlLogger:
    """Console plus JSONL event logger used by the public entry point."""

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "train.jsonl"
        self.console = logging.getLogger("rdea.train")

    def event(self, name: str, **values: Any) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": name,
            **values,
        }
        line = json.dumps(record, sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        self.console.info("%s %s", name, json.dumps(values, default=str))


def _metrics(values: Mapping[str, float]) -> dict[str, float]:
    metrics = {str(name): float(value) for name, value in values.items()}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("logged metrics must be finite")
    return metrics


def _run_sppc(
    config: TrainingConfig,
    backend: TrainingBackend,
    logger: TrainingLogger,
    state: RDEAState,
) -> RDEAState:
    assert config.sppc_epochs is not None
    backend.sppc.configure(config.optimizer, config.learning_rate)
    logger.event("stage_started", stage="stage_i_sppc")
    for epoch in range(1, config.sppc_epochs + 1):
        steps = 0
        for step, batch in enumerate(backend.data.sppc_batches(epoch), start=1):
            values = _metrics(backend.sppc.train_step(batch))
            steps = step
            if step == 1 or step % config.log_every == 0:
                logger.event(
                    "train_step",
                    stage="stage_i_sppc",
                    epoch=epoch,
                    step=step,
                    metrics=values,
                )
        if steps == 0:
            raise RuntimeError("Stage I received no training batches")
        logger.event("epoch_completed", stage="stage_i_sppc", epoch=epoch)
    artifact = backend.sppc.save(config.output_dir)
    logger.event("stage_completed", stage="stage_i_sppc", artifact=artifact)
    return complete_sppc(state)


def _run_adaptation(
    config: TrainingConfig,
    backend: TrainingBackend,
    logger: TrainingLogger,
    state: RDEAState,
) -> RDEAState:
    backend.sppc.freeze()
    backend.model.configure(config.optimizer, config.learning_rate)
    backend.model.initialize_teacher_from_student()
    state = begin_adaptation(state)
    logger.event("stage_started", stage="stage_ii_rdea")

    for epoch in range(1, config.segmentation_epochs + 1):
        steps = 0
        for step, batch in enumerate(
            backend.data.adaptation_batches(epoch), start=1
        ):
            paired = backend.sppc.construct_pairs(batch)
            student = backend.model.predict_student(paired)
            teacher = backend.model.predict_teacher(paired)
            reliability = backend.objectives.reliability(paired, student, teacher)
            supervised = backend.objectives.supervised(paired, student)
            esil = backend.objectives.esil(
                paired, student, teacher, reliability
            )
            demh = backend.objectives.demh(paired, student, reliability)
            terms = ObjectiveTerms(supervised.value, esil.value, demh.value)

            backend.model.update_student(
                (supervised.handle, esil.handle, demh.handle)
            )
            backend.model.update_teacher(config.ema_decay)
            state = record_adaptation_step(state)
            steps = step

            if step == 1 or step % config.log_every == 0:
                logger.event(
                    "train_step",
                    stage="stage_ii_rdea",
                    epoch=epoch,
                    step=step,
                    losses={
                        "supervised": terms.supervised,
                        "esil": terms.esil,
                        "demh": terms.demh,
                        "total": terms.total,
                    },
                )
        if steps == 0:
            raise RuntimeError("Stage II received no training batches")
        logger.event("epoch_completed", stage="stage_ii_rdea", epoch=epoch)

    state = select_inference_model(state)
    scores = _metrics(backend.evaluation.evaluate_teacher())
    artifact = backend.evaluation.save_teacher(config.output_dir)
    logger.event(
        "stage_completed",
        stage="stage_ii_rdea",
        metrics=scores,
        inference_model=state.inference_role.value,
        artifact=artifact,
    )
    return state


def run_training(
    config: TrainingConfig,
    backend: TrainingBackend,
    logger: TrainingLogger,
) -> RDEAState:
    """Execute Algorithm 1's stage order against a supplied backend."""

    config.validate(execution=True)
    logger.event(
        "run_started",
        dataset=config.dataset,
        source_domain=config.source_domain,
        target_domain=config.target_domain,
        stage=config.stage.value,
        image_size=config.image_size,
        slice_offsets=config.slice_offsets,
    )
    state = RDEAState()

    if config.stage in (TrainingStage.ALL, TrainingStage.SPPC):
        state = _run_sppc(config, backend, logger, state)
    else:
        backend.sppc.load()
        state = complete_sppc(state)
        logger.event("stage_loaded", stage="stage_i_sppc")

    if config.stage in (TrainingStage.ALL, TrainingStage.ADAPT):
        state = _run_adaptation(config, backend, logger, state)

    logger.event(
        "run_completed",
        student_updates=state.student_updates,
        teacher_updates=state.teacher_updates,
        inference_model=(
            state.inference_role.value if state.inference_role is not None else None
        ),
    )
    return state


def training_plan(config: TrainingConfig) -> Mapping[str, Any]:
    """Return the paper-aligned run plan used by ``--check``."""

    config.validate(execution=False)
    stages = []
    if config.stage in (TrainingStage.ALL, TrainingStage.SPPC):
        stages.append(
            {
                "name": "stage_i_sppc",
                "calls": [
                    "sppc.train_step",
                    "sppc.save",
                ],
            }
        )
    if config.stage in (TrainingStage.ALL, TrainingStage.ADAPT):
        stages.append(
            {
                "name": "stage_ii_rdea",
                "calls": [
                    "sppc.freeze",
                    "model.initialize_teacher_from_student",
                    "sppc.construct_pairs",
                    "model.predict_student",
                    "model.predict_teacher",
                    "objectives.reliability",
                    "objectives.supervised",
                    "objectives.esil",
                    "objectives.demh",
                    "model.update_student",
                    "model.update_teacher",
                    "evaluation.evaluate_teacher",
                    "evaluation.save_teacher",
                ],
            }
        )
    return {
        "method": "RDEA",
        "dataset": config.dataset,
        "domains": {
            "source": config.source_domain,
            "target": config.target_domain,
        },
        "input": {
            "image_size": config.image_size,
            "slice_offsets": config.slice_offsets,
        },
        "segmentation": {
            "epochs": config.segmentation_epochs,
            "batch_size": config.batch_size,
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "ema_decay": config.ema_decay,
        },
        "stages": stages,
    }


def load_backend(specification: str, config: TrainingConfig) -> TrainingBackend:
    """Load ``module:factory`` and construct a framework adapter."""

    if ":" not in specification:
        raise ValueError("backend must use module:factory syntax")
    module_name, factory_name = specification.rsplit(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    backend = factory(config)
    required = ("data", "sppc", "model", "objectives", "evaluation")
    if any(not hasattr(backend, name) for name in required):
        raise TypeError("backend factory did not return an RDEA training backend")
    return backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-aligned two-stage RDEA training entry point"
    )
    parser.add_argument("--dataset", required=True, choices=sorted(PAPER_IMAGE_SIZE))
    parser.add_argument("--source-domain", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument(
        "--stage", choices=[stage.value for stage in TrainingStage], default="all"
    )
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rdea"))
    parser.add_argument("--sppc-epochs", type=int)
    parser.add_argument("--segmentation-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--optimizer", choices=("adamw",), default="adamw")
    parser.add_argument(
        "--slice-offsets", nargs="+", type=int, default=(-1, 0, 1)
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--backend", help="framework adapter as module:factory")
    parser.add_argument(
        "--check", action="store_true", help="validate and print the training plan"
    )
    return parser


def config_from_args(arguments: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        dataset=arguments.dataset,
        source_domain=arguments.source_domain,
        target_domain=arguments.target_domain,
        stage=TrainingStage(arguments.stage),
        source_manifest=arguments.source_manifest,
        target_manifest=arguments.target_manifest,
        output_dir=arguments.output_dir,
        sppc_epochs=arguments.sppc_epochs,
        segmentation_epochs=arguments.segmentation_epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        ema_decay=arguments.ema_decay,
        optimizer=arguments.optimizer,
        slice_offsets=tuple(arguments.slice_offsets),
        log_every=arguments.log_every,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)
    config = config_from_args(arguments)
    if arguments.check:
        print(json.dumps(training_plan(config), indent=2, default=str))
        return 0
    if not arguments.backend:
        raise SystemExit("--backend module:factory is required unless --check is used")
    backend = load_backend(arguments.backend, config)
    run_training(config, backend, JsonlLogger(config.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
