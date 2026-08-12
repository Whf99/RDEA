from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rdea_train", PROJECT_ROOT / "train.py")
assert SPEC is not None and SPEC.loader is not None
train = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train
SPEC.loader.exec_module(train)


class MemoryLogger:
    def __init__(self) -> None:
        self.events = []

    def event(self, name, **values) -> None:
        self.events.append((name, values))


class FakeData:
    def sppc_batches(self, epoch):
        return [("source", "target")]

    def adaptation_batches(self, epoch):
        return [("source", "label", "target")]


class FakeSPPC:
    def __init__(self, calls):
        self.calls = calls

    def configure(self, optimizer, learning_rate):
        self.calls.append("sppc.configure")

    def train_step(self, batch):
        self.calls.append("sppc.train_step")
        return {"translator_loss": 1.0}

    def save(self, output_dir):
        self.calls.append("sppc.save")
        return "sppc-artifact"

    def load(self):
        self.calls.append("sppc.load")

    def freeze(self):
        self.calls.append("sppc.freeze")

    def construct_pairs(self, batch):
        self.calls.append("sppc.construct_pairs")
        return "paired"


class FakeModel:
    def __init__(self, calls):
        self.calls = calls

    def configure(self, optimizer, learning_rate):
        self.calls.append("model.configure")

    def initialize_teacher_from_student(self):
        self.calls.append("model.initialize_teacher")

    def predict_student(self, paired_batch):
        self.calls.append("model.predict_student")
        return "student"

    def predict_teacher(self, paired_batch):
        self.calls.append("model.predict_teacher")
        return "teacher"

    def update_student(self, losses):
        self.calls.append("model.update_student")

    def update_teacher(self, decay):
        self.calls.append("model.update_teacher")


class FakeObjectives:
    def __init__(self, calls):
        self.calls = calls

    def reliability(self, paired, student, teacher):
        self.calls.append("objectives.reliability")
        return "reliability"

    def supervised(self, paired, student):
        self.calls.append("objectives.supervised")
        return train.LossTerm(1.0, "supervised")

    def esil(self, paired, student, teacher, reliability):
        self.calls.append("objectives.esil")
        return train.LossTerm(2.0, "esil")

    def demh(self, paired, student, reliability):
        self.calls.append("objectives.demh")
        return train.LossTerm(3.0, "demh")


class FakeEvaluation:
    def __init__(self, calls):
        self.calls = calls

    def evaluate_teacher(self):
        self.calls.append("evaluation.evaluate_teacher")
        return {"DSC": 0.8, "ASSD": 1.2, "ECE": 0.1}

    def save_teacher(self, output_dir):
        self.calls.append("evaluation.save_teacher")
        return "teacher-artifact"


class TrainingEntrypointTests(unittest.TestCase):
    def test_check_plan_matches_paper_defaults(self):
        config = train.TrainingConfig("brats", "FLAIR", "T2")
        plan = train.training_plan(config)
        self.assertEqual(plan["input"]["image_size"], 240)
        self.assertEqual(plan["segmentation"]["epochs"], 300)
        self.assertEqual(plan["segmentation"]["batch_size"], 16)
        self.assertEqual(plan["segmentation"]["ema_decay"], 0.99)

    def test_stage_order_and_component_calls(self):
        calls = []
        backend = train.TrainingBackend(
            data=FakeData(),
            sppc=FakeSPPC(calls),
            model=FakeModel(calls),
            objectives=FakeObjectives(calls),
            evaluation=FakeEvaluation(calls),
        )
        with tempfile.TemporaryDirectory() as directory:
            config = train.TrainingConfig(
                dataset="npc",
                source_domain="MR",
                target_domain="CT",
                source_manifest=Path("source.json"),
                target_manifest=Path("target.json"),
                output_dir=Path(directory),
                sppc_epochs=1,
                segmentation_epochs=1,
            )
            state = train.run_training(config, backend, MemoryLogger())

        self.assertEqual(state.student_updates, 1)
        self.assertEqual(state.teacher_updates, 1)
        self.assertEqual(state.inference_role.value, "ema_teacher")
        self.assertLess(
            calls.index("sppc.freeze"), calls.index("model.initialize_teacher")
        )
        self.assertLess(
            calls.index("model.update_student"), calls.index("model.update_teacher")
        )
        self.assertLess(
            calls.index("objectives.esil"), calls.index("model.update_student")
        )
        self.assertLess(
            calls.index("objectives.demh"), calls.index("model.update_student")
        )


if __name__ == "__main__":
    unittest.main()
