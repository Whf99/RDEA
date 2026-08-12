from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdea_public import (  # noqa: E402
    ConsistencyPath,
    ModelRole,
    ObjectiveTerms,
    RDEAState,
    average_symmetric_surface_distance,
    begin_adaptation,
    build_demh_plan,
    build_esil_plan,
    clip_dirichlet,
    complete_sppc,
    dice_score,
    ema_update_values,
    expected_calibration_error,
    project_pixel_evidence,
    record_adaptation_step,
    select_inference_model,
)


class EvidentialTransformTests(unittest.TestCase):
    def test_zero_evidence_is_uniform_and_maximally_uncertain(self) -> None:
        result = project_pixel_evidence((0.0, 0.0), epsilon=1e-12)
        self.assertEqual(result.alpha, (1.0, 1.0))
        self.assertEqual(result.probability, (0.5, 0.5))
        self.assertAlmostEqual(result.uncertainty, 1.0)

    def test_more_evidence_reduces_inverse_strength_uncertainty(self) -> None:
        low = project_pixel_evidence((0.0, 0.0), epsilon=1e-12).uncertainty
        high = project_pixel_evidence((4.0, 2.0), epsilon=1e-12).uncertainty
        self.assertLess(high, low)

    def test_dirichlet_clipping_uses_explicit_bounds(self) -> None:
        self.assertEqual(clip_dirichlet((0.5, 3.0), 1.0, 2.0), (1.0, 2.0))


class LifecycleTests(unittest.TestCase):
    def test_paper_stage_order_and_teacher_only_inference(self) -> None:
        state = complete_sppc(RDEAState())
        state = begin_adaptation(state)
        state = record_adaptation_step(state)
        state = select_inference_model(state)
        self.assertTrue(state.translators_frozen)
        self.assertEqual(state.student_updates, state.teacher_updates)
        self.assertEqual(state.inference_role, ModelRole.EMA_TEACHER)

    def test_demh_anchor_direction_matches_paper(self) -> None:
        plan = build_demh_plan((2, 3, 64, 64))
        self.assertTrue(plan.original_anchor_detached)
        self.assertTrue(plan.clipping_required)
        self.assertEqual(plan.anchor_shape, plan.optimized_shape)

    def test_esil_stop_gradient_depends_on_path(self) -> None:
        teacher_student = build_esil_plan(
            (2, 3, 64, 64), ConsistencyPath.TEACHER_STUDENT
        )
        original_translated = build_esil_plan(
            (2, 3, 64, 64), ConsistencyPath.ORIGINAL_TRANSLATED
        )
        self.assertTrue(teacher_student.compared_target_detached)
        self.assertFalse(original_translated.compared_target_detached)

    def test_generic_ema_relation(self) -> None:
        self.assertEqual(
            ema_update_values((0.0, 2.0), (2.0, 0.0), momentum=0.5),
            (1.0, 1.0),
        )


class MetricTests(unittest.TestCase):
    def test_dice_and_ece_have_expected_simple_values(self) -> None:
        self.assertAlmostEqual(dice_score((1, 1, 0), (1, 0, 0)), 2 / 3)
        self.assertAlmostEqual(
            expected_calibration_error((0.9, 0.8), (True, False), bins=2),
            0.35,
        )
        self.assertAlmostEqual(
            average_symmetric_surface_distance(((0, 0),), ((3, 4),), (1, 1)),
            5.0,
        )

    def test_total_objective_matches_paper_assembly(self) -> None:
        self.assertEqual(ObjectiveTerms(1.0, 2.0, 3.0).total, 6.0)


if __name__ == "__main__":
    unittest.main()
