from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from stage2.evaluation.evaluator import EpisodeEvaluation, evaluate_actions, summarize_evaluations
from stage2.envs.dual_clearance.expert import rollout_expert
from stage2.envs.dual_clearance.geometry import generate_pair
from stage2.envs.dual_clearance.mujoco_env import DualClearanceMujocoEnv


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pair = generate_pair(0, "train", seed=42)

    def _expert_actions(self, scene):
        env = DualClearanceMujocoEnv(image_size=16)
        env.reset(scene)
        actions = rollout_expert(env).actions
        env.close()
        return actions

    def test_correct_actions_are_successful(self) -> None:
        for scene in self.pair.members:
            result = evaluate_actions(scene, self._expert_actions(scene))
            self.assertTrue(result.success, result)
            self.assertTrue(result.route_correct)
            self.assertEqual(result.failure_reasons, ())

    def test_early_release_is_a_drop(self) -> None:
        actions = list(self._expert_actions(self.pair.over_only))
        close_index = next(index for index, action in enumerate(actions) if action[6] < -0.5)
        early_release = np.zeros(7)
        early_release[6] = 1.0
        result = evaluate_actions(
            self.pair.over_only,
            [*actions[: close_index + 1], early_release],
        )
        self.assertFalse(result.success)
        self.assertTrue(result.drop)
        self.assertIn("drop", result.failure_reasons)

    def test_incomplete_trajectory_is_reported(self) -> None:
        result = evaluate_actions(self.pair.over_only, [])
        self.assertFalse(result.success)
        self.assertIn("incomplete", result.failure_reasons)

    def test_pair_metric_requires_both_members(self) -> None:
        first = evaluate_actions(self.pair.over_only, self._expert_actions(self.pair.over_only))
        second = evaluate_actions(self.pair.side_only, self._expert_actions(self.pair.side_only))
        summary = summarize_evaluations([first, second])
        self.assertEqual(summary["overall"]["episode_success_rate"], 1.0)
        self.assertEqual(summary["overall"]["pair_success_rate"], 1.0)
        failed_second = replace(second, success=False, failure_reasons=("incomplete",))
        failed_summary = summarize_evaluations([first, failed_second])
        self.assertEqual(failed_summary["overall"]["episode_success_rate"], 0.5)
        self.assertEqual(failed_summary["overall"]["pair_success_rate"], 0.0)

    def test_duplicate_or_incomplete_pairs_are_rejected(self) -> None:
        template = EpisodeEvaluation(
            episode_id="train:p:A",
            split="train",
            pair_id="p",
            pair_member="A",
            expected_route="over",
            route_decision=None,
            route_correct=False,
            success=False,
            collision=False,
            drop=False,
            bounds_violation=False,
            height_violation=False,
            final_placement_error_m=1.0,
            steps=0,
            submitted_actions=0,
            terminated=False,
            truncated=False,
            failure_reasons=("incomplete",),
        )
        with self.assertRaisesRegex(ValueError, "two distinct members"):
            summarize_evaluations([template])

    def test_swept_collision_closes_discrete_gap(self) -> None:
        start = np.array([-1.0, 0.0, 0.0])
        end = np.array([1.0, 0.0, 0.0])
        low = np.array([-0.1, -0.1, -0.1])
        high = np.array([0.1, 0.1, 0.1])
        self.assertFalse(np.all((start >= low) & (start <= high)))
        self.assertFalse(np.all((end >= low) & (end <= high)))
        self.assertTrue(DualClearanceMujocoEnv._segment_intersects_aabb(start, end, low, high))


if __name__ == "__main__":
    unittest.main()
