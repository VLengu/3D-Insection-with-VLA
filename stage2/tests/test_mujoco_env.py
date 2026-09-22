from __future__ import annotations

import unittest

import numpy as np

from stage2.data.schema import initial_robot_state
from stage2.envs.dual_clearance.expert import ExpertFailure, rollout_expert
from stage2.envs.dual_clearance.geometry import Route, generate_pair
from stage2.envs.dual_clearance.mujoco_env import DualClearanceMujocoEnv


class MujocoEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pair = generate_pair(0, "train", seed=42)

    def test_paired_policy_rgb_and_state_are_exactly_identical(self) -> None:
        images = []
        states = []
        for scene in self.pair.members:
            env = DualClearanceMujocoEnv(image_size=64)
            observation = env.reset(scene)
            images.append(env.render())
            states.append(observation["observation_state"])
            env.close()
        self.assertTrue(np.array_equal(images[0], images[1]))
        self.assertEqual(states[0], states[1])
        self.assertTrue(
            np.allclose(states[0], initial_robot_state(self.pair.over_only), atol=1e-12)
        )

    def test_critical_scene_elements_are_visible(self) -> None:
        env = DualClearanceMujocoEnv(image_size=256)
        env.reset(self.pair.over_only)
        metrics = env.visibility_metrics()
        self.assertGreaterEqual(metrics["object_pixels"], 120)
        self.assertGreaterEqual(metrics["goal_pixels"], 300)
        self.assertGreaterEqual(metrics["central_route_pixels"], 500)
        self.assertGreaterEqual(metrics["side_route_pixels"], 500)
        self.assertGreaterEqual(metrics["robot_pixels"], 3000)
        env.close()

    def test_correct_expert_completes_both_pair_members(self) -> None:
        for scene in self.pair.members:
            env = DualClearanceMujocoEnv(image_size=16)
            env.reset(scene)
            result = rollout_expert(env)
            self.assertTrue(result.success, result.final_info)
            self.assertEqual(result.route, scene.route_label.value)
            env.close()

    def test_wrong_routes_trigger_physical_failure(self) -> None:
        for scene in self.pair.members:
            env = DualClearanceMujocoEnv(image_size=16)
            env.reset(scene)
            wrong = Route.SIDE if scene.route_label is Route.OVER else Route.OVER
            with self.assertRaises(ExpertFailure):
                rollout_expert(env, forced_route=wrong)
            info = env.info()
            self.assertTrue(info["collision"] or info["height_violation"], info)
            env.close()

    def test_expert_actions_are_deterministic(self) -> None:
        action_runs = []
        for _ in range(2):
            env = DualClearanceMujocoEnv(image_size=16)
            env.reset(self.pair.over_only)
            action_runs.append(rollout_expert(env).actions)
            env.close()
        self.assertEqual(action_runs[0], action_runs[1])

    def test_action_contract_rejects_wrong_shape(self) -> None:
        env = DualClearanceMujocoEnv(image_size=16)
        env.reset(self.pair.over_only)
        with self.assertRaisesRegex(ValueError, "7D"):
            env.step(np.zeros(6))
        env.close()


if __name__ == "__main__":
    unittest.main()
