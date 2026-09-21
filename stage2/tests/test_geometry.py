from __future__ import annotations

import unittest

from stage2.envs.dual_clearance.geometry import Route, generate_pair


class GeometryTest(unittest.TestCase):
    def test_pair_has_exactly_one_feasible_route_per_member(self) -> None:
        for split in (
            "train",
            "val",
            "paired_test",
            "height_ood",
            "layout_ood",
            "camera_ood",
            "compound_ood",
        ):
            for index in range(100):
                pair = generate_pair(index, split, seed=10_000 + index)
                self.assertEqual(pair.over_only.feasible_route, Route.OVER)
                self.assertEqual(pair.side_only.feasible_route, Route.SIDE)
                self.assertNotEqual(
                    pair.over_only.over_feasible,
                    pair.over_only.side_feasible,
                )
                self.assertNotEqual(
                    pair.side_only.over_feasible,
                    pair.side_only.side_feasible,
                )

    def test_pair_shares_planar_layout_and_robot_limits(self) -> None:
        pair = generate_pair(3, "train", seed=42)
        self.assertEqual(pair.over_only.layout, pair.side_only.layout)
        self.assertEqual(pair.over_only.base_layout_id, pair.side_only.base_layout_id)

    def test_generation_is_deterministic(self) -> None:
        first = generate_pair(7, "train", seed=12345)
        second = generate_pair(7, "train", seed=12345)
        self.assertEqual(first, second)

    def test_fixed_route_baselines_are_balanced(self) -> None:
        scenes = [
            scene
            for index in range(64)
            for scene in generate_pair(index, "train", seed=index + 1).members
        ]
        over_accuracy = sum(scene.route_label is Route.OVER for scene in scenes) / len(scenes)
        side_accuracy = sum(scene.route_label is Route.SIDE for scene in scenes) / len(scenes)
        self.assertEqual(over_accuracy, 0.5)
        self.assertEqual(side_accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
