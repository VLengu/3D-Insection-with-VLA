"""Validate paired invariance, expert geometry, schema, and baseline inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from uwm3d.height_world import HeightWorldConfig


def batch_collisions(data: dict[str, np.ndarray], config: HeightWorldConfig, batch_size: int = 512) -> np.ndarray:
    """Vectorized conservative collision checks for a complete archive."""
    results = []
    fractions = np.linspace(0.0, 1.0, 33, dtype=np.float32)[None, None, :, None]
    expansion = config.object_radius + config.safety_margin
    for begin in range(0, len(data["trajectories"]), batch_size):
        end = min(begin + batch_size, len(data["trajectories"]))
        trajectories = data["trajectories"][begin:end]
        first = trajectories[:, :-1, None, :]
        points = first + fractions * (trajectories[:, 1:, None, :] - first)
        centers = data["obstacle_centers"][begin:end]
        halves = data["obstacle_halves"][begin:end]
        heights = data["obstacle_heights"][begin:end]
        inside = (
            (points[..., 0] > centers[:, None, None, 0] - halves[:, None, None, 0] - expansion)
            & (points[..., 0] < centers[:, None, None, 0] + halves[:, None, None, 0] + expansion)
            & (points[..., 1] > centers[:, None, None, 1] - halves[:, None, None, 1] - expansion)
            & (points[..., 1] < centers[:, None, None, 1] + halves[:, None, None, 1] + expansion)
        )
        below = points[..., 2] < heights[:, None, None] + config.vertical_margin - 1e-5
        results.append((inside & below).any(axis=(1, 2)))
    return np.concatenate(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data_height"))
    args = parser.parse_args()
    config = HeightWorldConfig()
    expected = ("train", "val", "height_ood", "layout_ood", "camera_ood", "paired_test")
    for split in expected:
        path = args.data / f"{split}.npz"
        with np.load(path) as archive:
            data = {key: archive[key] for key in archive.files}
        count = len(data["images"])
        assert count % 2 == 0
        assert data["actions"].shape == (count, config.horizon, config.action_dim)
        assert data["z2d"].shape == (count, config.z2d_dim)
        assert data["height"].shape == (count, config.height_dim)
        assert data["z3d"].shape == (count, config.z3d_dim)
        for index in range(0, count, 2):
            assert data["pair_ids"][index] == data["pair_ids"][index + 1]
            assert np.array_equal(data["images"][index], data["images"][index + 1])
            assert np.array_equal(data["z2d"][index], data["z2d"][index + 1])
            assert data["path_types"][index:index + 2].tolist() == [0, 1]
            assert not np.allclose(data["actions"][index], data["actions"][index + 1])
        assert not batch_collisions(data, config).any()
        assert np.max(np.abs(data["actions"])) <= 1.0001
        assert np.max(np.linalg.norm(data["trajectories"][:, -1] - data["goals"], axis=-1)) < 1e-5
        print(f"validated {split}: {count // 2} counterfactual pairs / {count} samples")
    print("height-world validation passed")


if __name__ == "__main__":
    main()
