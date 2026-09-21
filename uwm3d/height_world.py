"""Paired top-view scenes whose correct action depends on hidden height.

The RGB renderer intentionally omits every height cue.  Each base scene is
emitted twice with byte-identical RGB/XY geometry: a low obstacle that should
be crossed from above and a high obstacle that must be routed around.  This
makes 2D observation insufficient by construction while keeping the geometry
and expert policy fully auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


HeightSplit = Literal["train", "val", "height_ood", "layout_ood", "camera_ood", "paired_test"]


@dataclass(frozen=True)
class HeightWorldConfig:
    image_size: int = 64
    horizon: int = 24
    action_dim: int = 3
    action_scale_xyz: tuple[float, float, float] = (0.10, 0.10, 0.075)
    object_radius: float = 0.035
    safety_margin: float = 0.025
    vertical_margin: float = 0.04
    max_lift: float = 0.34
    ground_z: float = 0.04
    goal_radius: float = 0.075

    @property
    def z2d_dim(self) -> int:
        # start xy, goal xy, obstacle center xy and half extents xy
        return 8

    @property
    def height_dim(self) -> int:
        # obstacle height and maximum lift
        return 2

    @property
    def z3d_dim(self) -> int:
        # start xyz, goal xyz, relative xyz, obstacle center xyz,
        # obstacle half extents xyz, max lift, camera yaw, valid flag
        return 18


def _camera_xy(points: np.ndarray, yaw: float) -> np.ndarray:
    center = np.array([0.5, 0.5], dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (points - center) @ rotation.T + center


def _pixel(point: np.ndarray, size: int) -> tuple[int, int]:
    return int(round(point[0] * (size - 1))), int(round((1.0 - point[1]) * (size - 1)))


def render_height_scene(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_half: np.ndarray,
    camera_yaw: float,
    config: HeightWorldConfig,
) -> np.ndarray:
    """Render only the XY projection; obstacle height is deliberately absent."""
    size = config.image_size
    image = Image.new("RGB", (size, size), color=(238, 234, 222))
    draw = ImageDraw.Draw(image)
    for value in range(8, size, 8):
        draw.line((value, 0, value, size), fill=(224, 219, 207), width=1)
        draw.line((0, value, size, value), fill=(224, 219, 207), width=1)

    corners = np.array([
        obstacle_center + [-obstacle_half[0], -obstacle_half[1]],
        obstacle_center + [ obstacle_half[0], -obstacle_half[1]],
        obstacle_center + [ obstacle_half[0],  obstacle_half[1]],
        obstacle_center + [-obstacle_half[0],  obstacle_half[1]],
    ], dtype=np.float32)
    camera_corners = _camera_xy(corners, camera_yaw)
    draw.polygon([_pixel(point, size) for point in camera_corners], fill=(105, 105, 112), outline=(55, 55, 62))

    camera_goal = _camera_xy(goal_xy[None], camera_yaw)[0]
    gx, gy = _pixel(camera_goal, size)
    draw.rounded_rectangle((gx - 5, gy - 5, gx + 5, gy + 5), radius=2, fill=(45, 105, 225), outline=(22, 61, 155))
    camera_start = _camera_xy(start_xy[None], camera_yaw)[0]
    sx, sy = _pixel(camera_start, size)
    draw.rectangle((sx - 4, sy - 4, sx + 4, sy + 4), fill=(220, 48, 45), outline=(135, 24, 24))
    return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)


def _resample_waypoints(
    waypoints: list[np.ndarray], config: HeightWorldConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a polyline to fixed-horizon actions without cutting corners."""
    scale = np.asarray(config.action_scale_xyz, dtype=np.float32)
    points = np.asarray(waypoints, dtype=np.float32)
    normalized_delta = np.abs(np.diff(points, axis=0) / scale)
    segment_cost = normalized_delta.max(axis=1)
    steps = np.maximum(1, np.ceil(segment_cost).astype(np.int64))
    if int(steps.sum()) > config.horizon:
        raise ValueError(f"Path needs {steps.sum()} steps but horizon is {config.horizon}")
    remaining = config.horizon - int(steps.sum())
    if remaining:
        weights = segment_cost / max(float(segment_cost.sum()), 1e-8)
        ideal_extra = weights * remaining
        extra = np.floor(ideal_extra).astype(np.int64)
        for index in np.argsort(-(ideal_extra - extra))[: remaining - int(extra.sum())]:
            extra[index] += 1
        steps += extra
    trajectory = [points[0]]
    for index, count in enumerate(steps):
        for step in range(1, int(count) + 1):
            fraction = step / float(count)
            trajectory.append(points[index] + fraction * (points[index + 1] - points[index]))
    trajectory_array = np.asarray(trajectory, dtype=np.float32)
    actions = np.diff(trajectory_array, axis=0) / scale
    if np.max(np.abs(actions)) > 1.0001:
        raise AssertionError("Normalized action exceeds [-1, 1]")
    return actions.astype(np.float32), trajectory_array.astype(np.float32)


def _expanded_bounds(center: np.ndarray, half: np.ndarray, config: HeightWorldConfig) -> tuple[float, float, float, float]:
    expansion = config.object_radius + config.safety_margin
    return (
        float(center[0] - half[0] - expansion),
        float(center[0] + half[0] + expansion),
        float(center[1] - half[1] - expansion),
        float(center[1] + half[1] + expansion),
    )


def plan_expert(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_half: np.ndarray,
    obstacle_height: float,
    config: HeightWorldConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return actions, trajectory, and path type (0=over, 1=around)."""
    ground = config.ground_z
    start = np.array([start_xy[0], start_xy[1], ground], dtype=np.float32)
    goal = np.array([goal_xy[0], goal_xy[1], ground], dtype=np.float32)
    clearance_z = obstacle_height + config.vertical_margin
    if clearance_z <= config.max_lift:
        waypoints = [
            start,
            np.array([start_xy[0], start_xy[1], clearance_z], dtype=np.float32),
            np.array([goal_xy[0], goal_xy[1], clearance_z], dtype=np.float32),
            goal,
        ]
        path_type = 0
    else:
        left, right, bottom, top = _expanded_bounds(obstacle_center, obstacle_half, config)
        waypoint_margin = 0.015
        route_left = max(0.05, left - waypoint_margin)
        route_right = min(0.95, right + waypoint_margin)
        candidates = []
        side_values = []
        if bottom - waypoint_margin >= 0.05:
            side_values.append(bottom - waypoint_margin)
        if top + waypoint_margin <= 0.95:
            side_values.append(top + waypoint_margin)
        if not side_values:
            raise ValueError("Obstacle leaves no collision-free around route")
        for side_y in side_values:
            route = [
                start,
                np.array([route_left, side_y, ground], dtype=np.float32),
                np.array([route_right, side_y, ground], dtype=np.float32),
                goal,
            ]
            cost = sum(np.abs(route[i + 1] - route[i]).sum() for i in range(len(route) - 1))
            candidates.append((cost, route))
        waypoints = min(candidates, key=lambda item: item[0])[1]
        path_type = 1
    actions, trajectory = _resample_waypoints(waypoints, config)
    return actions, trajectory, path_type


def trajectory_collides(
    trajectory: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_half: np.ndarray,
    obstacle_height: float,
    config: HeightWorldConfig,
) -> bool:
    """Conservative point-sampled collision test for the carried object's center."""
    left, right, bottom, top = _expanded_bounds(obstacle_center, obstacle_half, config)
    fractions = np.linspace(0.0, 1.0, 33, dtype=np.float32)[None, :, None]
    first = trajectory[:-1, None, :]
    points = first + fractions * (trajectory[1:, None, :] - first)
    inside_xy = (
        (points[..., 0] > left)
        & (points[..., 0] < right)
        & (points[..., 1] > bottom)
        & (points[..., 1] < top)
    )
    below_clearance = points[..., 2] < obstacle_height + config.vertical_margin - 1e-5
    return bool(np.any(inside_xy & below_clearance))


def _ranges(split: HeightSplit) -> dict[str, tuple[float, float]]:
    if split == "height_ood":
        return {"low": (0.04, 0.08), "high": (0.38, 0.44), "half_x": (0.07, 0.11), "half_y": (0.11, 0.18)}
    if split == "layout_ood":
        return {"low": (0.10, 0.18), "high": (0.35, 0.42), "half_x": (0.11, 0.15), "half_y": (0.07, 0.21)}
    return {"low": (0.10, 0.18), "high": (0.35, 0.42), "half_x": (0.07, 0.11), "half_y": (0.11, 0.18)}


def _z_features(
    start: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    half: np.ndarray,
    height: float,
    yaw: float,
    config: HeightWorldConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start3 = np.array([start[0], start[1], config.ground_z], dtype=np.float32)
    goal3 = np.array([goal[0], goal[1], config.ground_z], dtype=np.float32)
    z2d = np.concatenate((start, goal, center, half), dtype=np.float32)
    height_only = np.array([height, config.max_lift], dtype=np.float32)
    z3d = np.concatenate((
        start3,
        goal3,
        goal3 - start3,
        np.array([center[0], center[1], height / 2], dtype=np.float32),
        np.array([half[0], half[1], height / 2], dtype=np.float32),
        np.array([config.max_lift, yaw, 1.0], dtype=np.float32),
    ), dtype=np.float32)
    return z2d, height_only, z3d


def generate_height_split(
    pair_count: int,
    split: HeightSplit,
    seed: int,
    config: HeightWorldConfig | None = None,
) -> dict[str, np.ndarray]:
    config = config or HeightWorldConfig()
    rng = np.random.default_rng(seed)
    ranges = _ranges(split)
    records: dict[str, list] = {key: [] for key in (
        "images", "actions", "trajectories", "z2d", "height", "z3d", "random_condition",
        "starts", "goals", "obstacle_centers", "obstacle_halves", "obstacle_heights",
        "camera_yaws", "path_types", "pair_ids",
    )}
    for pair_id in range(pair_count):
        start = np.array([rng.uniform(0.08, 0.20), rng.uniform(0.18, 0.82)], dtype=np.float32)
        goal = np.array([rng.uniform(0.80, 0.92), rng.uniform(0.18, 0.82)], dtype=np.float32)
        center_y_range = (0.30, 0.70) if split != "layout_ood" else (0.20, 0.80)
        center = np.array([rng.uniform(0.43, 0.57), rng.uniform(*center_y_range)], dtype=np.float32)
        half = np.array([rng.uniform(*ranges["half_x"]), rng.uniform(*ranges["half_y"])], dtype=np.float32)
        yaw = float(rng.uniform(-0.38, 0.38)) if split == "camera_ood" else 0.0
        image = render_height_scene(start, goal, center, half, yaw, config)
        heights = (rng.uniform(*ranges["low"]), rng.uniform(*ranges["high"]))
        pair_images = []
        for height in heights:
            actions, trajectory, path_type = plan_expert(start, goal, center, half, float(height), config)
            if trajectory_collides(trajectory, center, half, float(height), config):
                raise AssertionError(f"Expert trajectory collides for pair {pair_id}, height={height}")
            z2d, height_only, z3d = _z_features(start, goal, center, half, float(height), yaw, config)
            records["images"].append(image)
            records["actions"].append(actions)
            records["trajectories"].append(trajectory)
            records["z2d"].append(z2d)
            records["height"].append(height_only)
            records["z3d"].append(z3d)
            records["random_condition"].append(rng.standard_normal(config.z3d_dim).astype(np.float32))
            records["starts"].append(np.array([start[0], start[1], config.ground_z], dtype=np.float32))
            records["goals"].append(np.array([goal[0], goal[1], config.ground_z], dtype=np.float32))
            records["obstacle_centers"].append(center)
            records["obstacle_halves"].append(half)
            records["obstacle_heights"].append(float(height))
            records["camera_yaws"].append(yaw)
            records["path_types"].append(path_type)
            records["pair_ids"].append(pair_id)
            pair_images.append(image)
        if not np.array_equal(pair_images[0], pair_images[1]):
            raise AssertionError("Counterfactual pair RGB images must be byte-identical")
        if records["path_types"][-2:] != [0, 1]:
            raise AssertionError("Each pair must contain one over and one around trajectory")
    integer_keys = {"path_types", "pair_ids"}
    result = {}
    for key, value in records.items():
        if key == "images":
            result[key] = np.asarray(value, dtype=np.uint8)
        elif key in integer_keys:
            result[key] = np.asarray(value, dtype=np.int64)
        else:
            result[key] = np.asarray(value, dtype=np.float32)
    return result


def write_height_dataset(
    path: Path,
    pair_count: int,
    split: HeightSplit,
    seed: int,
    config: HeightWorldConfig | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **generate_height_split(pair_count, split, seed, config))


class HeightWorldDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str | Path, limit: int | None = None):
        archive = np.load(path)
        count = min(len(archive["images"]), limit) if limit else len(archive["images"])
        tensor_keys = (
            "actions", "trajectories", "z2d", "height", "z3d", "random_condition",
            "starts", "goals", "obstacle_centers", "obstacle_halves", "obstacle_heights",
            "camera_yaws", "path_types", "pair_ids",
        )
        self.images = torch.from_numpy(archive["images"][:count]).float().div_(255.0)
        self.values = {}
        for key in tensor_keys:
            tensor = torch.from_numpy(archive[key][:count])
            self.values[key] = tensor if key in {"path_types", "pair_ids"} else tensor.float()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index], **{key: value[index] for key, value in self.values.items()}}
