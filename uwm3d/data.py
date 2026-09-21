"""Offline synthetic push-to-goal data with a structured 3D oracle."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .config import WorldConfig


Split = Literal["train", "val", "ood", "ood_occluded", "ood_partial"]


def _sample_pair(rng: np.random.Generator, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """Draw start and goal that are separated enough for a meaningful trajectory."""
    for _ in range(100):
        start = rng.uniform(low, high, size=2)
        goal = rng.uniform(low, high, size=2)
        if np.linalg.norm(goal - start) > 0.28:
            return start.astype(np.float32), goal.astype(np.float32)
    raise RuntimeError("Could not sample separated start and goal")


def _pixel(point: np.ndarray, size: int) -> tuple[int, int]:
    return int(point[0] * (size - 1)), int((1.0 - point[1]) * (size - 1))


def render_scene(
    start: np.ndarray,
    goal: np.ndarray,
    distractors: np.ndarray,
    distractor_colors: list[tuple[int, int, int]],
    config: WorldConfig,
    occlusion: Literal["none", "full", "partial"] = "none",
) -> np.ndarray:
    """Render a top-down RGB observation. Physics state is intentionally not drawn as text."""
    size = config.image_size
    image = Image.new("RGB", (size, size), color=(238, 234, 222))
    draw = ImageDraw.Draw(image)

    # Light workspace grid helps the visual encoder retain position, not identity alone.
    for value in range(8, size, 8):
        draw.line((value, 0, value, size), fill=(224, 219, 207), width=1)
        draw.line((0, value, size, value), fill=(224, 219, 207), width=1)

    goal_xy = _pixel(goal, size)
    goal_half = 6
    draw.rounded_rectangle(
        (goal_xy[0] - goal_half, goal_xy[1] - goal_half, goal_xy[0] + goal_half, goal_xy[1] + goal_half),
        radius=2,
        fill=(45, 105, 225),
        outline=(22, 61, 155),
        width=1,
    )

    for pos, color in zip(distractors, distractor_colors, strict=True):
        xy = _pixel(pos, size)
        half = 4
        draw.rectangle((xy[0] - half, xy[1] - half, xy[0] + half, xy[1] + half), fill=color, outline=(80, 80, 80))

    start_xy = _pixel(start, size)
    cube_half = 5
    draw.rectangle(
        (start_xy[0] - cube_half, start_xy[1] - cube_half, start_xy[0] + cube_half, start_xy[1] + cube_half),
        fill=(220, 48, 45),
        outline=(135, 24, 24),
        width=1,
    )
    if occlusion == "full":
        # An opaque tabletop-colored card covers the semantic target cue. The
        # simulator state / Z_3D remains unchanged, making this a controlled
        # visual-reliability intervention rather than an unlabelled example.
        occlusion_half_x, occlusion_half_y = 7, 5
        draw.rectangle(
            (
                goal_xy[0] - occlusion_half_x,
                goal_xy[1] - occlusion_half_y,
                goal_xy[0] + occlusion_half_x,
                goal_xy[1] + occlusion_half_y,
            ),
            fill=(205, 199, 185),
            outline=(154, 147, 132),
            width=1,
        )
    elif occlusion == "partial":
        # Covers roughly half of the target. The remaining blue region is a
        # deliberately imperfect visual cue; Z_3D specifies the full target.
        draw.rectangle(
            (goal_xy[0], goal_xy[1] - 5, goal_xy[0] + 7, goal_xy[1] + 5),
            fill=(205, 199, 185),
            outline=(154, 147, 132),
            width=1,
        )
    return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)


def _expert_actions(start: np.ndarray, goal: np.ndarray, config: WorldConfig) -> np.ndarray:
    """Straight-line oracle with bounded continuous actions in [-1, 1]."""
    remaining = goal - start
    actions = []
    for step in range(config.horizon):
        delta = remaining / (config.horizon - step)
        normalized = np.clip(delta / config.action_scale, -1.0, 1.0)
        applied = normalized * config.action_scale
        actions.append(normalized)
        remaining = remaining - applied
    return np.asarray(actions, dtype=np.float32)


def _z3d(start: np.ndarray, goal: np.ndarray, distractors: np.ndarray, config: WorldConfig) -> np.ndarray:
    cube = np.array([start[0], start[1], config.cube_height], dtype=np.float32)
    target = np.array([goal[0], goal[1], 0.0], dtype=np.float32)
    relative = target - cube
    values = [cube, target, relative, np.array([np.linalg.norm(relative[:2])], dtype=np.float32)]
    for index in range(config.max_distractors):
        if index < len(distractors):
            values.append(np.array([distractors[index, 0], distractors[index, 1], config.cube_height, 1.0], dtype=np.float32))
        else:
            values.append(np.zeros(4, dtype=np.float32))
    return np.concatenate(values, dtype=np.float32)


def _distractors(
    rng: np.random.Generator,
    start: np.ndarray,
    goal: np.ndarray,
    split: Split,
    config: WorldConfig,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    if split == "train":
        count = int(rng.integers(0, 2))
        colors = [(124, 124, 124)]
    elif split == "val":
        count = int(rng.integers(0, 2))
        colors = [(124, 124, 124)]
    elif split in {"ood", "ood_occluded"}:
        count = int(rng.integers(1, config.max_distractors + 1))
        # OOD uses near-red and near-blue visual distractors.
        colors = [(190, 64, 55), (60, 105, 190), (155, 75, 85)]
    else:  # ood_partial
        count = 1
        colors = [(60, 105, 190)]

    placed: list[np.ndarray] = []
    for index in range(count):
        for _ in range(50):
            candidate = rng.uniform(0.12, 0.88, size=2).astype(np.float32)
            if np.linalg.norm(candidate - start) > 0.13 and np.linalg.norm(candidate - goal) > 0.13:
                placed.append(candidate)
                break
        else:
            placed.append(np.array([0.1 + 0.2 * index, 0.9], dtype=np.float32))
    return np.asarray(placed, dtype=np.float32).reshape(-1, 2), [colors[i % len(colors)] for i in range(count)]


def generate_split(count: int, split: Split, seed: int, config: WorldConfig | None = None) -> dict[str, np.ndarray]:
    config = config or WorldConfig()
    rng = np.random.default_rng(seed)
    images, actions, z3d, starts, goals = [], [], [], [], []
    low, high = (0.25, 0.75) if split in {"train", "val", "ood_partial"} else (0.10, 0.90)
    for _ in range(count):
        start, goal = _sample_pair(rng, low, high)
        distractors, colors = _distractors(rng, start, goal, split, config)
        occlusion: Literal["none", "full", "partial"] = "none"
        if split == "ood_occluded":
            occlusion = "full"
        elif split == "ood_partial":
            occlusion = "partial"
        images.append(render_scene(start, goal, distractors, colors, config, occlusion=occlusion))
        actions.append(_expert_actions(start, goal, config))
        z3d.append(_z3d(start, goal, distractors, config))
        starts.append(start)
        goals.append(goal)
    return {
        "images": np.asarray(images, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.float32),
        "z3d": np.asarray(z3d, dtype=np.float32),
        "starts": np.asarray(starts, dtype=np.float32),
        "goals": np.asarray(goals, dtype=np.float32),
    }


def write_dataset(path: Path, count: int, split: Split, seed: int, config: WorldConfig | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **generate_split(count=count, split=split, seed=seed, config=config))


class PushToGoalDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: str | Path, limit: int | None = None):
        archive = np.load(path)
        count = min(len(archive["images"]), limit) if limit else len(archive["images"])
        self.images = torch.from_numpy(archive["images"][:count]).float().div_(255.0)
        self.actions = torch.from_numpy(archive["actions"][:count])
        self.z3d = torch.from_numpy(archive["z3d"][:count])
        self.starts = torch.from_numpy(archive["starts"][:count])
        self.goals = torch.from_numpy(archive["goals"][:count])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[index],
            "action": self.actions[index],
            "z3d": self.z3d[index],
            "start": self.starts[index],
            "goal": self.goals[index],
        }
