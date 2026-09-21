from __future__ import annotations

import random

import numpy as np
import torch

from .config import WorldConfig
from .diffusion import DiffusionSchedule


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    inference_steps: int,
    seed: int = 0,
) -> dict[str, float]:
    model.eval()
    config = WorldConfig()
    # Generate inference noise from an isolated CPU RNG. Evaluation must not
    # consume the global RNG used by training, otherwise changing eval_every
    # silently changes subsequent batches, timesteps, and diffusion noise.
    eval_generator = torch.Generator(device="cpu").manual_seed(seed)
    successes, final_distances, first_direction_errors, total = 0, [], [], 0
    for batch in loader:
        image = batch["image"].to(device)
        z3d = batch["z3d"].to(device)
        initial_noise = torch.randn(
            len(image), model.horizon, model.action_dim, generator=eval_generator
        )
        predicted = schedule.sample(
            model,
            image,
            z3d,
            inference_steps=inference_steps,
            initial_noise=initial_noise,
        ).cpu()
        starts, goals = batch["start"], batch["goal"]
        position = starts.clone()
        for step in range(predicted.shape[1]):
            action = predicted[:, step]
            position = (position + action * config.action_scale).clamp(0.0, 1.0)
        distance = torch.linalg.vector_norm(position - goals, dim=-1)
        successes += int((distance <= config.goal_radius).sum())
        final_distances.extend(distance.tolist())
        target_direction = goals - starts
        first_direction_errors.extend(torch.linalg.vector_norm(predicted[:, 0] * config.action_scale - target_direction / config.horizon, dim=-1).tolist())
        total += len(image)
    return {
        "success": successes / max(total, 1),
        "final_distance": float(np.mean(final_distances)),
        "first_step_error": float(np.mean(first_direction_errors)),
    }
