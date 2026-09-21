"""Causal probe for whether the learned route embedding controls trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.train_height_world import collision_and_mode, predicted_trajectory, prepare_inputs
from uwm3d.diffusion import DiffusionSchedule
from uwm3d.height_world import HeightWorldConfig, HeightWorldDataset
from uwm3d.model import ActionDiffusionPolicy
from uwm3d.train_utils import choose_device, seed_everything


class RouteOverrideModel(nn.Module):
    """Expose a fixed route probability vector through the standard sampler API."""

    def __init__(self, model: ActionDiffusionPolicy, route_weights: torch.Tensor):
        super().__init__()
        self.model = model
        self.route_weights = route_weights
        self.horizon = model.horizon
        self.action_dim = model.action_dim
        self.prediction_type = model.prediction_type

    def forward(self, image, noisy_action, timesteps, z3d=None):
        return self.model(
            image, noisy_action, timesteps, z3d,
            route_override=self.route_weights,
        )


class IndependentTrajectoryClassifier(nn.Module):
    def __init__(self, horizon: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear((horizon + 1) * 3, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.network(trajectory)


def train_independent_classifier(
    data_path: Path, device: torch.device, epochs: int, seed: int
) -> tuple[IndependentTrajectoryClassifier, dict[str, float]]:
    seed_everything(seed)
    train = HeightWorldDataset(data_path / "train.npz")
    validation = HeightWorldDataset(data_path / "val.npz")
    model = IndependentTrajectoryClassifier(HeightWorldConfig().horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=256, shuffle=True, num_workers=0)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            trajectory = batch["trajectories"].to(device)
            label = batch["path_types"].to(device)
            loss = nn.functional.cross_entropy(model(trajectory), label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        trajectory = validation.values["trajectories"].to(device)
        label = validation.values["path_types"].to(device)
        prediction = model(trajectory).argmax(dim=-1)
        accuracy = float((prediction == label).float().mean().cpu())
    return model, {"validation_accuracy": accuracy}


def pair_swap(weights: torch.Tensor) -> torch.Tensor:
    if len(weights) % 2:
        raise ValueError("Route swap requires complete pairs")
    return weights.reshape(-1, 2, weights.shape[-1]).flip(1).reshape_as(weights)


def condition_metrics(
    trajectory: torch.Tensor,
    batch: dict[str, torch.Tensor],
    classifier: IndependentTrajectoryClassifier,
    device: torch.device,
) -> dict[str, float | list[float]]:
    world = HeightWorldConfig()
    centers = batch["obstacle_centers"].to(device)
    halves = batch["obstacle_halves"].to(device)
    heights = batch["obstacle_heights"].to(device)
    collision, semantic_mode = collision_and_mode(trajectory, centers, halves, heights, world)
    independent_mode = classifier(trajectory).argmax(dim=-1)
    endpoint_distance = torch.linalg.vector_norm(trajectory[:, -1] - batch["goals"].to(device), dim=-1)
    max_z = trajectory[:, :, 2].max(dim=1).values
    expansion = world.object_radius + world.safety_margin
    xy = trajectory[:, :, :2]
    q = (xy - centers[:, None, :]).abs() - (halves[:, None, :] + expansion)
    outside = torch.linalg.vector_norm(q.clamp_min(0.0), dim=-1)
    inside = torch.minimum(q.max(dim=-1).values, torch.zeros_like(outside))
    signed_distance = outside + inside
    minimum_xy_distance = signed_distance.min(dim=1).values
    steps = torch.linspace(0.0, 1.0, trajectory.shape[1], device=device).view(1, -1, 1)
    straight_xy = batch["starts"].to(device)[:, None, :2] + steps * (
        batch["goals"].to(device)[:, None, :2] - batch["starts"].to(device)[:, None, :2]
    )
    obstacle_zone = (
        (straight_xy[:, :, 0] >= centers[:, None, 0] - halves[:, None, 0] - expansion)
        & (straight_xy[:, :, 0] <= centers[:, None, 0] + halves[:, None, 0] + expansion)
    )
    lateral_distance = torch.linalg.vector_norm(xy - straight_xy, dim=-1)
    local_lateral = (lateral_distance * obstacle_zone).max(dim=1).values
    xy_velocity = xy[:, 1:] - xy[:, :-1]
    xy_curvature = torch.linalg.vector_norm(xy_velocity[:, 1:] - xy_velocity[:, :-1], dim=-1).sum(dim=1)
    terminal_direction = xy[:, -1] - xy[:, -2]
    terminal_target = batch["goals"].to(device)[:, :2] - xy[:, -2]
    terminal_alignment = nn.functional.cosine_similarity(terminal_direction, terminal_target, dim=-1, eps=1e-8)
    return {
        "footprint_crossing_rate": float((semantic_mode == 0).float().mean().cpu()),
        "semantic_over_rate": float((semantic_mode == 0).float().mean().cpu()),
        "independent_over_rate": float((independent_mode == 0).float().mean().cpu()),
        "semantic_classifier_agreement": float((semantic_mode == independent_mode).float().mean().cpu()),
        "semantic_mode_accuracy": float((semantic_mode == batch["path_types"].to(device)).float().mean().cpu()),
        "independent_mode_accuracy": float((independent_mode == batch["path_types"].to(device)).float().mean().cpu()),
        "collision_rate": float(collision.float().mean().cpu()),
        "max_z_mean": float(max_z.mean().cpu()),
        "max_z_std": float(max_z.std().cpu()),
        "endpoint_distance": float(endpoint_distance.mean().cpu()),
        "minimum_xy_signed_distance": float(minimum_xy_distance.mean().cpu()),
        "xy_clearance_rate": float((minimum_xy_distance >= 0).float().mean().cpu()),
        "obstacle_zone_lateral_deviation": float(local_lateral.mean().cpu()),
        "xy_curvature": float(xy_curvature.mean().cpu()),
        "terminal_direction_alignment": float(terminal_alignment.mean().cpu()),
    }


@torch.no_grad()
def run_probe(
    model: ActionDiffusionPolicy,
    classifier: IndependentTrajectoryClassifier,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    inference_steps: int,
    seed: int,
) -> dict:
    world = HeightWorldConfig()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    names = ("predicted", "forced_over", "forced_around", "swapped", "zero")
    trajectories = {name: [] for name in names}
    batches = []
    for batch in loader:
        image, condition = prepare_inputs(batch, "full-3d", device)
        if len(image) % 2:
            raise ValueError("Probe batches must preserve pairs")
        pair_noise = torch.randn(len(image) // 2, world.horizon, world.action_dim, generator=generator)
        initial_noise = pair_noise.repeat_interleave(2, dim=0).to(device)
        route_probabilities = model.predict_route(condition).softmax(dim=-1)
        overrides = {
            "predicted": route_probabilities,
            "forced_over": torch.tensor([1.0, 0.0], device=device).repeat(len(image), 1),
            "forced_around": torch.tensor([0.0, 1.0], device=device).repeat(len(image), 1),
            "swapped": pair_swap(route_probabilities),
            "zero": torch.zeros_like(route_probabilities),
        }
        start = batch["starts"].to(device)
        for name, weights in overrides.items():
            wrapper = RouteOverrideModel(model, weights)
            actions = schedule.sample(wrapper, image, condition, inference_steps, initial_noise=initial_noise)
            trajectories[name].append(predicted_trajectory(start, actions, world).cpu())
        batches.append({key: value.cpu() for key, value in batch.items()})

    combined_trajectories = {name: torch.cat(parts).to(device) for name, parts in trajectories.items()}
    combined_batch = {key: torch.cat([batch[key] for batch in batches]).to(device) for key in batches[0]}
    metrics = {
        name: condition_metrics(trajectory, combined_batch, classifier, device)
        for name, trajectory in combined_trajectories.items()
    }
    over = combined_trajectories["forced_over"]
    around = combined_trajectories["forced_around"]
    predicted = combined_trajectories["predicted"]
    _, over_mode = collision_and_mode(over, combined_batch["obstacle_centers"], combined_batch["obstacle_halves"], combined_batch["obstacle_heights"], world)
    _, around_mode = collision_and_mode(around, combined_batch["obstacle_centers"], combined_batch["obstacle_halves"], combined_batch["obstacle_heights"], world)
    metrics["causal_effect"] = {
        "forced_trajectory_l2": float(torch.linalg.vector_norm((over - around).flatten(1), dim=1).mean().cpu()),
        "forced_mode_flip_rate": float((over_mode != around_mode).float().mean().cpu()),
        "forced_max_z_delta": float((over[:, :, 2].max(1).values - around[:, :, 2].max(1).values).mean().cpu()),
        "forced_endpoint_delta": float(torch.linalg.vector_norm(over[:, -1] - around[:, -1], dim=-1).mean().cpu()),
        "zero_vs_predicted_l2": float(torch.linalg.vector_norm((combined_trajectories["zero"] - predicted).flatten(1), dim=1).mean().cpu()),
        "swapped_vs_predicted_l2": float(torch.linalg.vector_norm((combined_trajectories["swapped"] - predicted).flatten(1), dim=1).mean().cpu()),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_height/experiment_a/route_head_seed42/best_full-3d_base.pt"))
    parser.add_argument("--data", type=Path, default=Path("data_height"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--classifier-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=Path("outputs/route_control_probe.json"))
    args = parser.parse_args()
    if args.batch_size % 2:
        raise ValueError("batch-size must be even")
    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ActionDiffusionPolicy(checkpoint["config"], correction_mode=checkpoint["args"]["mode"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    classifier, classifier_metrics = train_independent_classifier(args.data, device, args.classifier_epochs, args.seed)
    loader = DataLoader(HeightWorldDataset(args.data / "paired_test.npz"), batch_size=args.batch_size, shuffle=False)
    schedule = DiffusionSchedule(model.diffusion_steps, device)
    metrics = run_probe(model, classifier, loader, schedule, device, args.inference_steps, args.seed + 1)
    result = {
        "checkpoint": str(args.checkpoint),
        "independent_classifier": classifier_metrics,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
