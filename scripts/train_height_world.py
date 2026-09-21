"""Train information-level baselines for the paired hidden-height world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader

from uwm3d.config import ModelConfig
from uwm3d.diffusion import DiffusionSchedule
from uwm3d.height_world import HeightWorldConfig, HeightWorldDataset
from uwm3d.model import ActionDiffusionPolicy
from uwm3d.train_utils import choose_device, seed_everything


CONDITIONS = ("rgb-only", "2d-oracle", "height-only", "full-3d", "3d-only", "random", "shuffled-3d")


class PairedBatchSampler(BatchSampler):
    """Shuffle complete counterfactual pairs without separating pair members."""

    def __init__(self, dataset_size: int, batch_size: int):
        if dataset_size % 2 or batch_size % 2:
            raise ValueError("Dataset and batch size must both preserve pairs")
        self.pair_count = dataset_size // 2
        self.pairs_per_batch = batch_size // 2

    def __iter__(self):
        order = torch.randperm(self.pair_count).tolist()
        for begin in range(0, self.pair_count, self.pairs_per_batch):
            batch_pairs = order[begin:begin + self.pairs_per_batch]
            yield [index for pair in batch_pairs for index in (2 * pair, 2 * pair + 1)]

    def __len__(self) -> int:
        return (self.pair_count + self.pairs_per_batch - 1) // self.pairs_per_batch


def condition_dim(name: str, world: HeightWorldConfig) -> int:
    return {
        "rgb-only": world.z3d_dim,
        "2d-oracle": world.z2d_dim,
        "height-only": world.height_dim,
        "full-3d": world.z3d_dim,
        "3d-only": world.z3d_dim,
        "random": world.z3d_dim,
        "shuffled-3d": world.z3d_dim,
    }[name]


def prepare_inputs(batch: dict[str, torch.Tensor], name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image = batch["image"].to(device)
    if name == "rgb-only":
        condition = torch.zeros(len(image), HeightWorldConfig().z3d_dim, device=device)
    elif name == "2d-oracle":
        condition = batch["z2d"].to(device)
    elif name == "height-only":
        condition = batch["height"].to(device)
    elif name in {"full-3d", "3d-only"}:
        condition = batch["z3d"].to(device)
    elif name == "random":
        condition = batch["random_condition"].to(device)
    elif name == "shuffled-3d":
        condition = batch["z3d"].roll(max(1, len(image) // 2), dims=0).to(device)
    else:
        raise ValueError(name)
    if name == "3d-only":
        image = torch.zeros_like(image)
    return image, condition


def predicted_trajectory(start: torch.Tensor, actions: torch.Tensor, world: HeightWorldConfig) -> torch.Tensor:
    scale = torch.tensor(world.action_scale_xyz, device=actions.device, dtype=actions.dtype)
    positions = start.unsqueeze(1) + torch.cumsum(actions * scale, dim=1)
    return torch.cat((start.unsqueeze(1), positions), dim=1)


def collision_and_mode(
    trajectory: torch.Tensor,
    centers: torch.Tensor,
    halves: torch.Tensor,
    heights: torch.Tensor,
    world: HeightWorldConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    expansion = world.object_radius + world.safety_margin
    xy = trajectory[:, :, :2]
    inside = (
        (xy[:, :, 0] > centers[:, None, 0] - halves[:, None, 0] - expansion)
        & (xy[:, :, 0] < centers[:, None, 0] + halves[:, None, 0] + expansion)
        & (xy[:, :, 1] > centers[:, None, 1] - halves[:, None, 1] - expansion)
        & (xy[:, :, 1] < centers[:, None, 1] + halves[:, None, 1] + expansion)
    )
    clear = trajectory[:, :, 2] >= heights[:, None] + world.vertical_margin
    collision = (inside & ~clear).any(dim=1)
    # Mode is a geometric route decision, independent of obstacle height:
    # crossing the XY footprint is an over attempt; avoiding it is around.
    # Feasibility of that attempt is measured separately by collision.
    crosses_footprint = inside.any(dim=1)
    # 0=over and 1=around, matching the dataset encoding.
    predicted_mode = (~crosses_footprint).long()
    return collision, predicted_mode


@torch.no_grad()
def evaluate(
    model: ActionDiffusionPolicy,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    condition_name: str,
    device: torch.device,
    inference_steps: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    world = HeightWorldConfig()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    endpoint_distances, collisions, bounds_violations, mode_correct, successes = [], [], [], [], []
    route_classifier_correct = []
    predicted_modes, pair_ids = [], []
    for batch in loader:
        image, condition = prepare_inputs(batch, condition_name, device)
        if model.route_classes > 0:
            route_prediction = model.predict_route(condition).argmax(dim=-1)
            route_classifier_correct.extend((route_prediction == batch["path_types"].to(device)).float().cpu().tolist())
        # Counterfactual pair members are adjacent and must share initial
        # diffusion noise. Otherwise an unconditioned stochastic policy could
        # appear to react to hidden height merely by sampling different modes.
        if len(image) % 2 != 0:
            raise ValueError("Evaluation batches must contain complete counterfactual pairs")
        pair_noise = torch.randn(len(image) // 2, world.horizon, world.action_dim, generator=generator)
        noise = pair_noise.repeat_interleave(2, dim=0).to(device)
        actions = schedule.sample(model, image, condition, inference_steps, initial_noise=noise)
        start = batch["starts"].to(device)
        goal = batch["goals"].to(device)
        trajectory = predicted_trajectory(start, actions, world)
        distance = torch.linalg.vector_norm(trajectory[:, -1] - goal, dim=-1)
        collision, predicted_mode = collision_and_mode(
            trajectory,
            batch["obstacle_centers"].to(device),
            batch["obstacle_halves"].to(device),
            batch["obstacle_heights"].to(device),
            world,
        )
        bounds_violation = (
            (trajectory[:, :, :2] < 0.0).any(dim=(1, 2))
            | (trajectory[:, :, :2] > 1.0).any(dim=(1, 2))
            | (trajectory[:, :, 2] < 0.0).any(dim=1)
            | (trajectory[:, :, 2] > world.max_lift).any(dim=1)
        )
        correct_mode = predicted_mode == batch["path_types"].to(device)
        success = (distance <= world.goal_radius) & ~collision & ~bounds_violation
        endpoint_distances.extend(distance.cpu().tolist())
        collisions.extend(collision.float().cpu().tolist())
        bounds_violations.extend(bounds_violation.float().cpu().tolist())
        mode_correct.extend(correct_mode.float().cpu().tolist())
        successes.extend(success.float().cpu().tolist())
        predicted_modes.extend(predicted_mode.cpu().tolist())
        pair_ids.extend(batch["pair_ids"].tolist())
    pair_results = []
    pair_decisions_differ = []
    for index in range(0, len(pair_ids), 2):
        if index + 1 < len(pair_ids) and pair_ids[index] == pair_ids[index + 1]:
            pair_results.append(mode_correct[index] == 1.0 and mode_correct[index + 1] == 1.0)
            pair_decisions_differ.append(predicted_modes[index] != predicted_modes[index + 1])
    result = {
        "success": float(np.mean(successes)),
        "endpoint_distance": float(np.mean(endpoint_distances)),
        "collision_rate": float(np.mean(collisions)),
        "bounds_violation_rate": float(np.mean(bounds_violations)),
        "path_mode_accuracy": float(np.mean(mode_correct)),
        "paired_both_modes_correct": float(np.mean(pair_results)) if pair_results else float("nan"),
        "paired_decisions_differ": float(np.mean(pair_decisions_differ)) if pair_decisions_differ else float("nan"),
    }
    if route_classifier_correct:
        result["route_classifier_accuracy"] = float(np.mean(route_classifier_correct))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hidden-height information baselines.")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--mode", choices=("base", "early", "middle", "middle_wide", "late", "all"), default="base")
    parser.add_argument("--correction-strength", type=float, default=0.025)
    parser.add_argument("--data", type=Path, default=Path("data_height"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--trajectory-loss-weight", type=float, default=2.0)
    parser.add_argument("--route-head", action="store_true")
    parser.add_argument("--geometry-cross-attention", action="store_true")
    parser.add_argument("--route-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-ranking-weight", type=float, default=0.25)
    parser.add_argument("--pair-ranking-margin", type=float, default=1.0)
    parser.add_argument("--spatial-consistency", action="store_true")
    parser.add_argument("--spatial-consistency-weight", type=float, default=2.0)
    parser.add_argument("--spatial-clearance-margin", type=float, default=0.015)
    parser.add_argument("--aux-min-alpha-bar", type=float, default=0.20)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_height"))
    args = parser.parse_args()

    world = HeightWorldConfig()
    if args.batch_size % 2 != 0:
        raise ValueError("batch-size must be even so evaluation pairs stay together")
    if args.limit_eval is not None and args.limit_eval % 2 != 0:
        raise ValueError("limit-eval must be even so evaluation pairs stay together")
    if args.geometry_cross_attention and args.condition != "full-3d":
        raise ValueError("Geometry cross-attention is defined for the full-3d condition")
    if args.geometry_cross_attention and not args.route_head:
        raise ValueError("Experiment A-prime requires --route-head")
    seed_everything(args.seed)
    device = choose_device()
    train_dataset = HeightWorldDataset(args.data / "train.npz", args.limit_train)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=PairedBatchSampler(len(train_dataset), args.batch_size),
        num_workers=0,
    )
    # Keep paired_test completely sealed during training. Checkpoint selection
    # uses validation only; the dedicated evaluator opens paired_test once the
    # checkpoint has been fixed.
    eval_splits = ("val", "height_ood", "layout_ood", "camera_ood")
    eval_loaders = {
        split: DataLoader(HeightWorldDataset(args.data / f"{split}.npz", args.limit_eval), batch_size=args.batch_size, shuffle=False)
        for split in eval_splits
    }
    config = ModelConfig(
        horizon=world.horizon,
        action_dim=world.action_dim,
        z_dim=condition_dim(args.condition, world),
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        prediction_type="v",
        # A-prime replaces global geometry AdaLN with local action-to-geometry
        # cross-attention. Time and route topology remain global conditions.
        use_3d_adaln=args.condition != "rgb-only" and not args.geometry_cross_attention,
        correction_strength=args.correction_strength,
        route_classes=2 if args.route_head else 0,
        geometry_tokens=args.geometry_cross_attention,
    )
    model = ActionDiffusionPolicy(config, correction_mode=args.mode).to(device)
    schedule = DiffusionSchedule(config.diffusion_steps, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_validation_key: tuple[float, float, float, float] | None = None
    print(f"device={device} condition={args.condition} mode={args.mode} parameters={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            image, condition = prepare_inputs(batch, args.condition, device)
            clean = batch["actions"].to(device)
            if args.spatial_consistency:
                if len(image) % 2:
                    raise ValueError("Spatial consistency requires complete counterfactual pairs")
                pair_timestep = torch.randint(1, model.diffusion_steps + 1, (len(image) // 2,), device=device)
                timestep = pair_timestep.repeat_interleave(2)
                pair_noise = torch.randn(len(image) // 2, world.horizon, world.action_dim, device=device)
                noise = pair_noise.repeat_interleave(2, dim=0)
            else:
                timestep = torch.randint(1, model.diffusion_steps + 1, (len(image),), device=device)
                noise = torch.randn_like(clean)
            noisy = schedule.add_noise(clean, noise, timestep)
            prediction = model(image, noisy, timestep, condition)
            target = schedule.prediction_target(clean, noise, timestep, model.prediction_type)
            diffusion_loss = nn.functional.mse_loss(prediction, target)
            clean_prediction = schedule.predict_clean(noisy, prediction, timestep, model.prediction_type)
            predicted_path = predicted_trajectory(batch["starts"].to(device), clean_prediction, world)
            expert_path = batch["trajectories"].to(device)
            per_example_path_loss = nn.functional.smooth_l1_loss(predicted_path, expert_path, reduction="none").mean((1, 2))
            valid = (schedule.alpha_bar(timestep) >= args.aux_min_alpha_bar).float()
            trajectory_loss = (per_example_path_loss * valid).sum() / valid.sum().clamp_min(1.0)
            loss = diffusion_loss + args.trajectory_loss_weight * trajectory_loss
            if args.route_head:
                route_logits = model.predict_route(condition)
                route_labels = batch["path_types"].to(device)
                route_loss = nn.functional.cross_entropy(route_logits, route_labels)
                if len(route_logits) % 2:
                    raise ValueError("Route ranking requires complete pairs")
                route_score = route_logits[:, 0] - route_logits[:, 1]
                low_score = route_score[0::2]
                high_score = route_score[1::2]
                ranking_loss = nn.functional.relu(args.pair_ranking_margin - low_score + high_score).mean()
                loss = loss + args.route_loss_weight * route_loss + args.pair_ranking_weight * ranking_loss
            if args.spatial_consistency:
                centers = batch["obstacle_centers"].to(device)
                halves = batch["obstacle_halves"].to(device)
                expansion = world.object_radius + world.safety_margin
                xy = predicted_path[:, :, :2]
                q = (xy - centers[:, None, :]).abs() - (halves[:, None, :] + expansion)
                outside = torch.linalg.vector_norm(q.clamp_min(0.0), dim=-1)
                inside = torch.minimum(q.max(dim=-1).values, torch.zeros_like(outside))
                signed_distance = outside + inside
                around = batch["path_types"].to(device) == 1
                clearance_loss = nn.functional.relu(
                    args.spatial_clearance_margin - signed_distance[around]
                ).mean()

                # Supervise XY specifically where the expert traverses the
                # obstacle's expanded X interval. This teaches which side to
                # take rather than merely repelling all points from the center.
                expert_x = expert_path[:, :, 0]
                local_mask = (
                    (expert_x >= centers[:, None, 0] - halves[:, None, 0] - expansion)
                    & (expert_x <= centers[:, None, 0] + halves[:, None, 0] + expansion)
                )
                local_xy_error = nn.functional.smooth_l1_loss(
                    predicted_path[:, :, :2], expert_path[:, :, :2], reduction="none"
                ).mean(dim=-1)
                local_xy_loss = (local_xy_error * local_mask).sum() / local_mask.sum().clamp_min(1)

                predicted_pair_delta = predicted_path[0::2, :, :2] - predicted_path[1::2, :, :2]
                expert_pair_delta = expert_path[0::2, :, :2] - expert_path[1::2, :, :2]
                pair_local_mask = local_mask[0::2] | local_mask[1::2]
                pair_delta_error = nn.functional.smooth_l1_loss(
                    predicted_pair_delta, expert_pair_delta, reduction="none"
                ).mean(dim=-1)
                pair_xy_loss = (pair_delta_error * pair_local_mask).sum() / pair_local_mask.sum().clamp_min(1)

                # v-prediction permits stable x0 reconstruction at high noise;
                # emphasize global-route learning early in reverse diffusion.
                noise_weight = 0.5 + 0.5 * timestep.float() / model.diffusion_steps
                spatial_loss = (clearance_loss + local_xy_loss + pair_xy_loss) * noise_weight.mean()
                loss = loss + args.spatial_consistency_weight * spatial_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record: dict[str, float | int] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        should_evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_evaluate:
            for index, (split, loader) in enumerate(eval_loaders.items()):
                metrics = evaluate(model, loader, schedule, args.condition, device, args.inference_steps, args.seed + 10_000 * (index + 1))
                record.update({f"{split}_{key}": value for key, value in metrics.items()})
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if should_evaluate:
            # The held-out paired test set must never choose a checkpoint.
            # Select on validation topology first, then success and endpoint.
            validation_key = (
                float(record["val_paired_both_modes_correct"]),
                float(record["val_paired_decisions_differ"]),
                float(record["val_success"]),
                -float(record["val_endpoint_distance"]),
            )
        if should_evaluate and (best_validation_key is None or validation_key > best_validation_key):
            best_validation_key = validation_key
            torch.save({"model": model.state_dict(), "args": vars(args), "config": config, "metrics": record}, args.checkpoint_dir / f"best_{args.condition}_{args.mode}.pt")
    (args.checkpoint_dir / f"history_{args.condition}_{args.mode}_seed{args.seed}.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
