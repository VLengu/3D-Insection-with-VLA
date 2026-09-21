from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader

from uwm3d.config import ModelConfig, WorldConfig
from uwm3d.data import PushToGoalDataset
from uwm3d.diffusion import DiffusionSchedule
from uwm3d.model import ActionDiffusionPolicy
from uwm3d.train_utils import choose_device, evaluate, seed_everything


def loader(path: Path, batch_size: int, limit: int | None, shuffle: bool) -> DataLoader:
    return DataLoader(PushToGoalDataset(path, limit=limit), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def masked_huber(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average one scalar loss per example over only numerically useful timesteps."""
    per_example = loss.mean(dim=tuple(range(1, loss.ndim)))
    return (per_example * mask).sum() / mask.sum().clamp_min(1.0)


def occlude_goal_in_image(
    image: torch.Tensor, goals: torch.Tensor, probability: float, style: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an RGB-only goal occlusion; returns image and a mask for clean examples."""
    if probability <= 0:
        return image, torch.ones(len(image), device=image.device)
    output = image.clone()
    occluded = torch.rand(len(image), device=image.device) < probability
    size = image.shape[-1]
    for index in torch.where(occluded)[0].tolist():
        x = int((goals[index, 0] * (size - 1)).item())
        y = int(((1.0 - goals[index, 1]) * (size - 1)).item())
        if style == "full":
            left, right = x - 7, x + 8
        else:  # partial
            left, right = x, x + 8
        output[index, :, max(0, y - 5):min(size, y + 6), max(0, left):min(size, right)] = torch.tensor(
            [205 / 255, 199 / 255, 185 / 255], device=image.device
        ).view(3, 1, 1)
    # A partial target remains visually identifiable, so train its auxiliary
    # visual head; a fully hidden target has no valid RGB-only target label.
    visual_loss_mask = torch.ones(len(image), device=image.device) if style == "partial" else (~occluded).float()
    return output, visual_loss_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UWM-Lite early 3D correction experiments.")
    parser.add_argument("--mode", choices=("base", "early", "middle", "middle_wide", "late", "all"), required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps; effective batch is batch_size * grad_accum.")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--prediction-type", choices=("epsilon", "v"), default="v")
    parser.add_argument("--three-d-adaln", action="store_true", help="Inject Z_3D into Transformer AdaLN with a late-strength gate.")
    parser.add_argument("--correction-strength", type=float, default=0.35, help="Global alpha for the bounded 3D residual correction.")
    parser.add_argument("--goal-loss-weight", type=float, default=3.0)
    parser.add_argument("--direction-loss-weight", type=float, default=0.5)
    parser.add_argument("--visual-goal-loss-weight", type=float, default=5.0)
    parser.add_argument("--aux-min-alpha-bar", type=float, default=0.20)
    parser.add_argument("--train-goal-occlusion-prob", type=float, default=0.0)
    parser.add_argument("--train-goal-occlusion-style", choices=("full", "partial"), default="full")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    if args.hidden_dim % args.heads != 0:
        raise ValueError("hidden_dim must be divisible by heads")
    if args.grad_accum < 1:
        raise ValueError("grad_accum must be at least 1")

    if not (args.data / "train.npz").exists():
        raise FileNotFoundError("Dataset missing. Run `./venv/bin/python scripts/generate_data.py` first.")
    seed_everything(args.seed)
    device = choose_device()
    train_loader = loader(args.data / "train.npz", args.batch_size, args.limit_train, shuffle=True)
    val_loader = loader(args.data / "val.npz", args.batch_size, args.limit_eval, shuffle=False)
    ood_loader = loader(args.data / "ood.npz", args.batch_size, args.limit_eval, shuffle=False)
    occluded_ood_loader = loader(args.data / "ood_occluded.npz", args.batch_size, args.limit_eval, shuffle=False)
    partial_ood_loader = loader(args.data / "ood_partial.npz", args.batch_size, args.limit_eval, shuffle=False)
    model = ActionDiffusionPolicy(
        ModelConfig(
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            heads=args.heads,
            prediction_type=args.prediction_type,
            use_3d_adaln=args.three_d_adaln,
            correction_strength=args.correction_strength,
        ),
        correction_mode=args.mode,
    ).to(device)
    world = WorldConfig()
    schedule = DiffusionSchedule(model.diffusion_steps, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_ood = float("-inf")
    history: list[dict[str, float | int]] = []
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} mode={args.mode} prediction_type={args.prediction_type} "
        f"three_d_adaln={args.three_d_adaln} correction_strength={args.correction_strength} "
        f"model={args.hidden_dim}x{args.depth} heads={args.heads} parameters={parameter_count:,} "
        f"train_examples={len(train_loader.dataset)} effective_batch={args.batch_size * args.grad_accum}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, diffusion_losses, goal_losses, direction_losses, visual_goal_losses = [], [], [], [], []
        diagnostic_sums = {
            f"{name}_{region}": 0.0
            for name in ("correction_ratio", "correction_saturation")
            for region in ("high_noise", "mid_noise", "low_noise")
        }
        diagnostic_counts = {region: 0 for region in ("high_noise", "mid_noise", "low_noise")}
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            image, action, z3d = batch["image"].to(device), batch["action"].to(device), batch["z3d"].to(device)
            start, goal = batch["start"].to(device), batch["goal"].to(device)
            image, clean_visual_mask = occlude_goal_in_image(
                image, goal, args.train_goal_occlusion_prob, args.train_goal_occlusion_style
            )
            timestep = torch.randint(1, model.diffusion_steps + 1, (len(image),), device=device)
            noise = torch.randn_like(action)
            noisy_action = schedule.add_noise(action, noise, timestep)
            prediction, correction_diagnostics = model(
                image=image,
                noisy_action=noisy_action,
                timesteps=timestep,
                z3d=z3d,
                return_diagnostics=True,
            )
            target = schedule.prediction_target(action, noise, timestep, model.prediction_type)
            diffusion_loss = nn.functional.mse_loss(prediction, target)

            # Epsilon loss alone can be minimized at high noise without learning
            # which visual target the action should reach. Convert the prediction
            # back to clean actions and apply task losses only when alpha_bar is
            # large enough that this conversion is well-conditioned.
            clean_prediction = schedule.predict_clean(noisy_action, prediction, timestep, model.prediction_type)
            valid_aux = (schedule.alpha_bar(timestep) >= args.aux_min_alpha_bar).float()
            predicted_endpoint = start + clean_prediction.sum(dim=1) * world.action_scale
            goal_loss = masked_huber(nn.functional.huber_loss(predicted_endpoint, goal, reduction="none"), valid_aux)
            desired_first_step = (goal - start) / model.horizon
            direction_loss = masked_huber(
                nn.functional.huber_loss(clean_prediction[:, 0] * world.action_scale, desired_first_step, reduction="none"), valid_aux
            )
            # Unlike the reconstructed-action losses above, this branch has no
            # access to a_t. It makes the image representation identify the
            # red cube and blue goal before the diffusion process starts.
            visual_goal_loss = masked_huber(
                nn.functional.huber_loss(model.predict_goal_delta(image), goal - start, reduction="none"), clean_visual_mask
            )
            loss = (
                diffusion_loss
                + args.goal_loss_weight * goal_loss
                + args.direction_loss_weight * direction_loss
                + args.visual_goal_loss_weight * visual_goal_loss
            )
            (loss / args.grad_accum).backward()
            is_update_step = (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == len(train_loader)
            if is_update_step:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            diffusion_losses.append(float(diffusion_loss.detach().cpu()))
            goal_losses.append(float(goal_loss.detach().cpu()))
            direction_losses.append(float(direction_loss.detach().cpu()))
            visual_goal_losses.append(float(visual_goal_loss.detach().cpu()))
            normalized_timestep = timestep.float() / model.diffusion_steps
            region_masks = {
                "high_noise": normalized_timestep > 2.0 / 3.0,
                "mid_noise": (normalized_timestep > 1.0 / 3.0) & (normalized_timestep <= 2.0 / 3.0),
                "low_noise": normalized_timestep <= 1.0 / 3.0,
            }
            for region, region_mask in region_masks.items():
                count = int(region_mask.sum().item())
                if count == 0:
                    continue
                diagnostic_counts[region] += count
                for name in ("correction_ratio", "correction_saturation"):
                    diagnostic_sums[f"{name}_{region}"] += float(
                        correction_diagnostics[name][region_mask].detach().sum().cpu()
                    )

        metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "train_diffusion_loss": sum(diffusion_losses) / len(diffusion_losses),
            "train_goal_loss": sum(goal_losses) / len(goal_losses),
            "train_direction_loss": sum(direction_losses) / len(direction_losses),
            "train_visual_goal_loss": sum(visual_goal_losses) / len(visual_goal_losses),
        }
        for name in ("correction_ratio", "correction_saturation"):
            for region in ("high_noise", "mid_noise", "low_noise"):
                metrics[f"train_{name}_{region}"] = (
                    diagnostic_sums[f"{name}_{region}"] / max(diagnostic_counts[region], 1)
                )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, schedule, device, args.inference_steps, seed=args.seed + 10_000)
            ood_metrics = evaluate(model, ood_loader, schedule, device, args.inference_steps, seed=args.seed + 20_000)
            occluded_ood_metrics = evaluate(
                model, occluded_ood_loader, schedule, device, args.inference_steps, seed=args.seed + 30_000
            )
            partial_ood_metrics = evaluate(
                model, partial_ood_loader, schedule, device, args.inference_steps, seed=args.seed + 40_000
            )
            metrics.update({f"val_{name}": value for name, value in val_metrics.items()})
            metrics.update({f"ood_{name}": value for name, value in ood_metrics.items()})
            metrics.update({f"occluded_ood_{name}": value for name, value in occluded_ood_metrics.items()})
            metrics.update({f"partial_ood_{name}": value for name, value in partial_ood_metrics.items()})
            if ood_metrics["success"] > best_ood:
                best_ood = ood_metrics["success"]
                torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, args.checkpoint_dir / f"best_{args.mode}.pt")
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

    result_path = args.checkpoint_dir / f"history_{args.mode}_seed{args.seed}.json"
    result_path.write_text(json.dumps(history, ensure_ascii=False, indent=2))
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
