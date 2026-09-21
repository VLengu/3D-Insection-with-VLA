"""Counterfactual diagnostics for whether an action policy uses Z_3D.

The four diagnostics are:
  * zero: replace Z_3D with zeros;
  * shuffle: pair each RGB observation with another sample's Z_3D;
  * shift: move only the internally consistent 3D target by a small amount;
  * conflict: put the 3D target opposite the RGB target around the start.

These tests establish dependence on the structured condition.  On the current
planar dataset they must not be interpreted as evidence of genuine 3D spatial
reasoning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from uwm3d.config import ModelConfig, WorldConfig
from uwm3d.data import PushToGoalDataset
from uwm3d.diffusion import DiffusionSchedule
from uwm3d.model import ActionDiffusionPolicy
from uwm3d.train_utils import choose_device


def load_model(path: Path, device: torch.device) -> tuple[ActionDiffusionPolicy, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    config = ModelConfig(
        hidden_dim=args["hidden_dim"],
        depth=args["depth"],
        heads=args["heads"],
        prediction_type=args.get("prediction_type", "epsilon"),
        use_3d_adaln=args.get("three_d_adaln", False),
        correction_strength=args.get("correction_strength", 0.35),
    )
    model = ActionDiffusionPolicy(config, correction_mode=args["mode"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, args


def update_target(z3d: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """Update target xyz and all derived target-relative fields consistently."""
    output = z3d.clone()
    output[:, 3:5] = target_xy
    output[:, 6:9] = output[:, 3:6] - output[:, 0:3]
    output[:, 9] = torch.linalg.vector_norm(output[:, 6:8], dim=-1)
    return output


def shifted_target(goal: torch.Tensor, amount: float) -> torch.Tensor:
    """Apply a deterministic horizontal intervention while remaining in bounds."""
    direction = torch.where(goal[:, 0] <= 0.5, 1.0, -1.0)
    shifted = goal.clone()
    shifted[:, 0] = (goal[:, 0] + amount * direction).clamp(0.05, 0.95)
    return shifted


def opposite_target(start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Place the 3D target opposite the visible goal with a safe in-frame scale."""
    direction = goal - start
    unit = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
    opposite = -unit
    positive_room = (0.95 - start) / opposite.clamp_min(1e-8)
    negative_room = (0.05 - start) / opposite.clamp_max(-1e-8)
    room = torch.where(opposite > 0, positive_room, negative_room)
    room = torch.where(opposite.abs() > 1e-8, room, torch.full_like(room, float("inf")))
    max_distance = room.min(dim=-1).values.clamp_min(0.05)
    original_distance = torch.linalg.vector_norm(direction, dim=-1)
    distance = torch.minimum(original_distance, 0.9 * max_distance).clamp_min(0.05)
    return (start + opposite * distance.unsqueeze(-1)).clamp(0.05, 0.95)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-8)


def sample_actions(
    model: ActionDiffusionPolicy,
    schedule: DiffusionSchedule,
    image: torch.Tensor,
    z3d: torch.Tensor,
    initial_noise: torch.Tensor,
    inference_steps: int,
) -> torch.Tensor:
    return schedule.sample(
        model, image, z3d, inference_steps=inference_steps, initial_noise=initial_noise
    )


def endpoint(actions: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    world = WorldConfig()
    return (start + actions.sum(dim=1) * world.action_scale).clamp(0.0, 1.0)


def summarize_standard(endpoints: torch.Tensor, goals: torch.Tensor) -> dict[str, float]:
    distance = torch.linalg.vector_norm(endpoints - goals, dim=-1)
    return {
        "success": float((distance <= WorldConfig().goal_radius).float().mean()),
        "final_distance": float(distance.mean()),
    }


@torch.no_grad()
def diagnose_checkpoint(
    checkpoint_path: Path,
    data_path: Path,
    batch_size: int,
    limit: int | None,
    inference_steps: int,
    eval_seed: int,
    shift_amount: float,
) -> tuple[dict, dict]:
    device = choose_device()
    model, train_args = load_model(checkpoint_path, device)
    dataset = PushToGoalDataset(data_path, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    schedule = DiffusionSchedule(model.diffusion_steps, device)
    generator = torch.Generator(device="cpu").manual_seed(eval_seed)

    collected: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "start", "goal", "shift_goal", "conflict_goal",
            "base_endpoint", "zero_endpoint", "shuffle_endpoint",
            "shift_endpoint", "conflict_endpoint",
        )
    }

    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device)
        z3d = batch["z3d"].to(device)
        start = batch["start"].to(device)
        goal = batch["goal"].to(device)
        noise = torch.randn(
            len(image), model.horizon, model.action_dim, generator=generator
        ).to(device)

        zero_z3d = torch.zeros_like(z3d)
        # Roll by a nonzero deterministic amount. Across batch boundaries this
        # is still a mismatched condition; the final short batch is handled too.
        offset = max(1, len(z3d) // 2)
        shuffle_z3d = z3d.roll(offset, dims=0)
        shift_goal = shifted_target(goal, shift_amount)
        conflict_goal = opposite_target(start, goal)
        shift_z3d = update_target(z3d, shift_goal)
        conflict_z3d = update_target(z3d, conflict_goal)

        conditions = {
            "base_endpoint": z3d,
            "zero_endpoint": zero_z3d,
            "shuffle_endpoint": shuffle_z3d,
            "shift_endpoint": shift_z3d,
            "conflict_endpoint": conflict_z3d,
        }
        for name, condition_z3d in conditions.items():
            actions = sample_actions(model, schedule, image, condition_z3d, noise, inference_steps)
            collected[name].append(endpoint(actions, start).cpu())
        for name, tensor in {
            "start": start,
            "goal": goal,
            "shift_goal": shift_goal,
            "conflict_goal": conflict_goal,
        }.items():
            collected[name].append(tensor.cpu())

    values = {name: torch.cat(parts) for name, parts in collected.items()}
    base = values["base_endpoint"]
    zero = values["zero_endpoint"]
    shuffled = values["shuffle_endpoint"]
    shifted = values["shift_endpoint"]
    conflicted = values["conflict_endpoint"]
    start, rgb_goal = values["start"], values["goal"]
    shift_goal, conflict_goal = values["shift_goal"], values["conflict_goal"]

    shift_response = shifted - base
    shift_intervention = shift_goal - rgb_goal
    shift_denominator = shift_intervention.square().sum(dim=-1).clamp_min(1e-8)
    shift_projection = (shift_response * shift_intervention).sum(dim=-1) / shift_denominator
    conflict_motion = conflicted - start
    rgb_direction = rgb_goal - start
    z_direction = conflict_goal - start
    conflict_rgb_cos = cosine(conflict_motion, rgb_direction)
    conflict_z_cos = cosine(conflict_motion, z_direction)

    metrics = {
        "baseline": summarize_standard(base, rgb_goal),
        "zero": {
            **summarize_standard(zero, rgb_goal),
            "endpoint_change": float(torch.linalg.vector_norm(zero - base, dim=-1).mean()),
        },
        "shuffle": {
            **summarize_standard(shuffled, rgb_goal),
            "endpoint_change": float(torch.linalg.vector_norm(shuffled - base, dim=-1).mean()),
        },
        "shift": {
            "shifted_goal_final_distance": float(torch.linalg.vector_norm(shifted - shift_goal, dim=-1).mean()),
            "endpoint_change": float(torch.linalg.vector_norm(shift_response, dim=-1).mean()),
            "response_alignment": float(cosine(shift_response, shift_intervention).mean()),
            "projection_sensitivity": float(shift_projection.mean()),
            "correct_direction_rate": float((shift_projection > 0).float().mean()),
        },
        "conflict": {
            "rgb_goal_final_distance": float(torch.linalg.vector_norm(conflicted - rgb_goal, dim=-1).mean()),
            "z3d_goal_final_distance": float(torch.linalg.vector_norm(conflicted - conflict_goal, dim=-1).mean()),
            "rgb_alignment": float(conflict_rgb_cos.mean()),
            "z3d_alignment": float(conflict_z_cos.mean()),
            "z3d_preference_rate": float((conflict_z_cos > conflict_rgb_cos).float().mean()),
        },
    }
    metadata = {
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "examples": len(dataset),
        "device": str(device),
        "inference_steps": inference_steps,
        "eval_seed": eval_seed,
        "train_args": {key: str(value) if isinstance(value, Path) else value for key, value in train_args.items()},
    }
    return metrics, metadata


def flatten_numeric(prefix: str, value: object, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten_numeric(f"{prefix}.{key}" if prefix else key, item, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)


def aggregate(records: list[dict]) -> dict[str, dict[str, float]]:
    flattened: list[dict[str, float]] = []
    for record in records:
        item: dict[str, float] = {}
        flatten_numeric("", record, item)
        flattened.append(item)
    return {
        key: {
            "mean": float(np.mean([record[key] for record in flattened])),
            "std": float(np.std([record[key] for record in flattened], ddof=1)) if len(records) > 1 else 0.0,
        }
        for key in flattened[0]
    }


def markdown_report(results: dict) -> str:
    lines = [
        "# Stage 2: structured-condition dependency diagnostics",
        "",
        "> These tests measure dependence on `Z_3D`. The current planar dataset cannot by itself prove genuine 3D reasoning.",
        "",
        "| Model | Base success | Zero success | Shuffle success | Shift sensitivity | Shift correct | Conflict Z3D preference |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results["models"].items():
        agg = result["aggregate"]
        cell = lambda key: f"{agg[key]['mean']:.3f} ± {agg[key]['std']:.3f}"
        lines.append(
            f"| {name} | {cell('baseline.success')} | {cell('zero.success')} | "
            f"{cell('shuffle.success')} | {cell('shift.projection_sensitivity')} | "
            f"{cell('shift.correct_direction_rate')} | {cell('conflict.z3d_preference_rate')} |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- A useful condition should lose performance under Zero and Shuffle.",
        "- Positive shift sensitivity and a high correct-direction rate mean the action changes coherently with a small 3D intervention.",
        "- Conflict Z3D preference above 0.5 means the counterfactual action is more aligned with the 3D target than the visible RGB target.",
        "- no-3D should remain invariant up to numerical noise; this is the negative control.",
    ])
    return "\n".join(lines) + "\n"


def parse_model_spec(value: str) -> tuple[str, list[Path]]:
    try:
        name, paths = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected NAME=PATH[,PATH,...]") from error
    result = [Path(path) for path in paths.split(",")]
    if not name or not all(str(path) for path in result):
        raise argparse.ArgumentTypeError("Expected NAME=PATH[,PATH,...]")
    return name, result


def validate_comparison(specs: list[tuple[str, list[Path]]]) -> None:
    reference: dict | None = None
    compared_keys = (
        "epochs", "batch_size", "grad_accum", "hidden_dim", "depth", "heads",
        "prediction_type", "train_goal_occlusion_prob", "train_goal_occlusion_style",
    )
    for name, paths in specs:
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            args = checkpoint["args"]
            current = {key: args.get(key) for key in compared_keys}
            if reference is None:
                reference = current
            elif current != reference:
                raise ValueError(f"Incomparable training configuration for {name}/{path}: {current} != {reference}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four counterfactual Z_3D dependency diagnostics.")
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True, help="NAME=CKPT[,CKPT,...]")
    parser.add_argument("--data", type=Path, default=Path("data/ood_partial.npz"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=20260916)
    parser.add_argument("--shift-amount", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=Path("outputs/stage2_diagnostics.json"))
    args = parser.parse_args()

    validate_comparison(args.model)
    results: dict = {
        "data": str(args.data),
        "models": {},
        "note": "Condition-dependence tests on the planar dataset; not proof of genuine 3D reasoning.",
    }
    for model_name, checkpoint_paths in args.model:
        records, metadata = [], []
        for checkpoint_path in checkpoint_paths:
            metrics, run_metadata = diagnose_checkpoint(
                checkpoint_path, args.data, args.batch_size, args.limit,
                args.inference_steps, args.eval_seed, args.shift_amount,
            )
            records.append(metrics)
            metadata.append(run_metadata)
            print(json.dumps({"model": model_name, "checkpoint": str(checkpoint_path), "metrics": metrics}, ensure_ascii=False))
        results["models"][model_name] = {
            "runs": records,
            "metadata": metadata,
            "aggregate": aggregate(records),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    report_path = args.output.with_suffix(".md")
    report_path.write_text(markdown_report(results))
    print(f"wrote {args.output}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
