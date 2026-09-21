"""Evaluate comparable Height World checkpoints with one fixed protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from scripts.train_height_world import evaluate
from uwm3d.diffusion import DiffusionSchedule
from uwm3d.height_world import HeightWorldDataset
from uwm3d.model import ActionDiffusionPolicy
from uwm3d.train_utils import choose_device


def parse_spec(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected NAME=CHECKPOINT") from error
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_spec, required=True)
    parser.add_argument("--data", type=Path, default=Path("data_height"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/height_core_e8.json"))
    args = parser.parse_args()
    if args.batch_size % 2:
        raise ValueError("batch-size must be even")
    device = choose_device()
    splits = ("val", "height_ood", "layout_ood", "camera_ood", "paired_test")
    results = {}
    reference_training = None
    compared_keys = ("epochs", "batch_size", "hidden_dim", "depth", "heads", "seed", "trajectory_loss_weight")
    for model_name, path in args.model:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        train_args = checkpoint["args"]
        current_training = {key: train_args.get(key) for key in compared_keys}
        if reference_training is None:
            reference_training = current_training
        elif current_training != reference_training:
            raise ValueError(f"Training configuration mismatch for {model_name}: {current_training} != {reference_training}")
        config = checkpoint["config"]
        model = ActionDiffusionPolicy(config, correction_mode=train_args["mode"])
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        schedule = DiffusionSchedule(config.diffusion_steps, device)
        model_metrics = {}
        for index, split in enumerate(splits):
            loader = DataLoader(HeightWorldDataset(args.data / f"{split}.npz"), batch_size=args.batch_size, shuffle=False)
            model_metrics[split] = evaluate(
                model, loader, schedule, train_args["condition"], device,
                args.inference_steps, train_args["seed"] + 10_000 * (index + 1),
            )
        results[model_name] = {
            "checkpoint": str(path),
            "condition": train_args["condition"],
            "checkpoint_epoch": checkpoint["metrics"]["epoch"],
            "metrics": model_metrics,
        }
        print(json.dumps({model_name: results[model_name]}, ensure_ascii=False))
    payload = {"protocol": {"paired_noise": "identical", "route_mode": "XY footprint", "training": reference_training}, "models": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    lines = [
        "# Height World core screening",
        "",
        "All counterfactual pair members share identical initial diffusion noise.",
        "",
        "| Model | Paired success | Collision | Mode accuracy | Both modes correct | Decisions differ | Height-OOD success |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        paired = result["metrics"]["paired_test"]
        height = result["metrics"]["height_ood"]
        lines.append(
            f"| {name} | {paired['success']:.3f} | {paired['collision_rate']:.3f} | "
            f"{paired['path_mode_accuracy']:.3f} | {paired['paired_both_modes_correct']:.3f} | "
            f"{paired['paired_decisions_differ']:.3f} | {height['success']:.3f} |"
        )
    report = args.output.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
