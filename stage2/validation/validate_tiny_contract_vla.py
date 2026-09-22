"""Validate the real-data TinyContractVLA training and checkpoint pipeline."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate

from stage2.data.action_chunks import LeRobotActionChunkDataset, PairedBatchSampler
from stage2.models.structured_projector import parameter_count
from stage2.models.tiny_contract_vla import (
    TINY_CONTRACT_VLA_VERSION,
    TaskTokenizer,
    TinyContractVLA,
    masked_action_mse,
)


CONFIGURATIONS = (
    ("VL-2D", "vl", "2d"),
    ("VL-3D", "vl", "3d"),
    ("A-2D", "action", "2d"),
    ("A-3D", "action", "3d"),
)


def _model_inputs(batch: dict[str, Any], device: torch.device, modality: str) -> dict[str, Any]:
    return {
        "images": batch["image"].to(device),
        "tasks": list(batch["task"]),
        "robot_state": batch["state"].to(device),
        "oracle": batch[f"oracle_{modality}"].to(device),
        "modality": modality,
    }


def _gradient_sum(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum().cpu())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def _train_configuration(
    name: str,
    site: str,
    modality: str,
    batch: dict[str, Any],
    *,
    device: torch.device,
    hidden_dim: int,
    steps: int,
) -> dict[str, Any]:
    torch.manual_seed(20260922)
    model = TinyContractVLA(
        tokenizer=TaskTokenizer(max_length=16),
        hidden_dim=hidden_dim,
        heads=4,
        context_layers=1,
        action_chunk=8,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    target = batch["action_chunk"].to(device)
    mask = batch["action_mask"].to(device)
    inputs = _model_inputs(batch, device, modality)
    losses: list[float] = []
    first_gradients: dict[str, float] = {}

    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs, injection_site=site)
        loss = masked_action_mse(output.actions, target, mask)
        loss.backward()
        if step == 0:
            first_gradients = {
                "image_encoder": _gradient_sum(model.image_encoder),
                "language_embedding": _gradient_sum(model.language_embedding),
                "structured_projector": _gradient_sum(
                    model.boundary_model.structured_projector
                ),
                "contextualizer": _gradient_sum(model.boundary_model.contextualizer),
                "action_expert": _gradient_sum(model.boundary_model.action_expert),
            }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        final_output = model(**inputs, injection_site=site)
        evaluated_loss = float(
            masked_action_mse(final_output.actions, target, mask).detach().cpu()
        )

    checkpoint = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "configuration": {"name": name, "site": site, "modality": modality},
            "steps": steps,
        },
        checkpoint,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.getvalue()).hexdigest()
    checkpoint.seek(0)
    restored = TinyContractVLA(
        tokenizer=TaskTokenizer(max_length=16),
        hidden_dim=hidden_dim,
        heads=4,
        context_layers=1,
        action_chunk=8,
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    restored.load_state_dict(payload["model"])
    restored.eval()
    with torch.no_grad():
        restored_actions = restored(**inputs, injection_site=site).actions
    checkpoint_max_error = float(
        (final_output.actions - restored_actions).abs().max().detach().cpu()
    )

    counts = parameter_count(model)
    return {
        "configuration": name,
        "injection_site": site,
        "modality": modality,
        "device": str(device),
        "steps": steps,
        "initial_loss": losses[0],
        "last_train_loss": losses[-1],
        "evaluated_loss": evaluated_loss,
        "loss_ratio": evaluated_loss / losses[0],
        "loss_curve_every_10_steps": losses[::10],
        "first_step_gradient_abs_sum": first_gradients,
        "all_required_gradients_nonzero": all(
            value > 0.0 for value in first_gradients.values()
        ),
        "parameters_total": counts[0],
        "parameters_trainable": counts[1],
        "action_shape": list(final_output.actions.shape),
        "finite": bool(torch.isfinite(final_output.actions).all()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_max_absolute_error": checkpoint_max_error,
        "checkpoint_roundtrip_within_1e-6": checkpoint_max_error <= 1e-6,
        "trace": {
            "structured_first_seen_at": final_output.boundary.trace.structured_first_seen_at,
            "contextualizer_input_tokens": final_output.boundary.trace.contextualizer_input_tokens,
            "action_memory_tokens": final_output.boundary.trace.action_memory_tokens,
        },
    }


def validate_tiny_vla(
    dataset_root: Path,
    output: Path,
    *,
    hidden_dim: int = 64,
    steps: int = 400,
) -> dict[str, Any]:
    dataset = LeRobotActionChunkDataset(
        dataset_root, split="train", action_chunk=8, frame_stride=1
    )
    selected_indices = dataset.paired_subset_indices(max_pairs=4, frames_per_member=2)
    sampler = PairedBatchSampler(selected_indices, batch_size=16)
    batches = list(sampler)
    if len(batches) != 1:
        raise AssertionError("Overfit validation expects exactly one paired batch")
    samples = [dataset[index] for index in batches[0]]
    batch = default_collate(samples)

    pair_members: dict[str, set[str]] = {}
    for sample in samples:
        pair_members.setdefault(sample["pair_id"], set()).add(sample["pair_member"])
    terminal_reference = next(
        reference
        for reference in reversed(dataset.references)
        if reference.frame_index == reference.episode_length - 1
    )
    terminal_sample = dataset[terminal_reference.sample_index]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    runs = [
        _train_configuration(
            name,
            site,
            modality,
            batch,
            device=device,
            hidden_dim=hidden_dim,
            steps=steps,
        )
        for name, site, modality in CONFIGURATIONS
    ]
    parameter_counts = {run["parameters_total"] for run in runs}
    checks = {
        "train_split_only": set(batch["split"]) == {"train"},
        "paired_batch_has_16_samples": len(samples) == 16,
        "paired_batch_has_four_complete_pairs": len(pair_members) == 4
        and all(members == {"A", "B"} for members in pair_members.values()),
        "real_rgb_shape_correct": tuple(batch["image"].shape) == (16, 3, 256, 256),
        "action_chunk_shape_correct": tuple(batch["action_chunk"].shape) == (16, 8, 7),
        "selected_training_chunks_are_unpadded": bool(batch["action_mask"].all()),
        "terminal_padding_mask_correct": int(terminal_sample["action_mask"].sum()) == 1,
        "all_four_configurations_run": len(runs) == 4,
        "equal_parameter_budgets": len(parameter_counts) == 1,
        "all_required_gradients_nonzero": all(
            run["all_required_gradients_nonzero"] for run in runs
        ),
        "all_outputs_finite": all(run["finite"] for run in runs),
        "all_losses_drop_by_at_least_80_percent": all(
            run["loss_ratio"] <= 0.2 for run in runs
        ),
        "all_checkpoint_roundtrips_pass": all(
            run["checkpoint_roundtrip_within_1e-6"] for run in runs
        ),
        "all_traces_match_configuration": all(
            (
                run["trace"]["structured_first_seen_at"] == "contextualizer_input"
                if run["injection_site"] == "vl"
                else run["trace"]["structured_first_seen_at"]
                == "action_expert_input"
            )
            for run in runs
        ),
    }
    summary = {
        "model_version": TINY_CONTRACT_VLA_VERSION,
        "dataset_root": str(dataset_root),
        "dataset_split": "train",
        "dataset_samples": len(dataset),
        "overfit_samples": len(samples),
        "overfit_pairs": len(pair_members),
        "action_chunk": 8,
        "hidden_dim": hidden_dim,
        "training_steps": steps,
        "device": str(device),
        "tokenizer_vocab_size": TaskTokenizer().vocab_size,
        "runs": runs,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    summary = validate_tiny_vla(
        args.dataset, args.output, hidden_dim=args.hidden_dim, steps=args.steps
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
