"""Validate the structured projector on every Tiny scene and available devices."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch

from stage2.data.lerobot_export import iter_scenes
from stage2.data.schema import FORBIDDEN_MODEL_KEYS, ORACLE_2D_NAMES, ORACLE_3D_NAMES, oracle_2d, oracle_3d
from stage2.models.structured_projector import (
    PROJECTOR_VERSION,
    RAW_WIDTH,
    TOKEN_COUNT,
    TOKEN_NAMES,
    StructuredProjector,
    parameter_count,
    tokenize_oracle_2d,
    tokenize_oracle_3d,
)


def _device_smoke(
    device: torch.device,
    two_d: torch.Tensor,
    three_d: torch.Tensor,
    hidden_dim: int,
) -> dict[str, Any]:
    torch.manual_seed(20260922)
    model = StructuredProjector(hidden_dim=hidden_dim).to(device)
    batch_2d = tokenize_oracle_2d(two_d.to(device))
    batch_3d = tokenize_oracle_3d(three_d.to(device))
    output_2d, attention_2d = model(batch_2d)
    output_3d, attention_3d = model(batch_3d)
    loss = output_2d.square().mean() + output_3d.square().mean()
    loss.backward()
    gradient_sum = sum(
        float(parameter.grad.detach().abs().sum().cpu())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return {
        "device": str(device),
        "output_shape_2d": list(output_2d.shape),
        "output_shape_3d": list(output_3d.shape),
        "attention_equal": bool(torch.equal(attention_2d, attention_3d)),
        "finite": bool(torch.isfinite(output_2d).all() and torch.isfinite(output_3d).all()),
        "loss": float(loss.detach().cpu()),
        "gradient_abs_sum": gradient_sum,
        "gradient_nonzero": gradient_sum > 0.0,
    }


def validate_projector(config: dict[str, Any], output: Path, hidden_dim: int = 128) -> dict[str, Any]:
    scenes = [scene for _, _, _, scene, _ in iter_scenes(config)]
    two_d = torch.tensor([oracle_2d(scene) for scene in scenes], dtype=torch.float32)
    three_d = torch.tensor([oracle_3d(scene) for scene in scenes], dtype=torch.float32)
    tokenized_2d = tokenize_oracle_2d(two_d)
    tokenized_3d = tokenize_oracle_3d(three_d)

    torch.manual_seed(20260922)
    projector_2d = StructuredProjector(hidden_dim=hidden_dim).eval()
    torch.manual_seed(20260922)
    projector_3d = StructuredProjector(hidden_dim=hidden_dim).eval()
    counts_2d = parameter_count(projector_2d)
    counts_3d = parameter_count(projector_3d)
    with torch.no_grad():
        tokens_2d, attention_2d = projector_2d(tokenized_2d)
        tokens_3d, attention_3d = projector_3d(tokenized_3d)

        altered_values = tokenized_2d.values.clone()
        altered_values[~tokenized_2d.feature_mask] = 123456.0
        altered_batch = type(tokenized_2d)(
            altered_values,
            tokenized_2d.feature_mask,
            tokenized_2d.type_ids,
            tokenized_2d.token_attention_mask,
            tokenized_2d.modality,
        )
        masked_slot_invariant = bool(torch.equal(tokens_2d, projector_2d(altered_batch)[0]))

    pair_2d_identical = 0
    pair_3d_different = 0
    pair_2d_max_error = 0.0
    pair_3d_min_difference = float("inf")
    for index in range(0, len(scenes), 2):
        error_2d = float((tokens_2d[index] - tokens_2d[index + 1]).abs().max())
        difference_3d = float((tokens_3d[index] - tokens_3d[index + 1]).abs().max())
        pair_2d_max_error = max(pair_2d_max_error, error_2d)
        pair_3d_min_difference = min(pair_3d_min_difference, difference_3d)
        pair_2d_identical += int(error_2d <= 1e-6)
        pair_3d_different += int(difference_3d > 1e-6)

    buffer = io.BytesIO()
    torch.save(projector_2d.state_dict(), buffer)
    checkpoint_sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()
    buffer.seek(0)
    restored = StructuredProjector(hidden_dim=hidden_dim).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        checkpoint_exact = bool(torch.equal(tokens_2d, restored(tokenized_2d)[0]))

    device_checks = [_device_smoke(torch.device("cpu"), two_d, three_d, hidden_dim)]
    mps_available = bool(torch.backends.mps.is_available())
    if mps_available:
        device_checks.append(
            _device_smoke(torch.device("mps"), two_d[:8], three_d[:8], hidden_dim)
        )

    schema_names = set(ORACLE_2D_NAMES) | set(ORACLE_3D_NAMES)
    checks = {
        "all_88_scenes_loaded": len(scenes) == 88,
        "identical_output_shapes": tokens_2d.shape == tokens_3d.shape,
        "expected_output_shape": tuple(tokens_2d.shape) == (88, TOKEN_COUNT, hidden_dim),
        "identical_attention_masks": bool(torch.equal(attention_2d, attention_3d)),
        "identical_parameter_counts": counts_2d == counts_3d,
        "all_parameters_trainable": counts_2d[0] == counts_2d[1],
        "masked_2d_slots_invariant": masked_slot_invariant,
        "paired_2d_tokens_identical": pair_2d_identical == 44,
        "paired_3d_tokens_different": pair_3d_different == 44,
        "checkpoint_roundtrip_exact": checkpoint_exact,
        "forbidden_fields_absent": not bool(schema_names & FORBIDDEN_MODEL_KEYS),
        "all_device_checks_pass": all(
            item["finite"] and item["gradient_nonzero"] and item["attention_equal"]
            for item in device_checks
        ),
    }
    summary = {
        "projector_version": PROJECTOR_VERSION,
        "hidden_dim": hidden_dim,
        "token_names": list(TOKEN_NAMES),
        "token_count": TOKEN_COUNT,
        "raw_width": RAW_WIDTH,
        "scenes": len(scenes),
        "pairs": len(scenes) // 2,
        "valid_feature_slots_per_scene": {
            "2d": int(tokenized_2d.feature_mask[0].sum()),
            "3d": int(tokenized_3d.feature_mask[0].sum()),
        },
        "parameter_counts": {
            "2d_total": counts_2d[0],
            "2d_trainable": counts_2d[1],
            "3d_total": counts_3d[0],
            "3d_trainable": counts_3d[1],
            "difference": counts_3d[0] - counts_2d[0],
        },
        "paired_token_checks": {
            "2d_equal_within_1e-6_pairs": pair_2d_identical,
            "2d_max_absolute_error": pair_2d_max_error,
            "3d_different_pairs": pair_3d_different,
            "3d_min_max_absolute_difference": pair_3d_min_difference,
        },
        "checkpoint_sha256": checkpoint_sha256,
        "mps_available": mps_available,
        "device_checks": device_checks,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()
    summary = validate_projector(
        json.loads(args.config.read_text()), args.output, hidden_dim=args.hidden_dim
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
