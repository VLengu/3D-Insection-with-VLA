"""Validate VL/action injection boundaries on the full Tiny counterfactual set."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch

from stage2.data.lerobot_export import iter_scenes
from stage2.data.schema import initial_robot_state, oracle_2d, oracle_3d
from stage2.models.injection_boundary import (
    INJECTION_BOUNDARY_VERSION,
    InjectionBoundaryModel,
)
from stage2.models.structured_projector import (
    StructuredBatch,
    parameter_count,
    tokenize_oracle_2d,
    tokenize_oracle_3d,
)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _mock_tokens(pair_indices: torch.Tensor, token_count: int, hidden_dim: int, offset: int) -> torch.Tensor:
    rows = []
    for pair_index in pair_indices.tolist():
        generator = torch.Generator().manual_seed(20260922 + offset + int(pair_index))
        rows.append(torch.randn(token_count, hidden_dim, generator=generator))
    return torch.stack(rows)


def _zero_values(batch: StructuredBatch) -> StructuredBatch:
    return StructuredBatch(
        values=torch.zeros_like(batch.values),
        feature_mask=batch.feature_mask,
        type_ids=batch.type_ids,
        token_attention_mask=batch.token_attention_mask,
        modality=batch.modality,
    )


def _run_device(
    device: torch.device,
    image_tokens: torch.Tensor,
    language_tokens: torch.Tensor,
    state: torch.Tensor,
    two_d: StructuredBatch,
    three_d: StructuredBatch,
    *,
    hidden_dim: int,
) -> dict[str, Any]:
    torch.manual_seed(20260922)
    model = InjectionBoundaryModel(
        hidden_dim=hidden_dim, heads=4, context_layers=2, action_chunk=4
    ).to(device)
    image_tokens = image_tokens.to(device)
    language_tokens = language_tokens.to(device)
    state = state.to(device)
    batches = {"2d": two_d.to(device), "3d": three_d.to(device)}
    configurations: dict[str, Any] = {}
    for site in ("vl", "action"):
        for modality, structured in batches.items():
            model.zero_grad(set_to_none=True)
            output = model(
                image_tokens,
                language_tokens,
                state,
                structured,
                injection_site=site,
            )
            loss = output.actions.square().mean()
            loss.backward()
            projector_gradient = sum(
                float(parameter.grad.detach().abs().sum().cpu())
                for parameter in model.structured_projector.parameters()
                if parameter.grad is not None
            )
            context_gradient = sum(
                float(parameter.grad.detach().abs().sum().cpu())
                for parameter in model.contextualizer.parameters()
                if parameter.grad is not None
            )
            action_gradient = sum(
                float(parameter.grad.detach().abs().sum().cpu())
                for parameter in model.action_expert.parameters()
                if parameter.grad is not None
            )
            configurations[f"{site}_{modality}"] = {
                "actions_shape": list(output.actions.shape),
                "context_shape": list(output.contextual_features.shape),
                "finite": bool(
                    torch.isfinite(output.actions).all()
                    and torch.isfinite(output.contextual_features).all()
                ),
                "projector_gradient_abs_sum": projector_gradient,
                "contextualizer_gradient_abs_sum": context_gradient,
                "action_expert_gradient_abs_sum": action_gradient,
                "all_required_gradients_nonzero": min(
                    projector_gradient, context_gradient, action_gradient
                )
                > 0.0,
                "trace": {
                    "structured_first_seen_at": output.trace.structured_first_seen_at,
                    "contextualizer_input_tokens": output.trace.contextualizer_input_tokens,
                    "action_memory_tokens": output.trace.action_memory_tokens,
                    "structured_context_span": output.trace.structured_context_span,
                    "structured_action_span": output.trace.structured_action_span,
                },
            }
    return {"device": str(device), "configurations": configurations}


def validate_boundary(config: dict[str, Any], output: Path, hidden_dim: int = 64) -> dict[str, Any]:
    scenes = [scene for _, _, _, scene, _ in iter_scenes(config)]
    scene_count = len(scenes)
    pair_indices = torch.arange(scene_count) // 2
    image_tokens = _mock_tokens(pair_indices, 4, hidden_dim, 0)
    language_tokens = _mock_tokens(pair_indices, 3, hidden_dim, 10000)
    state = torch.tensor([initial_robot_state(scene) for scene in scenes], dtype=torch.float32)
    two_d = tokenize_oracle_2d(
        torch.tensor([oracle_2d(scene) for scene in scenes], dtype=torch.float32)
    )
    three_d = tokenize_oracle_3d(
        torch.tensor([oracle_3d(scene) for scene in scenes], dtype=torch.float32)
    )

    torch.manual_seed(20260922)
    model = InjectionBoundaryModel(
        hidden_dim=hidden_dim, heads=4, context_layers=2, action_chunk=4
    ).eval()
    model_parameters = parameter_count(model)
    outputs: dict[str, Any] = {}
    with torch.no_grad():
        for site in ("vl", "action"):
            for modality, structured in (("2d", two_d), ("3d", three_d)):
                outputs[f"{site}_{modality}"] = model(
                    image_tokens,
                    language_tokens,
                    state,
                    structured,
                    injection_site=site,
                )

    action_context_exact_pairs = 0
    action_context_max_pair_error = 0.0
    vl_context_changed_pairs = 0
    action_output_changed_pairs = 0
    vl_output_changed_pairs = 0
    two_d_equal_pairs = {"vl": 0, "action": 0}
    for index in range(0, scene_count, 2):
        action_a = outputs["action_3d"].contextual_features[index]
        action_b = outputs["action_3d"].contextual_features[index + 1]
        vl_a = outputs["vl_3d"].contextual_features[index]
        vl_b = outputs["vl_3d"].contextual_features[index + 1]
        action_context_error = float((action_a - action_b).abs().max())
        action_context_max_pair_error = max(
            action_context_max_pair_error, action_context_error
        )
        action_context_exact_pairs += int(action_context_error <= 1e-6)
        vl_context_changed_pairs += int(not torch.allclose(vl_a, vl_b, atol=1e-7, rtol=0.0))
        action_output_changed_pairs += int(
            not torch.allclose(
                outputs["action_3d"].actions[index],
                outputs["action_3d"].actions[index + 1],
                atol=1e-7,
                rtol=0.0,
            )
        )
        vl_output_changed_pairs += int(
            not torch.allclose(
                outputs["vl_3d"].actions[index],
                outputs["vl_3d"].actions[index + 1],
                atol=1e-7,
                rtol=0.0,
            )
        )
        for site in ("vl", "action"):
            two_d_equal_pairs[site] += int(
                torch.allclose(
                    outputs[f"{site}_2d"].actions[index],
                    outputs[f"{site}_2d"].actions[index + 1],
                    atol=1e-6,
                    rtol=0.0,
                )
            )

    counterfactual_checks: dict[str, Any] = {}
    with torch.no_grad():
        for site in ("vl", "action"):
            for modality, structured in (("2d", two_d), ("3d", three_d)):
                baseline = outputs[f"{site}_{modality}"]
                zeroed = model(
                    image_tokens,
                    language_tokens,
                    state,
                    _zero_values(structured),
                    injection_site=site,
                )
                shuffled = model(
                    image_tokens,
                    language_tokens,
                    state,
                    StructuredBatch(
                        values=structured.values.roll(2, dims=0),
                        feature_mask=structured.feature_mask.roll(2, dims=0),
                        type_ids=structured.type_ids,
                        token_attention_mask=structured.token_attention_mask,
                        modality=structured.modality,
                    ),
                    injection_site=site,
                )
                counterfactual_checks[f"{site}_{modality}"] = {
                    "zero_changes_actions": not torch.allclose(
                        baseline.actions, zeroed.actions, atol=1e-7, rtol=0.0
                    ),
                    "shuffle_changes_actions": not torch.allclose(
                        baseline.actions, shuffled.actions, atol=1e-7, rtol=0.0
                    ),
                    "zero_preserves_action_context": (
                        torch.equal(baseline.contextual_features, zeroed.contextual_features)
                        if site == "action"
                        else None
                    ),
                    "shuffle_preserves_action_context": (
                        torch.equal(baseline.contextual_features, shuffled.contextual_features)
                        if site == "action"
                        else None
                    ),
                }

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    checkpoint_sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()
    buffer.seek(0)
    restored = InjectionBoundaryModel(
        hidden_dim=hidden_dim, heads=4, context_layers=2, action_chunk=4
    ).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        restored_actions = restored(
            image_tokens, language_tokens, state, three_d, injection_site="vl"
        ).actions
    checkpoint_exact = bool(torch.equal(outputs["vl_3d"].actions, restored_actions))

    device_checks = [
        _run_device(
            torch.device("cpu"),
            image_tokens,
            language_tokens,
            state,
            two_d,
            three_d,
            hidden_dim=hidden_dim,
        )
    ]
    mps_available = bool(torch.backends.mps.is_available())
    if mps_available:
        device_checks.append(
            _run_device(
                torch.device("mps"),
                image_tokens[:8],
                language_tokens[:8],
                state[:8],
                StructuredBatch(
                    two_d.values[:8],
                    two_d.feature_mask[:8],
                    two_d.type_ids[:8],
                    two_d.token_attention_mask[:8],
                    two_d.modality,
                ),
                StructuredBatch(
                    three_d.values[:8],
                    three_d.feature_mask[:8],
                    three_d.type_ids[:8],
                    three_d.token_attention_mask[:8],
                    three_d.modality,
                ),
                hidden_dim=hidden_dim,
            )
        )

    traces = {
        name: {
            "structured_first_seen_at": item.trace.structured_first_seen_at,
            "contextualizer_input_tokens": item.trace.contextualizer_input_tokens,
            "contextual_features_tokens": item.trace.contextual_features_tokens,
            "action_memory_tokens": item.trace.action_memory_tokens,
            "structured_context_span": item.trace.structured_context_span,
            "structured_action_span": item.trace.structured_action_span,
            "contextual_features_sha256": _sha256_tensor(item.contextual_features),
            "actions_sha256": _sha256_tensor(item.actions),
        }
        for name, item in outputs.items()
    }
    all_device_configs = [
        configuration
        for device in device_checks
        for configuration in device["configurations"].values()
    ]
    checks = {
        "all_88_scenes_loaded": scene_count == 88,
        "action_context_equal_within_1e-6_for_all_3d_pairs": action_context_exact_pairs
        == 44,
        "action_context_hash_equal_across_modalities": _sha256_tensor(
            outputs["action_2d"].contextual_features
        )
        == _sha256_tensor(outputs["action_3d"].contextual_features),
        "vl_context_changes_for_all_3d_pairs": vl_context_changed_pairs == 44,
        "action_outputs_change_for_all_3d_pairs": action_output_changed_pairs == 44,
        "vl_outputs_change_for_all_3d_pairs": vl_output_changed_pairs == 44,
        "paired_2d_outputs_equal": two_d_equal_pairs == {"vl": 44, "action": 44},
        "all_counterfactual_actions_change": all(
            item["zero_changes_actions"] and item["shuffle_changes_actions"]
            for item in counterfactual_checks.values()
        ),
        "action_context_survives_counterfactuals": all(
            item["zero_preserves_action_context"]
            and item["shuffle_preserves_action_context"]
            for name, item in counterfactual_checks.items()
            if name.startswith("action_")
        ),
        "all_configurations_have_equal_parameter_budget": model_parameters[0]
        == model_parameters[1],
        "all_device_configurations_finite_with_gradients": all(
            item["finite"] and item["all_required_gradients_nonzero"]
            for item in all_device_configs
        ),
        "checkpoint_roundtrip_exact": checkpoint_exact,
        "trace_boundaries_correct": all(
            (
                item["structured_first_seen_at"] == "contextualizer_input"
                and item["contextualizer_input_tokens"] == 13
            )
            if name.startswith("vl_")
            else (
                item["structured_first_seen_at"] == "action_expert_input"
                and item["contextualizer_input_tokens"] == 7
            )
            for name, item in traces.items()
        ),
        "action_memory_equal_length": all(
            item["action_memory_tokens"] == 14 for item in traces.values()
        ),
    }
    summary = {
        "boundary_version": INJECTION_BOUNDARY_VERSION,
        "scenes": scene_count,
        "pairs": scene_count // 2,
        "hidden_dim": hidden_dim,
        "model_parameters": {
            "total": model_parameters[0],
            "trainable": model_parameters[1],
            "per_configuration_difference": 0,
        },
        "pair_checks": {
            "action_context_equal_within_1e-6_3d_pairs": action_context_exact_pairs,
            "action_context_max_pair_error": action_context_max_pair_error,
            "vl_context_changed_3d_pairs": vl_context_changed_pairs,
            "action_output_changed_3d_pairs": action_output_changed_pairs,
            "vl_output_changed_3d_pairs": vl_output_changed_pairs,
            "2d_output_equal_pairs": two_d_equal_pairs,
        },
        "counterfactual_checks": counterfactual_checks,
        "traces": traces,
        "mps_available": mps_available,
        "device_checks": device_checks,
        "checkpoint_sha256": checkpoint_sha256,
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
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()
    summary = validate_boundary(
        json.loads(args.config.read_text()), args.output, hidden_dim=args.hidden_dim
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
