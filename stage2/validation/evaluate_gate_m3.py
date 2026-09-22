"""Run the full Gate-M3 evaluator validation over the Tiny scene set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from stage2.data.lerobot_export import iter_scenes
from stage2.evaluation.evaluator import (
    EVALUATOR_VERSION,
    METRIC_NAMES,
    EpisodeEvaluation,
    evaluate_actions,
    summarize_evaluations,
)
from stage2.envs.dual_clearance.expert import ExpertFailure, rollout_expert
from stage2.envs.dual_clearance.geometry import Route, SceneGeometry
from stage2.envs.dual_clearance.mujoco_env import DualClearanceMujocoEnv


def _actions_for_route(scene: SceneGeometry, route: Route) -> tuple[tuple[float, ...], ...]:
    env = DualClearanceMujocoEnv(image_size=16)
    env.reset(scene)
    try:
        actions = rollout_expert(env, forced_route=route).actions
    except ExpertFailure as failure:
        actions = failure.actions
    finally:
        env.close()
    if not actions:
        raise RuntimeError(f"No actions captured for {scene.pair_id}/{scene.member}/{route.value}")
    return actions


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: list[EpisodeEvaluation]) -> None:
    path.write_text(
        "".join(json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in records)
    )


def _markdown_summary(summaries: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Gate M3 evaluator results",
        "",
        f"Evaluator: `{EVALUATOR_VERSION}`",
        "",
        "| Policy | Episode success | Route accuracy | Pair success | Collision | Drop | Height violation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        values = summary["overall"]
        lines.append(
            f"| {name} | {values['episode_success_rate']:.3f} | "
            f"{values['route_accuracy']:.3f} | {values['pair_success_rate']:.3f} | "
            f"{values['collision_rate']:.3f} | {values['drop_rate']:.3f} | "
            f"{values['height_violation_rate']:.3f} |"
        )
    lines.extend(["", "## Metric names", "", "| Key | 中文 |", "|---|---|"])
    lines.extend(f"| `{key}` | {value} |" for key, value in METRIC_NAMES.items())
    return "\n".join(lines) + "\n"


def run_gate_m3(config: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    correct: list[EpisodeEvaluation] = []
    wrong: list[EpisodeEvaluation] = []
    fixed_over: list[EpisodeEvaluation] = []
    fixed_side: list[EpisodeEvaluation] = []

    for _, _, _, scene, _ in iter_scenes(config):
        correct_actions = _actions_for_route(scene, scene.route_label)
        other_route = Route.SIDE if scene.route_label is Route.OVER else Route.OVER
        wrong_actions = _actions_for_route(scene, other_route)
        over_actions = correct_actions if scene.route_label is Route.OVER else wrong_actions
        side_actions = correct_actions if scene.route_label is Route.SIDE else wrong_actions

        correct.append(evaluate_actions(scene, correct_actions))
        wrong.append(evaluate_actions(scene, wrong_actions))
        fixed_over.append(evaluate_actions(scene, over_actions))
        fixed_side.append(evaluate_actions(scene, side_actions))

    # Explicit failure probes use real expert prefixes, not fabricated result rows.
    probe_scene = next(iter(iter_scenes(config)))[3]
    probe_actions = list(_actions_for_route(probe_scene, probe_scene.route_label))
    close_index = next(index for index, action in enumerate(probe_actions) if action[6] < -0.5)
    release = np.zeros(7, dtype=np.float64)
    release[6] = 1.0
    drop_probe = evaluate_actions(
        probe_scene, [*probe_actions[: close_index + 1], release]
    )
    incomplete_probe = evaluate_actions(probe_scene, [])

    summaries = {
        "correct_expert": summarize_evaluations(correct),
        "wrong_route": summarize_evaluations(wrong),
        "fixed_over": summarize_evaluations(fixed_over),
        "fixed_side": summarize_evaluations(fixed_side),
    }
    episode_count = len(correct)
    gate_checks = {
        "correct_expert_all_success": all(item.success for item in correct),
        "wrong_routes_all_rejected": all(not item.success for item in wrong),
        "wrong_routes_all_physical_failure": all(
            item.collision or item.height_violation or item.bounds_violation or item.drop
            for item in wrong
        ),
        "fixed_over_episode_success_is_half": summaries["fixed_over"]["overall"][
            "episode_success_rate"
        ]
        == 0.5,
        "fixed_side_episode_success_is_half": summaries["fixed_side"]["overall"][
            "episode_success_rate"
        ]
        == 0.5,
        "fixed_over_pair_success_is_zero": summaries["fixed_over"]["overall"][
            "pair_success_rate"
        ]
        == 0.0,
        "fixed_side_pair_success_is_zero": summaries["fixed_side"]["overall"][
            "pair_success_rate"
        ]
        == 0.0,
        "drop_probe_detected": drop_probe.drop and not drop_probe.success,
        "incomplete_probe_detected": "incomplete" in incomplete_probe.failure_reasons,
        "metric_names_bilingual": set(METRIC_NAMES)
        >= {
            "episode_success_rate",
            "route_accuracy",
            "pair_success_rate",
            "collision_rate",
            "drop_rate",
            "mean_final_placement_error_m",
            "mean_executed_steps",
            "mean_completion_steps",
        },
    }
    summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "episodes_per_policy": episode_count,
        "pairs_per_policy": episode_count // 2,
        "summaries": summaries,
        "failure_probes": {
            "early_release": drop_probe.to_dict(),
            "empty_actions": incomplete_probe.to_dict(),
        },
        "gate_checks": gate_checks,
        "status": "PASS" if all(gate_checks.values()) else "FAIL",
    }

    _write_jsonl(output / "correct_expert.jsonl", correct)
    _write_jsonl(output / "wrong_route.jsonl", wrong)
    _write_jsonl(output / "fixed_over.jsonl", fixed_over)
    _write_jsonl(output / "fixed_side.jsonl", fixed_side)
    _write_json(output / "summary.json", summary)
    (output / "metrics.md").write_text(_markdown_summary(summaries))

    artifact_paths = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "manifest.sha256"
    )
    (output / "manifest.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in artifact_paths
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_gate_m3(json.loads(args.config.read_text()), args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
