"""Replay the complete Tiny geometry set in MuJoCo and save evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from stage2.data.generate import split_seed
from stage2.envs.dual_clearance.expert import ExpertFailure, rollout_expert
from stage2.envs.dual_clearance.geometry import Route, generate_pair
from stage2.envs.dual_clearance.mujoco_env import DualClearanceMujocoEnv


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _episode_id(split: str, pair_id: str, member: str) -> str:
    return f"{split}:{pair_id}:{member}"


def run_validation(config: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    preview_dir = output / "previews"
    preview_dir.mkdir(exist_ok=True)
    rollouts: list[dict[str, Any]] = []
    pair_checks: list[dict[str, Any]] = []
    correct_successes = 0
    wrong_route_failures = 0
    collision_failures = 0
    height_failures = 0

    for split, count_value in config["splits"].items():
        count = int(count_value)
        base_seed = split_seed(int(config["master_seed"]), split)
        for index in range(count):
            pair = generate_pair(index, split, base_seed + index)
            policy_images: list[np.ndarray] = []
            physical_images: list[np.ndarray] = []
            initial_states: list[list[float]] = []
            for scene in pair.members:
                env = DualClearanceMujocoEnv(image_size=256)
                observation = env.reset(scene)
                policy_images.append(env.render())
                physical_images.append(env.render_physical())
                initial_states.append(observation["observation_state"])
                result = rollout_expert(env)
                correct_successes += int(result.success)
                action_rows = [list(action) for action in result.actions]
                rollouts.append(
                    {
                        "episode_id": _episode_id(split, pair.pair_id, scene.member),
                        "split": split,
                        "pair_id": pair.pair_id,
                        "pair_member": scene.member,
                        "route": scene.route_label.value,
                        "steps": result.steps,
                        "success": result.success,
                        "initial_rgb_sha256": hashlib.sha256(
                            policy_images[-1].tobytes()
                        ).hexdigest(),
                        "actions_sha256": _json_hash(action_rows),
                        "actions": action_rows,
                        "final_info": result.final_info,
                    }
                )
                env.close()

                wrong_route = (
                    Route.SIDE if scene.route_label is Route.OVER else Route.OVER
                )
                wrong_env = DualClearanceMujocoEnv(image_size=16)
                wrong_env.reset(scene)
                try:
                    rollout_expert(wrong_env, forced_route=wrong_route)
                except ExpertFailure:
                    pass
                wrong_info = wrong_env.info()
                physically_failed = bool(
                    wrong_info["collision"]
                    or wrong_info["height_violation"]
                    or wrong_info["bounds_violation"]
                    or wrong_info["drop"]
                )
                wrong_route_failures += int(physically_failed)
                collision_failures += int(bool(wrong_info["collision"]))
                height_failures += int(bool(wrong_info["height_violation"]))
                wrong_env.close()

            rgb_equal = bool(np.array_equal(policy_images[0], policy_images[1]))
            state_equal = bool(np.allclose(initial_states[0], initial_states[1], atol=0.0))
            pair_checks.append(
                {
                    "pair_id": pair.pair_id,
                    "split": split,
                    "policy_rgb_byte_identical": rgb_equal,
                    "initial_robot_state_identical": state_equal,
                    "physical_rgb_byte_identical": bool(
                        np.array_equal(physical_images[0], physical_images[1])
                    ),
                }
            )
            if len(pair_checks) == 1:
                Image.fromarray(policy_images[0]).save(preview_dir / "policy_rgb.png")
                Image.fromarray(physical_images[0]).save(preview_dir / "physical_over.png")
                Image.fromarray(physical_images[1]).save(preview_dir / "physical_side.png")

    episode_count = len(rollouts)
    pair_count = len(pair_checks)
    summary = {
        "schema_version": "dual-clearance-mujoco-m2-v1",
        "master_seed": int(config["master_seed"]),
        "pairs": pair_count,
        "episodes": episode_count,
        "correct_expert_successes": correct_successes,
        "correct_expert_success_rate": correct_successes / episode_count,
        "wrong_route_physical_failures": wrong_route_failures,
        "wrong_route_rejection_rate": wrong_route_failures / episode_count,
        "wrong_route_collision_failures": collision_failures,
        "wrong_route_height_failures": height_failures,
        "policy_rgb_byte_identical_pairs": sum(
            int(item["policy_rgb_byte_identical"]) for item in pair_checks
        ),
        "initial_robot_state_identical_pairs": sum(
            int(item["initial_robot_state_identical"]) for item in pair_checks
        ),
        "physical_rgb_byte_identical_pairs": sum(
            int(item["physical_rgb_byte_identical"]) for item in pair_checks
        ),
        "min_episode_steps": min(item["steps"] for item in rollouts),
        "max_episode_steps": max(item["steps"] for item in rollouts),
        "pass": bool(
            correct_successes == episode_count
            and wrong_route_failures == episode_count
            and all(item["policy_rgb_byte_identical"] for item in pair_checks)
            and all(item["initial_robot_state_identical"] for item in pair_checks)
        ),
    }
    (output / "expert_rollouts.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rollouts)
    )
    (output / "pair_checks.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in pair_checks)
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest_paths = [
        output / "expert_rollouts.jsonl",
        output / "pair_checks.jsonl",
        output / "summary.json",
        preview_dir / "policy_rgb.png",
        preview_dir / "physical_over.png",
        preview_dir / "physical_side.png",
    ]
    (output / "manifest.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output)}\n"
            for path in manifest_paths
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_validation(json.loads(args.config.read_text()), args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
