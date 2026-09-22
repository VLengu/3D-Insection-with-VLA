"""Uniform episode and counterfactual-pair evaluation for Dual-Clearance."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from stage2.envs.dual_clearance.geometry import SceneGeometry
from stage2.envs.dual_clearance.mujoco_env import DualClearanceMujocoEnv


EVALUATOR_VERSION = "dual-clearance-evaluator-v1"

METRIC_NAMES = {
    "episode_success_rate": "单条轨迹成功率",
    "route_accuracy": "路线选择准确率",
    "pair_success_rate": "完整配对成功率",
    "collision_rate": "碰撞率",
    "drop_rate": "掉落率",
    "bounds_violation_rate": "越界率",
    "height_violation_rate": "超高率",
    "mean_final_placement_error_m": "平均最终放置误差（米）",
    "mean_executed_steps": "平均执行步数",
    "mean_completion_steps": "成功轨迹平均完成步数",
}


@dataclass(frozen=True)
class EpisodeEvaluation:
    episode_id: str
    split: str
    pair_id: str
    pair_member: str
    expected_route: str
    route_decision: str | None
    route_correct: bool
    success: bool
    collision: bool
    drop: bool
    bounds_violation: bool
    height_violation: bool
    final_placement_error_m: float
    steps: int
    submitted_actions: int
    terminated: bool
    truncated: bool
    failure_reasons: tuple[str, ...]
    evaluator_version: str = EVALUATOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failure_reasons"] = list(self.failure_reasons)
        return payload


def _failure_reasons(
    info: Mapping[str, Any], success: bool, terminated: bool, truncated: bool
) -> tuple[str, ...]:
    reasons = [
        name
        for name in ("collision", "drop", "bounds_violation", "height_violation")
        if bool(info[name])
    ]
    if not success and info["route_decision"] != info["route_label"]:
        reasons.append("wrong_or_missing_route")
    if not success and truncated:
        reasons.append("timeout")
    elif not success and not terminated:
        reasons.append("incomplete")
    return tuple(reasons)


def evaluate_actions(
    scene: SceneGeometry,
    actions: Iterable[Sequence[float]],
    *,
    image_size: int = 16,
    stop_on_terminal: bool = True,
) -> EpisodeEvaluation:
    """Replay one action sequence through the authoritative MuJoCo environment."""

    action_rows = tuple(tuple(float(value) for value in action) for action in actions)
    env = DualClearanceMujocoEnv(image_size=image_size)
    env.reset(scene)
    terminated = False
    truncated = False
    for action in action_rows:
        _, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float64))
        if stop_on_terminal and (terminated or truncated):
            break
    info = env.info()
    object_position = env.object_position()
    final_error = float(
        np.linalg.norm(object_position[:2] - np.asarray(scene.layout.goal_xy, dtype=np.float64))
    )
    success = bool(info["success"])
    result = EpisodeEvaluation(
        episode_id=f"{scene.split}:{scene.pair_id}:{scene.member}",
        split=scene.split,
        pair_id=scene.pair_id,
        pair_member=scene.member,
        expected_route=scene.route_label.value,
        route_decision=info["route_decision"],
        route_correct=info["route_decision"] == scene.route_label.value,
        success=success,
        collision=bool(info["collision"]),
        drop=bool(info["drop"]),
        bounds_violation=bool(info["bounds_violation"]),
        height_violation=bool(info["height_violation"]),
        final_placement_error_m=final_error,
        steps=int(info["steps"]),
        submitted_actions=len(action_rows),
        terminated=bool(terminated),
        truncated=bool(truncated),
        failure_reasons=_failure_reasons(info, success, terminated, truncated),
    )
    env.close()
    return result


def _rates(records: Sequence[EpisodeEvaluation]) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        raise ValueError("Cannot summarize an empty episode set")
    successful = [item for item in records if item.success]
    return {
        "episodes": count,
        "episode_success_rate": sum(item.success for item in records) / count,
        "route_accuracy": sum(item.route_correct for item in records) / count,
        "collision_rate": sum(item.collision for item in records) / count,
        "drop_rate": sum(item.drop for item in records) / count,
        "bounds_violation_rate": sum(item.bounds_violation for item in records) / count,
        "height_violation_rate": sum(item.height_violation for item in records) / count,
        "mean_final_placement_error_m": sum(
            item.final_placement_error_m for item in records
        )
        / count,
        "mean_executed_steps": sum(item.steps for item in records) / count,
        "mean_completion_steps": (
            sum(item.steps for item in successful) / len(successful) if successful else None
        ),
    }


def summarize_evaluations(records: Sequence[EpisodeEvaluation]) -> dict[str, Any]:
    """Aggregate episodes, requiring exactly two distinct members per pair."""

    if not records:
        raise ValueError("Cannot summarize an empty episode set")
    identifiers = [item.episode_id for item in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Episode identifiers must be unique")

    by_pair: dict[tuple[str, str], list[EpisodeEvaluation]] = defaultdict(list)
    by_split: dict[str, list[EpisodeEvaluation]] = defaultdict(list)
    for item in records:
        by_pair[(item.split, item.pair_id)].append(item)
        by_split[item.split].append(item)
    invalid_pairs = {
        f"{split}:{pair_id}": sorted(item.pair_member for item in items)
        for (split, pair_id), items in by_pair.items()
        if len(items) != 2 or len({item.pair_member for item in items}) != 2
    }
    if invalid_pairs:
        raise ValueError(f"Each pair requires exactly two distinct members: {invalid_pairs}")

    pair_successes = {
        f"{split}:{pair_id}": all(item.success for item in items)
        for (split, pair_id), items in by_pair.items()
    }
    overall = _rates(records)
    overall["pairs"] = len(by_pair)
    overall["pair_successes"] = sum(pair_successes.values())
    overall["pair_success_rate"] = sum(pair_successes.values()) / len(by_pair)
    split_summaries: dict[str, dict[str, Any]] = {}
    for split, items in sorted(by_split.items()):
        split_metrics = _rates(items)
        split_pairs = {
            pair_id: values
            for (pair_split, pair_id), values in by_pair.items()
            if pair_split == split
        }
        split_pair_successes = sum(all(item.success for item in values) for values in split_pairs.values())
        split_metrics["pairs"] = len(split_pairs)
        split_metrics["pair_successes"] = split_pair_successes
        split_metrics["pair_success_rate"] = split_pair_successes / len(split_pairs)
        split_summaries[split] = split_metrics
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "metric_names_zh": METRIC_NAMES,
        "overall": overall,
        "by_split": split_summaries,
        "pair_success": pair_successes,
    }
