"""Canonical Stage-2 model-input and evaluation schema."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from stage2 import SCHEMA_VERSION
from stage2.envs.dual_clearance.geometry import SceneGeometry


TASK_VARIANTS = (
    "Pick up the red object and place it in the blue target area.",
    "Move the red object into the blue target area.",
    "Grasp the red object, carry it safely, and place it in the blue target.",
    "Transport the red object to the blue goal region.",
)

STATE_NAMES = (
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_axis_angle_x",
    "eef_axis_angle_y",
    "eef_axis_angle_z",
    "gripper_left_qpos",
    "gripper_right_qpos",
)

ACTION_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_axis_angle_x",
    "delta_axis_angle_y",
    "delta_axis_angle_z",
    "gripper",
)

ORACLE_2D_NAMES = (
    "object_x",
    "object_y",
    "goal_x",
    "goal_y",
    "central_x",
    "central_y",
    "central_half_x",
    "central_half_y",
    "side_gate_x",
    "side_gate_y",
    "side_gate_half_x",
    "side_gate_half_y",
    "robot_base_x",
    "robot_base_y",
    "valid",
)

ORACLE_3D_NAMES = (
    "object_x",
    "object_y",
    "object_center_z",
    "object_half_x",
    "object_half_y",
    "object_half_z",
    "goal_x",
    "goal_y",
    "goal_z",
    "central_x",
    "central_y",
    "central_center_z",
    "central_half_x",
    "central_half_y",
    "central_half_z",
    "side_gate_x",
    "side_gate_y",
    "side_gate_half_x",
    "side_gate_half_y",
    "side_gate_clearance_z",
    "robot_base_x",
    "robot_base_y",
    "robot_base_z",
    "robot_max_carry_z",
    "safety_margin",
    "camera_yaw",
    "valid",
)

MODEL_INPUT_KEYS = frozenset(
    {"task", "observation_state", "oracle_2d", "oracle_3d"}
)
FORBIDDEN_MODEL_KEYS = frozenset(
    {
        "route_label",
        "pair_id",
        "pair_member",
        "over_feasible",
        "side_feasible",
        "expert_waypoints",
        "success",
    }
)


@dataclass(frozen=True)
class DataContract:
    schema_version: str = SCHEMA_VERSION
    coordinate_frame: str = "right-handed world: +x robot-to-goal, +y table lateral, +z up"
    length_unit: str = "meter"
    angle_unit: str = "radian"
    fps: int = 20
    max_episode_steps: int = 300
    image_shape: tuple[int, int, int] = (3, 256, 256)
    image_dtype: str = "uint8"
    state_names: tuple[str, ...] = STATE_NAMES
    action_names: tuple[str, ...] = ACTION_NAMES
    oracle_2d_names: tuple[str, ...] = ORACLE_2D_NAMES
    oracle_3d_names: tuple[str, ...] = ORACLE_3D_NAMES

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "coordinate_frame": self.coordinate_frame,
            "length_unit": self.length_unit,
            "angle_unit": self.angle_unit,
            "fps": self.fps,
            "max_episode_steps": self.max_episode_steps,
            "image_shape": list(self.image_shape),
            "image_dtype": self.image_dtype,
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "oracle_2d_names": list(self.oracle_2d_names),
            "oracle_3d_names": list(self.oracle_3d_names),
        }


def oracle_2d(scene: SceneGeometry) -> list[float]:
    layout = scene.layout
    return [
        *layout.object_xy,
        *layout.goal_xy,
        *layout.central_xy,
        *layout.central_half_xy,
        *layout.side_gate_xy,
        *layout.side_gate_half_xy,
        *layout.robot_base_xy,
        1.0,
    ]


def oracle_3d(scene: SceneGeometry) -> list[float]:
    layout = scene.layout
    object_center_z = layout.object_half_xyz[2]
    central_center_z = scene.central_top_z / 2.0
    central_half_z = scene.central_top_z / 2.0
    return [
        layout.object_xy[0],
        layout.object_xy[1],
        object_center_z,
        *layout.object_half_xyz,
        layout.goal_xy[0],
        layout.goal_xy[1],
        object_center_z,
        layout.central_xy[0],
        layout.central_xy[1],
        central_center_z,
        layout.central_half_xy[0],
        layout.central_half_xy[1],
        central_half_z,
        layout.side_gate_xy[0],
        layout.side_gate_xy[1],
        layout.side_gate_half_xy[0],
        layout.side_gate_half_xy[1],
        scene.side_gate_clearance_z,
        layout.robot_base_xy[0],
        layout.robot_base_xy[1],
        0.0,
        layout.robot_max_carry_z,
        layout.safety_margin,
        layout.camera_yaw,
        1.0,
    ]


def initial_robot_state(scene: SceneGeometry) -> list[float]:
    return [
        scene.layout.object_xy[0],
        scene.layout.object_xy[1],
        0.18,
        0.0,
        math.pi,
        0.0,
        0.04,
        -0.04,
    ]


def record_from_scene(scene: SceneGeometry, task_variant: int) -> dict[str, Any]:
    task = TASK_VARIANTS[task_variant % len(TASK_VARIANTS)]
    model_inputs = {
        "task": task,
        "observation_state": initial_robot_state(scene),
        "oracle_2d": oracle_2d(scene),
        "oracle_3d": oracle_3d(scene),
    }
    evaluation = {
        "pair_id": scene.pair_id,
        "pair_member": scene.member,
        "base_layout_id": scene.base_layout_id,
        "route_label": scene.route_label.value,
        "over_feasible": scene.over_feasible,
        "side_feasible": scene.side_feasible,
        "central_top_z": scene.central_top_z,
        "side_gate_clearance_z": scene.side_gate_clearance_z,
        "over_required_center_z": scene.over_required_center_z,
        "side_required_clearance_z": scene.side_required_clearance_z,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": f"{scene.pair_id}-{scene.member}",
        "split": scene.split,
        "seed": scene.seed,
        "model_inputs": model_inputs,
        "evaluation": evaluation,
    }


def _validate_vector(name: str, values: Any, expected_names: Sequence[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(values, list):
        return [f"{name} must be a list"]
    if len(values) != len(expected_names):
        errors.append(f"{name} length={len(values)} expected={len(expected_names)}")
    for index, value in enumerate(values):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{name}[{index}] is not finite numeric data")
    return errors


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    inputs = record.get("model_inputs")
    if not isinstance(inputs, dict):
        return errors + ["model_inputs must be a dictionary"]
    if set(inputs) != MODEL_INPUT_KEYS:
        errors.append(f"model_inputs keys={sorted(inputs)} expected={sorted(MODEL_INPUT_KEYS)}")
    leaked = FORBIDDEN_MODEL_KEYS & set(inputs)
    if leaked:
        errors.append(f"forbidden labels leaked into model_inputs: {sorted(leaked)}")
    if not isinstance(inputs.get("task"), str) or not inputs.get("task"):
        errors.append("task must be a non-empty string")
    errors += _validate_vector("observation_state", inputs.get("observation_state"), STATE_NAMES)
    errors += _validate_vector("oracle_2d", inputs.get("oracle_2d"), ORACLE_2D_NAMES)
    errors += _validate_vector("oracle_3d", inputs.get("oracle_3d"), ORACLE_3D_NAMES)
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        errors.append("evaluation must be a dictionary")
    return errors
