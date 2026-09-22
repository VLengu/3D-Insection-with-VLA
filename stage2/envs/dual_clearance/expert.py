"""Deterministic waypoint expert for MuJoCo Dual-Clearance scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from stage2.envs.dual_clearance.geometry import Route
from stage2.envs.dual_clearance.mujoco_env import (
    POSITION_STEP,
    DualClearanceMujocoEnv,
)


@dataclass(frozen=True)
class ExpertResult:
    success: bool
    route: str
    steps: int
    final_info: dict[str, object]
    actions: tuple[tuple[float, ...], ...]


class ExpertFailure(RuntimeError):
    """Raised when the deterministic local expert cannot finish a rollout."""


def _action_toward(env: DualClearanceMujocoEnv, target: np.ndarray) -> np.ndarray:
    if env.controller is None:
        raise RuntimeError("Environment is not reset")
    delta = target - env.controller.grip_position()
    action = np.zeros(7, dtype=np.float64)
    action[:3] = np.clip(delta / POSITION_STEP, -1.0, 1.0)
    return action


def _move(
    env: DualClearanceMujocoEnv,
    target: Iterable[float],
    actions: list[tuple[float, ...]],
    *,
    tolerance: float = 0.007,
    max_steps: int = 80,
) -> None:
    target_array = np.asarray(tuple(target), dtype=np.float64)
    if target_array.shape != (3,):
        raise ValueError("Cartesian waypoint must contain exactly three values")
    best_error = float("inf")
    stale_steps = 0
    for _ in range(max_steps):
        if env.controller is None:
            raise RuntimeError("Environment is not reset")
        error = float(np.linalg.norm(target_array - env.controller.grip_position()))
        if error <= tolerance:
            return
        if error + 1e-5 < best_error:
            best_error = error
            stale_steps = 0
        else:
            stale_steps += 1
        action = _action_toward(env, target_array)
        _, _, terminated, truncated, info = env.step(action)
        actions.append(tuple(float(value) for value in action))
        if (terminated or truncated) and not info["success"]:
            raise ExpertFailure(f"Expert move terminated: {info}")
        # A target can sit a few millimetres outside the fixed-orientation IK
        # workspace.  Accept a close, stalled transit waypoint; the final
        # placement waypoint remains strict.
        if stale_steps >= 8 and best_error <= 0.035:
            return
    raise ExpertFailure(
        f"Waypoint not reached: target={target_array.tolist()} best_error={best_error:.6f}"
    )


def _gripper(
    env: DualClearanceMujocoEnv,
    command: float,
    actions: list[tuple[float, ...]],
) -> None:
    action = np.zeros(7, dtype=np.float64)
    action[6] = command
    _, _, terminated, truncated, info = env.step(action)
    actions.append(tuple(float(value) for value in action))
    if (terminated or truncated) and not info["success"]:
        raise ExpertFailure(f"Expert gripper step terminated: {info}")


def rollout_expert(
    env: DualClearanceMujocoEnv,
    *,
    forced_route: Route | None = None,
) -> ExpertResult:
    """Execute a complete grasp/carry/place rollout in an already reset env.

    ``forced_route`` is used by validation to prove that the infeasible route
    fails.  Normal dataset generation leaves it unset and follows the oracle.
    """

    if env.scene is None:
        raise RuntimeError("Environment must be reset before expert rollout")
    scene = env.scene
    route = scene.route_label if forced_route is None else forced_route
    layout = scene.layout
    half_x, _, half_z = layout.object_half_xyz
    actions: list[tuple[float, ...]] = []

    pickup_z = half_z + 0.003
    _move(env, (*layout.object_xy, max(0.18, pickup_z + 0.08)), actions)
    _move(env, (*layout.object_xy, pickup_z), actions)
    _gripper(env, -1.0, actions)
    if not env.grasped:
        raise ExpertFailure("Assisted grasp did not attach the object")

    if route is Route.OVER:
        route_y = layout.central_xy[1]
        # Keep one controller-tolerance of reserve above the registered safety
        # envelope.  The generator guarantees at least 10 mm of over-route
        # feasibility slack, so this remains below robot_max_carry_z.
        carry_z = scene.over_required_center_z + 0.008
    else:
        route_y = layout.side_gate_xy[1]
        carry_z = pickup_z

    barrier_x = layout.central_xy[0]
    barrier_half_x = max(layout.central_half_xy[0], layout.side_gate_half_xy[0])
    crossing_span = barrier_half_x + half_x + layout.safety_margin + 0.015
    before_x = barrier_x - crossing_span
    after_x = barrier_x + crossing_span

    _move(env, (*layout.object_xy, carry_z), actions)
    _move(env, (before_x, route_y, carry_z), actions)
    _move(env, (after_x, route_y, carry_z), actions)

    # The fixed downward wrist orientation has less reach at high Z.  Once the
    # object is safely beyond the barrier, lower before travelling to the goal.
    transfer_z = max(0.20, pickup_z + 0.04)
    _move(env, (after_x, route_y, transfer_z), actions)
    _move(env, (after_x, layout.goal_xy[1], transfer_z), actions)
    _move(env, (*layout.goal_xy, transfer_z), actions)
    _move(env, (*layout.goal_xy, pickup_z), actions, tolerance=0.005)
    _gripper(env, 1.0, actions)

    final_info = env.info()
    return ExpertResult(
        success=bool(final_info["success"]),
        route=route.value,
        steps=len(actions),
        final_info=final_info,
        actions=tuple(actions),
    )
