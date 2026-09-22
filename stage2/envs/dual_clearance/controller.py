"""Deterministic Cartesian IK controller for the vendored Panda model."""

from __future__ import annotations

import math

import mujoco
import numpy as np


PANDA_HOME_QPOS = np.array(
    [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=np.float64
)


def quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = quaternion.copy()
    result[1:] *= -1.0
    return result


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = quat_multiply(target, quat_conjugate(current))
    if delta[0] < 0:
        delta *= -1.0
    vector_norm = np.linalg.norm(delta[1:])
    if vector_norm < 1e-10:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, max(delta[0], 1e-12))
    return angle * delta[1:] / vector_norm


class PandaCartesianController:
    """Maps a 6D target pose to seven Panda joint position targets."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self.hand_id < 0:
            raise ValueError("Panda hand body not found")
        self.grip_offset_local = np.array([0.0, 0.0, 0.12], dtype=np.float64)
        self.target_quat = np.zeros(4, dtype=np.float64)
        self.refresh_target_orientation()

    def refresh_target_orientation(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        self.target_quat[:] = self.data.xquat[self.hand_id]

    def grip_position(self) -> np.ndarray:
        rotation = self.data.xmat[self.hand_id].reshape(3, 3)
        return self.data.xpos[self.hand_id] + rotation @ self.grip_offset_local

    def solve(self, target_position: np.ndarray, iterations: int = 80) -> np.ndarray:
        target_position = np.asarray(target_position, dtype=np.float64)
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        damping = 2e-3
        def iterate(count: int) -> None:
            for _ in range(count):
                mujoco.mj_forward(self.model, self.data)
                grip = self.grip_position()
                position_error = target_position - grip
                rotation_error = quat_error(self.target_quat, self.data.xquat[self.hand_id])
                if np.linalg.norm(position_error) < 1e-4 and np.linalg.norm(rotation_error) < 2e-3:
                    break
                mujoco.mj_jac(
                    self.model,
                    self.data,
                    jacobian_position,
                    jacobian_rotation,
                    grip,
                    self.hand_id,
                )
                jacobian = np.vstack((jacobian_position[:, :7], jacobian_rotation[:, :7]))
                error = np.concatenate((position_error, 0.35 * rotation_error))
                system = jacobian @ jacobian.T + damping * np.eye(6)
                delta_q = jacobian.T @ np.linalg.solve(system, error)
                delta_q = np.clip(delta_q, -0.08, 0.08)
                self.data.qpos[:7] += delta_q
                for joint_index in range(7):
                    lower, upper = self.model.jnt_range[joint_index]
                    self.data.qpos[joint_index] = np.clip(
                        self.data.qpos[joint_index], lower, upper
                    )

        iterate(iterations)
        # Long waypoint sequences can drive a numerical IK seed into a local
        # minimum even though the pose is reachable.  Retry from the canonical
        # Panda posture; this changes only redundant arm joints, not the
        # requested Cartesian path or the object pose.
        if np.linalg.norm(target_position - self.grip_position()) > 0.01:
            self.data.qpos[:7] = PANDA_HOME_QPOS
            iterate(iterations * 2)
        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos[:7].copy()

    def move_kinematic(self, target_position: np.ndarray) -> np.ndarray:
        targets = self.solve(target_position)
        self.data.qvel[:7] = 0.0
        self.data.ctrl[:7] = targets
        mujoco.mj_forward(self.model, self.data)
        return targets
