"""Mac-compatible MuJoCo environment for Dual-Clearance transport."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import mujoco
import numpy as np

from stage2.data.schema import oracle_2d, oracle_3d
from stage2.envs.dual_clearance.controller import PANDA_HOME_QPOS, PandaCartesianController
from stage2.envs.dual_clearance.geometry import Route, SceneGeometry
from stage2.envs.dual_clearance.scene_builder import (
    WALL_HEIGHT,
    WORKSPACE_X,
    WORKSPACE_Y,
    build_model,
)


POSITION_STEP = 0.025
ROTATION_STEP = 0.08
PHYSICS_SUBSTEPS = 5


class DualClearanceMujocoEnv:
    """A deterministic local validation environment, not a training wrapper yet."""

    def __init__(self, image_size: int = 256):
        self.image_size = image_size
        self.scene: SceneGeometry | None = None
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.controller: PandaCartesianController | None = None
        self.renderer: mujoco.Renderer | None = None
        self.grasped = False
        self.collision = False
        self.drop = False
        self.bounds_violation = False
        self.height_violation = False
        self.steps = 0
        self.route_decision: str | None = None
        self._last_object_x: float | None = None
        self._last_object_position: np.ndarray | None = None
        self._barrier_geom_ids: set[int] = set()
        self._object_geom_id = -1
        self._object_joint_qpos_adr = -1
        self._object_joint_dof_adr = -1

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def reset(self, scene: SceneGeometry) -> dict[str, Any]:
        self.close()
        self.scene = scene
        self.model = build_model(scene)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:7] = PANDA_HOME_QPOS
        self.data.qpos[7:9] = 0.04
        object_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint"
        )
        self._object_joint_qpos_adr = int(self.model.jnt_qposadr[object_joint_id])
        self._object_joint_dof_adr = int(self.model.jnt_dofadr[object_joint_id])
        self._set_object_pose(
            np.array(
                [
                    scene.layout.object_xy[0],
                    scene.layout.object_xy[1],
                    scene.layout.object_half_xyz[2] + 0.001,
                ],
                dtype=np.float64,
            )
        )
        self.data.ctrl[:7] = PANDA_HOME_QPOS
        self.data.ctrl[7] = 255.0
        mujoco.mj_forward(self.model, self.data)
        self.controller = PandaCartesianController(self.model, self.data)
        self.renderer = mujoco.Renderer(
            self.model, height=self.image_size, width=self.image_size
        )
        self.grasped = False
        self.collision = False
        self.drop = False
        self.bounds_violation = False
        self.height_violation = False
        self.steps = 0
        self.route_decision = None
        self._last_object_x = float(self.object_position()[0])
        self._last_object_position = self.object_position()
        self._object_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"
        )
        self._barrier_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            and (
                name == "central_obstacle_collision"
                or name == "side_gate_beam_collision"
                or name.startswith("barrier_filler_")
                or name.startswith("outer_wall_")
            )
        }
        return self.observation()

    def _require_ready(self) -> tuple[mujoco.MjModel, mujoco.MjData, SceneGeometry]:
        if self.model is None or self.data is None or self.scene is None:
            raise RuntimeError("Environment must be reset before use")
        return self.model, self.data, self.scene

    def _set_object_pose(self, position: np.ndarray) -> None:
        if self.data is None:
            return
        start = self._object_joint_qpos_adr
        self.data.qpos[start : start + 3] = position
        self.data.qpos[start + 3 : start + 7] = [1.0, 0.0, 0.0, 0.0]
        if self._object_joint_dof_adr >= 0:
            self.data.qvel[self._object_joint_dof_adr : self._object_joint_dof_adr + 6] = 0.0

    def object_position(self) -> np.ndarray:
        _, data, _ = self._require_ready()
        start = self._object_joint_qpos_adr
        return data.qpos[start : start + 3].copy()

    def robot_state(self) -> list[float]:
        model, data, scene = self._require_ready()
        if self.controller is None:
            raise RuntimeError("Controller is not initialized")
        grip = self.controller.grip_position()
        current_quat = data.xquat[self.controller.hand_id]
        vector_norm = np.linalg.norm(current_quat[1:])
        if vector_norm < 1e-10:
            rotation = np.zeros(3)
        else:
            angle = 2.0 * math.atan2(vector_norm, current_quat[0])
            rotation = angle * current_quat[1:] / vector_norm
        return [
            *grip.tolist(),
            *rotation.tolist(),
            float(data.qpos[7]),
            float(data.qpos[8]),
        ]

    def observation(self) -> dict[str, Any]:
        _, _, scene = self._require_ready()
        return {
            "observation_state": self.robot_state(),
            "oracle_2d": oracle_2d(scene),
            "oracle_3d": oracle_3d(scene),
        }

    def _attach_object_to_grip(self) -> None:
        if not self.grasped or self.controller is None:
            return
        self._set_object_pose(self.controller.grip_position())

    def _update_contacts_and_limits(self) -> None:
        model, data, scene = self._require_ready()
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self._object_geom_id in pair and pair & self._barrier_geom_ids:
                self.collision = True
        position = self.object_position()
        half = scene.layout.object_half_xyz
        if self._last_object_position is not None:
            self.collision = self.collision or self._swept_barrier_collision(
                self._last_object_position, position
            )
        if (
            position[0] - half[0] < WORKSPACE_X[0]
            or position[0] + half[0] > WORKSPACE_X[1]
            or position[1] - half[1] < WORKSPACE_Y[0]
            or position[1] + half[1] > WORKSPACE_Y[1]
        ):
            self.bounds_violation = True
        if position[2] > scene.layout.robot_max_carry_z + 1e-6:
            self.height_violation = True
        # The local expert uses a deterministic grasp constraint: while the
        # fingers are closed, the object's free joint is attached to the grip
        # point.  A low Cartesian pose is therefore a table/contact problem,
        # not evidence that the object detached from the gripper.

        barrier_x = scene.layout.central_xy[0]
        if self._last_object_x is not None and self._last_object_x < barrier_x <= position[0]:
            y = position[1]
            central_low = scene.layout.central_xy[1] - scene.layout.central_half_xy[1]
            central_high = scene.layout.central_xy[1] + scene.layout.central_half_xy[1]
            side_low = scene.layout.side_gate_xy[1] - scene.layout.side_gate_half_xy[1]
            side_high = scene.layout.side_gate_xy[1] + scene.layout.side_gate_half_xy[1]
            if central_low <= y <= central_high:
                self.route_decision = Route.OVER.value
            elif side_low <= y <= side_high:
                self.route_decision = Route.SIDE.value
            else:
                self.route_decision = "invalid"
        self._last_object_x = float(position[0])
        self._last_object_position = position.copy()

    @staticmethod
    def _segment_intersects_aabb(
        start: np.ndarray,
        end: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
    ) -> bool:
        """Exact slab test used to close gaps between discrete MuJoCo poses."""

        direction = end - start
        t_low, t_high = 0.0, 1.0
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if start[axis] < low[axis] or start[axis] > high[axis]:
                    return False
                continue
            first = (low[axis] - start[axis]) / direction[axis]
            second = (high[axis] - start[axis]) / direction[axis]
            if first > second:
                first, second = second, first
            t_low = max(t_low, float(first))
            t_high = min(t_high, float(second))
            if t_low > t_high:
                return False
        return True

    def _swept_barrier_collision(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check the carried box's swept center against expanded obstacles."""

        if self.scene is None:
            return False
        scene = self.scene
        layout = scene.layout
        hx, hy, hz = layout.object_half_xyz
        safety = layout.safety_margin

        boxes: list[tuple[np.ndarray, np.ndarray]] = []
        cx, cy = layout.central_xy
        chx, chy = layout.central_half_xy
        boxes.append(
            (
                np.array([cx - chx - hx, cy - chy - hy, -hz - safety]),
                np.array(
                    [cx + chx + hx, cy + chy + hy, scene.central_top_z + hz + safety]
                ),
            )
        )
        gx, gy = layout.side_gate_xy
        ghx, ghy = layout.side_gate_half_xy
        boxes.append(
            (
                np.array(
                    [gx - ghx - hx, gy - ghy - hy, scene.side_gate_clearance_z - hz - safety]
                ),
                np.array([gx + ghx + hx, gy + ghy + hy, WALL_HEIGHT + hz + safety]),
            )
        )

        openings = sorted(
            [
                (cy - chy, cy + chy),
                (gy - ghy, gy + ghy),
            ]
        )
        cursor = WORKSPACE_Y[0]
        barrier_half_x = max(chx, ghx)
        for opening_low, opening_high in openings:
            if opening_low > cursor:
                boxes.append(
                    (
                        np.array([cx - barrier_half_x - hx, cursor - hy, -hz]),
                        np.array([cx + barrier_half_x + hx, opening_low + hy, WALL_HEIGHT + hz]),
                    )
                )
            cursor = max(cursor, opening_high)
        if cursor < WORKSPACE_Y[1]:
            boxes.append(
                (
                    np.array([cx - barrier_half_x - hx, cursor - hy, -hz]),
                    np.array([cx + barrier_half_x + hx, WORKSPACE_Y[1] + hy, WALL_HEIGHT + hz]),
                )
            )
        return any(
            self._segment_intersects_aabb(start, end, low, high) for low, high in boxes
        )

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        model, data, scene = self._require_ready()
        if self.controller is None:
            raise RuntimeError("Controller is not initialized")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite 7D vector")
        action = np.clip(action, -1.0, 1.0)
        current = self.controller.grip_position()
        target = current + POSITION_STEP * action[:3]
        target[0] = np.clip(target[0], WORKSPACE_X[0] + 0.03, WORKSPACE_X[1] - 0.03)
        target[1] = np.clip(target[1], WORKSPACE_Y[0] + 0.03, WORKSPACE_Y[1] - 0.03)
        target[2] = np.clip(target[2], 0.035, 0.52)
        self.controller.move_kinematic(target)

        if action[6] < -0.5:
            data.qpos[7:9] = 0.0
            data.ctrl[7] = 0.0
            if np.linalg.norm(self.controller.grip_position() - self.object_position()) <= 0.045:
                self.grasped = True
        elif action[6] > 0.5:
            data.qpos[7:9] = 0.04
            data.ctrl[7] = 255.0
            self.grasped = False

        self._attach_object_to_grip()
        mujoco.mj_forward(model, data)
        for _ in range(PHYSICS_SUBSTEPS):
            if self.grasped:
                self._attach_object_to_grip()
                mujoco.mj_forward(model, data)
            else:
                # Once released, the object is a normal MuJoCo free body.  In
                # particular, successful placement is checked after gravity
                # and table contact have been advanced, not at a teleported
                # pose that the simulator has never seen.
                mujoco.mj_step(model, data)
            self._update_contacts_and_limits()
        self.steps += 1
        terminated = self.success() or self.failure()
        truncated = self.steps >= 300
        reward = 1.0 if self.success() else 0.0
        return self.observation(), reward, terminated, truncated, self.info()

    def success(self) -> bool:
        if self.scene is None or self.grasped:
            return False
        position = self.object_position()
        goal = np.array(self.scene.layout.goal_xy)
        goal_distance = np.linalg.norm(position[:2] - goal)
        on_table = position[2] <= self.scene.layout.object_half_xyz[2] + 0.02
        return bool(
            goal_distance <= 0.055
            and on_table
            and not self.collision
            and not self.drop
            and not self.bounds_violation
            and not self.height_violation
            and self.route_decision == self.scene.route_label.value
        )

    def failure(self) -> bool:
        return self.collision or self.drop or self.bounds_violation or self.height_violation

    def info(self) -> dict[str, Any]:
        scene = self.scene
        return {
            "success": self.success(),
            "collision": self.collision,
            "drop": self.drop,
            "bounds_violation": self.bounds_violation,
            "height_violation": self.height_violation,
            "grasped": self.grasped,
            "route_decision": self.route_decision,
            "route_label": None if scene is None else scene.route_label.value,
            "steps": self.steps,
        }

    @staticmethod
    def _policy_camera(scene: SceneGeometry) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.50, 0.0, 0.08]
        camera.distance = 1.90
        camera.azimuth = 110.0 + math.degrees(scene.layout.camera_yaw)
        camera.elevation = -70.0
        camera.orthographic = 1
        return camera

    def _render_current_model(self) -> np.ndarray:
        _, data, scene = self._require_ready()
        if self.renderer is None:
            raise RuntimeError("Renderer is not initialized")
        self.renderer.update_scene(data, camera=self._policy_camera(scene))
        return self.renderer.render().copy()

    def render_segmentation(self) -> np.ndarray:
        """Return MuJoCo ``[object_id, object_type]`` policy-view pixels."""

        _, data, scene = self._require_ready()
        if self.renderer is None:
            raise RuntimeError("Renderer is not initialized")
        self.renderer.update_scene(data, camera=self._policy_camera(scene))
        self.renderer.enable_segmentation_rendering()
        try:
            return self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()

    def visibility_metrics(self) -> dict[str, int]:
        """Count visible pixels for the objects needed to understand the task."""

        model, _, _ = self._require_ready()
        segmentation = self.render_segmentation()
        geom_pixels = segmentation[..., 0][
            segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        ]

        def count(name: str) -> int:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            return int(np.count_nonzero(geom_pixels == geom_id))

        robot_visual_ids = np.flatnonzero(model.geom_group == 2)
        return {
            "object_pixels": count("object_geom"),
            "goal_pixels": count("goal"),
            "central_route_pixels": count("central_obstacle_visual"),
            "side_route_pixels": count("side_gate_beam_visual"),
            "robot_pixels": int(np.count_nonzero(np.isin(geom_pixels, robot_visual_ids))),
        }

    def render_physical(self) -> np.ndarray:
        """Render the actual member-specific obstacle heights for diagnostics."""

        model, data, _ = self._require_ready()
        central_collision = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "central_obstacle_collision"
        )
        gate_collision = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "side_gate_beam_collision"
        )
        central_visual = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "central_obstacle_visual"
        )
        gate_visual = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "side_gate_beam_visual"
        )
        ids = [central_collision, gate_collision, central_visual, gate_visual]
        saved = model.geom_rgba[ids].copy()
        model.geom_rgba[central_collision] = [0.90, 0.58, 0.12, 1.0]
        model.geom_rgba[gate_collision] = [0.10, 0.62, 0.52, 1.0]
        model.geom_rgba[central_visual, 3] = 0.0
        model.geom_rgba[gate_visual, 3] = 0.0
        mujoco.mj_forward(model, data)
        try:
            return self._render_current_model()
        finally:
            model.geom_rgba[ids] = saved
            mujoco.mj_forward(model, data)

    def render(self) -> np.ndarray:
        """Render the policy RGB with registered Z differences hidden.

        Counterfactual members expose the same RGB/XY evidence through shared
        visual shells.  Member-specific Z geometry remains active in separate,
        invisible collision geoms.
        """

        return self._render_current_model()

    def initial_rgb_sha256(self) -> str:
        return hashlib.sha256(self.render().tobytes()).hexdigest()
