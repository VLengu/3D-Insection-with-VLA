"""Simulator-independent geometry for the Stage-2 counterfactual task.

Each base XY layout produces two scenes.  In one scene only the route over the
central obstacle is feasible; in the other only the side gate is feasible.
The pair shares all planar geometry, object geometry, robot limits, language,
and camera pose.  Only registered Z geometry changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import random
from typing import Any


EPSILON = 1e-9


class Route(str, Enum):
    OVER = "over"
    SIDE = "side"


@dataclass(frozen=True)
class BaseLayout:
    object_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    central_xy: tuple[float, float]
    central_half_xy: tuple[float, float]
    side_gate_xy: tuple[float, float]
    side_gate_half_xy: tuple[float, float]
    robot_base_xy: tuple[float, float]
    object_half_xyz: tuple[float, float, float]
    robot_max_carry_z: float
    safety_margin: float
    camera_yaw: float


@dataclass(frozen=True)
class SceneGeometry:
    pair_id: str
    base_layout_id: str
    split: str
    member: str
    seed: int
    layout: BaseLayout
    central_top_z: float
    side_gate_clearance_z: float
    route_label: Route

    @property
    def object_full_height(self) -> float:
        return 2.0 * self.layout.object_half_xyz[2]

    @property
    def over_required_center_z(self) -> float:
        return self.central_top_z + self.layout.object_half_xyz[2] + self.layout.safety_margin

    @property
    def side_required_clearance_z(self) -> float:
        return self.object_full_height + self.layout.safety_margin

    @property
    def over_feasible(self) -> bool:
        return self.over_required_center_z <= self.layout.robot_max_carry_z + EPSILON

    @property
    def side_feasible(self) -> bool:
        return self.side_gate_clearance_z + EPSILON >= self.side_required_clearance_z

    @property
    def feasible_route(self) -> Route:
        if self.over_feasible == self.side_feasible:
            raise ValueError(
                "Scene must have exactly one feasible route: "
                f"over={self.over_feasible} side={self.side_feasible}"
            )
        return Route.OVER if self.over_feasible else Route.SIDE

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["route_label"] = self.route_label.value
        return payload


@dataclass(frozen=True)
class PairScenario:
    pair_id: str
    over_only: SceneGeometry
    side_only: SceneGeometry

    @property
    def members(self) -> tuple[SceneGeometry, SceneGeometry]:
        return self.over_only, self.side_only


def _uniform(rng: random.Random, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _layout_ranges(split: str) -> dict[str, tuple[float, float]]:
    if split in {"layout_ood", "compound_ood"}:
        return {
            "central_x": (0.39, 0.61),
            "central_y": (-0.11, 0.11),
            "central_half_x": (0.075, 0.105),
            "central_half_y": (0.13, 0.19),
            "side_abs_y": (0.30, 0.38),
        }
    return {
        "central_x": (0.45, 0.55),
        "central_y": (-0.05, 0.05),
        "central_half_x": (0.055, 0.08),
        "central_half_y": (0.10, 0.145),
        "side_abs_y": (0.235, 0.295),
    }


def _height_slack_ranges(split: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if split in {"height_ood", "compound_ood"}:
        return (0.04, 0.08), (0.04, 0.085)
    return (0.01, 0.035), (0.01, 0.04)


def _camera_yaw(rng: random.Random, split: str) -> float:
    if split in {"camera_ood", "compound_ood"}:
        magnitude = _uniform(rng, math.radians(18), math.radians(28))
    else:
        magnitude = _uniform(rng, 0.0, math.radians(8))
    return magnitude if rng.random() < 0.5 else -magnitude


def _generate_layout(rng: random.Random, split: str) -> BaseLayout:
    ranges = _layout_ranges(split)
    side_sign = -1.0 if rng.random() < 0.5 else 1.0
    # A wide object-height range deliberately overlaps absolute gate and
    # obstacle heights across route labels.  Feasibility must be inferred
    # relationally rather than from a single height threshold.
    object_half_z = _uniform(rng, 0.02, 0.09)
    return BaseLayout(
        object_xy=(_uniform(rng, 0.11, 0.21), _uniform(rng, -0.16, 0.16)),
        goal_xy=(_uniform(rng, 0.79, 0.89), _uniform(rng, -0.16, 0.16)),
        central_xy=(
            _uniform(rng, *ranges["central_x"]),
            _uniform(rng, *ranges["central_y"]),
        ),
        central_half_xy=(
            _uniform(rng, *ranges["central_half_x"]),
            _uniform(rng, *ranges["central_half_y"]),
        ),
        side_gate_xy=(
            _uniform(rng, 0.44, 0.56),
            side_sign * _uniform(rng, *ranges["side_abs_y"]),
        ),
        side_gate_half_xy=(_uniform(rng, 0.05, 0.075), _uniform(rng, 0.035, 0.055)),
        robot_base_xy=(0.0, 0.0),
        object_half_xyz=(
            _uniform(rng, 0.028, 0.052),
            _uniform(rng, 0.028, 0.052),
            object_half_z,
        ),
        robot_max_carry_z=_uniform(rng, 0.285, 0.43),
        safety_margin=_uniform(rng, 0.012, 0.035),
        camera_yaw=_camera_yaw(rng, split),
    )


def generate_pair(index: int, split: str, seed: int) -> PairScenario:
    """Generate one deterministic counterfactual pair.

    Absolute heights overlap across labels because feasibility is sampled
    relative to object size and robot reach.  A raw obstacle-height threshold
    is therefore not a reliable route label.
    """

    rng = random.Random(seed)
    layout = _generate_layout(rng, split)
    feasible_slack_range, blocked_slack_range = _height_slack_ranges(split)
    over_slack = _uniform(rng, *feasible_slack_range)
    over_block = _uniform(rng, *blocked_slack_range)
    side_slack = _uniform(rng, *feasible_slack_range)
    side_block = _uniform(rng, *blocked_slack_range)

    over_threshold_top = (
        layout.robot_max_carry_z - layout.object_half_xyz[2] - layout.safety_margin
    )
    side_threshold = 2.0 * layout.object_half_xyz[2] + layout.safety_margin

    over_only_top = over_threshold_top - over_slack
    side_only_top = over_threshold_top + over_block
    # Small objects in height-OOD can have a required clearance below the
    # requested blocking slack.  Clamp to a small physical opening rather
    # than emitting a negative gate clearance; it remains strictly blocked.
    blocked_gate_clearance = max(0.012, side_threshold - side_block)
    feasible_gate_clearance = side_threshold + side_slack
    if min(over_only_top, blocked_gate_clearance) <= 0.01:
        raise ValueError("Generated clearance is physically invalid")

    pair_id = f"{split}-{index:06d}"
    common = dict(
        pair_id=pair_id,
        base_layout_id=pair_id,
        split=split,
        seed=seed,
        layout=layout,
    )
    over_only = SceneGeometry(
        **common,
        member="A",
        central_top_z=over_only_top,
        side_gate_clearance_z=blocked_gate_clearance,
        route_label=Route.OVER,
    )
    side_only = SceneGeometry(
        **common,
        member="B",
        central_top_z=side_only_top,
        side_gate_clearance_z=feasible_gate_clearance,
        route_label=Route.SIDE,
    )

    if over_only.feasible_route is not Route.OVER:
        raise AssertionError("Over-only member is not exclusively over-feasible")
    if side_only.feasible_route is not Route.SIDE:
        raise AssertionError("Side-only member is not exclusively side-feasible")
    return PairScenario(pair_id=pair_id, over_only=over_only, side_only=side_only)
