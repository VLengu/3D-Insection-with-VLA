"""Build a MuJoCo scene from a Dual-Clearance geometry record."""

from __future__ import annotations

from pathlib import Path
import tempfile
from xml.sax.saxutils import escape

import mujoco

from stage2.envs.dual_clearance.geometry import SceneGeometry


VENDOR_PANDA_XML = (
    Path(__file__).resolve().parents[2] / "vendor" / "franka_emika_panda" / "panda.xml"
)

WORKSPACE_X = (0.04, 0.96)
WORKSPACE_Y = (-0.50, 0.50)
WALL_HEIGHT = 0.50


def _box(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: str,
    *,
    collision: bool = True,
) -> str:
    contact = 'contype="2" conaffinity="4"' if collision else 'contype="0" conaffinity="0"'
    return (
        f'<geom name="{escape(name)}" type="box" '
        f'pos="{pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f}" '
        f'size="{size[0]:.9f} {size[1]:.9f} {size[2]:.9f}" '
        f'rgba="{rgba}" {contact} friction="1 0.01 0.001"/>'
    )


def _barrier_fillers(scene: SceneGeometry, half_x: float) -> list[str]:
    layout = scene.layout
    openings = sorted(
        [
            (
                layout.central_xy[1] - layout.central_half_xy[1],
                layout.central_xy[1] + layout.central_half_xy[1],
            ),
            (
                layout.side_gate_xy[1] - layout.side_gate_half_xy[1],
                layout.side_gate_xy[1] + layout.side_gate_half_xy[1],
            ),
        ]
    )
    cursor = WORKSPACE_Y[0]
    segments: list[tuple[float, float]] = []
    for low, high in openings:
        if low > cursor:
            segments.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < WORKSPACE_Y[1]:
        segments.append((cursor, WORKSPACE_Y[1]))

    geoms = []
    for index, (low, high) in enumerate(segments):
        half_y = (high - low) / 2.0
        center_y = (high + low) / 2.0
        geoms.append(
            _box(
                f"barrier_filler_{index}",
                (layout.central_xy[0], center_y, WALL_HEIGHT / 2.0),
                (half_x, half_y, WALL_HEIGHT / 2.0),
                "0.33 0.35 0.38 1",
            )
        )
    return geoms


def build_scene_xml(scene: SceneGeometry) -> str:
    if not VENDOR_PANDA_XML.is_file():
        raise FileNotFoundError(f"Vendored Panda XML is missing: {VENDOR_PANDA_XML}")
    layout = scene.layout
    barrier_half_x = max(layout.central_half_xy[0], layout.side_gate_half_xy[0])
    central = _box(
        "central_obstacle",
        (layout.central_xy[0], layout.central_xy[1], scene.central_top_z / 2.0),
        (layout.central_half_xy[0], layout.central_half_xy[1], scene.central_top_z / 2.0),
        "0.42 0.43 0.46 1",
    )
    beam_top = WALL_HEIGHT
    beam_half_z = (beam_top - scene.side_gate_clearance_z) / 2.0
    gate_beam = _box(
        "side_gate_beam",
        (
            layout.side_gate_xy[0],
            layout.side_gate_xy[1],
            scene.side_gate_clearance_z + beam_half_z,
        ),
        (layout.side_gate_half_xy[0], layout.side_gate_half_xy[1], beam_half_z),
        "0.42 0.43 0.46 1",
    )
    fillers = "\n      ".join(_barrier_fillers(scene, barrier_half_x))
    object_pos = (
        layout.object_xy[0],
        layout.object_xy[1],
        layout.object_half_xyz[2] + 0.001,
    )
    goal_radius = max(layout.object_half_xyz[0], layout.object_half_xyz[1]) + 0.025

    return f"""<mujoco model="dual-clearance panda">
  <include file="{escape(str(VENDOR_PANDA_XML))}"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.72 0.72 0.72" diffuse="0.28 0.28 0.28" specular="0 0 0"/>
    <global offwidth="256" offheight="256" orthographic="true" fovy="1.15"/>
  </visual>
  <worldbody>
    <camera name="top_camera" mode="fixed" pos="0.5 0 1.8" quat="1 0 0 0" fovy="42"/>
    <light name="flat_light" pos="0.5 0 2" dir="0 0 -1" directional="true"
      castshadow="false" ambient="0.7 0.7 0.7" diffuse="0.3 0.3 0.3" specular="0 0 0"/>
    <geom name="table" type="plane" size="1 1 0.05" rgba="0.86 0.84 0.78 1"
      contype="1" conaffinity="5" friction="1 0.01 0.001"/>
    <geom name="goal" type="cylinder"
      pos="{layout.goal_xy[0]:.9f} {layout.goal_xy[1]:.9f} 0.002"
      size="{goal_radius:.9f} 0.002" rgba="0.08 0.28 0.90 1"
      contype="0" conaffinity="0"/>
    <body name="object" pos="{object_pos[0]:.9f} {object_pos[1]:.9f} {object_pos[2]:.9f}">
      <freejoint name="object_freejoint"/>
      <geom name="object_geom" type="box"
        size="{layout.object_half_xyz[0]:.9f} {layout.object_half_xyz[1]:.9f} {layout.object_half_xyz[2]:.9f}"
        rgba="0.85 0.06 0.04 1" mass="0.08" contype="4" conaffinity="3"
        friction="1.2 0.02 0.002"/>
    </body>
    {central}
    {gate_beam}
    {fillers}
    {_box("outer_wall_top", (0.5, WORKSPACE_Y[1] + 0.025, WALL_HEIGHT / 2), (0.5, 0.025, WALL_HEIGHT / 2), "0.25 0.27 0.30 1")}
    {_box("outer_wall_bottom", (0.5, WORKSPACE_Y[0] - 0.025, WALL_HEIGHT / 2), (0.5, 0.025, WALL_HEIGHT / 2), "0.25 0.27 0.30 1")}
    {_box("outer_wall_right", (WORKSPACE_X[1] + 0.025, 0, WALL_HEIGHT / 2), (0.025, 0.525, WALL_HEIGHT / 2), "0.25 0.27 0.30 1")}
  </worldbody>
</mujoco>"""


def build_model(scene: SceneGeometry) -> mujoco.MjModel:
    # Parse beside the vendored Panda XML so its relative meshdir="assets"
    # resolves exactly as in MuJoCo Menagerie.
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        prefix="dual_clearance_",
        dir=VENDOR_PANDA_XML.parent,
    ) as handle:
        handle.write(build_scene_xml(scene))
        handle.flush()
        return mujoco.MjModel.from_xml_path(handle.name)
