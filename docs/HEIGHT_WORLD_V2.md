# Height World V2 specification

## Research question

Can a conditional action diffusion policy use height information to choose a
physically valid 3D route when the RGB/XY projection is insufficient?

## Counterfactual construction

Every base layout creates a pair with byte-identical RGB, start/goal XY, and
obstacle footprint:

- low obstacle: lift, cross above it, and descend (`path_type=0`);
- high obstacle: route around it because the required clearance exceeds the
  maximum lift (`path_type=1`).

Thus every pair satisfies:

`same RGB + same XY + different height -> different expert action`.

The renderer deliberately contains no height-dependent color, size, shadow,
or texture. Camera OOD uses an orthographic yaw transform and still omits
height cues.

## State and action schema

- image: `[3, 64, 64]` RGB top view;
- action: `[24, 3]` normalized `(dx, dy, dz)`;
- trajectory: `[25, 3]` world-space object-center positions;
- z2d: start XY, goal XY, obstacle center XY, obstacle half extents XY;
- height: obstacle height and maximum lift;
- z3d: start/goal/relative XYZ, obstacle 3D center and half extents, maximum
  lift, camera yaw, and validity;
- path type: over or around;
- pair ID: identifies the two counterfactual samples.

## Expert planner

The analytic planner selects over when

`obstacle_height + vertical_margin <= max_lift`.

Otherwise it evaluates top and bottom routes around an obstacle expanded by
the object radius and safety margin, then selects the shorter route. Polyline
segments are discretized separately so fixed-horizon sampling cannot cut
across a safety waypoint. Every generated trajectory is collision checked.

## Splits

- train: paired in-distribution layouts and heights;
- val: held-out paired ID scenes;
- height_ood: lower low obstacles and higher high obstacles;
- layout_ood: unseen obstacle sizes and vertical placements;
- camera_ood: unseen orthographic camera yaw;
- paired_test: dedicated paired causal evaluation.

## Information baselines

- `rgb-only`;
- `2d-oracle`;
- `height-only`;
- `full-3d`;
- `3d-only`;
- `random`;
- `shuffled-3d`.

The decisive comparison is full-3D versus 2D-oracle. Height-only determines
whether a model learns a relational geometric policy or merely reads a binary
height-like switch. Formal experiments must not start until generated data pass
`scripts/validate_height_world.py`.

## Evaluation

- task success: endpoint within the goal radius and no collision;
- endpoint distance;
- collision rate;
- workspace-bounds violation rate;
- over/around path-mode accuracy;
- paired both-modes-correct rate;
- paired decision-difference rate.
