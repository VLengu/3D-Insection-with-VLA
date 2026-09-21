# V1 freeze: planar structured-condition study

V1 is frozen as the completed planar baseline. Its purpose is to establish
whether the diffusion policy causally uses a structured condition, not whether
it performs genuine 3D reasoning.

## Frozen comparison

- no-3D: RGB-only, 384 hidden dimension, 6 blocks, 6 heads, 6 epochs;
- AdaLN-only: `Z_3D` injected through block-wise AdaLN;
- AdaLN + middle: the same AdaLN path plus a middle-gated residual with
  `correction_strength=0.025`.

All three use v-prediction, the same partial-goal-occlusion augmentation,
12,000 training examples, and seeds 42/43/44.

## Frozen conclusion

Zero, Shuffle, Shift, and RGB-3D Conflict interventions show that the AdaLN
models causally and strongly follow `Z_3D`. Because V1 actions and useful
geometry are planar, this is evidence of structured-condition dependence and
must not be presented as proof of 3D spatial reasoning.

The machine-readable result is `outputs/stage2_diagnostics.json`; the compact
table is `outputs/stage2_diagnostics.md`.

V1 should receive only bug fixes. New task design, data, and model comparisons
belong to the height-world V2 pipeline.
