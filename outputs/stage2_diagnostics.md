# Stage 2: structured-condition dependency diagnostics

> These tests measure dependence on `Z_3D`. The current planar dataset cannot by itself prove genuine 3D reasoning.

| Model | Base success | Zero success | Shuffle success | Shift sensitivity | Shift correct | Conflict Z3D preference |
|---|---:|---:|---:|---:|---:|---:|
| no-3d | 0.333 ± 0.026 | 0.333 ± 0.026 | 0.333 ± 0.026 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| adaln-only | 0.897 ± 0.173 | 0.000 ± 0.000 | 0.038 ± 0.002 | 0.885 ± 0.078 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| adaln-middle-0025 | 0.998 ± 0.004 | 0.000 ± 0.000 | 0.043 ± 0.004 | 0.895 ± 0.070 | 1.000 ± 0.000 | 1.000 ± 0.000 |

Interpretation:

- A useful condition should lose performance under Zero and Shuffle.
- Positive shift sensitivity and a high correct-direction rate mean the action changes coherently with a small 3D intervention.
- Conflict Z3D preference above 0.5 means the counterfactual action is more aligned with the 3D target than the visible RGB target.
- no-3D should remain invariant up to numerical noise; this is the negative control.
