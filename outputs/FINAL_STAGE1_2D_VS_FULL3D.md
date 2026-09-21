# Final Stage 1: 2D versus 2D+3D action-side injection

## Question

What changes when explicit height-aware 3D state is injected through the same
global AdaLN mechanism as explicit 2D geometry?

This is an incremental information comparison. Both conditions include RGB.
The 2D condition contains start, goal, obstacle center, and obstacle size in
XY. The full-3D condition contains that planar information plus height-aware
world-state features.

## Controlled protocol

- Post-vision policy-side global AdaLN injection.
- Hidden dimension 256, 4 blocks, 4 heads.
- 8 epochs, batch size 32, 10 reverse-diffusion steps.
- Seeds 42/43/44.
- No route head, spatial loss, geometry tokens, or cross-attention.
- Counterfactual pair members share initial diffusion noise.
- Validation-only checkpoint selection.
- `paired_test` remains sealed until each checkpoint is fixed.

## Paired-test results

Mean and sample standard deviation over three training seeds:

| Metric / 中文 | 2D oracle | Full 3D | Mean change |
|---|---:|---:|---:|
| success / 任务成功率 | 0.341 ± 0.020 | 0.382 ± 0.112 | +0.041 |
| endpoint distance / 终点距离误差 | 0.0325 ± 0.0048 | 0.0316 ± 0.0054 | -0.0008 |
| collision rate / 碰撞率 | 0.656 ± 0.022 | 0.608 ± 0.116 | -0.047 |
| bounds violation rate / 边界违规率 | 0.0018 ± 0.0015 | 0.0058 ± 0.0060 | +0.0040 |
| path-mode accuracy / 路径模式准确率 | 0.500 ± 0.000 | 0.559 ± 0.009 | +0.059 |
| paired both modes correct / 配对双模式同时正确率 | 0.000 ± 0.000 | 0.172 ± 0.021 | +0.172 |
| paired decisions differ / 配对决策差异率 | 0.000 ± 0.000 | 0.226 ± 0.031 | +0.226 |

Full-3D paired-test success for seeds 42/43/44 is 0.506, 0.289, and 0.349.
The corresponding paired-decision-difference rates are 0.245, 0.243, and
0.191.

## Interpretation

The central causal observation is stable across seeds: with identical RGB,
identical planar geometry, and matched diffusion noise, 2D conditioning never
changes the route within a pair, whereas naive full-3D conditioning changes it
for about 22.6% of pairs. The added 3D state therefore reaches and influences
the action generator through the ordinary global conditioning pathway.

The influence is limited. Path-mode accuracy improves by only 5.9 percentage
points and remains far below reliable height-conditioned route selection.
Task success and collision exhibit large training-seed variance, so the
three-seed evidence does not establish a stable success-rate improvement.

The supported conclusion is therefore narrow: naive 3D injection is causally
used and changes action decisions, but it does not by itself produce robust
height-aware spatial behavior. Because the current task permits a valid
always-around strategy, success must remain a secondary quality metric rather
than evidence that 3D is necessary for feasibility.
