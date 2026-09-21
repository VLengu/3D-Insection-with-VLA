# Final Stage 1: 2D-oracle action-side injection

## Protocol

- Input: RGB plus explicit 2D start, goal, obstacle center, and obstacle size.
- Injection: global AdaLN in the post-vision diffusion policy trunk.
- No route head, spatial-consistency loss, geometry tokens, or cross-attention.
- Architecture: hidden dimension 256, 4 blocks, 4 heads.
- Training: 8 epochs, batch size 32, seeds 42/43/44.
- Sampling: 10 reverse-diffusion steps; counterfactual pair members share noise.
- Checkpoint selection: validation metrics only. `paired_test` is not loaded
  during training and is evaluated once after the checkpoint is fixed.

## Selected checkpoints

| Seed | Selected epoch |
|---:|---:|
| 42 | 8 |
| 43 | 6 |
| 44 | 8 |

## Paired-test result

Mean and sample standard deviation over three training seeds:

| Metric / 中文 | Mean ± std |
|---|---:|
| success / 任务成功率 | 0.341 ± 0.020 |
| endpoint distance / 终点距离误差 | 0.0325 ± 0.0048 |
| collision rate / 碰撞率 | 0.656 ± 0.022 |
| bounds violation rate / 边界违规率 | 0.0018 ± 0.0015 |
| path-mode accuracy / 路径模式准确率 | 0.500 ± 0.000 |
| paired both modes correct / 配对双模式同时正确率 | 0.000 ± 0.000 |
| paired decisions differ / 配对决策差异率 | 0.000 ± 0.000 |

Per-seed paired-test success is 0.331, 0.364, and 0.327 for seeds 42,
43, and 44 respectively.

## Interpretation

The 2D-oracle is a stable negative control for the height-conditioned route
decision. Counterfactual pair members have identical RGB and explicit 2D
geometry, and the model receives identical initial diffusion noise. It
therefore emits the same route decision for both heights: mode accuracy stays
at chance, no pair has both modes correct, and no paired decision differs.

This result does not mean 2D information is generally useless: endpoint error
is low and some samples complete successfully. It establishes the narrower
claim required by Stage 1: 2D information alone cannot reproduce a route
choice that is labeled by hidden height under the matched-noise protocol.
