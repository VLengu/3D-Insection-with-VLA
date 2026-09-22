# Gate M4A：统一 Structured Projector 验证报告

日期：2026-09-22

## 结论

统一结构化投影器 `typed-structured-projector-v1` 已实现并通过验证。2D 与 3D
oracle 现在使用完全相同的 token 数量、原始槽位宽度、hidden dimension、
attention mask 形状和网络参数量，可以公平地进入后续 VL/A 注入实验。

这是 Gate M4 的第一部分。VL/A 注入边界和 TinyContractVLA 尚未实现，因此
Gate M4 整体尚未放行。

## Token 设计

两个模态统一产生 6 个 typed tokens：

1. `object`；
2. `goal`；
3. `central_obstacle`；
4. `side_gate`；
5. `robot`；
6. `global`。

每个 token 固定为 8 个原始槽位：

```text
[position_x, position_y, position_z,
 extent_x, extent_y, extent_z,
 scalar_0, scalar_1]
```

3D 使用 26 个有效槽位，2D 使用 14 个有效槽位。2D 缺失的高度、厚度、净空、
机械臂最大携带高度、安全余量和相机偏航由 feature mask 屏蔽，不改变张量宽度。
改变被屏蔽槽位的任意值不会改变 projector 输出。

输入接口只接受严格的 15 维或 27 维浮点 tensor。路线标签、pair ID、成功状态、
可行性和专家 waypoint 均不属于可接受字段，附加宽度会被直接拒绝。

## 归一化与网络结构

归一化尺度来自预先声明的物理工作区，而非 validation、paired-test 或 OOD
统计。例如位置按 1 m、高度按 0.5 m、物体尺寸按 0.1 m、偏航按 π 缩放。

投影器输入是：

```text
concat(masked_normalized_values, feature_mask)
```

之后经过共享的两层 MLP、token type embedding 和 LayerNorm，输出：

```text
structured_tokens     [batch, 6, hidden_dim]
token_attention_mask  [batch, 6]
```

2D 与 3D 都使用同一个 `StructuredProjector` 类，不存在模态专属参数分支。

## 全量验证结果

使用全部 44 pair / 88 scenes，`hidden_dim=128`：

| 项目 | 2D | 3D |
|---|---:|---:|
| 输出 shape | `[88,6,128]` | `[88,6,128]` |
| 总参数 | 38,272 | 38,272 |
| 可训练参数 | 38,272 | 38,272 |
| 参数差 | 0 | 0 |
| 有效原始槽位/scene | 14 | 26 |

配对检查：

- 44 / 44 的 2D 配对 token 在 `1e-6` 内一致；
- 2D 配对最大绝对误差为 `1.1920928955078125e-07`；
- 44 / 44 的 3D 配对 token 不同；
- 3D 配对最小的最大绝对差为 `0.008812189102172852`。

2D 的极小非零差来自批量矩阵乘法的浮点累加顺序；原始 2D oracle 和 tokenized
slots 均逐位相同。该误差比验收容差小约一个数量级，并比最小 3D 信号差异小约
五个数量级。

设备与训练链路：

- CPU：88 scenes 前向/反向，有限值且梯度非零；
- Apple MPS：8 scenes 前向/反向，有限值且梯度非零；
- checkpoint 保存/恢复后 CPU 输出逐位相同；
- 固定 seed 下初始化和验证可复现；
- 35 项项目自动测试全部通过。

机器证据位于 `fixtures/structured_projector/summary.json`。

## 已知边界

- 当前 projector 只验证接口、公平性和梯度，不证明这些 token 足以训练出成功策略；
- feature mask 会让网络知道哪些语义字段缺失，这是 2D/3D 条件本身的一部分，
  但不会通过 shape 或参数量泄漏模型类别；
- normalization scales 是固定物理先验，后续不得根据 paired-test 表现调整；
- Gate M4 仍需证明 VL tokens 出现在 contextualization 前，而 A tokens 只在
  VLM 输出后进入 action expert。

## 复现命令

```bash
venv_stage2/bin/python -m stage2.validation.validate_structured_projector \
  --config stage2/configs/dataset_tiny.json \
  --output stage2/fixtures/structured_projector \
  --hidden-dim 128
```

## 下一步

实现 VL/A 两条严格分离的注入 API，并用 hook 与反事实输入验证：A 侧的 VLM
contextual features 不随 oracle 改变，VL 侧 contextual features 必须随 oracle
改变。随后把相同接口接入 TinyContractVLA 做四配置 forward/backward。
