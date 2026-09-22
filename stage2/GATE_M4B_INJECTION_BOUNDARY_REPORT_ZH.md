# Gate M4B：VL/A 注入边界验证报告

日期：2026-09-22

## 结论

VL 侧与 Action 侧的结构化 token 注入边界已经实现，并通过全部 44 pair / 88
scene 的反事实验证。`vl-action-injection-boundary-v1` 使用同一套模型参数，仅改变
structured tokens 首次进入网络的位置。

- VL 模式：structured tokens 首次出现在 `contextualizer_input`；
- A 模式：structured tokens 首次出现在 `action_expert_input`；
- A-2D 与 A-3D 的 contextual features 整批 SHA-256 完全相同；
- VL contextual features 会随 oracle 改变；
- 两种注入模式的最终动作都会随 oracle 改变。

这是 Gate M4 的第二部分。完整 TinyContractVLA 的图像、语言、action loss 与
训练/checkpoint 链路尚未完成，因此 Gate M4 整体仍未最终放行。

## 统一模型结构

当前边界模型接收预编码的：

```text
image tokens     [B, 4, H]
language tokens  [B, 3, H]
robot state      [B, 8]
structured       [B, 6, H]
```

VL 路径：

```text
[4 image + 3 language + 6 structured]
                  ↓
          contextualizer (13 tokens)
                  ↓
       action expert (+ 1 state token)
```

A 路径：

```text
[4 image + 3 language]
          ↓
 contextualizer (7 tokens)
          ↓
[7 contextual + 6 structured + 1 state]
          ↓
     action expert
```

两条路径进入 action expert 的 memory 都是 14 tokens。action expert、contextualizer、
structured projector、状态投影和动作头全部共享相同结构与参数。

## 注入追踪

forward 输出明确的 `InjectionTrace`，记录：

- `structured_first_seen_at`；
- contextualizer 输入 token 数；
- contextual features token 数；
- action memory token 数；
- structured tokens 在 context/action memory 中的索引区间；
- 实际有效的 structured attention token 数。

forward hooks 额外确认：

| 模式 | Contextualizer 实际输入 | Action expert 实际输入 |
|---|---:|---:|
| VL | 13 | 14 |
| A | 7 | 14 |

## 反事实验证

固定 image、language 和 robot state，只改变 oracle：

| 检查 | VL | A |
|---|---|---|
| 3D pair 改变 contextual features | 44/44 改变 | 44/44 在 `1e-6` 内一致 |
| 3D pair 改变动作输出 | 44/44 改变 | 44/44 改变 |
| 2D pair 动作一致 | 44/44 | 44/44 |
| oracle 置零改变动作 | 是 | 是 |
| oracle batch shuffle 改变动作 | 是 | 是 |
| 置零后 contextual features 不变 | 不适用 | 逐位不变 |
| shuffle 后 contextual features 不变 | 不适用 | 逐位不变 |

A-2D 与 A-3D 的整批 contextual feature hash 均为：

```text
f93de953f5f7928a1faf446ab58b70c2d37ce658ec84755367040336f444ba9b
```

pair-row 的最大数值误差为 `4.76837158203125e-07`，来自批量矩阵的浮点累加；
更强的“同一输入、仅替换 oracle”检查为逐位相同。

## 公平性与梯度

验证配置使用 `hidden_dim=64`、2 层 contextual transformer、4 个 action queries：

- 每个配置总参数：137,991；
- 每个配置可训练参数：137,991；
- 配置间参数差：0；
- 四配置 action memory 长度：14；
- VL-2D、VL-3D、A-2D、A-3D 的 projector、contextualizer 和 action expert
  梯度均存在且非零；
- CPU 使用 88 scenes 完成四配置 forward/backward；
- Apple MPS 使用 8 scenes 完成四配置 forward/backward；
- 所有输出有限，无 NaN/Inf；
- checkpoint 恢复后输出逐位一致。

项目自动测试现为 41 项，全部通过。机器证据位于
`fixtures/injection_boundary/summary.json`。

## 已知边界

- 这里使用的是预编码 image/language tokens，还不是完整视觉和语言编码器；
- 随机初始化动作变化只能证明信息通路存在，不能证明已经学会正确控制；
- VL 路径 contextualizer 的序列比 A 路径长，这是注入位置定义带来的必要计算差异，
  但两者参数量和 action memory 长度相同；
- 正式比较仍必须使用相同 batch、更新步数、优化器和 checkpoint 选择规则。

## 复现命令

```bash
venv_stage2/bin/python -m stage2.validation.validate_injection_boundary \
  --config stage2/configs/dataset_tiny.json \
  --output stage2/fixtures/injection_boundary \
  --hidden-dim 64
```

## 下一步

进入 Gate M4C：实现 TinyContractVLA，把真实 RGB、task 文本、robot state、
structured oracle 和 action chunk 接成完整 dataloader→model→loss→optimizer→
checkpoint 链路，并让四个配置都完成小规模过拟合测试。
