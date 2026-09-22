# Gate M4C：TinyContractVLA 端到端验证报告

日期：2026-09-22

## 结论

TinyContractVLA 已完成真实数据的端到端训练验证：

```text
LeRobotDataset
→ RGB / task / state / oracle / action chunk
→ TinyContractVLA
→ VL-2D / VL-3D / A-2D / A-3D
→ masked action loss
→ backward / optimizer / checkpoint
```

四个配置均能在 Apple MPS 上完成 forward、backward 和小样本过拟合；所有必要
模块获得非零梯度，checkpoint 恢复后输出逐位一致。结合 M4A projector 和 M4B
注入边界证据，计划中的 Gate M4 现已完整通过。

这不是正式模型效果实验。过拟合 loss 不得用于判断 VL 与 A 哪一侧更优。

## 数据链路

`LeRobotActionChunkDataset` 从本地 LeRobotDataset v3 读取：

- `observation.images.image`：`3×256×256` float RGB；
- `observation.state`：8 维；
- task：自然语言字符串；
- `oracle_2d`：15 维；
- `oracle_3d`：27 维；
- 当前帧开始的未来 8 步、每步 7 维动作；
- action mask：episode 尾部不足 8 步时屏蔽 padding。

训练 batch 只来自 `train`：4 个 pair，每个 pair 同时包含 A/B，每个成员选取两个
内部时间点，共 16 samples。validation、paired-test 和所有 OOD 未参与训练或调参。

完整 train split 共有 2,491 个可采样 frame。终止帧的 action mask 只有 1 个有效
位置，证明 episode 边界没有跨越到下一条轨迹。

## TinyContractVLA

### 图像

轻量三层卷积把 `256×256 RGB` 转换为 `8×8=64` 个视觉 tokens。

### 语言

固定词表来自预先登记的 4 条等义任务指令：

- 词表大小 23；
- 最大长度 16；
- 明确 PAD 和 UNK；
- 不包含 `over`、`side`、`height`、`clearance` 或 `route`；
- padding 由 language attention mask 屏蔽。

### 结构化条件与动作

- 使用 M4A 的 6-token structured projector；
- 使用 M4B 的 VL/A 注入边界；
- robot state 作为 action expert 的独立 state token；
- 输出 8 步、每步 7 维 action chunk；
- loss 为只统计有效位置的 masked MSE。

本轮 `hidden_dim=64`、1 层 contextual transformer。每个配置均为 133,927 个
总参数和可训练参数，参数差为 0。

## 四配置过拟合结果

固定相同初始化、相同 16 samples、AdamW、学习率 `3e-3`、400 步：

| 配置 | 初始 loss | 最终评估 loss | 最终/初始 |
|---|---:|---:|---:|
| VL-2D | 0.151827 | 0.024330 | 0.1602 |
| VL-3D | 0.151669 | 0.027765 | 0.1831 |
| A-2D | 0.145954 | 0.001877 | 0.0129 |
| A-3D | 0.145842 | 0.002288 | 0.0157 |

四配置 loss 均至少下降 80%，通过预先设置的工程门槛。A 路径在这个随机初始化、
固定 16 样本的记忆实验中下降更快，但这不代表泛化、闭环成功或注入位置优势。

## 梯度与边界

四种配置首步均确认以下模块梯度存在且非零：

- image encoder；
- language embedding；
- structured projector；
- contextualizer；
- action expert。

实际 token 路径：

| 模式 | Contextualizer 输入 | Action memory | 首次看到结构信息 |
|---|---:|---:|---|
| VL | 86 | 87 | `contextualizer_input` |
| A | 80 | 87 | `action_expert_input` |

因此真实 RGB 与语言接入后，M4B 的边界仍然保持成立。

## Checkpoint 与设备

- 训练设备：Apple MPS；
- 四配置输出均无 NaN/Inf；
- checkpoint 包含模型、优化器、配置和步数；
- 四配置重新加载后的最大动作误差均为 `0.0`；
- 项目自动测试增至 46 项，全部通过。

机器证据位于 `fixtures/tiny_contract_vla/summary.json`。实际 checkpoint 只用于本次
内存往返验证，不提交 Git，避免把 Tiny 工程模型误当成正式结果。

## 已知边界

- TinyContractVLA 使用直接 action MSE，不是 SmolVLA 的正式 flow/diffusion 训练；
- 图像和语言 encoder 是项目内小模型，不含预训练视觉语言知识；
- 16 样本过拟合只证明链路可训练；
- 未进行 validation checkpoint selection 或闭环 MuJoCo rollout；
- 不得根据本报告中的四个 loss 修改正式实验假设或宣称注入优劣。

## 复现命令

```bash
venv_stage2/bin/python -m stage2.validation.validate_tiny_contract_vla \
  --dataset stage2/generated_data/lerobot_tiny \
  --output stage2/fixtures/tiny_contract_vla \
  --hidden-dim 64 \
  --steps 400
```

## 下一步

进入 Gate M5：准备 Linux/GPU 可执行交接包。重点是锁定 SmolVLA 版本与依赖、
四配置正式适配方案、Linux 安装和 smoke 命令、训练/评估配置、输出目录保护、
paired-test 锁定规则以及数据和代码版本绑定。
