# UWM-Lite: Early 3D Diffusion Correction

> 当前项目进度、正式结论、已知限制和下一阶段交接请优先阅读
> `docs/PROJECT_STATUS_HANDOFF_ZH.md`。该报告高于早期探索文档中的历史解释。
>
> 第二阶段的 Mac 本地构建、数据设计、验证门和 GPU 实验交接方案见
> `docs/STAGE2_MAC_BUILD_VALIDATION_PLAN_ZH.md`。
> 第二阶段 M1 几何与数据契约已通过验证，报告见
> `stage2/M1_GEOMETRY_DATA_CONTRACT_REPORT_ZH.md`。
> M3 LeRobotDataset 全量往返验证报告见
> `stage2/M3_LEROBOT_ROUNDTRIP_REPORT_ZH.md`。
> Gate M3 统一评价器验证报告见
> `stage2/GATE_M3_EVALUATOR_REPORT_ZH.md`。

这是一个面向 Apple Silicon 本地实验的最小验证项目。它不复现大规模 UWM 预训练，而是隔离检验一个假设：**在高噪声的反向扩散初期，用显式 3D 关系修正动作噪声预测，是否比后期或全程修正更有效。**

## 实验世界

- `64 x 64` 顶视 RGB 图像：红色可推动方块、蓝色目标区、可选干扰物。
- 动作是长度为 8 的二维连续动作序列，每步为方块的 `(dx, dy)`。
- 训练集中的位置范围较窄，OOD 集扩大目标位置范围，并加入颜色相近的视觉干扰物。
- `Z_3D` 是模拟器给出的结构化几何真值：方块、目标和干扰物的 3D 坐标、相对位移、距离和有效掩码。第一阶段用它隔离验证扩散门控；后续可替换为真正的 3D encoder 输出。

## 模型

`uwm3d.model.ActionDiffusionPolicy` 是一个 UWM-Lite 动作分支：图像编码器条件化一个带 AdaLN 时间条件的 Transformer。基础模型预测动作噪声；`ThreeDCorrectionAdapter` 根据 Transformer 表征和 `Z_3D` 预测有界残差：

`prediction = prediction_base + strength * gate(t) * tanh(correction)`

训练保留标准噪声预测损失，同时在中低噪声时间步增加终点到达与首步目标方向辅助损失。另有一个训练期的纯视觉目标位移预测头，迫使共享视觉编码器辨别红色方块与蓝色目标，即使动作已高度加噪。这些损失只使用模拟器训练标签，不作为 Baseline 的输入，因此不会把几何真值泄露给纯视觉基线。

支持四种门控：

- `base`：无 3D 矫正。
- `early`：高噪声步骤强、低噪声步骤弱。
- `middle`：仅中噪声步骤强，两个端点弱。
- `middle_wide`：在大部分中噪声区间维持强 correction。
- `late`：高噪声步骤弱、低噪声步骤强。
- `all`：所有步骤同样强。

扩散步 `t = T` 表示最高噪声。因此 `early` 使用 `t / T`，这与反向采样的“早期”一致。

默认使用 `v-prediction`。与 epsilon-prediction 相比，它在高噪声步直接保留干净动作的信息，更适合这个 50 步的本地小型实验；可用 `--prediction-type epsilon` 做公平对照。

`all` 的常数门控值为 `0.5`，使其跨全部扩散时间步的平均修正预算与线性 `early` / `late` 门控相同；否则它会因总注入强度翻倍而不是注入时机获益。

`--train-goal-occlusion-prob 0.5` 会在一半训练样本中仅遮挡 RGB 内的蓝色目标区，而保留动作标签和 `Z_3D`。评估会额外报告 `occluded_ood_*`，用以检验高噪声阶段的 3D correction 是否能在视觉目标不可见时恢复正确的动作意图。

`--train-goal-occlusion-style partial` 会只遮挡目标约一半。对应的 `partial_ood_*` 评估集保持 ID 位置范围、仅有一个近蓝色干扰物，避免把位置外推、多物体混淆和完全遮挡叠加为同一个难题。

`--three-d-adaln` 会将 `Z_3D` 融入每个 Transformer block 的 AdaLN 调制条件。其门控在高噪声时为 0.15、在低噪声时递增至 1.0；它与 residual correction 的时序门控互补。

`--correction-strength` 对应有界残差的全局系数 \(\alpha\)。例如 `--mode early --three-d-adaln --correction-strength 0.05` 实现“高噪声时残差主导、低噪声时 AdaLN 主导”的两阶段版本，同时避免过强的早期残差破坏去噪轨迹。

中等容量实验建议使用 `--hidden-dim 384 --depth 6 --heads 6 --batch-size 16 --grad-accum 2`，即约两倍物理 batch 的有效梯度累积，适合本机统一内存训练。

## 本地运行

在项目根目录运行，使用已有虚拟环境：

```bash
./venv/bin/python scripts/generate_data.py
./venv/bin/python scripts/visualize_data.py --split train
./venv/bin/python scripts/train.py --mode base --epochs 20
./venv/bin/python scripts/train.py --mode early --epochs 20
./venv/bin/python scripts/train.py --mode late --epochs 20
./venv/bin/python scripts/train.py --mode all --epochs 20
```

每次训练会自动在 ID 和 OOD 集上评估，并将最佳 OOD checkpoint 写入 `checkpoints/`。第一轮建议先用 `--epochs 3 --limit-train 512 --limit-eval 128` 做冒烟测试。

受控规模比较可使用完整 12,000 条训练样本、2 个 epoch（约 750 次更新），并对每种门控运行 3 个随机种子；完成后运行：

```bash
./venv/bin/python scripts/summarize_sweep.py --root checkpoints/scaled
```

## 判据

关键结论不是单次的绝对成功率，而是固定种子下多次运行时，`early` 是否在 OOD 成功率、最终距离和首步方向误差上稳定优于 `base`、`late` 与 `all`。

## 第二阶段：结构化条件依赖诊断

`scripts/diagnose_3d_dependency.py` 对同一初始扩散噪声运行四种反事实干预：

- `Zero 3D`：将 `Z_3D` 全部置零；
- `Shuffle 3D`：将 RGB 与 batch 内其他样本的 `Z_3D` 错配；
- `Shift target`：只平移结构化目标，并同步重算相对位移与距离；
- `RGB-3D conflict`：保持 RGB 目标不变，将结构化目标放到相反方向。

推荐用三个同配置、三随机种子的模型组进行对照：no-3D、AdaLN-only、AdaLN + middle residual (`alpha=0.025`)。完整命令示例：

```bash
./venv/bin/python scripts/diagnose_3d_dependency.py \
  --model 'no-3d=checkpoints/stage2/no3d_seed42/best_base.pt,checkpoints/stage2/no3d_seed43/best_base.pt,checkpoints/stage2/no3d_seed44/best_base.pt' \
  --model 'adaln-only=checkpoints/alpha_sweep_e6/alpha_0_seed42/best_base.pt,checkpoints/alpha_sweep_e6/alpha_0_seed43/best_base.pt,checkpoints/alpha_sweep_e6/alpha_0_seed44/best_base.pt' \
  --model 'adaln-middle-0025=checkpoints/alpha_sweep_e6/alpha_0p025_seed42/best_middle.pt,checkpoints/alpha_sweep_e6/alpha_0p025_seed43/best_middle.pt,checkpoints/alpha_sweep_e6/alpha_0p025_seed44/best_middle.pt' \
  --data data/ood_partial.npz \
  --output outputs/stage2_diagnostics.json
```

脚本会检查训练配置是否可比，并同时生成 JSON 原始结果和 Markdown 汇总表。这些测试证明模型是否因果性地使用结构化条件；由于当前任务仍是平面运动，它们不能单独证明真正的 3D 空间推理。

## Height World V2：真正需要高度的配对任务

V1 已冻结为平面结构化条件研究，具体边界见 `docs/V1_FREEZE.md`。V2 使用字节级相同的顶视 RGB/XY 配对场景：低障碍要求从上方越过，高障碍因超过最大抬升高度而要求绕行。动作扩展为 24 步 `(dx, dy, dz)`，使高度通过碰撞约束真正决定正确轨迹。完整规范见 `docs/HEIGHT_WORLD_V2.md`。

生成并验证数据：

```bash
./venv/bin/python scripts/generate_height_data.py
./venv/bin/python scripts/validate_height_world.py
```

统一训练入口支持 `rgb-only`、`2d-oracle`、`height-only`、`full-3d`、`3d-only`、`random` 和 `shuffled-3d`。例如：

```bash
./venv/bin/python scripts/train_height_world.py \
  --condition full-3d \
  --mode base \
  --checkpoint-dir checkpoints_height/full3d_seed42
```

该入口已经通过小规模端到端冒烟测试；正式基线训练应在确认 V2 数据统计和配对难度后再开始。

### V2 第一轮核心筛选

固定 seed 42、256 hidden dimension、4 blocks、8 epochs 的 RGB-only、2D-oracle 和 full-3D AdaLN-only 已完成。反事实配对成员使用相同初始扩散噪声，路径模式只按 XY 是否穿过障碍 footprint 判定，不使用高度参与分类。

结果见 `outputs/height_core_e8.json` 和 `outputs/height_core_e8.md`。full-3D 的 paired-test 成功率为 0.404，高于 2D-oracle 的 0.317，但路径模式准确率仅为 0.533，尚不足以证明模型可靠掌握高度驱动的 over/around 切换。因此在扩大到多种子和门控消融前，应先改善 full-3D 的路径学习。
