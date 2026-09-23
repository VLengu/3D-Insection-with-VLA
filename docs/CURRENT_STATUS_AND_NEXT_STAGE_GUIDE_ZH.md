# 当前工作总结与下一阶段指导报告

更新日期：2026-09-23
当前代码基线：`main@59153b2380729b44ce964980174232f2a53514db`

> 本文是从 Mac 构建验证阶段转入正式环境/GPU 阶段的主交接文档。
> 新任务应先阅读本文，再根据文中链接查阅单项验收报告。

---

## 1. 一句话状态

项目已完成第一阶段保护，并在 Mac 上打通了第二阶段的几何、MuJoCo、
LeRobotDataset、评价器、结构化 2D/3D token、VL/A 注入边界和 Tiny 训练链路；
GPU 交接包的 Mac 静态部分已完成，但正式复杂环境、正式数据和真实
SmolVLA GPU 适配尚未完成。

---

## 2. 当前研究问题

第二阶段只回答一个受控问题：

> 在同一个小型 VLA、同一批 RGB/语言/机器人状态和同一训练预算下，
> 结构化 3D 信息相比 2D 信息是否改善操作，且它进入 VLM contextual
> stream 与只进入 action expert 时的作用是否不同？

固定的 2×2 对照是：

| 配置 | 结构化信息 | 注入位置 |
|---|---|---|
| VL-2D | 2D | VLM contextual stream |
| VL-3D | 3D | VLM contextual stream |
| A-2D | 2D | action expert |
| A-3D | 3D | action expert |

主分析量：

```text
3D@VL = metric(VL-3D) - metric(VL-2D)
3D@A  = metric(A-3D)  - metric(A-2D)
注入位置交互 = 3D@VL - 3D@A
```

本阶段不研究新的 3D encoder、点云网络、双侧联合注入、额外 route head、
从零预训练 VLA，也不重启 A1b/A-prime 架构搜索。

---

## 3. 已完成的工作

### 3.1 第一阶段：完成并冻结

第一阶段的代码、数据、checkpoint 和结果已建立 checksum 保护。
当前校验结果：

```text
STAGE1_FREEZE_VERIFY=PASS
files=303
bytes=5,224,662,284
```

第一阶段的正式 paired-test 结果：

| 条件 | Paired decisions differ / 配对决策差异率 |
|---|---:|
| 2D-oracle | `0.000 ± 0.000` |
| Full-3D | `0.226 ± 0.031` |
| Height-only | `0.239 ± 0.026` |

支持的结论：朴素 3D 条件确实会改变动作决策，但当前路径响应主要可由
高度维度解释，不能表述为已形成完整 3D 理解。

### 3.2 M1：几何和数据契约通过

- 构建 Dual-Clearance 反事实 pair；
- 同一 pair 共享平面布局、语言、初始机器人状态和策略 RGB；
- 仅隐藏 3D 物理关系变化，使 A/B 分别只能走 over/side；
- Tiny 共 44 pair / 88 scene；
- 2D 快捷分类为随机水平，3D relational oracle 可完全判断路线。

### 3.3 M2/M2.5：MuJoCo 环境、专家轨迹和视觉质量通过

- Franka Panda、7D 动作、抓取—搬运—放置链路完整；
- 88/88 正确专家 episode 物理回放成功；
- 错误路线会产生物理失败；
- 44/44 pair 的初始策略 RGB 字节级相同；
- 44/44 pair 通过关键场景元素可见性检查。

### 3.4 LeRobotDataset v3 往返通过

- 44 pair，88 episode，6,998 frame；
- RGB、数值、episode 边界和动作逐帧精确回读；
- 6,998/6,998 frame 通过；
- 88/88 episode 从导出数据重放成功；
- pair ID、route label 和成功标签仅在评价 sidecar，不进入模型特征。

### 3.5 Gate M3：统一评价器通过

- 正确专家：88/88 episode 成功；
- 错误路线：88/88 被拒绝；
- fixed-over 和 fixed-side：episode success 各 50%，pair success 均 0%；
- 使用连续扫掠碰撞检测，避免离散时间点漏检。

### 3.6 Gate M4：模型边界通过

- 2D/3D 统一为 6 个 typed token，raw width 均为 8；
- `hidden_dim=128` 时两种模态均为 38,272 个 projector 参数；
- VL 侧 oracle 变化会先改变 contextual features；
- A 侧 oracle 变化不改变 VLM contextual features，只改变动作侧；
- TinyContractVLA 四配置均完成 forward/backward、checkpoint 往返和 MPS 小样本过拟合；
- 当时项目的 46 项 Stage 2 回归测试全部通过。

### 3.7 Gate M5-Mac：GPU 交接包静态部分通过

- 固定 LeRobot `0.6.1`/commit 与 SmolVLA snapshot；
- 四组配置只允许实验处理差异；
- Formal 配置硬性拒绝 Tiny 路径；
- train/val 与 paired-test/OOD 实现访问隔离；
- 测试解锁绑定 checkpoint 和 dataset manifest 的 SHA-256；
- 输出目录禁止覆盖；
- Dockerfile、Linux 安装、preflight 和分阶段脚本已准备。

---

## 4. 非常重要的数据定位

### 4.1 当前 Tiny 数据是什么

`stage2/generated_data/lerobot_tiny` 只是 **pipeline validation data**，用于证明：

- 仿真状态能转换成 RGB、state、action、2D 和 3D oracle；
- LeRobotDataset 能正确存储和读取；
- 四种注入配置具有一致的形状和预算；
- 训练、评价和测试锁的工程逻辑正确。

### 4.2 Tiny 数据不是什么

它不是：

- 正式 SmolVLA 训练数据；
- 第二阶段论文的主实验数据；
- 用来选择正式 checkpoint 的 validation set；
- 能独立证明通用 VLA 能力的 benchmark。

GPU 阶段必须从后续确定的更复杂环境重新生成或导出 Formal 数据。

### 4.3 Formal 3D 信息如何获得

正式仿真数据的 3D 不由 SmolVLA 从 RGB 中猜测，而由正式仿真环境的同一时刻状态生成：

```text
正式仿真器状态
  → 物体/目标/障碍/机器人位姿
  → 统一坐标系与单位
  → 同源 2D 投影与 3D 结构化字段
  → 固定 6-token 合同
  → VL 侧或 A 侧注入
```

主实验先用 oracle 3D，将“3D 信息是否有效”与“3D 感知器是否准确”分开。
Noisy 3D 是后续鲁棒性消融；estimated 3D/RGB-D 属于再后一阶段。

---

## 5. 当前尚未完成的工作

1. 尚未选定最终 Formal 操作环境/数据集；
2. 尚未生成 Formal train/val/paired-test/OOD 数据；
3. 尚未在 Linux/CUDA 加载固定的 SmolVLA checkpoint；
4. 项目内的 SmolVLA 真实 VL/A adapter 尚未实现并通过 G0；
5. 尚未测量显存、step time、吞吐和实际 GPU 成本；
6. 尚未进行 SmolVLA smoke、overfit、pilot 或正式三种子训练；
7. paired-test/OOD 尚未解锁，也不应现在解锁。

因此当前不能声称“Gate M5 整体完成”或“第二阶段实验完成”。

---

## 6. 下一阶段的第一个任务：Formal Environment/Data Gate

下一个任务不应直接开始大规模训练，而应先用小规模 probe 选定正式环境。

### 6.1 硬性准入标准

候选环境必须同时满足：

1. 是真正的 3D 机器人操作，而非二维点运动；
2. 具有抓取、搬运、障碍/遮挡、放置等多步骤行为；
3. 有语言任务描述；
4. 可取得可审计的物体、目标、障碍和机器人 3D 真值；
5. 能构造只改变隐藏 3D 关系的成对场景；
6. 一条固定路线不能在 pair 两个成员上都成功；
7. 有可重现的专家轨迹或数据采集办法；
8. 能闭环评价抓取、碰撞、掉落、放置和路线决策；
9. 可转换为 LeRobotDataset v3 或有稳定的适配路径；
10. 授权、安装、存储和训练成本可接受。

### 6.2 候选类型

下一任务应至少审计三类方案，不应直接默认某一个：

- 标准语言操作 benchmark 中可改造为 paired-3D 的子任务；
- 可读取 MuJoCo/物理真值的标准化操作环境；
- 将当前 Dual-Clearance 逻辑迁移到更复杂、更丰富的机器人场景。

标准 benchmark 提供外部说服力，自建 paired task 提供严格因果控制；
最终可能需要“一个主 paired 环境 + 少量标准任务外部检查”，但必须经 probe
后再决定。

### 6.3 候选环境 probe

对每个候选方案只做 5–10 个 episode 的小型证明，检查：

- 环境能否 reset/step/render；
- RGB、语言、state、action 形状和坐标系；
- 2D/3D 字段能否从同一仿真状态导出；
- 专家或 scripted policy 能否生成完整成功轨迹；
- 错误 3D 决策是否真的造成物理失败；
- A/B pair 是否避免 RGB、文本、顺序和 episode 长度泄漏；
- 数据量、生成速度和预计 GPU 存储。

### 6.4 Environment/Data Gate 交付物

完成后必须产出：

1. 候选环境对比表和最终选择理由；
2. Formal 任务和 pair 生成规则；
3. 坐标系、单位、2D/3D 字段和 action 定义；
4. train/val/paired-test/OOD 按 layout/task 分组的划分方案；
5. 小型 probe 数据、回放视频/图像和评价结果；
6. Formal 数据量、生成时间、存储和预计训练成本；
7. 一份冻结的 `dataset-manifest.sha256`。

---

## 7. Formal 数据集的建议划分

具体数量由 environment probe 决定，但划分逻辑必须先固定：

| split | 用途 | 训练期可见 | 分组原则 |
|---|---|---:|---|
| train | 优化参数 | 是 | pair 两成员同 split |
| val | checkpoint/超参数选择 | 是 | layout/task 与 train 分组隔离 |
| paired_test | 正式同分布因果比较 | 否 | 锁定，选完 checkpoint 只评一次 |
| geometry_ood | 新物体/障碍几何 | 否 | 几何家族隔离 |
| camera_ood | 新相机位姿 | 否 | 相机簇隔离 |
| appearance_ood | 纹理/背景/光照变化 | 否 | 外观参数隔离 |
| compound_ood | 多因素组合 | 否 | 不用于任何调参 |

禁止随机逐帧切分。必须按任务、layout 或场景家族分组，避免相邻帧和
pair 成员跨 split 泄漏。

---

## 8. 环境选定后的 GPU 执行顺序

### G0：真实 Linux/CUDA 预检与 adapter

1. 记录 GPU、VRAM、driver、CUDA、PyTorch 和 Linux 版本；
2. 使用固定 revision 加载 SmolVLA base；
3. 核对 VLM 和 action-expert 实际 hidden width；
4. 在 `stage2_gpu/adapters/` 实现项目内 adapter，不修改 `site-packages`；
5. 四配置各运行一个 batch forward/backward；
6. 重做 VL/A 反事实边界 probe；
7. 记录总/可训练参数、峰值显存、step time 和吞吐；
8. 实测后再固定 batch size、梯度累积和容器 image。

### G1：500-step smoke

- 四配置、单种子；
- 检查 loss 下降、checkpoint 保存/恢复和 rollout 能否执行；
- 只修复接口和明确 bug，不看 locked test。

### G2：2,000-step 小样本过拟合

- 证明四种配置都具有学习能力；
- 若基础抓取和放置都不能学会，停止进入 pilot。

### G3：5,000–10,000-step pilot

- 四配置、单种子；
- 只使用 train/val 决定训练步数、checkpoint 规则和统一训练预算；
- 检查 zero/shuffle/pair-swap 3D 的初步响应；
- G3 后冻结协议。

### G4：正式 2×2 三种子实验

```text
VL-2D × seeds 42/43/44
VL-3D × seeds 42/43/44
A-2D  × seeds 42/43/44
A-3D  × seeds 42/43/44
```

四组必须使用相同预训练 checkpoint、训练步数、batch、优化器、数据增强、
checkpoint 选择规则和 rollout 初始随机源。

### G5：checkpoint 冻结与锁定评估

1. 仅根据 validation 选择 checkpoint；
2. 记录 checkpoint SHA-256 和 dataset manifest SHA-256；
3. 生成 unlock audit record；
4. 一次性运行 paired-test 和 OOD；
5. 禁止根据测试结果回头调整架构或超参数。

### G6：外部有效性检查

在少量标准任务上比较最佳 3D 模型、同侧 2D 对照和未改造基线，
检查结构改造是否破坏普通操作能力。不扩张为完整排行榜实验。

---

## 9. 正式指标和决策规则

### 9.1 主指标

| English | 中文 | 作用 |
|---|---|---|
| Paired both tasks successful | 配对双任务同时成功率 | 一个 pair 的两个隐藏 3D 配置都成功 |
| Paired decisions differ | 配对决策差异率 | 只改变 3D 时是否改变决策 |
| Feasible-route accuracy | 可行路线准确率 | 是否选择唯一可行路线 |
| Task success | 任务成功率 | 抓取、无碰撞搬运并放置成功 |

还应报告 grasp success、transport collision rate、object drop rate、
placement success、endpoint distance、completion time 和 path efficiency。

### 9.2 因果诊断

- Zero oracle；
- Shuffle oracle；
- Pair swap；
- 只替换 Z/高度字段；
- 固定 RGB、语言、初始机器人状态和策略随机源；
- A 侧 VLM feature 不变性；
- VL 侧 contextual feature 对 3D 的响应。

不能只根据 task success 声称模型使用了 3D。

---

## 10. 用户在什么时候需要提供信息

当前不需要用户做额外操作。进入 GPU/G0 前，需要一次性确认：

1. GPU 型号与显存；
2. Linux 发行版和可用 CUDA/driver；
3. 机器的连接方式；
4. 数据传输方式：直接拷贝、私有 Hugging Face dataset 或其他对象存储；
5. 可接受的最大 GPU 时长/预算。

在 Formal Environment/Data Gate 需要在多个候选中做最终取舍时，应先提供对比证据和
成本表，再请用户确认，不应只给环境名称让用户盲选。

---

## 11. 下一任务的建议执行指令

新任务应按以下顺序工作：

1. 阅读本报告、`stage2_gpu/HANDOFF_CHECKLIST_ZH.md` 和 M5 报告；
2. 运行 `git status` 并确认基线 commit；
3. 运行 Stage 1 freeze verify；
4. 不要修改或重新训练第一阶段；
5. 不要将 Tiny 作为 Formal GPU 数据；
6. 先完成 Formal Environment/Data Gate 的候选审计和 5–10 episode probe；
7. 环境选定后冻结 schema、split、generator 和 manifest；
8. 获得 GPU 信息后才执行 G0；
9. G0 通过前，`stage2_gpu.runtime` 应保持 fail closed；
10. 任何时候都不允许训练阶段读取 paired-test/OOD。

---

## 12. 必读文件

| 文件 | 用途 |
|---|---|
| `stage1_freeze/verify.py` | 第一阶段保护校验 |
| `docs/STAGE2_MAC_BUILD_VALIDATION_PLAN_ZH.md` | 第二阶段总实验协议 |
| `stage2/GATE_M4C_TINY_CONTRACT_VLA_REPORT_ZH.md` | Mac Tiny 端到端训练边界 |
| `stage2/GATE_M5_GPU_HANDOFF_REPORT_ZH.md` | Gate M5 当前状态 |
| `stage2_gpu/README_GPU_ZH.md` | GPU 交接包使用方法 |
| `stage2_gpu/HANDOFF_CHECKLIST_ZH.md` | M5-Mac/M5-GPU 检查项 |
| `stage2_gpu/manifests/upstream.json` | LeRobot/SmolVLA 固定版本 |
| `docs/METRICS_GLOSSARY_ZH.md` | 指标中英对照 |

---

## 13. 当前最终交接结论

Mac 阶段已经完成它应该完成的核心工作：不是证明正式 VLA 性能，而是将
研究问题、数据契约、反事实 pair、评价器、2D/3D 公平投影、VL/A 边界、
测试锁和 GPU 交接格式全部做成可验证的工程合同。

下一阶段的主线只有两件事：

```text
先选定有说服力的 Formal 复杂环境并冻结数据
                         ↓
再在 Linux/CUDA 上完成 SmolVLA adapter、分级训练和锁定评估
```

只要保持这个顺序，后续结果就能明确区分：环境是否有 3D 必要性、模型是否
使用了 3D，以及 3D 注入位置是否真的影响了操作能力。
