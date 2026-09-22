# 第二阶段：Mac 本地构建、验证与 GPU 实验交接方案

更新日期：2026-09-22

## 0. 报告结论

第二阶段采用“两层证据、三档数据、分级放行”的路线：

1. **主证据**来自一个可审计的、严格成对的三维机械臂搬运任务；
2. **外部证据**来自少量标准 LIBERO-Spatial 任务，用于验证改造后的模型没有只适用于自建环境；
3. 当前 Mac 只负责环境逻辑、数据格式、评估器、注入边界和小模型接口验证；
4. SmolVLA 正式训练、微调、大规模推理和闭环评估全部留到 Linux GPU 阶段；
5. GPU 阶段开始前，数据和代码协议必须冻结，不能一边看 paired-test 一边修改模型。

本阶段的核心不是追求最大的环境，而是建立一条可解释、可复现、能移交 GPU
运行的实验链路。

一句话原则：

> 让三维决策保持简单且可证伪，让视觉场景和机器人执行具有足够复杂度。

---

## 1. 第二阶段要回答的问题

正式研究问题定义为：

> 在同一个小型 VLA 中，使用完全相同的 oracle 空间信息和训练预算时，把
> 结构化 3D tokens 放入 VLM contextual stream，是否比只送入 action expert
> 更容易形成稳定、正确的三维条件化动作？

主实验为固定的 2×2：

| 模型 | 结构化信息 | 注入位置 |
|---|---|---|
| VL-2D | 二维状态 | VLM contextual stream |
| VL-3D | 二维状态加三维状态 | VLM contextual stream |
| A-2D | 二维状态 | action expert |
| A-3D | 二维状态加三维状态 | action expert |

需要分别回答两个问题：

1. **3D 信息效应**：VL-3D 相对 VL-2D、A-3D 相对 A-2D 是否改善；
2. **注入位置效应**：VL 侧的 3D 增益是否大于 A 侧的 3D 增益。

形式上重点分析：

```text
3D@VL = metric(VL-3D) - metric(VL-2D)
3D@A  = metric(A-3D)  - metric(A-2D)

位置交互效应 = 3D@VL - 3D@A
```

第二阶段不研究：

- 新型 3D encoder；
- 深度估计、点云重建或 3D 检测误差；
- VL+A 双侧联合注入；
- route head、额外空间损失或复杂融合结构；
- 从零预训练 VLA；
- A1b、A-prime 的继续优化。

---

## 2. 为什么不能直接使用当前 Height World 作为最终数据集

当前 Height World 非常适合作为快速单元测试，但不足以作为第二阶段最终主证据。

主要原因：

1. 当前动作是抽象的 24 步 `(dx,dy,dz)`，不是真实机械臂控制；
2. 没有语言模块、夹爪状态、关节或末端执行器状态；
3. 高障碍的绕行路线用于低障碍时仍然合法，因此存在“永远绕行”的通用解；
4. Success 不能证明 3D 是完成任务所必需的；
5. 专家标签与 footprint 路径判定不完全一致；
6. 数据验证和模型评估的碰撞检测精度不同。

因此当前数据保留为 **L0 接口数据集**，用于快速验证转换、张量形状、配对采样、
注入逻辑和评价代码，不用于第二阶段最终结论。

---

## 3. 总体证据结构

```text
L0：现有 Height World
    目的：Mac 上快速验证代码和数据契约
            ↓
L1：自建 Dual-Clearance Robot Transport
    目的：第二阶段主因果证据
            ↓
L2：少量 LIBERO-Spatial 标准任务
    目的：外部有效性和退化检查
```

三层承担不同角色，不能相互替代：

| 层级 | 是否是主结果 | 主要价值 | 不承担的结论 |
|---|---:|---|---|
| L0 Height World | 否 | 快速、便宜、可审计 | 不代表真实 VLA 操作 |
| L1 自建机器人任务 | 是 | 严格反事实、真实控制、闭环成功 | 不代表通用机器人能力 |
| L2 LIBERO 子集 | 辅助 | 标准环境兼容性和外部有效性 | 不单独证明 3D 因果性 |

---

## 4. 主环境：Dual-Clearance Robot Transport

### 4.1 任务描述

机器人需要：

1. 抓取桌面上的目标物体；
2. 在两个候选通道中选择一条；
3. 搬运过程中保持抓取；
4. 避免与障碍物碰撞；
5. 把物体放入目标区域。

固定语言指令示例：

```text
Pick up the red object and place it in the blue target area.
```

训练时可以使用 4 至 8 个等义改写，但任何指令都不能透露高度、净空状态或
正确路线。

### 4.2 互斥路线设计

场景具有两条二维投影不同的候选路线：

- `route_over`：经过中央障碍上方；
- `route_side`：经过侧方门洞或隧道。

每个基础布局生成严格配对的两个三维配置：

#### 配置 A：over-only

- 中央障碍足够低，可以安全抬升通过；
- 侧方门洞的垂直净空不足，携带物体无法通过；
- 唯一可行路线是 `route_over`。

#### 配置 B：side-only

- 中央障碍过高，受机器人抬升上限限制，无法从上方通过；
- 侧方门洞具有足够净空；
- 唯一可行路线是 `route_side`。

这两个配置必须保持：

- 起点、目标、物体二维位置相同；
- 障碍物二维 footprint 相同；
- 初始机器人状态相同；
- 语言指令相同；
- 主相机初始 RGB 在隐藏高度版本中相同或达到规定的像素容差；
- 仅允许预先登记的 Z 轴几何和净空字段变化。

这样固定选择 over 或固定选择 side 的策略理论上最多只能完成一半配对，消除
当前环境中的通用绕行解。

### 4.3 难度应该放在哪里

科学问题保持简单：只判断哪条三维路线可行。

执行过程保持真实：

- 机械臂抓取；
- 末端位姿连续控制；
- 夹爪开合；
- 物体动力学；
- 多阶段动作；
- 碰撞和掉落；
- 精确放置。

视觉复杂度逐步增加：

1. 物体与目标位置随机化；
2. 障碍物尺寸和通道宽度随机化；
3. 材质、光照和背景随机化；
4. 无关干扰物；
5. 相机位置和偏航小幅随机化；
6. OOD 中再扩大布局、视角和几何范围。

不能同时在第一版加入多机器人、多物体长序列、移动障碍和复杂语言规划。
这些因素会把“注入位置”问题淹没在整体控制难度中。

### 4.4 建议的动作和观测

面向 LIBERO/Franka 风格接口：

```text
observation.images.image     主相机 RGB，建议 256×256
observation.images.image2    可选腕部 RGB
observation.state            8 维机器人本体状态
action                       7 维连续控制
task                         自然语言指令
```

7 维动作采用：

```text
[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
```

第一版主实验建议只使用主相机。腕部相机可以在后续作为单独消融加入，避免它
在决策发生前泄露隐藏高度。

### 4.5 结构化 oracle 字段

#### oracle_2d

至少包括：

```text
object_xy
goal_xy
central_obstacle_xy_and_footprint
side_gate_xy_and_footprint
robot_base_xy_or_relevant_planar_state
valid_mask
```

#### oracle_3d

在相同二维字段基础上增加：

```text
object_xyz_and_extent
goal_xyz_and_extent
central_obstacle_xyz_and_extent
side_gate_xyz_and_clearance
robot_reachable_z_range
carried_object_clearance_requirement
camera_extrinsics_if_used
valid_mask
```

模型输入不得直接包含：

- `route_label`；
- `over_feasible`；
- `side_feasible`；
- 专家 waypoint；
- success 标签；
- 任何由正确答案直接编码得到的二值字段。

几何关系必须由模型从基础 oracle 字段中计算或学习，不能把答案伪装成 3D
状态输入。

---

## 5. 数据集格式

### 5.1 标准格式

最终数据采用 LeRobotDataset v3 风格组织：

- 图像或视频；
- Parquet 中的状态、动作、时间戳和扩展 oracle 字段；
- episode 元数据；
- task 语言文本；
- 全局统计量和特征 schema。

本地构建阶段优先使用 `use_videos=False` 的图像形式，降低视频编码和解码带来的
额外变量。协议冻结后再生成或转换为 MP4 视频存储版本。

每个 frame 至少包含：

```text
observation.images.image
observation.state
action
timestamp
frame_index
episode_index
task
oracle.z2d
oracle.z3d
```

每个 episode 的独立元数据至少包含：

```text
pair_id
pair_member                 over-only / side-only
base_layout_id
seed
split
route_label
geometry_parameters
expert_valid
initial_rgb_hash
generator_version
simulator_version
```

`route_label` 和评估元数据可以存储，但数据加载器必须保证它们不会进入策略模型。

### 5.2 配对关系

一个 `pair_id` 必须恰好对应两个 episode：

```text
pair_id = K
  ├── K-A：over-only
  └── K-B：side-only
```

训练时可把两条 episode 放在相邻 batch 或同一 paired sampler 中，但这不是必须的。
评估时必须使用相同的模型随机源或固定 action-noise seed，减少采样噪声。

---

## 6. 三档数据规模

所有数量均以“反事实配对”为单位，一个 pair 包含两个 episode。

### 6.1 Tiny：Mac 持续集成数据

| 划分 | pairs | episodes |
|---|---:|---:|
| train | 16 | 32 |
| val | 4 | 8 |
| paired_test | 8 | 16 |
| 每个 OOD | 4 | 8 |

用途：

- 每次修改后的快速生成；
- schema 和 round-trip 检查；
- evaluator 单元测试；
- 配对不变量检查；
- 小模型端到端冒烟测试。

Tiny 数据可以进入仓库或由固定 seed 随时重建。

### 6.2 Pilot：GPU 预实验数据

| 划分 | pairs | episodes |
|---|---:|---:|
| train | 250 | 500 |
| val | 50 | 100 |
| paired_test | 100 | 200 |
| 每个 OOD | 50 | 100 |

用途：

- 500 至 10,000 步预实验；
- 估计显存、吞吐、学习曲线和过拟合风险；
- 检查任务是否过难；
- 决定正式数据量和训练步数。

### 6.3 Formal：正式实验数据

| 划分 | pairs | episodes |
|---|---:|---:|
| train | 2,000 | 4,000 |
| val | 250 | 500 |
| paired_test | 500 | 1,000 |
| height_ood | 250 | 500 |
| layout_ood | 250 | 500 |
| camera_ood | 250 | 500 |
| compound_ood | 250 | 500 |

这是初始目标，不是未经测量就必须达到的硬指标。Pilot 阶段如果显示 500 至
1,000 个 train pairs 已经饱和，可以降低正式规模；如果学习仍受数据限制，才
增加到 3,000 至 5,000 train pairs。

---

## 7. 数据划分原则

不能按 frame 随机切分，也不能让同一个基础布局的两个配对成员进入不同划分。

### 7.1 分组单位

划分以 `base_layout_id` 为最小分组单位：

```text
同一 base_layout_id 的：
  所有 pair 成员
  所有语言改写
  所有相邻轨迹变体
必须属于同一个 split
```

### 7.2 ID 划分

- `train`：训练范围内的布局、几何和相机；
- `val`：独立布局，用于 checkpoint 选择和早停；
- `paired_test`：独立布局，只在 checkpoint 锁定后打开一次。

### 7.3 OOD 划分

- `height_ood`：中央障碍高度和门洞净空超出训练范围；
- `layout_ood`：障碍尺寸、通道位置、起终点范围超出训练范围；
- `camera_ood`：未见过的相机位置或偏航；
- `compound_ood`：高度与布局同时变化，仅用于最终压力测试。

正式结论以 paired_test 为主，OOD 结果用于讨论泛化，不能取代 ID 因果比较。

### 7.4 防止选择偏差

- paired_test 生成后立刻写入 manifest 和 checksum；
- 训练脚本默认拒绝读取 paired_test；
- checkpoint 只由 val 指标选择；
- paired_test 不参与超参数选择；
- 如果测试协议发生变化，必须增加数据版本号，而不是覆盖旧结果。

---

## 8. Mac 本地阶段的工程目录建议

建议在现有项目中新增独立的第二阶段目录，避免污染第一阶段：

```text
stage2/
  configs/
    dataset_tiny.yaml
    dataset_pilot.yaml
    dataset_formal.yaml
    experiment_2x2.yaml
  envs/
    dual_clearance/
      geometry.py
      task.py
      expert.py
      renderer.py
      collision.py
  data/
    schema.py
    generate.py
    split.py
    export_lerobot.py
    validate.py
  models/
    structured_projector.py
    vl_injection.py
    action_injection.py
    boundary_probe.py
    tiny_contract_vla.py
  eval/
    rollout.py
    metrics.py
    paired_metrics.py
  tests/
    test_geometry.py
    test_pair_invariants.py
    test_collision.py
    test_dataset_roundtrip.py
    test_no_label_leakage.py
    test_injection_boundary.py
    test_metrics.py
  manifests/
  README.md
```

第一阶段的 `uwm3d/`、`data_height/` 和 `checkpoints_height/` 保持冻结。

---

## 9. Mac 本地构建步骤

### 步骤 1：版本与环境隔离

目标：第二阶段可以独立重建，不依赖当前未记录的环境状态。

具体工作：

1. 为第二阶段建立独立 Python 环境；
2. 固定 Python、PyTorch、LeRobot、Transformers 和模拟器版本；
3. 保存 lockfile 和环境报告；
4. 初始化 Git 或以其他方式保存基准快照；
5. 第一阶段目录改为只读约定，不再原位修改。

验收：在一个新目录中能根据 lockfile 安装最小开发依赖并运行单元测试。

### 步骤 2：先定义数据契约

目标：在环境实现前锁定策略真正看到什么。

具体工作：

1. 定义 observation、action、oracle 和 metadata schema；
2. 明确归一化单位、坐标系、旋转表示和 action convention；
3. 明确哪些字段只供 evaluator 使用；
4. 编写 schema validator；
5. 编写最小伪数据和 round-trip 测试。

验收：错误 shape、dtype、缺失字段、非法范围和标签泄漏都能被自动拒绝。

### 步骤 3：实现与模拟器无关的几何核心

目标：先证明任务在数学上成立，再接物理模拟器。

具体工作：

1. 生成基础二维布局；
2. 从同一布局生成 over-only 和 side-only 三维配置；
3. 计算机器人和携带物体的净空条件；
4. 实现解析 waypoint 专家；
5. 实现连续线段或 swept-volume 碰撞检测；
6. 检查两条互斥路线是否分别可行。

验收：每个 pair 恰好一条 route_over 可行、恰好一条 route_side 可行，不能出现
两条都可行或两条都不可行。

### 步骤 4：实现 Tiny 环境和专家回放

目标：在 Mac 上完成完整环境逻辑验证。

具体工作：

1. 接入 Mac 可运行的 MuJoCo 或轻量模拟层；
2. 实现 reset、step、render、success 和 collision；
3. 专家动作从抓取开始，完整执行到放置结束；
4. 生成 Tiny 数据；
5. 回放所有专家 episode。

验收：

- 专家成功率必须为 100%；
- 无碰撞、无掉落、无越界；
- 配对成员初始状态满足规定的不变量；
- 固定 over-only 和固定 side-only 基线都不能超过约 50% 的配对成功。

### 步骤 5：导出 LeRobot 格式

目标：GPU 端无需再次解释自定义 NPZ。

具体工作：

1. 将 Tiny 数据写成 LeRobotDataset；
2. 写入 task、episode、pair 和生成器版本元数据；
3. 关闭 writer 并 finalize；
4. 重新加载全部 episode；
5. 验证 frame 数、时间戳、视频/图像、统计量和 episode 边界；
6. 比较导出前后的动作与 oracle 字段。

验收：导出再读取后，数值字段在容差内相同，episode 和 pair 数量完全一致。

### 步骤 6：实现统一 structured projector

目标：四个模型共用同一套输入规范。

具体工作：

1. 把 2D 和 3D 字段整理为固定数量的 typed tokens；
2. 2D/3D 使用相同 hidden dimension 和 token 数量；
3. 对缺少的三维字段使用明确 mask，而不是改变网络宽度；
4. projector 参数量尽可能一致；
5. 输出 token mask 和类型 embedding。

验收：2D/3D projector 输出 shape 完全相同，参数量差异在预先规定范围内。

### 步骤 7：实现严格的 VL/A 注入边界

#### VL 侧

结构化 tokens 必须在 VLM contextualization 完成前加入视觉/语言 token stream，
并进入相同的上下文计算。action expert 接收已经被这些 tokens 影响的 contextual
features。

#### A 侧

VLM 输入中不能出现结构化 tokens。VLM 输出后，结构化 tokens 才能进入 action
expert。对于相同 RGB 和语言，A-2D 与 A-3D 的 VLM contextual features 应完全
一致或在数值容差内一致。

验收测试：

- hook 记录结构化 tokens 首次出现的层；
- 检查 attention mask；
- 检查 projector 梯度非零；
- 检查 A 侧 VLM features 不随 oracle 改变；
- 检查 VL 侧 features 会随 oracle 改变；
- 输出各模型总参数和可训练参数；
- 禁止通过 action normalization、padding 或 task text 泄漏模型类别。

### 步骤 8：使用 Tiny Contract VLA 做端到端测试

Mac 阶段不要求运行完整 SmolVLA。实现一个接口相同、规模很小的
`TinyContractVLA`：

- 小型视觉 patch encoder；
- 简单语言 embedding；
- 两至四层 contextual transformer；
- 小型 flow/diffusion action expert；
- 与正式模型相同的 structured projector 和 VL/A 注入 API。

它只用于验证：

- dataloader 到模型的字段连接；
- 四种配置能运行 forward/backward；
- paired sampler 正确；
- checkpoint 保存和恢复；
- evaluator 能消费模型输出；
- 注入边界测试可自动执行。

Tiny 模型结果不能写入正式实验表。

### 步骤 9：做数据泄漏与捷径检查

必须至少包含：

1. `RGB + 2D` 简单分类器预测 route，准确率应接近机会水平；
2. `3D oracle` 简单分类器应接近 100%，证明 oracle 中确有决定信息；
3. 仅使用 episode 长度、语言文本、文件顺序、pair_id 不能预测 route；
4. 打乱 3D 后，route 可预测性应消失；
5. 检查训练/验证/测试的 `base_layout_id` 无交集；
6. 对初始 RGB 做 hash 或像素差统计；
7. 检查动作长度和专家路径形态不会在输入阶段提前泄漏标签。

### 步骤 10：冻结实验协议并打包 GPU 交接物

交接物必须包括：

```text
代码快照或 Git commit
环境 lockfile
容器或 Linux 安装脚本
Tiny 数据
Pilot 数据或生成命令
Formal 数据生成配置
数据 manifest 和 checksum
四模型配置文件
训练命令模板
评估命令模板
测试报告
已知问题列表
```

---

## 10. Mac 阶段必须通过的验证门

### Gate M0：第一阶段保护

- 第一阶段关键 JSON、checkpoint 和报告 checksum 已记录；
- 第二阶段代码不覆盖第一阶段文件；
- 主报告仍可访问。

### Gate M1：几何有效

- 所有 pair 满足互斥可行性；
- 专家轨迹通过连续碰撞检测；
- 起终点和动作范围合法；
- 固定路线不存在通用解。

### Gate M2：数据有效

- schema 全部通过；
- pair 和 split 分组正确；
- LeRobot round-trip 正确；
- 数据无标签泄漏；
- 生成结果可由 seed 重现。

### Gate M3：评价有效

- 构造的人工轨迹能触发预期成功/失败；
- 连续碰撞和离散点碰撞差异已消除；
- paired 指标与手工计算一致；
- metric 名称有中英对照。

### Gate M4：模型边界有效

- VL/A 注入位置由自动测试证明；
- 四模型输入 shape 相同；
- 参数量和可训练预算报告完整；
- A 侧 VLM 不受 oracle 改变；
- VL 侧上下文受 oracle 改变；
- Tiny 模型四配置均可完成 forward/backward。

### Gate M5：GPU 可执行包完整

- Linux 安装脚本可解析；
- 所有命令只依赖配置文件，不依赖手工修改；
- paired-test 默认锁定；
- 数据版本与代码版本绑定；
- 实验输出目录不会相互覆盖。

只有 M0 至 M5 全部通过，才进入 GPU 阶段。

---

## 11. GPU 阶段计划

Mac 阶段结束并不直接开始正式三种子训练。GPU 端仍按以下顺序放行。

### G0：Linux 环境预检

1. 安装锁定版本；
2. 加载 Tiny 数据；
3. 加载 SmolVLA 预训练权重；
4. 四种配置各执行一个 batch 的 forward；
5. 四种配置各执行一次 backward；
6. 记录显存峰值、step time 和吞吐。

### G1：500 步接口训练

四模型、单种子、Pilot 数据：

- 检查 loss 是否下降；
- 检查 checkpoint 是否可恢复；
- 检查 VL/A 边界 probe 在真实 SmolVLA 中仍成立；
- 检查评估 rollout 能运行。

### G2：2,000 步小样本过拟合

- 使用小规模数据；
- 目标不是泛化，而是证明四种模型都有学习能力；
- 如果基础抓取和放置都学不会，禁止进入正式比较。

### G3：5,000 至 10,000 步 Pilot

四模型、单种子：

- 判断数据量和训练步数是否合理；
- 选择统一训练预算；
- 检查是否存在明显的 3D 响应；
- 只使用 val 调整协议。

G3 完成后冻结所有超参数。

### G4：正式 2×2 三种子实验

```text
VL-2D × seeds 42/43/44
VL-3D × seeds 42/43/44
A-2D  × seeds 42/43/44
A-3D  × seeds 42/43/44
```

- 使用相同预训练 checkpoint；
- 相同 batch、优化器、训练步数和增强；
- 相同 checkpoint 选择规则；
- checkpoint 锁定后一次性运行 paired-test；
- 共享配对 rollout 初始随机源。

### G5：标准环境外部检查

只对关键模型运行 2 至 4 个 LIBERO-Spatial 任务：

- 最佳 3D 模型；
- 同侧 2D 对照；
- 必要时加入未修改的 SmolVLA 基线。

这一步检查模型改造是否造成普通操作能力退化，不扩张为完整 LIBERO 排行榜实验。

LIBERO 的 LeRobot 官方集成要求 Linux，因此该步骤只能在 GPU/Linux 阶段执行。

---

## 12. 训练策略与成本控制

### 12.1 不从头训练

使用 SmolVLA 预训练权重。第二阶段研究的是注入边界，不是 VLA 预训练方法。

### 12.2 第一轮冻结大部分骨干

推荐起始策略：

- 冻结视觉编码器的大部分参数；
- 冻结或只用低秩适配器更新 VLM；
- 四模型使用相同 action expert 微调策略；
- VL/A structured projector 参数量匹配；
- 报告总参数和可训练参数。

如果完全冻结 VLM 导致 VL tokens 无法被利用，则在不查看 paired-test 的前提下，
统一开放预先指定的上层 VLM blocks 或 LoRA，而不是只为某一模型单独调参。

### 12.3 分级预算

| 阶段 | 模型/种子 | 建议步数 | 是否允许修改协议 |
|---|---|---:|---:|
| G0 | 4×1 | 1 batch | 是，仅修接口 |
| G1 | 4×1 | 500 | 是 |
| G2 | 4×1 | 2,000 | 是 |
| G3 | 4×1 | 5,000–10,000 | 是，只看 val |
| G4 | 4×3 | 初始约 20,000 | 否 |

最终步数必须由 G3 学习曲线决定。不能因为官方示例使用某个步数就直接照搬。

### 12.4 实际成本公式

```text
GPU 小时
= 模型数 × 种子数 × 训练步数 × 实测每步秒数 ÷ 3600
```

正式 4×3×20,000 步共 240,000 steps：

- 1 秒/step：约 67 GPU 小时；
- 2 秒/step：约 133 GPU 小时；
- 4 秒/step：约 267 GPU 小时。

G0 必须记录真实 step time 和峰值显存，再决定 GPU 类型、batch size、梯度累积
和最终预算。

---

## 13. 正式评价指标

### 13.1 主指标

| English | 中文 | 作用 |
|---|---|---|
| Paired both tasks successful | 配对双任务同时成功率 | 一个 pair 的两种隐藏 3D 配置都成功 |
| Paired decisions differ | 配对决策差异率 | 只改变 3D 后是否改变路线 |
| Feasible-route accuracy | 可行路线准确率 | 选择唯一可行路线的比例 |
| Task success | 任务成功率 | 抓取、无碰撞搬运并正确放置 |

新的主任务中只有一条路线可行，因此 Task success 不再像第一阶段那样可以由
始终绕行策略取得。

### 13.2 质量指标

| English | 中文 |
|---|---|
| Grasp success | 抓取成功率 |
| Transport collision rate | 搬运碰撞率 |
| Object drop rate | 物体掉落率 |
| Placement success | 放置成功率 |
| Endpoint distance | 终点距离误差 |
| Completion time | 完成时间 |
| Path efficiency | 路径效率 |

### 13.3 因果诊断

- Zero oracle；
- Shuffle oracle；
- Pair swap；
- 只替换 Z 轴字段；
- 固定 RGB、语言、初始机器人状态和策略随机源；
- A 侧检查 VLM features 不变；
- VL 侧检查 contextual features 随 3D 改变。

这些诊断用于证明模型确实依赖正确的结构化信息，而不是利用文件顺序或其他捷径。

---

## 14. Checkpoint 和测试纪律

checkpoint 仅根据 validation 选择。建议采用预先固定的词典序规则：

```text
1. validation paired both tasks successful 最大
2. validation task success 最大
3. validation collision rate 最小
4. validation endpoint distance 最小
```

禁止：

- 用 paired-test 选择 epoch；
- 看到某个 seed 的测试结果后补训练；
- 为 VL-3D 和 A-3D 使用不同的调参预算；
- 因结果不理想而改变 route 定义；
- 只汇报最好 seed；
- 把普通随机 rollout 当作配对因果评估。

---

## 15. 推荐的实际执行顺序

### 当前 Mac 阶段

1. 保存第一阶段基准并隔离第二阶段目录；
2. 编写第二阶段 schema 与配置；
3. 实现 Dual-Clearance 几何生成器；
4. 实现互斥可行性和连续碰撞验证；
5. 实现解析专家；
6. 接入 Mac 可运行的轻量模拟；
7. 生成 Tiny 数据并完成专家回放；
8. 导出并重新读取 LeRobotDataset；
9. 实现 paired evaluator；
10. 实现统一 structured projector；
11. 实现 VL/A 两种注入 API；
12. 实现 Tiny Contract VLA；
13. 运行 M0 至 M5 全部测试；
14. 生成 Pilot 数据和 GPU 交接包。

### 后续 GPU 阶段

1. Linux 环境与 SmolVLA 单 batch 预检；
2. 500 步四配置 smoke；
3. 2,000 步过拟合测试；
4. 5,000 至 10,000 步单种子 Pilot；
5. 冻结协议；
6. 正式四模型三种子训练；
7. 锁定 checkpoint 后运行 paired-test 和 OOD；
8. 关键模型运行少量 LIBERO-Spatial；
9. 汇总均值、标准差、配对效应和失败案例；
10. 达到停止边界后结束第二阶段。

---

## 16. 第二阶段停止边界

完成以下内容即停止：

1. 一个明确区分 VLM 与 action expert 的小型 VLA；
2. 一个消除了固定路线通用解的三维机械臂任务；
3. VL-2D、VL-3D、A-2D、A-3D 四模型；
4. 三种子、统一预算、validation 选 checkpoint；
5. 一次锁定后的 paired-test；
6. 至少一个高度、布局和相机 OOD；
7. 少量标准 LIBERO-Spatial 外部检查；
8. 对“3D 是否有效”和“VL/A 哪侧更有效”给出有边界的结论。

不因阶段性结果不好而继续加入更复杂融合结构。若四个模型均失败，应首先判断：

- 专家数据是否足够；
- 基础抓取控制是否学会；
- oracle 是否真正进入预期网络层；
- 任务是否超出模型和预算能力；
- 评价器是否正确。

只有这些问题排除后，才能把失败解释为注入方法的限制。

---

## 17. 当前已知平台事实与参考

- 本机为 Apple Silicon，MPS 可用但无 CUDA；M3 环境已安装并锁定 LeRobot
  0.6.1、Datasets 4.8.5、PyTorch 2.11.0 等数据链路依赖。Transformers、
  Accelerate 和 PEFT 留到模型接口阶段按实际版本兼容性加入。
- LeRobotDataset v3 使用 Parquet 保存低维时序数据、MP4 保存视觉流，并通过
  metadata 保存 schema、统计量、任务和 episode 边界：
  <https://huggingface.co/docs/lerobot/lerobot-dataset-v3>
- SmolVLA 输入包括多相机图像、机器人状态和语言指令，VLM contextual features
  条件化 action expert：
  <https://huggingface.co/docs/lerobot/v0.4.3/en/smolvla>
- LeRobot 的 LIBERO 集成提供 8 维状态、双 256×256 相机和 7 维动作，但官方
  环境要求 Linux：
  <https://huggingface.co/docs/lerobot/libero>

因此，Mac 阶段的完成标准是“工程与科学协议已验证并可移交”，而不是在本机
复现最终 SmolVLA 训练结果。
