# UWM-3D视觉矫正：项目状态、结论与交接报告

更新日期：2026-09-19

> 本文是当前项目的主状态报告。若旧文档中的解释与本文冲突，以本文为准。

## 1. 当前项目要回答的问题

项目最初研究扩散动作模型中3D残差的注入时机，随后收敛为一个更清楚、
更有限的问题：

> 在RGB和二维几何已经存在的情况下，把额外3D状态通过与二维条件相同的
> 简单全局条件机制注入动作策略，会产生怎样的动作响应？3D信息被注入、
> 被模型读取，与形成可靠3D空间行为之间有多大差距？

当前正式比较不是“纯2D vs纯3D”，而是：

```text
RGB + 显式2D几何
        vs
RGB + 显式2D几何 + 高度/3D状态
```

纯3D或无RGB模型只适合作为“结构化状态是否充分”的辅助消融，不是主比较。

## 2. 当前模型的准确定位

当前实现不是完整VLA：它有视觉编码器和动作扩散策略，但没有语言模块，
视觉编码器也不是预训练VLM。更准确的名称是：

- V-A policy；
- VLA-like action diffusion model；
- 视觉编码后、策略侧条件化的动作模型。

核心信息流如下：

```text
RGB -> CNN VisualEncoder -> image token
                                  |
noisy actions -> action tokens -> diffusion Transformer -> actions
                                  ^
                         structured condition -> AdaLN
```

结构化2D/3D条件没有进入CNN，而是在视觉编码完成后调制联合策略Transformer。
因此当前主实验应归类为：

> post-vision policy-side global conditioning，中文可称“视觉编码后的策略/动作
> 侧全局条件注入”。

它更接近A侧注入，但AdaLN会调制联合Transformer中的image token与action
tokens，因此比“只修改最终动作头”稍宽。

历史模块的归类：

| 模块 | 注入位置 |
|---|---|
| 全局Z3D AdaLN | 策略/A侧为主的全局条件 |
| ThreeDCorrectionAdapter | 直接修正动作噪声，纯A侧 |
| route embedding | 全局动作模式条件，纯A侧 |
| A-prime action-to-geometry cross-attention | 动作token查询几何，纯A侧 |

当前项目没有完成真正的V/VL侧3D注入实验。

## 3. 代码与数据概况

### V1平面环境

- 64x64顶视RGB；
- 8步二维动作；
- 结构化条件包含起点、目标、相对位置和干扰物；
- 支持base、early、middle、middle-wide、late、all门控；
- 支持epsilon/v-prediction、目标遮挡、AdaLN和残差修正。

V1已冻结，只允许修复bug。规范见 `docs/V1_FREEZE.md`。

### V2 Height World

- 图像：`[3,64,64]`顶视RGB；
- 动作：24步`(dx,dy,dz)`；
- 轨迹：25个三维位置；
- 每个布局生成低/高障碍反事实配对；
- 配对成员具有字节级相同RGB和相同XY几何；
- 低障碍专家选择over，高障碍专家选择around；
- train 12,000条，val/height_ood/layout_ood/camera_ood/paired_test各1,500条；
- 所有数据通过配对、动作范围、专家碰撞和终点验证。

规范见 `docs/HEIGHT_WORLD_V2.md`，生成与验证入口为：

```bash
./venv/bin/python scripts/generate_height_data.py
./venv/bin/python scripts/validate_height_world.py
```

主要代码：

| 文件 | 作用 |
|---|---|
| `uwm3d/height_world.py` | V2数据、专家规划、碰撞定义 |
| `uwm3d/model.py` | 视觉编码、动作Transformer、AdaLN、历史A-prime模块 |
| `scripts/train_height_world.py` | V2训练与validation checkpoint选择 |
| `scripts/evaluate_height_checkpoints.py` | 锁定checkpoint后的统一评估 |
| `scripts/probe_route_control.py` | route强制干预诊断 |
| `docs/METRICS_GLOSSARY_ZH.md` | 指标中英对照 |

当前目录没有Git仓库，因此checkpoint与训练历史存在，但无法通过commit完整恢复
每次历史实验对应的源代码版本。

## 4. V1完成了什么

V1包含约97份训练历史，探索过模型容量、遮挡、v-prediction、注入时机、
AdaLN和残差强度。不同预算的探索不能混在一起排序，也没有建立“early注入
稳定最好”的结论。

V1最可信的结果是三种子结构化条件依赖诊断：

| 模型 | 原始成功率 | Zero后 | Shuffle后 | Conflict时偏向结构化目标 |
|---|---:|---:|---:|---:|
| no-3D | 0.333 | 0.333 | 0.333 | 0.000 |
| AdaLN-only | 0.897 | 0.000 | 0.038 | 1.000 |
| AdaLN+middle 0.025 | 0.998 | 0.000 | 0.043 | 1.000 |

支持的结论：模型确实因果性地使用结构化条件。

不支持的结论：V1动作和有效几何仍是二维的，因此不能据此声称模型完成3D
空间推理。

正式记录：`outputs/stage2_diagnostics.json`与
`outputs/stage2_diagnostics.md`。

## 5. V2探索性实验历史

### 核心单种子筛选

早期seed 42筛选显示：RGB-only和2D-oracle不能改变配对路线，Full-3D会产生
有限响应。这批checkpoint曾使用paired-test参与选择，只保留为探索性证据。

### Experiment A：route head

增加route classification、paired ranking和route embedding。route classifier
达到100%，但路径模式准确率约0.541、配对决策差异率约0.179。

结论：能从3D条件分类over/around，不等于动作生成器会执行对应路线。

### A1a：推理干预

强制over/around显著改变最大Z，但footprint定义的路线翻转仅约0.169。说明
全局route主要控制“抬高/保持低位”的动作原语，没有可靠形成XY绕行。

### A1b：空间一致性训练

加入扩张footprint有符号距离、障碍局部XY监督和配对XY差异监督：

| 指标 | Experiment A | A1b |
|---|---:|---:|
| 任务成功率 / Success | 0.463 | 0.600 |
| 路径模式准确率 / Mode accuracy | 0.541 | 0.747 |
| 配对决策差异率 / Paired decisions differ | 0.179 | 0.567 |
| 配对双模式同时正确率 / Paired both correct | 0.131 | 0.531 |

结论：直接空间监督能够显著改善路线行为，说明任务可学；它不是朴素3D注入
的结果，因此不属于当前主比较。

### A-prime：geometry cross-attention

将结构化状态拆成几何tokens，让动作tokens逐层查询几何。结果：

| 指标 | A1b | A-prime |
|---|---:|---:|
| 任务成功率 / Success | 0.600 | 0.401 |
| 终点距离误差 / Endpoint distance | 0.040 | 0.067 |
| 路径模式准确率 / Mode accuracy | 0.747 | 0.779 |
| 配对决策差异率 / Paired decisions differ | 0.567 | 0.587 |

它提升模式指标但损害终点与总体成功率。此前提出的“几何持续注入干扰终点”
和“route与geometry竞争”只是候选解释，未通过单因素消融证实。A-prime已停止，
不继续优化。

完整探索记录见 `docs/EXPERIMENT_A_ANALYSIS.md`。

## 6. 第一阶段最终正式实验

### 统一协议

- 视觉编码后的策略/A侧全局AdaLN；
- hidden dimension 256，4 blocks，4 heads；
- 8 epochs，batch size 32，10步推理；
- seeds 42/43/44；
- 配对成员共享初始扩散噪声；
- 无route head、空间损失、geometry tokens或cross-attention；
- checkpoint只由validation选择；
- 训练过程中不加载paired-test；
- checkpoint锁定后，每个模型只评估一次paired-test。

三档输入：

1. `2D-oracle`：RGB + 显式二维几何；
2. `Full-3D`：RGB + 完整三维真值状态；
3. `Height-only`：RGB + 障碍物高度和最大抬升高度。

最大抬升高度在当前数据中固定，因此Height-only真正变化的信息近似一维高度。

### Paired-test三种子结果

| 指标 / English | 2D-oracle | Full-3D | Height-only |
|---|---:|---:|---:|
| 任务成功率 / Success | 0.341 ± 0.020 | 0.382 ± 0.112 | 0.367 ± 0.092 |
| 终点距离误差 / Endpoint distance | 0.0325 ± 0.0048 | 0.0316 ± 0.0054 | 0.0455 ± 0.0069 |
| 碰撞率 / Collision rate | 0.656 ± 0.022 | 0.608 ± 0.116 | 0.586 ± 0.099 |
| 边界违规率 / Bounds violation rate | 0.0018 ± 0.0015 | 0.0058 ± 0.0060 | 0.0411 ± 0.0701 |
| 路径模式准确率 / Mode accuracy | 0.500 ± 0.000 | 0.559 ± 0.009 | 0.562 ± 0.017 |
| 配对双模式同时正确率 / Paired both correct | 0.000 ± 0.000 | 0.172 ± 0.021 | 0.181 ± 0.030 |
| 配对决策差异率 / Paired decisions differ | 0.000 ± 0.000 | 0.226 ± 0.031 | 0.239 ± 0.026 |

### OOD摘要

| 划分与指标 | 2D-oracle | Full-3D | Height-only |
|---|---:|---:|---:|
| 高度OOD成功率 / Height-OOD success | 0.440 ± 0.020 | 0.516 ± 0.104 | 0.395 ± 0.130 |
| 高度OOD模式准确率 | 0.500 ± 0.000 | 0.577 ± 0.009 | 0.574 ± 0.025 |
| 布局OOD成功率 / Layout-OOD success | 0.387 ± 0.016 | 0.414 ± 0.084 | 0.379 ± 0.089 |
| 布局OOD模式准确率 | 0.500 ± 0.000 | 0.563 ± 0.015 | 0.562 ± 0.013 |
| 视角OOD成功率 / Camera-OOD success | 0.330 ± 0.011 | 0.384 ± 0.124 | 0.126 ± 0.029 |
| 视角OOD模式准确率 | 0.500 ± 0.000 | 0.565 ± 0.005 | 0.570 ± 0.024 |

逐种子原始结果位于：

- `outputs/final_stage1_2d_oracle_seed{42,43,44}.json`；
- `outputs/final_stage1_full3d_seed{42,43,44}.json`；
- `outputs/final_stage1_height_only_seed{42,43,44}.json`。

checkpoint位于：

- `checkpoints_height/final_stage1/2d-oracle_seed{42,43,44}/`；
- `checkpoints_height/final_stage1/full-3d_seed{42,43,44}/`；
- `checkpoints_height/final_stage1/height-only_seed{42,43,44}/`。

正式简表：`outputs/FINAL_STAGE1_INFORMATION_LADDER.md`。

## 7. 当前能够支持的核心结论

### 结论一：朴素3D条件被动作模型使用

在RGB、二维布局和初始扩散噪声相同时：

- 2D-oracle配对决策差异率为`0.000 ± 0.000`；
- Full-3D为`0.226 ± 0.031`。

额外3D状态稳定改变了约四分之一配对的over/around决策。因此3D条件通过
普通全局AdaLN进入并因果性地影响了动作生成。

### 结论二：利用效率较低

Full-3D路径模式准确率只有`0.559 ± 0.009`。约77%的反事实配对仍没有根据
高度改变路线。朴素注入没有自然形成可靠的高度感知空间行为。

### 结论三：路线响应主要由高度维度解释

Height-only配对决策差异率为`0.239 ± 0.026`，与Full-3D同量级；模式准确率
也几乎相同。这说明当前消费端的路线响应主要可以由高度标量解释，更像
“高度触发的模式开关”，不是多维3D关系推理的证据。

### 结论四：完整几何仍影响轨迹执行

Height-only终点误差高于Full-3D，Camera-OOD成功率也明显较低，但两者
Camera-OOD模式准确率相近。高度标量足以触发模式变化，完整结构化几何和
camera yaw可能帮助终点控制与视角对齐。当前实验没有进一步拆分这两项贡献。

## 8. 不能声称的结论

当前结果不能支持：

- 模型已经理解完整3D空间规律；
- 多维3D关系被有效利用；
- Full-3D稳定提高任务成功率，因success种子方差较大；
- A-prime失败原因已经被定位；
- 当前模型代表完整VLA中的VL侧3D注入；
- 没有3D就无法完成当前任务。

## 9. 当前任务与评价的已知限制

### 始终绕行是合法解

把高障碍的around专家路径复用于同一布局的低障碍时，按现有终点、碰撞和
边界规则，paired-test解析成功率为100%。因此高度不是“任务可行性”的必要
信息，而是专家路线选择/效率的条件。

论文主张应是“高度条件化专家行为或路线选择”，不能写成“没有3D就无法
完成任务”。Success是次要轨迹质量指标。

### 专家标签与footprint模式判定不完全一致

paired-test中低障碍专家轨迹约79.1%穿过扩张footprint。部分起点/终点布局
使直线路径自然避开障碍，因此专家标签为over，几何模式判定却为around。
Mode accuracy存在评价口径上限问题，应结合配对差异和连续轨迹指标解释。

### 碰撞检测精度不统一

数据验证沿轨迹线段加密采样，模型评估主要检查25个离散轨迹点。正式论文前
若继续使用collision，应统一为连续线段采样。

## 10. 第一阶段状态

第一阶段核心工作已经闭环，当前不应继续优化A1b或A-prime，也不应增加新的
token、loss、route结构或注入门控。已完成的问题是：

> 在理想真值条件下，普通A侧全局注入能否使用额外3D信息，以及有效响应
> 主要需要多少输入维度？

答案是：会使用，但响应有限；当前路线响应主要由高度这一维度解释，完整
几何更多体现在轨迹执行质量上。

## 11. 下一阶段建议：在小型VLA中比较VL侧与A侧

下一阶段不是继续修当前toy模型，而是在有明确VLM/action-expert边界的小型
VLA上验证结论是否迁移。

优先候选：Hugging Face LeRobot中的SmolVLA。它约450M参数，有明确的VLM
contextual features和flow-matching action expert，官方实现支持MPS推理；正式
训练仍可能需要单GPU环境。TinyVLA可作为备选。

第一轮只使用模拟器或数据集提供的oracle 3D真值，暂不加入3D提取器。这样
可以隔离VLA消费3D信息的能力上限。确定最佳注入位置后，再用深度/点云/3D
检测输出替换oracle真值，测量感知噪声造成的下降。

### 最小2x2矩阵

| 模型 | 结构化信息 | 注入位置 |
|---|---|---|
| VL-2D | 2D状态 | VL contextual stream |
| VL-3D | 2D+3D状态 | VL contextual stream |
| A-2D | 2D状态 | action expert |
| A-3D | 2D+3D状态 | action expert |

真正的VL侧注入要求3D tokens与图像、语言tokens共同进入VLM上下文并发生
交互；如果3D只在VLM编码完成后拼到action expert输入，属于边界融合或A侧，
不能命名为VL侧。

A侧注入应保持VLM上下文不变，只把同一份3D投影送入action expert。两侧需
使用相同oracle 3D、相近projector参数量、相同训练预算和种子。

### 下一阶段停止边界

完成一个小型VLA、一个明确依赖空间信息的任务集、上述四个模型和统一反事实
评估后即停止。暂不加入VL+A双侧复杂融合、新型3D encoder或新的空间loss。

## 12. 建议新窗口首先做什么

1. 阅读本文与`outputs/FINAL_STAGE1_INFORMATION_LADDER.md`；
2. 不重复运行第一阶段，也不恢复A1b/A-prime优化；
3. 检查本机/可用GPU、SmolVLA依赖和可用模拟任务；
4. 写出第二阶段的数据接口、oracle 3D字段、VL/A注入边界和2x2实验协议；
5. 协议锁定后再开始实现与训练。

## 13. 一句话项目结论

> 在受控的隐藏高度任务中，把理想3D真值像普通2D条件一样注入动作策略，
> 会稳定改变部分动作决策，但约四分之三的配对仍不响应；这种响应主要可以
> 由单一高度变量解释，说明“3D信息被注入并被使用”与“形成多维3D空间理解”
> 之间仍有明显差距。
