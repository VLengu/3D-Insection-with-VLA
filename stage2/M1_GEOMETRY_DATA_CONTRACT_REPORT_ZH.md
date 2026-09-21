# M1：几何与数据契约验证报告

更新日期：2026-09-21

## 状态

`PASS`

本里程碑完成了与模拟器无关的 Dual-Clearance 几何核心、数据 Schema、Tiny
数据生成、配对验证和统计捷径审计。当前结果不包含 MuJoCo、机器人动力学、
图像渲染、LeRobotDataset 导出或 VLA 模型。

## 已实现内容

- 第二阶段独立工程目录；
- 右手世界坐标、米/弧度单位约定；
- 8维机器人状态与7维动作契约；
- 15维 `oracle_2d`；
- 27维 `oracle_3d`；
- over-only / side-only 互斥配对生成；
- ID、height/layout/camera/compound OOD 配置；
- 模型输入 allowlist 和评估字段隔离；
- Tiny 数据生成和可复现性检查；
- 分组划分、配对不变量和标签泄漏检查；
- 单字段阈值捷径审计。

## Tiny 数据

| split | pairs | episodes | over | side |
|---|---:|---:|---:|---:|
| train | 16 | 32 | 16 | 16 |
| val | 4 | 8 | 4 | 4 |
| paired_test | 8 | 16 | 8 | 8 |
| height_ood | 4 | 8 | 4 | 4 |
| layout_ood | 4 | 8 | 4 | 4 |
| camera_ood | 4 | 8 | 4 | 4 |
| compound_ood | 4 | 8 | 4 | 4 |
| 总计 | 44 | 88 | 44 | 44 |

固定 over 或固定 side 的路线策略在这个平衡集合上的理论路线准确率均为
`0.500`，且一个固定路线无法完成同一 pair 的两个成员。

## 互斥逻辑

模型只获得基础几何，不获得以下布尔结果：

```text
over_feasible
side_feasible
route_label
```

可行性由基础字段关系决定：

```text
over feasible:
central_top_z + object_half_z + safety_margin <= robot_max_carry_z

side feasible:
side_gate_clearance_z >= 2*object_half_z + safety_margin
```

每个 pair 的 A 成员只允许 over，B 成员只允许 side。两个成员的二维 oracle、
初始机器人状态、任务文本、物体尺寸、机器人能力和相机参数完全相同，仅登记的
中央障碍 Z 几何与门洞净空发生变化。

## 捷径审计

审计只使用 train split 的32条记录：

| 输入或规则 | 路线预测准确率 |
|---|---:|
| 语言文本查表 | 0.500 |
| 初始状态最佳单字段阈值 | 0.500 |
| 2D oracle最佳单字段阈值 | 0.500 |
| 3D oracle最佳单字段阈值 | 0.781 |
| 注册的3D关系规则 | 1.000 |

最强3D单字段是 `side_gate_clearance_z`，准确率为`0.781`，低于预设的
`0.950`拒绝阈值。完整关系规则恢复全部标签，说明决定信息存在于字段关系中。

第一版生成器曾出现两项问题，均由测试发现并修复：

1. Height-OOD 小物体可能产生负门洞净空；现已限制为物理有效的最小开口；
2. 初始门洞高度分布可被单字段以`0.969`准确率预测路线；现已扩大物体尺寸、
   机器人能力和净空范围的交叠，使单字段准确率降至`0.781`。

## 自动验证结果

```text
16 tests: PASS
STAGE2_DATA_VALIDATE=PASS records=88 pairs=44
STAGE2_SHORTCUT_AUDIT=PASS 2d=0.500 3d_single=0.781 3d_relational=1.000
STAGE1_FREEZE_VERIFY=PASS files=303 bytes=5224662284
```

## 当前边界

通过M1只证明任务几何和数据字段设计在逻辑上成立。尚未证明：

- 机械臂在动力学环境中能执行专家轨迹；
- 障碍碰撞体与抽象净空公式一致；
- 初始 RGB 在配对成员间一致；
- LeRobotDataset 导出与读取正确；
- VL/A 注入边界正确；
- SmolVLA 能学习该任务。

## 下一里程碑

M2 接入 Mac 可运行的 MuJoCo 环境，完成：

1. 场景实体和碰撞体；
2. reset/step/render；
3. 抓取、搬运和放置解析专家；
4. 连续碰撞、掉落和成功评价；
5. Tiny episode 回放；
6. 配对初始 RGB 一致性或像素容差验证。

