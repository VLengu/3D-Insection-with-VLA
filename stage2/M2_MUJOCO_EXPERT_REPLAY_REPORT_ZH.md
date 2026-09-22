# M2 MuJoCo 环境与专家回放验收报告

## 结论

M2 已在 Mac arm64 本地完成。这一里程碑证明的是：M1 的抽象
Dual-Clearance 几何可以落到真实 MuJoCo 场景，Franka Panda 能以连续
7 维动作接口完成抓取、搬运和放置，正确路线全部成功，错误路线
全部被碰撞或机器人最大搬运高度拒绝。

M2 不是 SmolVLA 训练结果，也不是完整力矩控制基准。它是在进入数据导出和
模型实现前，对任务物理逻辑和数据因果边界的本地验证。

## 实现范围

- MuJoCo `3.13.0` 本地物理与离屏渲染；
- MuJoCo Menagerie 的 Franka Emika Panda 官方资产；
- 中央越障通道、侧方门洞、填充墙、外围墙、物体和目标区；
- `reset / step / render / success / collision` 环境闭环；
- 7 维动作 `[dx,dy,dz,drx,dry,drz,gripper]`；
- 固定向下姿态的 DLS 逆运动学和解析 waypoint 专家；
- 离散 MuJoCo contact 加连续 swept-AABB 碰撞检查；
- 抓取约束、释放后自由体重力与桌面接触；
- 正确路线回放与强制错误路线反例回放；
- 配对策略 RGB、初始机器人状态和生成结果的自动审计。

## 为什么这样设计

### 1. 先校准真实可达域，再生成数据

M1 的早期物体起点靠近机器人基座，在抽象几何中合法，但固定向下抓取
时不在 Panda 的稳定 IK 可达域。现在起点、障碍和目标都依据实测可达域
重新校准，最大搬运高度上限设为 `0.38 m`。这不改变互斥路线公式，
只排除数学上合法但机械臂无法执行的布局。

### 2. 错误路线必须物理失败

只看终点或 route label 会让“路线错但也到达”的轨迹蒙混过关。M2 为此强制
回放反路线：over-only 场景走 side 必须碰到门洞的安全净空边界；
side-only 场景走 over 必须超过注册的最大搬运高度。

运动学控制可能在两个离散姿态之间跨过薄碰撞体，因此除 MuJoCo contact 外，
又对物体中心轨迹与扩张障碍体做精确线段/AABB 检查。

### 3. 不让 RGB 泄漏隐藏高度

普通顶视物理 RGB 中，障碍高度会通过与机械臂的遮挡关系留下微小像素差异。
审计中最坏配对有约 `0.82%` 像素发生变化，这足以让分类器学到标签捷径。

因此正式策略 RGB 在渲染时将两个登记的 Z 变量暂时投影到该 pair 共享的
可行性边界；碰撞和执行仍使用成员各自的真实高度。这样做到：

- `RGB + 2D` 不能从图像偷看路线答案；
- `3D oracle` 是唯一登记的决定信息；
- 物理环境仍保持两个配对成员的真实差异。

`render()` 返回策略 RGB，`render_physical()` 仅供诊断实际高度。

## Tiny 全量结果

| 检查项 | 结果 |
|---|---:|
| 反事实 pair | 44 |
| 正确专家 episode | 88 / 88 |
| 正确专家成功率 | 1.000 |
| 错误路线物理拒绝 | 88 / 88 |
| 其中：碰撞失败 | 44 |
| 其中：高度越界 | 44 |
| 策略 RGB 字节级相同 | 44 / 44 pair |
| 初始机器人状态相同 | 44 / 44 pair |
| episode 步数范围 | 49–117 |
| 自动化单元测试 | 21 / 21 |

完整结果见 `fixtures/mujoco_tiny/summary.json`，每条专家动作及其 hash 见
`fixtures/mujoco_tiny/expert_rollouts.jsonl`。

## 环境与资产来源

- Python `3.12.14`；
- MuJoCo `3.13.0`；
- NumPy `2.4.3`；
- Pillow `12.1.1`；
- macOS arm64；
- Panda 资产来自 MuJoCo Menagerie commit
  `822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7`，保留 Apache-2.0 许可证。

可重现命令：

```bash
python3 -m venv venv_stage2
venv_stage2/bin/pip install -r stage2/requirements-macos.txt
venv_stage2/bin/python -m unittest discover -s stage2/tests -v
venv_stage2/bin/python -m stage2.validation.replay_mujoco \
  --config stage2/configs/dataset_tiny.json \
  --output stage2/fixtures/mujoco_tiny
```

## 已知边界

1. 专家使用运动学 Cartesian IK 和确定性抓取约束，不是力矩控制器；
2. 7 维动作契约已固定，但 M2 专家保持末端姿态不变，旋转三维在本里程碑
   未主动使用；
3. 没有导出 LeRobotDataset；
4. 没有实现 TinyContractVLA 或 SmolVLA；
5. 没有 GPU 训练、微调或模型推理结果。

## 下一步

进入 M3：把 M2 的 RGB、状态、7D 动作和 pair metadata 导出成 LeRobotDataset，
做导出—重载—逐帧对比，并完成 episode 边界、时间戳、统计量和 checksum 验证。
