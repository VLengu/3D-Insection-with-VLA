# M3：LeRobotDataset 导出与全量往返验证报告

日期：2026-09-22

## 结论

Dual-Clearance Tiny 数据已经成功导出为 LeRobotDataset v3，并通过全量磁盘
重载、逐帧精确比较和物理动作回放。当前数据接口可供后续 evaluator、
TinyContractVLA 和 GPU 端训练代码直接消费。

本次结果同时满足计划中“步骤 5：导出 LeRobot 格式”和 Gate M2 的
LeRobot round-trip 条件。这里沿用开发工作编号 M3；下一项工作是实现
Gate M3 所要求的 paired evaluator。

## 数据规模

| 项目 | 结果 |
|---|---:|
| pair | 44 |
| episode | 88 |
| frame | 6,998 |
| 图像 | 单主相机，256×256 RGB，无损 PNG |
| 状态 | 8 维 float32 |
| 动作 | 7 维 float32 |
| oracle_2d | 15 维 float32 |
| oracle_3d | 27 维 float32 |
| 频率 | 20 Hz |
| 本地数据大小 | 128,390,169 bytes |

划分 episode 数为：train 32、val 8、paired_test 16，以及 layout、height、
camera、compound 四类 OOD 各 8。

## 为什么这样设计

1. **采用 LeRobotDataset v3**：后续 GPU 环境能直接使用统一的 episode、
   frame、task、时间戳和特征接口，不再解释自定义 JSON/NPZ。
2. **Mac 验证集使用 `use_videos=False`**：PNG 保留逐像素可审计性，避免视频
   编解码误差干扰“配对图像是否相同”的科学检查。正式大数据可另行评估 MP4。
3. **模型输入与评价元数据隔离**：pair、split、route label 和 seed 只写入
   `meta/dual_clearance_episodes.jsonl`，不进入 frame feature，防止答案泄漏。
4. **记录动作前观测**：每个 frame 表示策略看到的观测及其应输出的动作，符合
   行为克隆的因果顺序。
5. **完整数据不提交 Git**：本地生成目录含近 7,000 张图，Git 仅保存生成代码、
   版本锁和可核查的摘要/hash；任何机器均可用固定 seed 重建。

## 验证方法与结果

验证器关闭 writer 后，从磁盘重新构造 `LeRobotDataset`，对每个 episode：

1. 用固定配置和 seed 重建原始场景；
2. 重新生成专家动作；
3. 对每一帧比较解码 RGB、状态、2D/3D oracle、动作、时间戳和索引；
4. 将动作重新送入 MuJoCo，检查最终 success；
5. 检查每个 pair 两个成员的首帧 RGB 完全一致；
6. 为完整目录生成 SHA-256 manifest。

结果：

- 6,998 / 6,998 帧通过；
- state、oracle_2d、oracle_3d、action 最大绝对误差均为 `0.0`；
- 44 / 44 pair 的解码首帧 RGB 完全一致；
- 88 / 88 episode 物理回放成功；
- 错误列表为空，最终状态 `PASS`；
- 数据树摘要：`d674e0d33ade6f39c7d57a8701f0592af76fcc9cc94a26a2abf58be2aeb0c8f8`。

机器可读证据位于 `fixtures/lerobot_tiny/summary.json` 和
`fixtures/lerobot_tiny/selected_hashes.json`。

## 复现命令

```bash
python3 -m venv venv_stage2
venv_stage2/bin/pip install -r stage2/requirements-macos.lock.txt

venv_stage2/bin/python -m stage2.data.lerobot_export \
  --config stage2/configs/dataset_tiny.json \
  --output stage2/generated_data/lerobot_tiny

venv_stage2/bin/python -m stage2.data.validate_lerobot \
  --config stage2/configs/dataset_tiny.json \
  --dataset stage2/generated_data/lerobot_tiny \
  --evidence stage2/fixtures/lerobot_tiny
```

导出器拒绝覆盖任何已存在的目标目录，以避免误覆盖或把两次数据混在一起。

## 已知边界

- 当前 Mac 没有 TorchCodec 所需的 FFmpeg shared libraries；LeRobot 会回退到
  PyAV。由于本数据使用 PNG 而非视频，这不影响本报告的导出和读取结果。
- 这一步证明数据格式和物理轨迹保持正确，不证明任何 VLA 已学会任务。
- 当前提交的 hash 是本机本次构建证据；跨平台渲染是否字节级相同仍需在 Linux
  GPU 环境单独检查。

## 下一步

实现 paired evaluator（计划中的 Gate M3）：构造明确成功/失败的人工轨迹，
统一连续碰撞判定，核对 pair success、route accuracy 和中英指标名称，再进入
structured projector 与 VL/A 注入边界。
