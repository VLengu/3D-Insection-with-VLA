# Stage 2: controlled VLA injection

本目录实现第二阶段的独立数据与模型接口。第一阶段的 `uwm3d/`、`scripts/`、
数据和 checkpoint 不在此处修改。

当前已完成 M1 几何/数据契约、M2 MuJoCo/专家回放、LeRobot 往返验证和
Gate M3 统一评价器验证；Gate M4A projector 与 M4B 注入边界也已通过：

- Dual-Clearance 成对几何生成；
- 模型输入与评估元数据 Schema；
- Tiny 数据生成；
- 配对、可行性、划分、复现性和标签泄漏检查。
- Franka Panda MuJoCo 场景、7D 动作环境和解析专家；
- 44 pair / 88 episode 正确与错误路线全量回放；
- 配对策略 RGB 字节级相同审计。
- LeRobotDataset v3 导出、全量重载和 6,998 帧逐帧精确核验；
- 评价元数据与模型 frame features 隔离。
- 统一 episode、pair、split 评价与中英指标协议；
- 正确、错误和固定路线策略的全量物理重放验证。
- 统一 6-token structured projector，2D/3D shape 与参数量完全一致；
- CPU 与 Apple MPS 前向、反向和 checkpoint 往返验证。
- VL/A 两条注入路径由 trace、hook 和反事实 oracle 自动验证；
- 四种配置参数预算与 action memory 长度一致。

M2 设计和结果见 `M2_MUJOCO_EXPERT_REPLAY_REPORT_ZH.md`，M2.5 视觉质量整改见
`M2_5_VISUAL_QUALITY_REPORT_ZH.md`。
LeRobot 导出设计和结果见 `M3_LEROBOT_ROUNDTRIP_REPORT_ZH.md`。
统一评价器设计和 Gate M3 结果见 `GATE_M3_EVALUATOR_REPORT_ZH.md`。
Projector 设计和 Gate M4A 结果见 `GATE_M4A_STRUCTURED_PROJECTOR_REPORT_ZH.md`。
注入边界设计和 Gate M4B 结果见 `GATE_M4B_INJECTION_BOUNDARY_REPORT_ZH.md`。

生成并验证 Tiny 数据：

```bash
./venv/bin/python -m stage2.data.generate \
  --config stage2/configs/dataset_tiny.json \
  --output stage2/fixtures/tiny

./venv/bin/python -m stage2.data.validate \
  --data stage2/fixtures/tiny/records.jsonl

./venv/bin/python -m stage2.data.audit \
  --data stage2/fixtures/tiny/records.jsonl \
  --output stage2/fixtures/tiny/shortcut_audit.json
```

运行测试：

```bash
./venv/bin/python -m unittest discover -s stage2/tests -v

shasum -a 256 -c stage2/fixtures/tiny/manifest.sha256
```

MuJoCo 环境和 M2 全量回放：

```bash
python3 -m venv venv_stage2
venv_stage2/bin/pip install -r stage2/requirements-macos.txt
venv_stage2/bin/python -m stage2.validation.replay_mujoco \
  --config stage2/configs/dataset_tiny.json \
  --output stage2/fixtures/mujoco_tiny
```

导出并验证 LeRobotDataset（输出目录必须不存在）：

```bash
venv_stage2/bin/python -m stage2.data.lerobot_export \
  --config stage2/configs/dataset_tiny.json \
  --output stage2/generated_data/lerobot_tiny

venv_stage2/bin/python -m stage2.data.validate_lerobot \
  --config stage2/configs/dataset_tiny.json \
  --dataset stage2/generated_data/lerobot_tiny \
  --evidence stage2/fixtures/lerobot_tiny
```

运行 Gate M3 全量评价验证：

```bash
venv_stage2/bin/python -m stage2.validation.evaluate_gate_m3 \
  --config stage2/configs/dataset_tiny.json \
  --output stage2/fixtures/evaluator_tiny
```

运行 Gate M4A projector 验证：

```bash
venv_stage2/bin/python -m stage2.validation.validate_structured_projector \
  --config stage2/configs/dataset_tiny.json \
  --output stage2/fixtures/structured_projector \
  --hidden-dim 128
```

运行 Gate M4B 注入边界验证：

```bash
venv_stage2/bin/python -m stage2.validation.validate_injection_boundary \
  --config stage2/configs/dataset_tiny.json \
  --output stage2/fixtures/injection_boundary \
  --hidden-dim 64
```

## 坐标与动作契约

- 世界坐标：右手系，`+x` 从机器人指向目标，`+y` 沿桌面横向，`+z` 向上；
- 长度单位：米；角度单位：弧度；
- 图像：单主相机，`CHW uint8`，正式目标尺寸 `3×256×256`；
- 机器人状态：8维，末端位置、轴角和双指夹爪位置；
- 动作：7维，`[dx,dy,dz,drx,dry,drz,gripper]`；
- 正式环境目标频率：20 Hz；最大 episode 长度：300步。

路线标签、可行性和完整评估几何只存在于 `evaluation`，不得传给模型。
