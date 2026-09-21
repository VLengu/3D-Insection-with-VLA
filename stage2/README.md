# Stage 2: controlled VLA injection

本目录实现第二阶段的独立数据与模型接口。第一阶段的 `uwm3d/`、`scripts/`、
数据和 checkpoint 不在此处修改。

当前里程碑只包含与模拟器无关的部分：

- Dual-Clearance 成对几何生成；
- 模型输入与评估元数据 Schema；
- Tiny 数据生成；
- 配对、可行性、划分、复现性和标签泄漏检查。

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

## 坐标与动作契约

- 世界坐标：右手系，`+x` 从机器人指向目标，`+y` 沿桌面横向，`+z` 向上；
- 长度单位：米；角度单位：弧度；
- 图像：单主相机，`CHW uint8`，正式目标尺寸 `3×256×256`；
- 机器人状态：8维，末端位置、轴角和双指夹爪位置；
- 动作：7维，`[dx,dy,dz,drx,dry,drz,gripper]`；
- 正式环境目标频率：20 Hz；最大 episode 长度：300步。

路线标签、可行性和完整评估几何只存在于 `evaluation`，不得传给模型。
