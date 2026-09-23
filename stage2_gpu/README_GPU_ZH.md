# Stage 2 Linux/GPU 交接包

本目录是 Gate M5 的 GPU 交接边界，不是已完成的 GPU 训练结果。

## 数据边界

- `stage2/generated_data/lerobot_tiny` 只用于 Mac 工程链路验证；
- GPU 训练必须指向后续选定的 Formal 环境数据；
- 配置将数据路径写为 `${FORMAL_DATASET_ROOT}`，未设置时拒绝运行；
- 配置校验器会拒绝任何包含 `tiny` 的 Formal 数据路径。

## 固定上游

- LeRobot `0.6.1` / commit `7e241bd630a3719a56157a497ce5d08f244784f1`；
- `lerobot/smolvla_base` snapshot `d9f33c94a60fb382c90dea2164c96845bd955e28`；
- 完整记录见 `manifests/upstream.json`。

## Mac 静态验收

```bash
venv_stage2/bin/python -m unittest discover -s stage2_gpu/tests -v
venv_stage2/bin/python -m stage2_gpu.preflight --mode static \
  --output stage2_gpu/fixtures/m5_static_preflight.json
```

## GPU 上的放行顺序

1. `install_linux.sh`：创建隔离环境并安装固定依赖；
2. `scripts/preflight.sh`：记录 Linux、CUDA、GPU、PyTorch 和 LeRobot；
3. G0：加载预训练权重，四配置各做一次 forward/backward 和边界 probe；
4. G1/G2/G3：500-step smoke、2,000-step overfit、5k–10k pilot；
5. 仅使用 validation 选 checkpoint，然后冻结协议；
6. G4：四配置×三种子正式训练；
7. 生成绑定 checkpoint 和 dataset manifest 的 unlock 记录；
8. `scripts/locked_evaluate.sh` 一次性评估 paired-test/OOD。

## 当前故意的 fail-closed 状态

`stage2_gpu.runtime` 在真正训练前会主动停止。原因是 SmolVLA 的 VL 隐层宽度、
action-expert 宽度和显存行为必须在目标 GPU 加载固定 checkpoint 后实测。
在 G0 前解除此保护会把“未验证”伪装成“可运行”。
