# Gate M5：GPU 可执行交接包报告

## 结论

2026-09-23 完成 **M5-Mac**：GPU 交接包的静态完整性、四组对照配置、
数据边界、测试集锁和输出隔离均有自动验证。

**Gate M5 整体仍为条件通过，而非已完成。** M5-GPU 需要目标 Linux/CUDA
机器和正式环境数据，无法在当前 Mac 上真实完成。

## 为什么不使用 Tiny 做正式 GPU 数据

Tiny Dual-Clearance 数据的目的是验证 schema、LeRobot round-trip、注入边界、
forward/backward 和评价器。它的环境和规模不足以支持正式科研结论。
因此 GPU 配置要求 `dataset.role=formal`，并在路径中发现 `tiny` 时立即拒绝。

## 交接内容

- `stage2_gpu/configs/`：VL-2D、VL-3D、A-2D、A-3D 四组等预算配置；
- `stage2_gpu/contracts.py`：配置、Formal/Tiny 隔离和输出目录合同；
- `stage2_gpu/test_lock.py`：测试集锁与 checkpoint/data hash 绑定；
- `stage2_gpu/preflight.py`：Mac 静态检查和 Linux/CUDA 运行时检查；
- `stage2_gpu/adapters/interface.py`：与 LeRobot 内部实现解耦的 tensor 边界；
- Dockerfile、Linux 安装脚本、分阶段命令入口和交接检查表。

## 版本固定

- LeRobot `v0.6.1` commit：`7e241bd630a3719a56157a497ce5d08f244784f1`；
- SmolVLA base snapshot：`d9f33c94a60fb382c90dea2164c96845bd955e28`；
- 不使用 `main`、`latest` 或浮动 HEAD。

## 安全停止边界

当前 runtime 入口会在实际训练前 fail closed。SmolVLA 的 VLM 和 action expert
隐层宽度不同，且显存限制尚未实测；项目不应在 Mac 上猜测适配后宣称可训练。
到 GPU 后先完成 G0 adapter 和四组单 batch 验证，然后才解锁 smoke training。

## 验收命令

```bash
venv_stage2/bin/python -m unittest discover -s stage2_gpu/tests -v
venv_stage2/bin/python -m stage2_gpu.preflight --mode static \
  --output stage2_gpu/fixtures/m5_static_preflight.json
stage1_freeze/verify.py
```
