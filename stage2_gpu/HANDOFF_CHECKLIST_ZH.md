# Gate M5 GPU 交接检查表

## M5-Mac（本机完成）

- [x] LeRobot tag/commit 与 SmolVLA snapshot 使用不可变 revision。
- [x] 四配置只允许 condition、name、site、modality 不同。
- [x] Formal 训练配置拒绝 Tiny 路径。
- [x] 训练阶段只能读取 train/val。
- [x] paired-test 和 OOD 需要绑定 checkpoint/data hash 的 unlock 记录。
- [x] 输出目录不允许覆盖。
- [x] Linux 安装脚本和 shell 语法通过。
- [x] Dockerfile、版本 manifest、四配置和命令入口完整。
- [x] Stage 1 checksum 仍然通过。

## M5-GPU / G0（需目标 Linux/CUDA 机器）

- [ ] 记录 GPU 型号、显存、driver、CUDA 和 PyTorch。
- [ ] 校验 SmolVLA 权重的 snapshot 和文件 hash。
- [ ] 记录 VLM/action-expert 实际 hidden width。
- [ ] 完成项目内 adapter，不修改 `site-packages`。
- [ ] 四配置各一个 batch forward/backward。
- [ ] VL/A 反事实边界 probe 在真实 SmolVLA 通过。
- [ ] 报告总参数、可训练参数、峰值显存、step time 和吞吐。
- [ ] 根据实测显存锁定 PyTorch/CUDA image 和 batch/accumulation。

M5-GPU 未全部勾选前，不得声称 Gate M5 整体通过，不进入正式训练。
