# 第一阶段冻结保护

本目录记录第一阶段正式结果的逐文件 SHA-256 校验信息。

冻结范围：

- `uwm3d/`：第一阶段模型与环境代码；
- `scripts/`：第一阶段生成、训练、评估与诊断入口；
- 第一阶段正式文档；
- `outputs/`：结果 JSON、Markdown 与预览图；
- `data/`、`data_height/`：V1/V2 数据；
- `checkpoints/`、`checkpoints_height/`：全部历史与正式 checkpoint。

不纳入冻结范围：虚拟环境、Python 缓存、Git 元数据、README 导航，以及
第二阶段新增的报告和代码。

校验命令：

```bash
./venv/bin/python stage1_freeze/verify.py
```

校验器会同时检查：

1. 冻结文件是否缺失；
2. 文件大小是否变化；
3. SHA-256 是否变化；
4. 冻结目录内是否出现清单之外的新文件。

`manifest.json` 是机器生成文件。除非明确建立新的冻结版本，否则不要使用
`--write` 覆盖它。

