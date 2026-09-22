# Gate M3：统一评价器验证报告

日期：2026-09-22

## 结论

统一评价器 `dual-clearance-evaluator-v1` 已实现，并通过 Tiny 全量验证。
它可以接收任意 7D 动作序列，在权威 MuJoCo 环境中重新执行，并以相同协议
评价后续 VL-2D、VL-3D、A-2D、A-3D 四个模型。

计划中的 Gate M3“评价有效”现已通过。该结论只说明评分工具可靠，不说明
任何学习模型已经具备任务能力。

## 评价协议

每条 episode 输出：

- 成功、预期路线、实际路线和路线正确性；
- 碰撞、掉落、越界、超高；
- 最终二维放置误差；
- 实际执行步数、提交动作数、终止和截断状态；
- 可同时出现的失败原因列表。

汇总层输出：

- 单条轨迹成功率 `episode_success_rate`；
- 路线选择准确率 `route_accuracy`；
- 完整配对成功率 `pair_success_rate`；
- 碰撞、掉落、越界和超高率；
- 平均最终放置误差；
- 平均执行步数；
- 仅在成功轨迹上计算的平均完成步数；
- overall、各 split 和逐 pair 结果。

完整配对成功要求同一 pair 的 A、B 两个成员都成功。评价器拒绝缺少成员、
重复 episode ID 或成员不唯一的输入，避免把不完整样本误算成 pair 指标。

## 环境边界补强

本阶段补充了两项此前不完整的环境事件：

1. 物体被抓住后，如果在目标范围外打开夹爪，累计标记为 `drop`；
2. 专家因碰撞或超高失败时保留已执行的动作前缀，使失败轨迹也能在全新的
   环境中通过统一 evaluator 重放，而不是直接复用专家内部结论。

连续碰撞仍由 carried box 的 swept segment 与扩张障碍 AABB 相交测试负责。
自动测试包含“线段两端均在障碍外、但中间穿过障碍”的样例，确认离散 waypoint
不会漏掉穿越碰撞。

## 全量结果

每种策略覆盖 44 pair / 88 episode：

| 策略 | 单条成功率 | 路线准确率 | Pair 成功率 | 碰撞率 | 超高率 |
|---|---:|---:|---:|---:|---:|
| 正确专家 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| 每条都走错误路线 | 0.000 | 0.000 | 0.000 | 0.500 | 0.500 |
| 固定只走 over | 0.500 | 0.500 | 0.000 | 0.000 | 0.500 |
| 固定只走 side | 0.500 | 0.500 | 0.000 | 0.500 | 0.000 |

这验证了 pair 指标的必要性：固定路线策略在 episode 层面看似有 50% 成功，
但没有完成任何一个反事实 pair。

额外人工探针：

- 抓取后立即在起点松爪：`drop=true`，失败；
- 提交空动作序列：同时报告 `wrong_or_missing_route` 与 `incomplete`；
- 错误 side 进入低净空门洞：连续碰撞；
- 错误 over 超过机械臂携带高度：超高失败。

## 自动验收

- 29 项单元测试通过；
- 88 / 88 正确专家轨迹成功；
- 88 / 88 错误路线被拒绝，并且全部具有物理失败原因；
- 固定 over 和固定 side 均为 50% episode success、0% pair success；
- 七个 split 的正确专家 episode success 与 pair success 均为 100%；
- 第一阶段 303 个冻结文件校验通过，共 5,224,662,284 bytes。

机器可读结果位于 `fixtures/evaluator_tiny/`：

- `correct_expert.jsonl`、`wrong_route.jsonl`；
- `fixed_over.jsonl`、`fixed_side.jsonl`；
- `summary.json`、`metrics.md`、`manifest.sha256`。

## 复现命令

```bash
venv_stage2/bin/python -m stage2.validation.evaluate_gate_m3 \
  --config stage2/configs/dataset_tiny.json \
  --output stage2/fixtures/evaluator_tiny
```

## 下一步

进入 Gate M4 的前半部分：实现统一 structured projector。2D 与 3D 输入必须
输出完全相同的 token 数、hidden dimension 和 mask 结构，并报告参数量差异。
在 projector 合同通过后，再实现 VL 侧与 A 侧的严格注入边界。
