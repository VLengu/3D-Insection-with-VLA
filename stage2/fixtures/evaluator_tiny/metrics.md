# Gate M3 evaluator results

Evaluator: `dual-clearance-evaluator-v1`

| Policy | Episode success | Route accuracy | Pair success | Collision | Drop | Height violation |
|---|---:|---:|---:|---:|---:|---:|
| correct_expert | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| wrong_route | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 0.500 |
| fixed_over | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 |
| fixed_side | 0.500 | 0.500 | 0.000 | 0.500 | 0.000 | 0.000 |

## Metric names

| Key | 中文 |
|---|---|
| `episode_success_rate` | 单条轨迹成功率 |
| `route_accuracy` | 路线选择准确率 |
| `pair_success_rate` | 完整配对成功率 |
| `collision_rate` | 碰撞率 |
| `drop_rate` | 掉落率 |
| `bounds_violation_rate` | 越界率 |
| `height_violation_rate` | 超高率 |
| `mean_final_placement_error_m` | 平均最终放置误差（米） |
| `mean_executed_steps` | 平均执行步数 |
| `mean_completion_steps` | 成功轨迹平均完成步数 |
