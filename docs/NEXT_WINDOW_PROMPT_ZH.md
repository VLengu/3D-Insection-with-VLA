# 给新Codex窗口的交接提示词

请完整阅读项目主报告：

`/Users/vlengu/Desktop/UWM-3D视觉矫正/docs/PROJECT_STATUS_HANDOFF_ZH.md`

并将它视为当前项目状态的唯一主依据；旧文档中的A/A1/A-prime内容是探索性
历史，不要自动继续那些架构优化。

项目第一阶段已经完成并冻结：在当前V-A扩散策略中，使用相同的策略/A侧
全局AdaLN比较了2D-oracle、Full-3D和Height-only，全部为seeds 42/43/44，
validation选checkpoint，paired-test在锁定后评估一次。核心paired-test结果：

- 2D-oracle配对决策差异率：`0.000 ± 0.000`；
- Full-3D：`0.226 ± 0.031`；
- Height-only：`0.239 ± 0.026`；
- 对应路径模式准确率：`0.500 / 0.559 / 0.562`。

结论：朴素3D条件会被动作模型使用，但响应有限；路线响应主要由高度这一维
解释，完整几何更多影响终点控制和视角泛化。不要将其表述为完整3D理解。
当前任务允许始终绕行，因此Success不是3D必要性的证据。

接下来的目标是设计第二阶段：选择一个有明确VLM/action-expert边界的小型
VLA（优先评估SmolVLA，TinyVLA作为备选），先使用oracle 3D真值，比较：

1. VL-2D；
2. VL-3D；
3. A-2D；
4. A-3D。

真正的VL侧注入要让3D tokens与视觉、语言tokens在VLM contextual stream中
交互；A侧注入保持VLM不变，只把3D送入action expert。先完成环境、算力、
数据集、oracle字段和注入边界的可行性审计，再给出固定实验协议。不要直接
开始大规模下载或训练，不要重新设计复杂3D融合结构，也不要重新开启A1b或
A-prime方向。

所有指标在汇报时同时提供中文名和英文名。指标词典位于：

`/Users/vlengu/Desktop/UWM-3D视觉矫正/docs/METRICS_GLOSSARY_ZH.md`
