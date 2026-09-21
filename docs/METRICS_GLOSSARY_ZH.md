# Height World 指标中英对照

| English metric | 中文名称 | 含义 |
|---|---|---|
| Success | 任务成功率 | 终点进入目标半径、轨迹无碰撞且不越界的样本比例。 |
| Endpoint distance | 终点距离误差 | 生成轨迹终点到目标位置的平均欧氏距离，越低越好。 |
| Collision rate | 碰撞率 | 生成轨迹与障碍物碰撞的样本比例，越低越好。 |
| Bounds violation rate | 边界违规率 | 轨迹离开工作空间或超出最大抬升高度的样本比例。 |
| Path-mode accuracy / Mode accuracy | 路径模式准确率 | 生成路径的 over/around 模式与专家标签一致的比例。 |
| Paired both modes correct | 配对双模式同时正确率 | 同一反事实配对中，低障碍和高障碍两条路径都选择正确模式的配对比例。 |
| Paired decisions differ | 配对决策差异率 | 同一RGB和二维布局下，仅改变高度后，模型是否改变over/around决策的配对比例。 |
| Height-OOD success | 高度分布外成功率 | 在训练范围之外的更低/更高障碍上取得任务成功的比例。 |
| Layout-OOD success | 布局分布外成功率 | 在未见过的障碍物尺寸和位置范围上的成功率。 |
| Camera-OOD success | 视角分布外成功率 | 在未见过的正交相机偏航角下的成功率。 |

在当前研究中，`Paired decisions differ（配对决策差异率）`是判断额外
高度信息是否改变动作的主要指标；`Success（任务成功率）`属于次要质量
指标，因为当前环境允许不依赖高度的始终绕行策略。
