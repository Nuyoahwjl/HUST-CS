# 🧠 深度学习

本目录为华中科技大学《深度学习》课程实验归档，包含 4 次课程实验的报告、代码、训练日志与结果图。每个实验的具体任务、模型设计、训练过程与结果分析以对应目录中的 `report.pdf` 为准。

| 实验 | 主题 | 主要内容 | 报告 |
| --- | --- | --- | --- |
| [1](./exp-1/) | MLP | 基于 MNIST 数据集实现多层感知机，分析隐藏层规模、层数、激活函数、批大小和学习率对分类性能的影响，并展示前向传播、反向传播和参数更新过程。 | [report.pdf](./exp-1/report.pdf) |
| [2](./exp-2/) | CNN | 基于 CIFAR-10 数据集实现卷积神经网络，比较卷积核大小、通道数、池化方式、激活函数、Batch Normalization 和 Dropout 等设置对模型收敛与泛化性能的影响。 | [report.pdf](./exp-2/report.pdf) |
| [3](./exp-3/) | VAE | 实现基础 VAE 对 MNIST 数字 0 进行重构与生成，在此基础上实现 Conditional VAE，通过标签条件生成指定数字类别图像。 | [report.pdf](./exp-3/report.pdf) |
| [4](./exp-4/) | DQN | 在 CartPole-v1 环境中实现 DQN 智能体，使用经验回放、目标网络、Double DQN、epsilon-greedy、Huber Loss 和梯度裁剪完成训练，并以 100 回合滑动平均回报评估通关效果。 | [report.pdf](./exp-4/report.pdf) |
