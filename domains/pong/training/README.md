# Pong 训练实现

训练入口见 [train_pong_nn.py](/Users/zhangyuang/Desktop/ICLR/XAI/scripts/train_pong_nn.py)，直接运行命令见 [Pong README](/Users/zhangyuang/Desktop/ICLR/XAI/domains/pong/README.md)。导入这些模块不会自动训练。

v2.1 的已确认问题与修正位置：

| 已核实的问题 | v2.2 修正 |
| --- | --- |
| 完整 90 秒专项回合使指定大球的第一次接球与后续结果混在一起 | [evaluation.py](/Users/zhangyuang/Desktop/ICLR/XAI/domains/pong/training/evaluation.py) 用初始机会 ID 单独报告第一次结果，整局另报 |
| 重复追小球原指标只看几何估算目标 | `evaluation.py` 加入双方真实动作；仍无法确认的意图不作为 NN 明确目标 |
| v2.1 只有纯 NN 训练与评估，没有可审计的辅助数据流 | [corrections.py](/Users/zhangyuang/Desktop/ICLR/XAI/domains/pong/training/corrections.py) 独立收集，成对回放筛选；[runner.py](/Users/zhangyuang/Desktop/ICLR/XAI/domains/pong/training/runner.py) 分开计数与损失 |
| 旧课程专项状态也训练整局、人工截断边界未进入 GAE | `runner.py` 在结算安全时截断短片段，末状态 bootstrap，切断跨回合优势递推 |

v2.1 已保存的验证结果是综合漏接 41.625、NN–NN 漏接 47.8125、小球成功率 34.4%、大球成功率约 32.6%，RCPD 反馈更新 0 次。这些是父实验的记录，并不是 v2.2 训练结果。模型是否提高仍需正式训练后验证。
