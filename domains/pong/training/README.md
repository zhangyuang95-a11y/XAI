# Pong training boundary

格子五球版本目前只提供公开观察适配器和 `rule_demo` 控制器：三个小球 A1-A3 与两个合作大球 B1-B2。没有Pong训练入口、checkpoint或冻结NN。

未来训练必须使用 `domains/pong/adapters/core_adapter.py` 的五球特征签名，独立使用 `domain_id=pong` 和 `pong-continuous-24x14-v7` 版本，不能加载 Warehouse Actor。训练、评估、导出和运行时必须共享完全相同的球槽位顺序、动作顺序、连续物理和尺寸配置。本目录当前只记录训练边界，不启动正式训练。
