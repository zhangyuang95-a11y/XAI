# Cooperative Pong

这是与 Warehouse 共存的合作接球 domain。场地是 **24×14 格**：A1、A2、A3 是 1×1 小球；B1、B2 是 2×2 合作大球；两块板子都是 4×1 格，位于第12行，下面保留两行空间。球和板子用连续坐标移动，格子只定义尺寸和边界。

漏小球计 1 次；漏大球计 3 次。大球的两个接触点需要由两块不同的板子同时覆盖。球会反弹且不会消失。

## 本地运行

在项目根目录运行：

```bash
python3 -m ui.domain_hub_server --host 127.0.0.1 --port 8765
```

打开 <http://127.0.0.1:8765/pong>。按住 A/D 或左右方向键移动机器人1，松开停止；浏览器失焦会暂停。

## A/B 与解释

- A 组的 Task 1 显示机器人2当前分工提示和目标球高亮。
- 气泡说明机器人2负责哪颗球、为什么这样分工、哪颗球更适合机器人1，以及大球是否需要一起接；它不显示不断变化的秒数。
- B 组没有气泡和解释性高亮，Task 1 后直接进入 Task 2。
- Task 2 两组都不提供解释或回放问答。

机器人2当前是透明的规则控制器，不是已训练的 Pong NN。它对每次大球下落建立承诺：进入大球接球准备或就位等待后，不会仅因新小球出现就离开，直到该次接球结算或自身已确定无法到位。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `config.py` 与 `configs/default.json` | 24×14地图、连续速度、球和板子尺寸 |
| `environment/engine.py` | 唯一的连续物理、反弹、接球与计分逻辑 |
| `policies/rule_demo.py` | 机器人2的分工、大球承诺和可行性判断 |
| `study.py` | A/B权限、Task流程、气泡、回放和问答绑定 |
| `explanation/evidence.py` | 由保存的决策证据生成解释 |
| `web/` | 游戏界面；浏览器以逐帧绘制显示服务器连续状态 |
| `tests/test_pong_domain.py` | 物理、承诺、解释、A/B和服务生命周期测试 |

## 测试

```bash
pytest -q tests/test_pong_domain.py
python3 -m py_compile domains/pong/config.py domains/pong/environment/engine.py domains/pong/policies/rule_demo.py domains/pong/study.py domains/pong/web/server.py
```

这些测试验证代码和固定场景，不证明解释已经提高玩家成绩。正式发布前仍需要实际试玩和A/B数据验证。
