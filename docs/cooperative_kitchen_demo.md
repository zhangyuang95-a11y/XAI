# PolicyLens · 协作厨房可玩 Demo

这是一个独立的合作玩法原型：两名厨师通过共享工作台交接物品，在 120 个联合步内完成两份洋葱汤。橙色队友使用固定程序规则，不是强化学习模型。本页不接入 Qwen、正式 A/B 实验或研究数据库。

## 启动

在项目根目录运行：

```sh
python3 -m http.server 8002 --bind 127.0.0.1 --directory ui/cooperative_kitchen_demo
```

打开 <http://127.0.0.1:8002/>。页面没有构建步骤、外部图片或第三方脚本依赖。现有仓库和糖豆环境仍使用各自的启动方式。

## 操作与合作

- 方向键 / WASD：移动并调整朝向。碰到设施或墙壁也会调整朝向，但不会移动。
- E：与面前的设施交互。空格：等待一步。
- 每个操作推进一个联合步，撞墙、无效交互也消耗一步。
- 默认玩家在左侧，负责取洋葱、交接和出餐。橙色队友在右侧，负责往锅里加料、取盘装汤和交回成汤。
- 洋葱放到中央任一共享工作台后，对侧可以取走。双方每人只能拿一件物品，每张工作台只能放一件物品。
- 三份洋葱入锅后，再经过四个联合步煮熟。拿着盘子面向锅按 E 装汤，然后通过工作台交回左侧，送至出餐口。
- 拿错的盘子或多余洋葱可以丢入本侧垃圾桶。若两个工作台都已占满，持汤队友会等待，需要腾出交接位置。
- “交换分工”会重开本次试玩，将玩家放在右侧。“重新开始”保留分工、清空当前试玩；此前正式环境的日志不会受影响。
- “自动演示”使用相同引擎控制双方；暂停后可以接管蓝色玩家。点击解释、进入回放或查看说明会暂停演示；切换语言、页面进入后台也会暂停。

## 解释与回放

三个问题按钮绑定当前画面（包括选中的历史帧）。“为什么选择这个动作”和“你在等什么”调用程序队友对该状态的真实决策函数，解释的是**该帧之后的下一步决策**。“如果我等待会怎样”复制所选状态，模拟玩家连续等待最多三个联合步，终局时提前停止；回答明确列出这个假设及模拟结果。它们不改变当前厨房状态、历史或计数。

时间滑块和前后帧按钮查看已确认状态；历史回放期间不能操作，点击“回到当前”继续。主地图与回放共用 Canvas 绘制函数。角色动画仅发生在确认一个联合步之后。

试玩状态与完整帧历史保存在当前站点的独立 localStorage 键 `policylens-kitchen-demo-v1` 中。刷新后恢复已确认进度，并保持自动演示暂停。删除浏览器站点数据会删除这份本机试玩记录。

## 实现与验证

- `ui/cooperative_kitchen_demo/engine.js`：纯状态引擎、固定优先级程序队友、最短路径、快照与解释。双方均依据行动前状态决策和解析交互；物品竞争按步数奇偶确定优先方，禁止同帧放入后立即取走。
- `ui/cooperative_kitchen_demo/renderer.js`：共用 Canvas 绘制；蓝色玩家、橙色队友、厨师帽、设施与手持物品。
- `ui/cooperative_kitchen_demo/app.js`：双语页面、键盘操作、自动演示、回放和本机存储。
- `tests/cooperative_kitchen_engine.cjs`：动力学、程序完成任务和解释隔离测试。
- `tests/cooperative_kitchen_browser.cjs`：真实页面操作、两种分工自动完成、双语尺寸、恢复与截图。

```sh
node tests/cooperative_kitchen_engine.cjs
node tests/cooperative_kitchen_browser.cjs http://127.0.0.1:8002 output/cooperative_kitchen_demo/v1
```

浏览器测试需要 Playwright 与 Chromium；可通过 `KITCHEN_PLAYWRIGHT_MODULE` 指定 Playwright 模块目录，通过 `KITCHEN_CHROME` 指定 Chrome 可执行文件。验收产物放在 `output/cooperative_kitchen_demo/v1/`。这是玩法与界面验收，不代表训练策略或解释实验效果已经得到研究验证。

当前这台 Mac 可直接使用已有的运行时，无需另行安装依赖：

```sh
cd /Users/zhangyuang/Desktop/ICLR/XAI
KITCHEN_NODE=/Users/zhangyuang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node
"$KITCHEN_NODE" --test tests/cooperative_kitchen_engine.cjs
KITCHEN_PLAYWRIGHT_MODULE=/Users/zhangyuang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright \
KITCHEN_CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
"$KITCHEN_NODE" tests/cooperative_kitchen_browser.cjs http://127.0.0.1:8002 output/cooperative_kitchen_demo/v1
```
