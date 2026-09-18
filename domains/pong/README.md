# Cooperative Pong：本地 v2.3 候选版

机器人2先用公开球位比较接球机会、给两块球拍分工，再参考**冻结 NN** 的动作建议；建议会破坏分工时，规则可以改选。小球漏接计 1，大球需要双方同时覆盖两侧，漏接计 3。Group A 在 Task 1 看分工气泡、暂停或局后提问；Group B 不看解释；Task 2 两组都不看解释。两组的游戏和控制器相同。

冻结 Actor 来自 `output/pong_nn/v22_hybrid_seed_260920/best_hybrid_candidate.pt`，参数 SHA-256 为 `90f86aa11ef66cba65e87d24f3eab05bb2f3ae05cd4e3e4b0258574573a07667`。本版没有重新训练；该 Actor 没有匹配的已抽取解释程序，因此气泡解释的是**实际规则分工**，不声称知道 NN 的内部想法。

以下命令可以从任意目录直接运行。在 PyCharm 终端选择装有 PyTorch、NumPy、PyYAML 的解释器。Apple Silicon 使用 `mps`；也可将设备改为 `cpu`。

先用同一批场景比较纯 NN、旧有限辅助、纯规则和 v2.3：

```bash
python3 /Users/zhangyuang/Desktop/ICLR/XAI/scripts/evaluate_pong_controller_v23.py --config /Users/zhangyuang/Desktop/ICLR/XAI/configs/pong_controller_v23.json --output /Users/zhangyuang/Desktop/ICLR/XAI/output/pong_controller/v23 --split validation --device mps
```

首次本地试玩先导出冻结 Actor；如果 `output/pong_controller/v23/export` 已存在，跳过此步：

```bash
python3 /Users/zhangyuang/Desktop/ICLR/XAI/scripts/export_pong_nn.py --run /Users/zhangyuang/Desktop/ICLR/XAI/output/pong_nn/v22_hybrid_seed_260920 --checkpoint best_hybrid_candidate.pt --controller-mode coordinated --controller-config /Users/zhangyuang/Desktop/ICLR/XAI/configs/pong_controller_v23.json --allow-candidate --output /Users/zhangyuang/Desktop/ICLR/XAI/output/pong_controller/v23/export --device mps
```

启动本地网站：

```bash
python3 /Users/zhangyuang/Desktop/ICLR/XAI/scripts/serve_pong_nn.py --bundle /Users/zhangyuang/Desktop/ICLR/XAI/output/pong_controller/v23/export --port 18768
```

启动上述命令后，再打开 [本地 Pong](http://127.0.0.1:18768/pong/)；也可以直接打开 [在线 Pong](https://policylens-warehouse-study.onrender.com/pong/)。不要双击 `index.html`：`file://` 页面无法读取模型文件。网页加载模型后，游戏在浏览器内运行，无需把每一步发送到服务器。A 组可点“暂停”，选择单帧或拖动“这段过程从”选择时间段，并输入自己的问题；回放区的“下载本局决策与问答记录”保存逐帧动作、规则依据、接球结果和问答。比较结果在 [`comparison_validation.json`](/Users/zhangyuang/Desktop/ICLR/XAI/output/pong_controller/v23/comparison_validation.json)。

固定验证集的 16 局中，v2.3 双自动球拍平均加权漏接 **18.5**；相同场景纯规则双球拍 **23.25**、旧有限辅助双自动球拍 **48.56**。独立的 16 局最终集分别为 **18.31、25.81、48.63**。10 个受控合作大球机会全部接住。v2.3 的规则改选约占 65%，因此它是**规则协调的混合控制器**，不是纯 NN。验证检查已通过，但它仍是候选版：真人操作、解释是否改善得分尚未验证，不要自动用于正式实验。详细结果在 [`comparison_final_test.json`](/Users/zhangyuang/Desktop/ICLR/XAI/output/pong_controller/v23/comparison_final_test.json)。训练命令与历史 v2.2 说明保留在 [`training/README.md`](/Users/zhangyuang/Desktop/ICLR/XAI/domains/pong/training/README.md)。
