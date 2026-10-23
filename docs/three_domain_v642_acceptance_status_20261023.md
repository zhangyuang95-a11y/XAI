# v6.42 三域验收状态与已有资产复核（2026-10-23）

本次只读核验既有文件，重算内容哈希、收据数值和模型绑定。没有重新运行Pong／Kitchen已使用的留出集，没有环境初始化／reset、NN前向、程序拟合、训练更新或Warehouse新随机流。

证据：`experiments/warehouse/evidence/three-domain-v642-existing-assets-20261023.json`，SHA-256 `75053b4f1046b207b1d083f8371a7c227d490d16596cb3dfa3e34fe41d320654`。
发布清单：`releases/policylens-three-domain-20261016.v6.0/catalog.json`，SHA-256 `402726d459db90c1347f0e4a00d34500631b106b19937c08d2010f803e2af77b`；核验前后字节一致。

## Pong、Kitchen已有证据

内容寻址索引的 **18 个唯一文件**均匹配登记SHA及目录SHA，仍在本地的源文件与打包副本逐字节相同。四变体均通过当前运行时收据判定和保存绑定复核。Pong JSON和Kitchen NPZ重新计算的双角色权重哈希与清单一致。A/B绑定相同，C的Actor及两个角色哈希分别独立。

| Domain | 原验收收据SHA-256 | raw Actor SHA-256 | regularized Actor SHA-256 |
| --- | --- | --- | --- |
| pong | `ad6e4882964c84cbc565fc72dd33a16e79399f5984b416f84026bf0da1fd9958` | `1b7d950ba9ff9eef33c6b726eee57eb0ff793af55fcbf192465d9baf09c490d7` | `1da5dd98fde4206b320306e85f98a340476d010ab0f105dc972db446dda999a6` |
| kitchen | `012af827e8416f293bec97670bb3aec1b752de58dc0155051e98d33c56eabc9a` | `30e4e4b58f99997bba03005061b9d5d9762f59f01c75b42d9d527032390541eb` | `0ecd13da18d00347104a3a6e05456ceb8bb1c10ba41b5621a3fcb44fb4fc5d85` |

### 原留出结果只读复核

| Domain／版本 | 既有结果 | 本次核验范围 |
| --- | --- | --- |
| Pong raw | 300局，平均容量92.0125%，最小83.75%；平均AI等待率1.7037%，最大11.1111% | 既有300行及逐task汇总数值重算；未执行环境 |
| Pong regularized | 300局，平均容量91.8569%，最小83.75%；平均AI等待率1.7704%，最大11.1111% | 既有300行及逐task汇总数值重算；未执行环境 |
| Kitchen raw | NN–NN 300/300；离线代理 300/300 | 原收据汇总、种子列表、成功／失败数关系；未重建底层episode |
| Kitchen regularized | NN–NN 300/300；离线代理 300/300 | 原收据汇总、种子列表、成功／失败数关系；未重建底层episode |

原绑定收据登记每版本200k联合步、raw程序反馈为零、regularized双角色反馈为正、动作覆盖计数为零。本次没有重放历史训练；这些历史过程字段继续按原收据的证明范围解释。Kitchen代理结果不表示真人协作效果。

## Warehouse仍未正式准入

`warehouse.status=pending`，清单没有Warehouse政策。`warehouse_policy_artifacts()`按预期拒绝：`warehouse_release_pending`。本次没有观察20k完成或执行恢复，训练状态由单独恢复链核验。
本地`output/**/budget.lock*`无匹配，`output/warehouse_v6_sealed_claims`不存在；这个本地库存检查不证明其他检出中没有消费。

必要后续顺序：

1. 完成并审计真实20k双版本训练及显式安全恢复链。
2. 最终20k NN的四角色程序绑定及内部预检。
3. 全新75场景×reference/raw/regularized的225局任务及程序预检，逐帧独立重放。
4. 三开发继续训练随机流全部通过后，锁最小200k／300k／400k预算。
5. 正式三配对只按任务质量选定模型，固定Actor和程序。
6. 先程序预审，再一次性独立程序审计，再每task100个新种子的单次留出。
7. 三域演示、真人接管、恢复、回放、反事实的逐帧绑定检查，工程门及内容寻址发布包。

通用`experiments/warehouse/protocol.py`和`admission.py`仍登记v6.18方法`independent_selective_masked_distribution_v4`和旧BC／程序池。v6.42深12／256叶及参考KL继续训练需要独立版本化开发、正式、准入和打包兼容层，不能修改全局常量重解释旧收据。

## 结论

**Pong／Kitchen既有资产完整性与绑定复核通过；Warehouse正式验收未完成，v6发布与pilot继续关闭。**冻结参考NN的程序确认、训练池程序指标、20k校准均不等于正式模型验收。

本次没有运行逐帧测试，已有测试按原日期和原范围解释。程序、报告、物理配置、双角色哈希及核验源码SHA见机器证据。
