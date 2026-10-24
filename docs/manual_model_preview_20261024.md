# 三域指定模型线上试玩

目标网站：`https://policylens-warehouse-study.onrender.com/`。
此文件记录手动试玩部署，不是模型验收收据。

## 不可变模型

- Warehouse：v6.42 实际20k训练后的raw和regularized，双Actor NPZ SHA分别为
  `ca7bfddfca1eb89d3c73935cd22d3828600009053e80663cca58e66de9703677`、
  `2ff0ed2778ecc14b3cd170aef4eb0410ab6a30382e9109b94fb0531874c93ab1`。
  regularized通过本轮校准，raw停机率5/75，配对校准未过；正式发布仍未获准。
- Pong、Kitchen：原已验收的v5 raw/regularized，Actor与原验收文件逐字节复制。
- 内容包：`releases/policylens-manual-model-preview-20261024.v1/catalog.json`，
  SHA `ec7b6370454321748f9767c5cbbce4f630d539afb5763da742f6da6f7f543d43`。
  该ID只表示手动预览，不是v6或v7科学发布；所有域均不能据此包获得pilot权限。

## 运行

```sh
python -m pip install -r requirements-manual-preview.txt
OPENBLAS_NUM_THREADS=1 python -m ui.model_preview_server --host 0.0.0.0 --port "$PORT"
```

页面允许选择三域、两个版本、三个任务和24个明确的试玩场景。
WASD/方向键控制角色1，空格等待，Kitchen使用E操作工位。
双NN播放与真人接管使用同一冻结绑定，只有角色1被替换。
每个提交使用command_id和expected_turn，重试不重复推进。
下载记录时独立重新执行本局所有NN动作和环境物理步骤，拒绝不一致记录。

Warehouse NumPy适配器的类和函数体与原`runtime_v642/warehouse.py`一致，
数据快照的四个函数与原冻结物理读者一致；只分离会引入Torch的训练导入。
显示和事件函数也保持原函数体。AST一致性、原实现逐帧数值一致性及禁用Torch导入测试分别覆盖。
未修改原447文件的科学绑定源码、模型权重或物理规则。

## 数据与边界

- 独立SQLite预览存储，不读取或修改Render既有`DATABASE_URL`所指参与者数据库。
- 只支持手动预览，不挂载参与者报名、Prolific、奖励或pilot授权。
- 页面和`/api/release`明确报告pilot关闭、Warehouse验收未完成、Pong/Kitchen历史验收范围。
- 此次不运行训练、程序拟合或旧留出集，不生成新的模型质量成绩。
- 不提供程序问答，避免将开发校准程序冒充最终独立审计解释。
- Render Free的预览存储不承诺持久化，用户可下载本局记录；原数据库和已有历史记录留存。

## 上线前的线上状态

2026-10-24只读核对：网站原来运行`policylens-three-domain-20260922.v3.10-prolific`，
commit `36d3173f054043811f9dc24a8b134370736006f6`；不是本次指定模型。
Render服务为`srv-da66ggbl550s738j2q9g`，Free Singapore。
部署完成与否以独立保存的Render及线上HTTP核验记录为准。
