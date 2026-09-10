# 仓库 A/B 预实验 Render 部署

## 部署范围

公网服务继续使用 XAI 主库与现有 Render 服务：
`https://policylens-warehouse-study.onrender.com`。页面先展示数据说明并要求显式勾选同意，再原子登记用户 ID、分组并进入操作练习，之后完成 Task 1 三局、Task 2 三局、冻结问卷和回放。四人区组覆盖 `A/XY`、`A/YX`、`B/XY`、`B/YX`；A 组只能在 Task 1 进行中及回顾时询问，B 组只看回放，Task 2 两组均不能读取解释。

当前部署是在线技术试玩，`formal_ready=false`。免费 Render 实例没有持久磁盘，SQLite 位于 `/tmp`，休眠、重启或重新部署后记录可能丢失，因此此地址不能直接用于需要持久保存的正式人类实验。

## 冻结产物

- 冻结 Actor SHA-256：`309b6e53fe682bead8d3443015aca27eae60e561175e71d7c25f57314ac69d5b`
- 父级本地发布 manifest SHA-256：`cdcd44e3f8db950f5d5e36b175ed389ea54bd16748b504d7660d9e44cc86ad28`
- 在线 ZIP SHA-256：`011f9bd9fe93d857e960145b5e6fc2b68b049b8a269345c20942e82484260e15`
- 在线 manifest SHA-256：`2a3aedf224d617eeb346a666a84d628d52f7473818457e2272dcb9d497677726`
- Secret File：`/etc/secrets/warehouse_alignment_release.b64`

Secret File 包含冻结 Actor、完整协议、12 个试玩场景、抽取程序和 8 道带私有答案的冻结题目。它不进入 Git；加载器会校验 Base64、ZIP 白名单、解压大小、逐项哈希、父级身份和当前运行源码。

## 连续动画

每个已确认联合动作在 Canvas 内用 380 ms 平滑插值显示两台机器人的实际起点和终点。双方同时移动时同步播放；等待、撞墙、机器人冲突、刷新、提问和历史回放不会产生假移动。动画期间操作保持锁定，结束后才采用服务器的新状态，页面布局和滚动位置不重建。

## 本机复现

```bash
python scripts/build_warehouse_alignment_online_release.py \
  --parent-release-root output/warehouse_native/alignment_local_pilot_release_r3_animation_395m_20260910 \
  --expected-parent-manifest-sha256 cdcd44e3f8db950f5d5e36b175ed389ea54bd16748b504d7660d9e44cc86ad28 \
  --output-package /new/path/warehouse_alignment_online.zip \
  --output-base64 /new/path/warehouse_alignment_release.b64

python -m ui.warehouse_alignment_online_server \
  --base64 /etc/secrets/warehouse_alignment_release.b64 \
  --expected-package-sha256 011f9bd9fe93d857e960145b5e6fc2b68b049b8a269345c20942e82484260e15 \
  --expected-manifest-sha256 2a3aedf224d617eeb346a666a84d628d52f7473818457e2272dcb9d497677726 \
  --database /tmp/warehouse_alignment_online.sqlite3 \
  --storage-mode ephemeral \
  --public-origin https://policylens-warehouse-study.onrender.com
```

Render 构建只安装 NumPy。训练、模型选择和 RCPD 拟合仍在本机完成，线上进程不会导入 PyTorch、scikit-learn 或训练模块。

## 验证

- 线上发布、轻量运行时、解释器和 A/B 存储测试：20 项通过。
- 连续动画及既有回放/权限前端测试：12 项通过。
- 冻结 Actor 在 12 个场景的 120 次决策与原运行时动作、概率和推进后状态一致。
- 中文、英文的原因、反事实及规则回答与原已验收解释器一致。
- 真实发布包通过四人区组、ID 大小写判重、幂等操作、Task 1 权限、Task 2 隔离、回放和进程重启恢复测试。
