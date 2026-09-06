# 协作厨房操作时页面位移修复

每次正常请求都曾插入等待确认提示，令地图和操作区上下移动约 56.8 px。现在正常请求使用已有连接状态；只有请求失败、需要重试或确有通知时才显示提示。按住方向键时继续阻止浏览器滚动，同时只提交一次动作，保留输入框和按钮的原生键盘行为。

运行时代码相对 `355e2757d78559135527d55f02027e8b4f45957b` 仅改动 `ui/cooperative_kitchen_web/app.js`。新的布局夹具在 1365×900、1280×800 和中英文下通过 21 项检查，覆盖点击、键盘、按住方向键以及丢失响应后的重试；采样中的地图、按钮和滚动位置最大变化均为 0 px。同一夹具对旧脚本检出 12 项失败。A/B 页面协议夹具 23 项、用户 ID 入口夹具 9 项通过。这些是前端夹具验证，不是新一轮神经策略或云端问答验证。

另用现有远程服务的自由试玩，对浏览器中的脚本作本地响应覆盖预览，点击与键盘操作均未再产生地图位移。该记录明确标记 `actual_remote_freeplay_with_local_frontend_overlay`，不构成修复已部署的证据；部署后的实测另行记录。

本次仍为 candidate。环境、后端、Actor、程序、场景、冻结题库和 QA 配置保持不变。四份涉及 runtime 绑定的报告采用明确的前端兼容性封装，保存原 runtime、manifest 和报告 SHA；原有训练、QA、数据库恢复、四会话和二十会话数字没有重测或改写。历史失败结论继续保留，不据此宣称支持更多并发参与者。原训练、抽取、校准和题库验收报告字节不变。

复现命令（需已安装 Playwright，可用 `KITCHEN_PLAYWRIGHT_MODULE` 和 `KITCHEN_CHROME` 指定本机运行时）：

```sh
node tests/cooperative_kitchen_study_browser.cjs --layout
node tests/cooperative_kitchen_study_browser.cjs
node tests/cooperative_kitchen_enrollment_browser.cjs
```

原始报告及旧发布文件保存在私密输出目录，公开 Git 仅包含代码、此说明和不含内容的发布校验描述文件。新 manifest 继续遵循现有会话版本检查，历史数据保留。
