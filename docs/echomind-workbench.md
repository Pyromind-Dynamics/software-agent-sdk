# EchoMind 当前工作台链路

当前入口是“对话准备 → 审阅 workflowPLAN → 创建并启动 Project → 阶段工作台”。
不再使用旧实验计划版本面板、SQL 后台工作队列面板或旧 baseline 验收客户端。

## 准备与执行

- 创建 Project 前，Manager 根据 preparation policy 收集事实、探查资源并生成计划。
- 用户通过创建按钮确认当前 preparation revision；Runtime 创建 Project，
  激活开发工具策略并通知 Manager。
- Manager 使用原生 `subagent` 委派已授权角色；角色工具和 Skill 由 Profile 限定。
- pi-subagents 管理运行、并发、续跑和原生通知；Product Runtime 不另建角色调度器。
- 工作流提交、权限确认、平台任务关联与回调继续使用通用 Runtime 和 Adapter 端口。
- 场景差异只在 Profile、Prompt 和场景 Skill 中，前端不根据角色硬编码业务逻辑。

## 状态与前端

前端通过 Snapshot/SSE 显示准备、Project、阶段会话和每列的运行记录。
列内下拉优先显示运行中的任务，展开后可查看当前快照中的运行和续跑历史。
记录只反映已接收和已观测到的运行，不承诺 FIFO，也不包含被插件拒绝的委派。

子 Agent 的一次运行结束不代表平台任务完成：

- 提交后等待平台结果，显示“等待平台结果”。
- 平台结果返回后，区分等待分析、正在分析和报告交付。
- `resume` 保留原任务上下文，界面显示续跑次数，不代表清空上下文重建任务。

已有报告、失败、暂停等状态仍需结合执行和平台任务证据判断。
只读状态投影不得派发任务、唤醒 Manager 或重提平台工作。

## 保留的兼容边界

当前 preparation / Project 仍复用 `domain/experiment_plans.py` 中的工作流模型。
该模块名不意味着旧实验计划执行链仍在使用。

公开 Snapshot 的旧可选字段与已有数据库迁移链保留，以免破坏其他客户端、
已有部署或持久化解析。删除本地历史会话不等于这些外部契约已全部退役。
本次清理不重置数据库、不删除迁移、不改变公开协议版本。

## 验证

```sh
uv run pytest tests/pyromind_runtime
```

在 `minimal-chat-frontend` 中运行 `npm test`、`npm run lint` 和 `npm run build`。
实际平台验收通过当前工作台进行，核对平台任务 ID、状态回调、报告和续跑上下文；
不再依赖旧 `experiment_queue` 或已撤销的实验产物读取接口。
