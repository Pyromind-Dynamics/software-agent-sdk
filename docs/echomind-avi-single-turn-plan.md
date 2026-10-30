# EchoMind AVI 单轮训练闭环方案

> 状态：P1.5 已完成，P2 进入阶段执行与前端验证
> 目标：先跑通 Dataset -> Inference -> Evaluator -> Train 的最小纵向闭环

## 1. 目标

第一阶段不追求完整生产调度，而是打通以下路径：

```text
对话说明目标
  -> EchoMind Manager 读取 avi-single-turn Skill
  -> 生成可确认的 WorkflowPlan
  -> 用户确认
  -> 创建 Project/Graph
  -> 同一会话获得完整开发能力
  -> 逐阶段启动和验证
  -> 保存输入、Run、Artifact、错误和结果
```

要验证的核心不是一次生成多复杂的工作流，而是每个阶段都有明确契约、可以独立
运行、可以观察结果，并且开发态改动不会污染生产态执行。

## 2. 当前基线

已实现：

- 两个独立 Chat API；
- EchoMind Manager 规划 Profile；
- SQL PlanProposal / PlanRevision；
- 前端计划草稿、Diff、确认、拒绝和历史 Revision；
- 确认后立即创建 Project/Graph；
- 后续 Revision 更新同一个 Project/Graph；
- 确认后同一会话切换到完整开发 Tool/Skill；
- Pi 远端 Sandbox 执行面；
- Project Graph 查询和节点启动/运行 API；
- AVI 四阶段 Graph 模板；
- Stage Contract Probe、Run 和 Artifact；
- 前端确认后读取真实 Graph。

尚未实现：

- Dataset 真实平台工作流执行器；
- Inference、Evaluator、Train 真实执行器；
- 生产发布协议和事件调度；
- 版本回滚；
- 多场景 Skill 自动路由。

## 3. 阶段计划

### P0：双 Agent 隔离

内容：

- 拆分 Pyromind 与 EchoMind Manager 的 Chat 路由；
- 独立 Profile、System Prompt、Tool 和 Skill allowlist；
- 删除旧 EchoMind Chat 兼容入口；
- 两条路径共享底座但不互相转交会话。

验证：

- EchoMind 创建、列表、快照、命令和 SSE 可用；
- Pyromind 原链路无变化；
- 两边不能混用会话；
- 删除的旧路由返回 404。

### P1：计划与 Project

内容：

- 增加 `create_experiment_plan`、`revise_experiment_plan`、
  `get_experiment_plan`；
- 规划阶段只开放 `read` 和三个计划工具；
- 前端显示计划、待确认提案、更改点、历史和确认按钮；
- 确认时提交 SQL Revision；
- 从 Revision 生成 Project/Graph。

验证：

- 未确认时不存在 Project；
- 首次确认后 Project ID 等于 Revision.project_id；
- 重复确认命令幂等；
- 第二个 Revision 更新同一个 Project/Graph；
- Graph 节点与计划阶段一致。

### P1.5：确认后的开发策略切换

内容：

- Product Store 持久化 planning/development allowlist；
- 确认后原子激活开发 allowlist；
- Pi Adapter 持久化 `session.json`，下一次 prompt 前重启 runner；
- 恢复会话后策略保持；
- OpenHands Adapter 明确不支持 runtime policy change，避免通用层分支。

验证：

- 规划 prompt 看不到 `terminal`、`sandbox`、`df_submit_pipeline`；
- 确认后的下一条 prompt 可以看到完整开发工具；
- 实际工具调用受同一 allowlist 约束；
- 重启后仍加载开发策略；
- 开发工具集不低于原 Pyromind Agent 能力。

### P2：Graph 与阶段壳

内容：

- Project 文件持久化 `project.json` 与 `graph.json`；
- Graph 节点带 `launch_mode`、`executor`、`config`；
- Dataset 节点启动 EchoMind 开发任务；
- 其他节点可由确定性 `contract_probe` 执行；
- 每次 Stage Run 记录状态、错误和 Artifact；
- 前端显示真实 Graph 和逐阶段操作。

验证：

- 四个阶段都能独立触发；
- Dataset 启动后事件进入当前 EchoMind 会话；
- 每个 executor 节点能产生一条 Run；
- 成功与失败状态都能在 Graph 中刷新；
- Artifact 可读取；
- 刷新页面后 Graph 和 Run 仍可恢复。

### P3：Dataset 接通 data-processing

内容：

- 开发态 Dataset 任务进入通用 `data-processing` Skill；
- 探查真实 Storage 数据，不使用静态 AVI requirements；
- 小样、用户确认、全量验证只在开发态执行；
- 固化后的 Graph 保存已验证执行规格。

验证：

- 能从对话启动 Dataset 开发；
- Agent 能读取真实 Sandbox/Storage 数据；
- Pipeline 小样可以确认；
- 全量任务可以提交和查询；
- 通用 Skill/Tool 不出现 EchoMind 或 AVI 专属分支。

### P4：Inference、Evaluator、Train

内容：

- Inference 执行器接入平台模型推理；
- Evaluator 输出标准 ATIF/ATIF-like 产物；
- Train 创建并跟踪 SFT/DPO 平台工作流；
- 阶段间只用 Artifact/Run 引用传递数据；
- 失败可重试，错误结构可回传 Agent。

验证：

- 固定评测集口径一致；
- Evaluator 可渲染标准结果；
- Train 能读取 Dataset 和 Evaluator 产物；
- 任一阶段失败不会伪造成功；
- Run 历史可追溯。

### P5：发布态

内容：

- 已验证 Graph 转换为稳定执行协议；
- 事件、定时、循环和状态机由执行引擎负责；
- Agent 只按独立任务读取错误并辅助修复；
- 配置、Revision、Run 和 Artifact 保持可追溯。

验证：

- 发布后不依赖 Chat 保持在线；
- 同一冻结定义可重复执行；
- 失败事件可重新唤起独立 Agent 任务；
- Agent 不成为生产环境唯一调度器。

## 4. AVI 计划契约

AVI 的 `WorkflowPlan`：

```text
skill = avi-single-turn
stages = dataset, inference, evaluator, train
edges = dataset -> inference -> evaluator -> train
```

节点配置来自计划：

- `resources` 表达数据集、模型或其他外部资源；
- `tasks` 表达阶段内工作、输入、预期产物和完成条件；
- `unresolved` 表达尚未确认的信息；
- 不生成 `project_request.json`；
- 不维护另一份 AVI requirements 配置。

Project ID 由 PlanRevision 提供，Graph ID 由服务端首次创建时生成。后续 Revision
保留这两个 ID。

## 5. 节点与执行边界

| 节点 | 当前 launch mode | 首期执行器 | 后续 |
|---|---|---|---|
| Dataset | Agent | 启动当前 EchoMind 开发会话 | 接通用 data-processing |
| Inference | Executor | `contract_probe` | 平台推理 |
| Evaluator | Executor | `contract_probe` | 标准评测与报告 |
| Train | Executor | `contract_probe` | SFT/DPO 平台工作流 |

Contract Probe 只证明：

- 节点 ID、Graph 和 Stage API 契约可执行；
- Run 状态能持久化；
- Artifact 能持久化并读取；
- 前端能刷新并显示结果。

Contract Probe 不证明业务阶段已经完成。

## 6. 前后端交互

EchoMind Chat：

```text
GET/POST /api/v2/echomind/conversations
GET      /api/v2/echomind/conversations/{id}/snapshot
POST     /api/v2/echomind/conversations/{id}/commands
GET      /api/v2/echomind/conversations/{id}/events
```

Project/Graph：

```text
GET  /api/echomind/projects/{project_id}
POST /api/echomind/projects/{project_id}/nodes/{node_id}/start
POST /api/echomind/projects/{project_id}/nodes/{node_id}/run
GET  /api/echomind/projects/{project_id}/runs
GET  /api/echomind/projects/{project_id}/artifacts/{artifact_id}
```

前端在计划确认后读取 Project，而不是把计划临时拼成第二份 Graph。

## 7. 测试命令

```bash
uv run pytest -q tests/pyromind_runtime
uv run pytest -q tests/agent_server/test_echomind_router.py
```

```bash
cd harness-adapter/pi-runtime
npm test -- --run
```

```bash
cd /Users/zhangpeng/PycharmProjects/minimal-chat-frontend
npm test
npm run build
npm run lint
```

## 8. 当前验证重点

P2 验收时至少手工确认：

1. 新 EchoMind 会话不带任何业务 ID；
2. 对话生成 AVI 计划，确认前没有 Project；
3. 点击确认后立即出现 Project ID 和真实 Graph；
4. 同一会话下一条消息可以使用开发工具；
5. Dataset 节点可启动到当前会话；
6. 其余节点可独立产生 Contract Probe Run；
7. Graph、Run、Artifact 在刷新后仍存在；
8. 修改计划并确认后仍只有一个 Project；
9. Pyromind Workbench 行为和旧 `/api/v2/pyromind` 入口不受影响。
