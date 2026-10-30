# EchoMind 双 Agent 改动设计

> 状态：P1.5 已实现
> 范围：Pyromind Workbench 与 EchoMind Manager 的入口、Profile、会话策略、
> Project/Graph 生命周期和前端边界

## 1. 结论

系统只有两个独立 Agent：

| Agent | 入口 | 职责 |
|---|---|---|
| Pyromind | `/api/v2/pyromind/conversations` | 原有 Workbench 开发对话 |
| EchoMind Manager | `/api/v2/echomind/conversations` | 从实验规划进入 Project/Graph 开发 |

它们是两个 Agent Profile、两个 Chat API 和两条独立执行契约，不是两套框架。
两者可以共用底层 SDK、Harness、Tool、Skill 和 Sandbox，但不会互相转交会话，
也不能读取或续接对方的 Chat。

EchoMind Manager 不是第三个 Agent，也不是 Pyromind 的子 Agent。它是 EchoMind
业务自己的主 Agent，规划阶段和确认后的开发阶段仍属于同一个 EchoMind 会话。

## 2. 产品生命周期

```text
EchoMind Chat
  -> 规划阶段：澄清需求，生成、修改、查询实验计划
  -> 用户在计划面板确认
  -> SQL PlanRevision 提交
  -> 以同一 Revision.project_id 创建或更新 Project/Graph
  -> 同一 EchoMind 会话切换到开发工具与 Skill
  -> 用户和 Agent 逐阶段调试
  -> 后续按稳定协议发布到生产执行引擎
```

关键边界：

- 计划草稿由 `create_experiment_plan` / `revise_experiment_plan` 产生。
- 只有前端面板提交 `confirm_experiment_plan` 才算确认。
- SQL 中的 PlanRevision 是计划、Project ID 和 Revision 的事实源。
- 文件型 Project/Graph 从已确认的 `WorkflowPlan` 生成，不维护第二份 AVI
  requirements 配置。
- 首次确认创建 Project；后续 Revision 确认更新同一个 Project 和 Graph ID。
- 已确认计划不等于已执行，各阶段仍需显式启动。
- 生产态由平台执行引擎持续运行，Agent 不承担生产调度。

## 3. 会话策略切换

### 3.1 规划阶段

EchoMind Manager 只加载：

```text
read
terminal
create_experiment_plan
revise_experiment_plan
get_experiment_plan
```

Skill：

```text
experiment-planning
```

这个白名单同时约束模型可见工具、实际工具调用、Skill 可见范围和工作区读取。
规划阶段不能写工作流文件、创建 Sandbox 或提交平台任务。

### 3.2 确认后的开发阶段

确认成功后，Runtime 按顺序执行：

1. 提交 SQL PlanRevision；
2. 调用 Project binder，从 Revision 生成或更新 Project/Graph；
3. 原子切换 Product Store 中的会话 allowlist；
4. 更新 Harness 会话策略；
5. 在下一个 prompt 前重启 Pi runner，使新工具和 Skill 生效。

开发工具集等于 Pyromind Agent 的完整开发工具集，再加三个计划工具。开发 Skill
集等于 Pyromind Agent 的完整 Skill 集，再加 `experiment-planning`。

确认命令使用 `command_id` 保证幂等。若确认在 SQL 提交后、文件绑定或策略切换时
中断，同一条命令重试会继续完成剩余步骤，不会创建第二个 Project。

## 4. 路由与 Profile 隔离

两个入口分别由自己的 `ConversationProfile` 注册：

| 项目 | Pyromind | EchoMind Manager |
|---|---|---|
| Chat API | `/api/v2/pyromind/conversations` | `/api/v2/echomind/conversations` |
| Harness | 按现有配置 | Pi |
| Profile ID | Workbench 默认 | `echomind` |
| Usage ID | `pyromind` | `echomind-manager` |
| Conversation marker | Workbench marker | `echomind` |
| 前端入口 | Workbench | `?chat=echomind` |

EchoMind 的 Project 查询和 Stage API 位于 `/api/echomind/projects`，它们读取已经
由计划确认生成的资源，不是第二个 Chat API，也不提供手动创建 Project。

不保留以下兼容面：

- 旧 `/api/echomind/conversations`；
- 旧 EchoMind Chat Profile；
- `project_request.json` 创建流程；
- 单独 `POST /api/echomind/projects`；
- `?chat=echomind-legacy`；
- 旧 EchoMind 会话迁移。

## 5. Sandbox 与共享底座

两个 Product Runtime 会话都由 Pi Adapter 管理：

```text
Chat API
  -> Product Runtime
  -> PiAdapter
  -> resolve_pi_terminal_backend()
  -> 平台 Sandbox
```

默认执行面必须是远端平台 Sandbox，而不是本机 LocalWorkspace。EchoMind 与
Pyromind 使用同一个 Sandbox 实现和同一套 Storage 挂载语义。

共享底座只提供中立能力：

- Skill/Tool 不能判断调用方是 EchoMind 还是 Pyromind；
- EchoMind 的场景提示和工具差异放在自己的 Profile、Prompt 和 Skill；
- 通用 `data-processing`、`sandbox`、`generate-workflow-dsl` 等 Skill 不加入
  AVI 或 EchoMind 分支；
- 两条链路的测试分别维护，公共能力只通过独立测试验证。

## 6. 实验 Graph

通用 Skill 先识别场景，再按需读取 reference。AVI reference 首期使用四阶段：

```text
dataset -> inference -> evaluator -> train
```

Graph 生成规则：

- `skill=experiment-planning` 只记录规划入口，不用于选择后端模板；
- 后端统一从计划中的 stages 和 edges 生成节点与边；
- 节点 title 来自计划；
- 节点 `config.resources` 和 `config.tasks` 来自计划；
- Dataset 节点使用 `launch_mode=agent`，绑定 `data-processing` 能力；
- Inference、Evaluator、Train 首期使用确定性执行器契约壳；
- Evaluator 标准产物格式为 `atif`；
- 节点和边不另读 Project Request 或 AVI requirements 文件。

首期 Stage Run 仍以 `contract_probe` 验证状态、Run、Artifact 和前端交互。
这不代表真实数据处理、推理、评估或训练执行器已经完成。

## 7. 前端

EchoMind 页面只有一个正式入口：`?chat=echomind`。

页面结构：

- 左侧：EchoMind 会话历史；
- 中间：对话；
- 右侧：实验计划和已持久化 Project Graph。

计划面板负责草稿、历史 Revision、Diff 和确认操作。计划确认后，前端使用
`experiment_plan.project_id` 读取 `/api/echomind/projects/{project_id}`，
展示真实 Graph，并可逐节点启动开发或运行验证。

旧的 legacy 页面、project-request 读取和手动创建 Project 不再暴露。

## 8. 代码落点

| 内容 | 位置 |
|---|---|
| EchoMind Profile 与工具清单 | `openhands-agent-server/openhands/agent_server/agent_profiles/echomind_manager.py` |
| Product Profile 注册 | `pyromind-agent-server/pyromind_agent_server/profiles.py` |
| 通用会话策略 | `pyromind-runtime/pyromind_runtime/domain/session_policy.py` |
| 确认与策略切换 | `pyromind-runtime/pyromind_runtime/application/conversation_runtime.py` |
| Project binder | `pyromind-agent-server/pyromind_agent_server/experiment_plan_projects.py` |
| Graph 生成 | `openhands-agent-server/openhands/agent_server/echomind_graph_templates.py` |
| Project/Graph 持久化 | `openhands-agent-server/openhands/agent_server/echomind_projects.py` |
| Project/Stage API | `pyromind-agent-server/pyromind_agent_server/api/echomind_projects.py` |
| 通用实验规划 Skill | `.agents/skills/experiment-planning/` |
| AVI 场景 reference | `.agents/skills/experiment-planning/references/avi-single-turn.md` |

## 9. 验证

后端：

```bash
uv run pytest -q tests/pyromind_runtime
uv run pytest -q tests/agent_server/test_echomind_router.py
```

前端：

```bash
npm test
npm run build
npm run lint
```

重点验证：

1. 两个 Chat API 独立创建、列出和恢复会话；
2. EchoMind 规划阶段看不到开发工具；
3. 计划确认后同一会话加载完整开发工具；
4. Project ID 等于 PlanRevision.project_id；
5. 后续 Revision 更新同一个 Project/Graph；
6. 前端确认后显示真实 Graph，并可按阶段触发验证；
7. 旧 EchoMind 路由和兼容入口不存在。
