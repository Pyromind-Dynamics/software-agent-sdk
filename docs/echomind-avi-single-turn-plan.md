# EchoMind 实验工作台技术方案

> 状态：P0、P1、P1.5 和 P2 基础能力已落地；下一步进入 M1。
> 更新时间：2026-10-30
> 首期场景：AVI 单轮 SFT/DPO，AVI 仅作为 `experiment-planning` 的一个 reference。
>
> 文件名保留历史原因。本文是 EchoMind 当前技术方案，内容以当前代码实现为准。
> 可执行 Graph、生产异常降级和主/子 Agent 结构化契约见
> `docs/echomind-graph-agent-contract.md`。
>
> 执行层最新渐进方案见 `docs/echomind-four-node-graph-execution-plan.md`。
> 开发态只保留一张可变的 Working Graph，不引入 GraphRevision；发布时才生成冻结
> 快照。本文第 9 节中开发态先做 GraphRevision 的路线以新文档 S1-S8 为准。

## 1. 结论

当前已经打通：

```text
独立 EchoMind Chat
  -> 通用 experiment-planning Skill
  -> 按需加载 AVI reference
  -> 生成、修改和确认 WorkflowPlan
  -> 创建或更新 Project/Graph
  -> 同一 EchoMind 会话切换到完整开发能力
  -> 在远端 Sandbox 中继续开发
  -> Graph 节点可分别启动 Agent 开发或执行验证
```

但当前完成的是“Graph 和阶段执行契约”，还不是“四个业务阶段都已执行”。

准确状态：

- Dataset：可以把阶段开发任务投递给当前 EchoMind 会话，但还没有创建
  StageRun，也没有把 `data-processing` 小样、用户确认和全量任务绑定到阶段；
- Inference、Evaluator、Train：`/run` 只执行 `contract_probe`，用于验证 Run、
  Artifact、状态和前端接口，不产生真实推理、评估或训练结果；
- Project 生命周期已有 `development/frozen` 字段，但尚未实现冻结和发布协议；
- 当前整体目标是先把 Dataset 做成第一条“从阶段启动到 Run 可追溯”的纵向
  闭环；执行上先通过 M1 收敛现有 Graph 契约，再逐步接入可执行 GraphRevision 和
  NodeRun，不直接修改 Dataset Executor。

目标设计进一步将当前 Stage 级 Graph 升级为 Task 级 GraphRevision，并明确：

- GraphRevision 与 Run 状态分离；
- Node 通过 Artifact Port 和 Edge Binding 连接；
- 生产异常暂停后 Fork Development Branch，不原地修改冻结 Graph；
- 主 Agent 做业务决策，Runtime 做确定性调度和状态记账；
- 主/子 Agent 使用结构化 Decision、Command、Task 和 Result 契约。

## 2. 产品边界

EchoMind 以 Project 为单位承载训练和实验工作台，分为两个状态：

### 2.1 开发态

用户和 Agent 澄清目标、准备数据、搭建 Graph、逐阶段运行和调试。允许修改计划、
节点配置、脚本和参数。

开发态包含两个会话策略：

1. 计划确认前：只能做需求澄清、只读探查和计划提案；
2. 计划确认后：同一个 EchoMind 会话获得完整开发工具和 Skill。

这是同一个 EchoMind Manager 会话的策略切换，不是两个 Agent，也不是会话交接。

### 2.2 生产态

已验证的 Graph 被冻结并发布，由平台执行引擎按稳定协议运行。生产运行不依赖某个
Chat 会话持续在线，Agent 不承担生产调度器职责。

当前尚未实现生产态发布。首期只要求开发态中每个阶段都能启动、观察和追溯。

## 3. 双 Agent 与独立入口

系统有两个相互独立的 Agent Profile 和两个独立 Chat API：

| Agent | Chat API | 职责 |
|---|---|---|
| Pyromind Workbench | `/api/v2/pyromind/conversations` | 原有 Workbench 开发对话 |
| EchoMind Manager | `/api/v2/echomind/conversations` | 实验规划、Project/Graph 和 EchoMind 开发 |

两者共享 SDK、Harness、Tool、Skill 和 Sandbox，但：

- 不互相转交会话；
- 不能读取或续接对方的 Chat；
- EchoMind 不会自动调用 Workbench Agent；
- 通用 Skill 和 Tool 不判断调用方是 EchoMind 还是 Workbench。

当前 EchoMind 只有一个 Profile：

```text
profile_id = echomind
usage_id   = echomind-manager
```

EchoMind 页面入口：

```text
/?chat=echomind&userId=<debug-user-id>
```

## 4. 当前执行架构

```text
EchoMind UI
  -> /api/v2/echomind/conversations
  -> Product Runtime
  -> PiAdapter
  -> PYROMIND_PI_TERMINAL_BACKEND=sandbox
  -> 远端平台 Sandbox
```

Pyromind Workbench 使用同样形态的 Pi + Sandbox 执行面，但入口、Profile 和会话
策略独立。

Skill 和知识库正文读取由 host mirror 提供，不因为没有启动 Sandbox 而失败；
`terminal`、Skill 脚本、DataFlow 和业务工具调用仍在远端 Sandbox 中执行。
Sandbox 会从归档恢复会话资源，不逐文件上传。

## 5. 规划、确认与开发策略

### 5.1 规划阶段

Tool：

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

通用 Skill 先识别场景，再按需读取 reference。当前只有：

```text
.agents/skills/experiment-planning/references/avi-single-turn.md
```

规划阶段可以只读探查 Storage，但不能写文件、提交平台任务、创建 Sandbox 或伪造
Project/Run/Artifact。该限制目前主要由 System Prompt 和 Skill 约束，`terminal`
本身没有被平台权限做只读隔离；这是后续需要评估的安全边界。

### 5.2 计划确认

SQL `PlanProposal/PlanRevision` 是计划事实源。用户必须通过前端计划面板确认，
聊天里的“确认”不替代面板操作。

确认顺序：

1. 校验 `expected_revision`；
2. 确认或推进 SQL PlanRevision；
3. 使用 `Revision.project_id` 创建或更新文件型 Project/Graph；
4. 原子切换 Product Store 中的开发 Tool/Skill allowlist；
5. 更新 Pi session policy；
6. 在下一次 prompt 前重启 Pi runner，使新能力生效。

确认命令通过 `command_id` 幂等。首次确认创建 Project；后续 Revision 保留同一个
Project ID 和 Graph ID，并更新 Graph。

计划确认是“开发准备”动作，不等于执行：

- 确认后立即存在 Project 和 Graph，但不会自动创建 GraphRun；
- 确认只解锁当前 EchoMind 会话的开发工具，不会自动启动节点或提交平台任务；
- Graph/节点的真正执行仍由明确的启动动作触发；
- 用户确认后续 Revision 时只更新同一个 Project 的 Graph，不自动重跑已有 Run。

### 5.3 开发阶段

开发 Tool 集等于 Pyromind Workbench 的完整开发 Tool 集，再加三个计划工具。
开发 Skill 集等于 Workbench 的完整 Skill 集，再加 `experiment-planning`。

重点能力包括：

- Storage 和 Sandbox 读写；
- `data-processing`；
- `generate-workflow-dsl` 和 Workflow 调试；
- 推理、评测和训练分析；
- 平台任务提交、观察和停止。

EchoMind 特有语义只放在 EchoMind Profile、System Prompt、`experiment-planning`
及后续 EchoMind 专属阶段服务中，不进入通用 Tool 或通用 Skill 的业务分支。

## 6. Project/Graph 数据模型

Project 存储于 `conversations_dir.parent / "projects"`：

```text
projects/<project_id>/
  project.json
  graph.json
  runs/<run_id>.json
  artifacts/<artifact_id>.json
```

当前 `graph.json` 已经是一份可用的 Stage 级 Graph 快照，包含节点、边、
`launch_mode`、`executor`、资源和任务配置。后续 GraphRevision 以此为基础演进，
不会另造一套与当前 Graph 无关的模型。

Graph 从已确认 `WorkflowPlan` 的 `stages`、`edges`、`resources` 和 `tasks` 生成，
不根据 Skill 名选择 AVI 模板。

当前模板标识：

```text
experiment-planning/v1
```

当前支持的节点类型：

```text
dataset
inference
evaluator
train
```

节点名称无法识别为以上类型时不会进入首期 Graph。Graph 要求无环，节点 ID 必须
唯一。

当前 Graph 生成规则：

- Dataset 节点使用 `launch_mode=agent`，绑定 `data-processing`；
- 其他节点使用 `launch_mode=executor`；
- 所有节点的 `executor` 当前都是 `contract_probe`；
- Evaluator 节点配置 `artifact_format=atif`；
- Project/Graph 中的节点状态初始为 `idle`，前端优先用最新 StageRun 覆盖展示。

## 7. 当前阶段执行能力

### 7.1 节点矩阵

| 节点 | launch mode | 启动接口 | 当前实际行为 |
|---|---|---|---|
| Dataset | Agent | `POST .../nodes/{node_id}/start` | 向当前 EchoMind 会话投递阶段开发任务 |
| Inference | Executor | `POST .../nodes/{node_id}/run` | 生成 `contract_probe` Run 和 Artifact |
| Evaluator | Executor | `POST .../nodes/{node_id}/run` | 生成 `contract_probe` Run 和 Artifact |
| Train | Executor | `POST .../nodes/{node_id}/run` | 生成 `contract_probe` Run 和 Artifact |

`contract_probe` 只证明：

- 节点 ID、Graph 和 API 契约可执行；
- Run 状态能持久化；
- JSON Artifact 能持久化并读取；
- 前端能刷新并显示结果。

它不证明任何业务阶段已经完成。

### 7.2 StageRun

当前 StageRun 状态：

```text
running | succeeded | failed
```

StageRun 会记录：

- `project_id`、`node_id`、`node_type`、`executor`；
- 状态、消息、错误和时间；
- `input_refs` 字段；
- Artifact 元数据。

当前限制：

- `input_refs` 没有实际填充；
- 没有 `waiting_for_user`、`waiting_for_task`、`cancelled` 等状态；
- 没有外部任务 ID、平台 Run ID 或 conversation command 绑定；
- 没有依赖门禁、重试、取消和并发控制；
- Graph Revision 更新后不会自动让旧 Run 失效；
- `shell` executor 只是 `contract_probe` 的别名，不执行 shell。

### 7.3 Dataset `/start`

当前 `POST /api/echomind/projects/{project_id}/nodes/{node_id}/start` 的调用链：

```text
校验 Project、Graph、Dataset 节点和 data-processing capability
  -> 校验 EchoMind 会话归属
  -> 检查远端 Sandbox 所需的 Cookie/Bearer 凭证
  -> 构造 Project/Stage 上下文
  -> 向当前 EchoMind 会话提交 UserMessageCommand
  -> 返回 command ID 作为 event_id
```

注入给 Agent 的开发顺序：

1. 检查真实 Storage 数据；
2. 按 `data-processing` 选择 Pipeline；
3. 跑小样并展示结果；
4. 等待用户显式确认；
5. 提交全量验证任务；
6. 报告 task/run/output 标识。

当前缺口：

- 启动 Dataset 时没有创建 StageRun；
- 返回的 `event_id` 实际是 command ID，不是 Run ID；
- “正在开发、等待确认、任务已提交、成功、失败”没有阶段级持久化；
- `df_submit_pipeline` 产生的 task/output 不会自动关联到 StageRun 或 Artifact；
- 页面刷新后只能从会话和 Graph 恢复上下文，不能从 Dataset StageRun 恢复；
- 五步顺序主要靠 Prompt，不是平台状态机。

## 8. 当前前端能力

EchoMind 页面已经具备：

- EchoMind 独立会话创建、历史选择和恢复；
- 对话、计划草稿、Diff、确认、拒绝和历史 Revision；
- 确认后读取真实 Project/Graph；
- Agent 节点显示“启动开发”；
- Executor 节点显示“运行验证”；
- 展示最近 Run、错误和 JSON Artifact。

当前限制：

- Dataset 启动后只提示“已交给当前 EchoMind 会话继续开发”，Graph 不会立即出现
  Dataset Run；
- Project Run 没有实时推送或自动轮询，开发过程主要看 Chat；
- 页面无法区分“Agent 正在开发”和“平台任务正在运行”；
- Artifact 目前只支持 JSON。

## 9. 开发实施路线

后续按 M0 到 M8 推进。每一步都以前一步通过验收为前提，不在一个阶段同时引入
Graph、Agent、平台任务和前端的大范围改动。

每个阶段的完成目标、验证内容、验证流程和通过证据见
`docs/echomind-milestone-verification.md`。其中 M4 是协议闭环门禁，M7 是真实业务
闭环门禁。

### M0：当前基线，已完成

- 双 EchoMind/Workbench 独立入口；
- 通用 `experiment-planning` 与 AVI reference；
- 计划确认后创建或更新 Project/Graph；
- Stage 级 Graph、StageRun、Artifact 和 Contract Probe；
- Dataset `/start` 向当前 EchoMind 会话投递开发任务；
- 前端展示计划、Project、Graph、Run 和 Artifact。

### M1：Graph 契约收敛，下一步

目标：以当前 `ProjectGraph` 为基线收敛 Graph 定义，不另造一套平行模型，也不改变
现有接口行为。

内容：

- 复用现有 `ProjectGraph`、`ProjectNode`、`ProjectEdge` 和拓扑校验；
- 定义带 `revision`、`parent_revision` 等版本信息的 `GraphRevision`；
- 将 Graph 定义与节点运行状态分离，运行状态继续由 Run 投影；
- 保留当前 `ProjectGraph` 作为兼容读取结构和前端 Stage 视图；
- 兼容读取现有 `schema_version=1` 的 `graph.json`；
- 为当前 AVI Graph 增加 round-trip 和旧数据兼容测试。

不做：

- 不改现有 `/start` 和 `/run` 行为；
- 不切换到 Revision 文件持久化；
- 不生成 Task 级节点和端口绑定；
- 不增加 GraphRun、NodeRun、Incident 或 DevelopmentBranch；
- 不增加 ManagerCommand 或 Agent 执行信封；
- 不改前端。

验收：

- 当前 AVI `ProjectGraph` 可以无损转换为 `GraphRevision` 并投影回兼容视图；
- 现有 `graph.json` 无需迁移即可读取；
- Graph 定义中不再依赖易失的节点运行状态；
- 非法 Revision、重复节点、未知边和环会被拒绝；
- 现有 EchoMind/Workbench 测试全部通过；
- 新契约只有独立单元测试，没有接口行为副作用。

### M2：GraphRevision 编译

目标：在 M1 的 Graph 契约上，把当前 Stage 级 `ProjectGraph` 编译成 Task 级
`GraphRevision`，同时保留前端 Stage 聚合能力。

内容：

- 从 PlanRevision 的 stages、tasks 和 resources 编译 Task Node；
- 复用 `WorkflowPlan` 已有的 task inputs、outputs 和上下游引用；
- 建立明确的 Input/Output Port 和 Edge Binding；
- 引入首期内置 AVI TaskType；
- 将不可变 GraphRevision 持久化为独立 Revision，不再覆盖历史；
- 过渡期继续生成当前 `ProjectGraph` 作为兼容视图；
- 运行 Graph 校验：缺输入、类型不匹配、环、未解析 Executor；
- 前端继续读取当前聚合视图，新增接口可以后置。

验收：

- 当前 AVI Plan 可以稳定生成同一份 GraphRevision；
- 每次修订产生新 Revision，不覆盖历史；
- 校验失败会指出具体 node、input 或 edge；
- 旧 Project/Graph 读取接口保持兼容。

### M3：Durable NodeRun

目标：在可执行 GraphRevision 上引入最小 GraphRun/NodeRun，把 Dataset `/start`
从“只投递会话命令”升级为真正的可恢复运行。

内容：

- 增加最小 `GraphRun`、`NodeRun` 和运行状态契约；
- `/start` 在投递命令前创建 NodeRun；
- 返回真实 `run_id`、`node_run_id` 和 command 关联；
- 支持 `running`、`waiting_for_user`、`succeeded`、`failed` 最小状态；
- 分发失败时立即落失败状态；
- Readiness 校验 Graph 依赖、Artifact 和输入指纹；
- 前端根据 NodeRun 刷新节点状态。

验收：

- 点击 Dataset 启动后立即出现可查询的 NodeRun；
- 刷新、重启和会话切换后 Run 仍然存在；
- 失败不会留下永久 running；
- 旧 StageRun 展示可由兼容投影生成。

### M4：完整 Graph 的 Agent 协同闭环

目标：使用 M2 生成的完整 AVI GraphRevision 和 M3 的 GraphRun/NodeRun，让 EchoMind
Manager 与节点子 Agent 按结构化协议协同跑完一次完整开发态 Graph。该里程碑优先
证明 Graph 执行协议，不以所有业务 Executor 都已生产化为前提。

内容：

- 为已确认的 AVI Graph 创建一次开发态 GraphRun；
- Runtime 根据 GraphRevision 计算 ready 节点；
- Runtime 向 EchoMind Manager 生成 ManagerDecisionContext；
- Manager 只通过 ManagerCommand 请求启动、重试或停止节点；
- Runtime 校验 Command，创建 NodeRun，并为对应节点生成
  AgentExecutionEnvelope；
- `dataset_agent`、`inference_agent`、`evaluator_agent` 和 `train_agent`
  分别执行自己的节点切片，返回 AgentResultEnvelope；
- Runtime 校验 Artifact、Schema、完成条件和幂等键后推进下游节点；
- 所有节点成功后 GraphRun 结束，Manager 读取最终摘要；
- 任一节点失败时 GraphRun 暂停，Manager 决定重试、停止或请求用户处理。

M4 可以使用小样、固定 fixture 或确定性开发 Executor 完成协议闭环，但每个节点的
输入输出必须是真实 Artifact，不能把 `contract_probe` 返回的占位结果当作业务成功。
后续 M5-M7 只替换业务 Executor，不改变主/子 Agent 协议和 Runtime 状态机。

验收：

- 能从一份已确认的 AVI Plan 创建完整 GraphRevision 和 GraphRun；
- `dataset -> inference -> evaluator -> train` 每个节点都有独立 NodeRun；
- 每个 NodeRun 都能追踪到 manager command、子 Agent task、输入 Artifact 和输出
  Artifact；
- 主 Agent 不能直接修改 Run 状态，子 Agent 不能修改 Graph 或下游节点状态；
- 子 Agent 的 `succeeded` 只是声明，Runtime 校验通过后才能结束 NodeRun；
- 相同 command/result 重放不会产生重复 NodeRun 或 Artifact；
- 一次成功执行和一次失败后重试都留下完整事件序列；
- Runtime 重启后可以根据持久化状态继续同一个 GraphRun。

### M5：Dataset 纵向闭环

目标：完成 `dataset.clean_label -> dataset.validate -> dataset_version`。

内容：

- Dataset Node 使用 `data-processing` Skill 和远端 Sandbox；
- 小样 Run 记录为 NodeRun；
- 用户确认门禁写入结构化状态；
- 全量 `df_submit_pipeline` 保存 task、run、output 和 fingerprint；
- Kafka 终态驱动 NodeRun；
- 生成 `avi.dataset_version` Artifact；
- Graph 固化后不再要求小样人工确认。

验收：

- 探查、小样、确认、全量和报告位于同一个 NodeRun 链路；
- 会话断开和 Runtime 重启后可以恢复；
- 任务失败有结构化错误和 Artifact 证据；
- 通用 `data-processing` Skill/Tool 不出现 EchoMind/AVI 分支。

### M6：Inference 与 Evaluator 纵向闭环

目标：把 M4 中的推理和评估开发 Executor 替换为真实执行器，完成
`dataset_version -> inference.predictions -> evaluator.eval_result`。

内容：

- Inference 消费 DatasetVersion 和模型资源，保存逐样本预测、模型版本和输入指纹；
- Evaluator 消费预测、真值和固定评测规则；
- 生成 `avi.eval_result` 和 `avi.atif_report`；
- 每个阶段都有独立 NodeRun、结构化错误和可重试边界；
- 结果写入 Artifact，不依赖报告自然语言；
- 首期只做固定评测集批量推理和确定性评分，不做在线推理。

验收：

- 推理结果可以通过 `sample_id` 关联回 DatasetVersion；
- Inference 失败不会伪造 Evaluator 成功；
- 同一 predictions 重复评分结果一致；
- Evaluator 不调用被测模型；
- 评测 Artifact 可以被前端直接加载。

### M7：Train 纵向闭环

目标：完成 `dataset_version -> train.train_run -> checkpoint`，并在真实执行器
替换完成后形成首条完整业务闭环。

内容：

- Train Node 由 Agent 生成、验证并提交平台 Workflow；
- 保存 workflow revision、task_id 和 correlation；
- Kafka 终态更新 NodeRun；
- 保存 checkpoint 或训练输出 Artifact；
- 支持暂停和取消的平台确认。

验收：

- Train 只消费已通过的 DatasetVersion；
- Agent 不负责持续调度训练任务；
- 任务终态、错误和输出可以反查；
- 重启后不会重复提交同一个训练任务；
- M4 的主/子 Agent 协议和 GraphRevision 不需要因真实 Executor 接入而修改。

### M8：生产暂停、开发分支与发布

目标：在 AVI 单轮链路稳定后再开放生产态。

内容：

- 创建不可变 Deployment；
- GraphRun 异常时进入 paused 并生成 Incident；
- 从 Run 快照 Fork DevelopmentBranch；
- 修复后创建新 GraphRevision；
- 首期只允许新 Production Run 从起点重跑；
- 支持手工触发，事件、定时和 continuous 后续扩展。

验收：

- 暂停后不再派发新的下游任务；
- 原 GraphRevision、Run 和失败现场不被修改；
- 开发分支有完整父引用；
- 新 GraphRevision 未发布前不能替换生产 Deployment；
- 生产主 Agent 可以决策，但不能直接写状态或修改冻结 Graph。

详细的 Graph Schema、状态机、结构化 Agent 契约和生产异常 Fork 规则见
`docs/echomind-graph-agent-contract.md`。

## 10. 当前 API

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

`/start` 只用于 Agent-led Dataset；`/run` 只用于 executor 节点。后续接入其他阶段的
Agent 开发入口时，不应把两种语义继续硬塞进当前 `launch_mode`。

## 11. 代码落点

| 内容 | 位置 |
|---|---|
| EchoMind Profile 和 Tool/Skill 清单 | `openhands-agent-server/openhands/agent_server/agent_profiles/echomind_manager.py` |
| EchoMind 路由注册 | `pyromind-agent-server/pyromind_agent_server/profiles.py` |
| 计划确认和策略切换 | `pyromind-runtime/pyromind_runtime/application/conversation_runtime.py` |
| Project binder | `pyromind-agent-server/pyromind_agent_server/experiment_plan_projects.py` |
| Graph 生成 | `openhands-agent-server/openhands/agent_server/echomind_graph_templates.py` |
| Project/Graph 持久化 | `openhands-agent-server/openhands/agent_server/echomind_projects.py` |
| StageRun/Artifact | `openhands-agent-server/openhands/agent_server/echomind_stage_runs.py` |
| 阶段 executor | `openhands-agent-server/openhands/agent_server/echomind_stage_executors.py` |
| Project/Stage API | `pyromind-agent-server/pyromind_agent_server/api/echomind_projects.py` |
| 通用实验规划 Skill | `.agents/skills/experiment-planning/SKILL.md` |
| AVI reference | `.agents/skills/experiment-planning/references/avi-single-turn.md` |

## 12. 首期默认决策

为继续开发，首期采用以下默认值：

1. 生产态只在阶段边界使用 `manager` 决策；checkpoint、回调和重试由 Runtime 自动
   处理；
2. GraphRun 暂停时先停止新调度，已提交的平台任务默认继续，用户可以显式取消；
3. DevelopmentBranch 修复后首期只允许创建新 Production Run，不支持原地恢复；
4. 主 Agent 可以决定启动、重试和跳过；取消平台任务、回退和发布新 Revision 需要
   用户确认；
5. TaskType 首期内置 AVI 任务类型，节点库注册机制后置；
6. 前端首期使用短轮询刷新 Project Run，SSE 后置。
