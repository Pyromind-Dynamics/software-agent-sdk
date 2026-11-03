# EchoMind Graph 与主/子 Agent 交互契约

> 状态：设计稿，待评审
> 更新时间：2026-10-30
> 范围：可执行 Graph、运行状态、生产异常暂停与开发分支、主/子 Agent
> 结构化交互契约
>
> 当前轻量实施以 `docs/echomind-four-node-graph-execution-plan.md` 为准：开发态
> 不创建 GraphRevision，只维护可变 Working Graph；本文的 GraphRevision、
> Deployment 和 DevelopmentBranch 属于后续生产化契约。
>
> 本文不定义任务队列、Agent 进程调度、Kafka Transport 和子 Agent 生命周期实现。
> 这些属于后续执行层设计。

## 1. 结论

生产态遇到异常时可以暂停，并恢复到开发态排查，但不能把正在运行的冻结 Graph
原地改成开发 Graph。

推荐流程：

```text
生产 GraphRun 遇到异常
  -> 暂停新的下游调度
  -> 记录 Incident 和异常快照
  -> 从当前 GraphRevision、输入和 Artifact Fork Development Branch
  -> 在开发分支修改节点实现或流程
  -> 重新小样/全量验证
  -> 用户确认新的 GraphRevision
  -> 发布新的 Deployment
  -> 新 Production Run 从合适节点恢复或重新执行
```

这样同时满足：

- 生产运行可暂停、可恢复；
- 故障现场可复现、可追溯；
- 开发修复不会修改历史生产定义；
- 修改完成前旧生产版本仍可查询和回滚；
- 主 Agent 参与生产决策，但不直接修改运行状态。

这里的“主 Agent / 子 Agent”是 EchoMind 内部的执行角色契约，不新增对外 Chat
入口，也不改变当前 Pyromind Workbench 与 EchoMind Manager 两个独立 Agent Profile
的边界。

## 2. 职责边界

### 2.1 主 Agent

主 Agent 负责业务判断：

- 读取 Graph、Run 和 Artifact 的摘要；
- 判断某个 ready 节点是否启动；
- 决定暂停、重试、跳过或请求用户确认；
- 判断是否从生产异常进入开发分支；
- 汇总阶段结果并提出下一轮计划；
- 提交结构化命令，不直接写 Run 状态。

主 Agent 不负责：

- 维护 NodeRun 状态机；
- 处理 Kafka 重复消息；
- 管理 lease、幂等和并发；
- 判定 checkpoint 文件是否写完；
- 绕过 Runtime 直接启动平台任务；
- 修改已经冻结的 GraphRevision。

### 2.2 Runtime

Runtime 负责确定性执行：

- 校验 Graph、输入 Artifact、依赖和权限；
- 创建 GraphRun、NodeRun 和 Incident；
- 计算 ready 节点；
- 按 Decision Policy 请求主 Agent 决策或自动推进；
- 派发 Executor，保存 external task ID；
- 消费 Kafka 终态和 Observer 事件；
- 校验子 Agent 返回的结构化结果；
- 持久化 Artifact、事件和状态；
- 处理重试、取消、幂等、恢复和失效传播。

### 2.3 子 Agent

子 Agent 负责一个任务切片：

- 读取任务信封中的明确输入；
- 使用被授权的 Tool 和 Skill；
- 完成数据、训练、评测或分析任务；
- 执行任务级校验；
- 返回结构化结果和 Artifact 引用；
- 不直接修改 Graph、Run 或下游节点状态。

## 3. Graph 的四层表示

### 3.1 PlanRevision

用户确认的计划版本，表达“想做什么”。

当前 SQL `PlanRevision` 已经承担这一职责。它包含阶段、任务、资源和边，但任务
描述仍偏规划语义。

确认 PlanRevision 只负责提交计划、创建或更新 Graph，并解锁 EchoMind 开发能力。
它不自动创建 GraphRun，也不自动启动任何节点。执行必须由独立的 Run 启动动作
触发。

### 3.2 GraphRevision

可执行、不可变的流程版本，表达“按什么依赖和契约执行”。

GraphRevision 由 PlanRevision 编译产生，必须满足：

- 所有节点引用已知 TaskType；
- 输入输出端口可绑定；
- Graph 无环；
- 必需输入有来源；
- Executor 已解析；
- `contract_probe`、未决资源和未绑定参数不能进入发布版本；
- 每个输出有可校验的 Artifact Schema。

### 3.3 Deployment

生产绑定，表达“什么时候、用什么策略运行哪个 GraphRevision”。

至少包含：

- `graph_id` 和 `graph_revision`；
- 触发器：manual、cron、event、continuous；
- 生产决策策略；
- 并发、重试、超时和暂停策略；
- 当前是否启用；
- 发布人和发布时间。

### 3.4 Run

一次实际执行，表达“这次发生了什么”。

```text
GraphRun
  -> NodeRun
  -> Artifact
  -> Incident
```

GraphRevision 和 Deployment 不保存运行状态。运行状态只存在于 Run 体系，避免修改
Graph 定义来记录执行结果。

## 4. GraphRevision 数据模型

### 4.1 顶层结构

```json
{
  "schema_version": 1,
  "project_id": "project-uuid",
  "graph_id": "graph-uuid",
  "revision": 3,
  "parent_revision": 2,
  "mode": "production",
  "lifecycle": "frozen",
  "nodes": [],
  "edges": [],
  "created_at": "2026-10-30T00:00:00Z",
  "created_by": "42",
  "contract_digest": "sha256:..."
}
```

`contract_digest` 覆盖节点定义、边映射、参数、Executor 引用和 Artifact Schema。
任何会改变执行结果的内容变化都必须产生新 Revision。

### 4.2 Node

Node 是 Task 实例，不是整个 Stage。

```json
{
  "id": "dataset.clean_label",
  "stage_id": "dataset",
  "task_type_ref": "echomind.task.dataset.clean_label@1",
  "title": "数据清洗与预打标",
  "kind": "task",
  "agent_role": "dataset_agent",
  "executor": {
    "kind": "platform_workflow",
    "ref": "workflow-version://avi-clean-label/v12",
    "params": {
      "model_profile": "vision",
      "output_schema": "structured"
    }
  },
  "inputs": {
    "raw_data": {
      "artifact_type": "avi.raw_data",
      "schema": "avi.raw_data/v1",
      "required": true,
      "binding": "artifact://raw-data/batch-2026-10"
    }
  },
  "outputs": {
    "dataset_version": {
      "artifact_type": "avi.dataset_version",
      "schema": "avi.dataset/v1"
    }
  },
  "decision_policy": "manager",
  "trigger_policy": {
    "kind": "manual"
  },
  "failure_policy": {
    "max_attempts": 1,
    "on_exhausted": "pause_and_propose_development"
  },
  "timeout_seconds": 86400
}
```

字段语义：

| 字段 | 含义 |
|---|---|
| `stage_id` | 前端聚合展示和业务归属，不参与执行判断 |
| `task_type_ref` | 不可变任务契约引用 |
| `agent_role` | 该节点需要唤醒哪个逻辑 Agent；自动化节点可为空 |
| `executor` | 真实执行后端；不是聊天会话 |
| `inputs` | 上游 Artifact 或固定 Resource 的绑定 |
| `outputs` | 运行结束后必须产出的 Artifact 契约 |
| `decision_policy` | 谁决定启动该节点 |
| `trigger_policy` | 节点如何变为 ready |
| `failure_policy` | 失败后的重试、暂停和开发分支策略 |

### 4.3 Node Kind

首期只定义三类：

| kind | 用途 |
|---|---|
| `source` | 外部数据、模型或固定资源入口 |
| `task` | 有明确输入输出和 Executor 的业务任务 |
| `gate` | 用户确认、指标门禁或人工审批，不执行模型任务 |

复杂平台工作流作为 `task` 的 Executor 实现，不把内部节点默认展开到 Project Graph。
需要查看时，通过 NodeRun 关联的 Workflow Run 展示子图。

### 4.4 Edge

Edge 是 Artifact 绑定，不是简单的前后顺序。

```json
{
  "id": "edge-dataset-train",
  "from": {
    "node_id": "dataset.clean_label",
    "output": "dataset_version"
  },
  "to": {
    "node_id": "train.train_run",
    "input": "train_dataset"
  }
}
```

校验规则：

- source node 和 output port 必须存在；
- target node 和 input port 必须存在；
- Artifact Type 必须兼容；
- 一个 input port 只能有一个有效绑定；
- Graph 不允许环；
- 条件分支必须通过 gate 或显式条件边表达，不能在 Agent Prompt 中隐式决定。

### 4.5 Trigger Policy

| trigger | 含义 |
|---|---|
| `manual` | 需要主 Agent 或用户显式启动 |
| `auto` | 依赖满足后由 Runtime 自动启动 |
| `event` | 外部事件到达后启动 |
| `cron` | 定时启动 |
| `on_checkpoint` | 发现完整 checkpoint 后启动 |
| `continuous` | 对持续数据流创建批次运行 |

Graph 表达依赖，Trigger Policy 表达何时产生候选 NodeRun。二者合起来由 Runtime
计算 readiness。

### 4.6 Decision Policy

| policy | 行为 |
|---|---|
| `auto` | 依赖满足后直接派发 |
| `manager` | Runtime 通知主 Agent，等待结构化决定 |
| `user` | 等待用户确认 |
| `manager_on_failure` | 正常自动执行，失败时请求主 Agent |

生产态可以大量使用 `manager` 做阶段边界决策，但不能让主 Agent 代替 Runtime
处理每次心跳、回调和重试。

## 5. Artifact 契约

Artifact 是阶段之间唯一稳定的数据交换方式。

```json
{
  "artifact_id": "artifact-uuid",
  "project_id": "project-uuid",
  "graph_run_id": "run-uuid",
  "node_run_id": "node-run-uuid",
  "type": "avi.dataset_version",
  "schema": "avi.dataset/v1",
  "uri": "storage://datasets/avi/versions/v3",
  "fingerprint": "sha256:...",
  "parent_artifact_ids": ["artifact-parent"],
  "producer": {
    "node_id": "dataset.clean_label",
    "executor": "platform_workflow"
  },
  "metadata": {},
  "created_at": "2026-10-30T00:00:00Z"
}
```

首期 AVI Artifact 类型：

| Artifact | 关键内容 |
|---|---|
| `avi.raw_data` | 原始批次、manifest、数据范围 |
| `avi.dataset_version` | 清洗后数据、split、schema、统计和指纹 |
| `model.base` | 基模标识和版本 |
| `model.checkpoint` | checkpoint 路径、step、完整性证据 |
| `avi.prediction_set` | 评测集逐样本预测 |
| `avi.eval_result` | 指标、混淆矩阵和 badcase 引用 |
| `avi.atif_report` | 标准评测展示产物 |
| `avi.promote_decision` | 择优结果和门禁证据 |

Artifact 一旦被下游引用就不可原地修改。

## 6. 运行状态机

### 6.1 GraphRun

```text
queued
  -> running
  -> paused
  -> running
  -> succeeded

running -> failed
running -> cancelled
running -> stale
```

`paused` 表示不再派发新的下游任务，但不等价于已停止所有外部任务。

### 6.2 NodeRun

```text
pending
  -> ready
  -> waiting_decision
  -> running
  -> waiting_external
  -> succeeded

ready/running/waiting_external -> failed
pending/ready -> skipped
ready/running -> cancelled
ready/running/waiting_external -> stale
```

节点可由多个 NodeRun 尝试，NodeRun 必须记录：

- `attempt`
- `graph_revision`
- `input_artifact_ids`
- `input_fingerprint`
- `external_task_id`
- `agent_session_id`
- `decision_id`
- `error`
- `artifacts`

### 6.3 Readiness

Node 变为 ready 必须同时满足：

- GraphRun 处于 running；
- 所有必需输入已绑定且 Artifact 可读；
- 上游依赖完成状态满足 Edge 规则；
- Gate 条件满足；
- 输入 fingerprint 没有触发 stale；
- 没有未完成的前置 NodeRun；
- Decision Policy 已满足。

主 Agent 的自然语言判断不能替代上述确定性校验。

## 7. 生产异常与开发分支

### 7.1 合理流程

```text
1. NodeRun 失败、校验失败、指标门禁失败或主 Agent 请求介入
2. Runtime 创建 Incident
3. GraphRun 进入 paused
4. 停止派发新的下游 NodeRun
5. 已运行的外部任务按 pause_policy 继续等待或请求平台取消
6. Runtime 向主 Agent 提供异常视图和允许动作
7. 主 Agent 可以 retry、resume、cancel 或 open_development_branch
8. open_development_branch 生成 DevelopmentBranch
9. 开发分支继承 GraphRevision、Run 快照、输入和 Artifact 引用
10. Agent 在开发分支修改流程或实现并重新验证
11. 用户确认新的 GraphRevision
12. 发布新 Deployment，新 Production Run 恢复或重跑
```

### 7.2 不能原地降级

不能执行：

```text
production_run.lifecycle = development
graph.lifecycle = development
```

原因：

- 会修改正在被运行引用的不可变定义；
- 无法判断当前 Run 使用的是修改前还是修改后的 Graph；
- 旧失败现场会被覆盖；
- 回滚和审计会失去稳定引用；
- 多个并发 Run 可能引用同一 GraphRevision。

正确方式是 Fork：

```json
{
  "branch_id": "development-branch-uuid",
  "source_project_id": "project-uuid",
  "source_graph_id": "graph-uuid",
  "source_graph_revision": 3,
  "source_run_id": "production-run-uuid",
  "source_node_run_id": "failed-node-run-uuid",
  "base_artifacts": ["artifact-uuid"],
  "reason": "dataset output schema validation failed",
  "status": "open"
}
```

生产 Deployment 保持冻结；开发分支产生新的 PlanRevision 和 GraphRevision。

### 7.3 恢复生产

开发修复完成后有两种恢复方式：

1. **新 Run 从起点重跑**：最安全，首期默认；
2. **从失败节点恢复**：新 GraphRevision 必须显式声明旧 Artifact 与恢复点的兼容性，
   并由 Runtime 校验 fingerprint。

不能让主 Agent 仅凭自然语言判断“旧产物可以复用”。

## 8. 主/子 Agent 结构化交互

本节定义数据契约，不定义传输、队列和 Agent 进程生命周期。

### 8.1 ManagerDecisionContext

Runtime 给主 Agent 的只读决策视图：

```json
{
  "schema_version": 1,
  "project_id": "project-uuid",
  "graph_id": "graph-uuid",
  "graph_revision": 3,
  "run_id": "production-run-uuid",
  "mode": "production",
  "run_status": "paused",
  "node_summary": [
    {
      "node_id": "dataset.clean_label",
      "status": "failed",
      "attempt": 1,
      "artifact_refs": []
    }
  ],
  "ready_nodes": [],
  "incident_id": "incident-uuid",
  "attention": {
    "kind": "data_quality",
    "summary": "Dataset validation failed"
  },
  "allowed_actions": [
    "retry_node",
    "open_development_branch",
    "cancel_run"
  ],
  "evidence_refs": [
    "artifact://validation-report",
    "run://node-run/logs"
  ]
}
```

原则：

- 默认不向主 Agent 注入整张原始 Graph 和全部日志；
- 只提供决策所需摘要、候选节点和证据引用；
- 主 Agent 可以调用只读查询工具获取节点切片；
- Context 必须绑定 `graph_revision` 和 `run_id`，避免旧状态污染决策。

### 8.2 ManagerCommand

主 Agent 输出的结构化命令：

```json
{
  "schema_version": 1,
  "command_id": "command-uuid",
  "project_id": "project-uuid",
  "graph_id": "graph-uuid",
  "graph_revision": 3,
  "run_id": "production-run-uuid",
  "expected_run_revision": 17,
  "action": "start_node",
  "target_node_id": "train.train_run",
  "reason": "Dataset v3 passed validation and baseline is current",
  "input_overrides": {},
  "metadata": {}
}
```

首期动作：

| action | 含义 |
|---|---|
| `start_node` | 请求启动一个 ready 节点 |
| `retry_node` | 请求创建新的 NodeRun attempt |
| `skip_node` | 跳过允许跳过的节点 |
| `pause_run` | 暂停新调度 |
| `resume_run` | 恢复暂停的 Run |
| `cancel_run` | 请求取消 Run |
| `open_development_branch` | 从当前现场创建开发分支 |
| `request_user_input` | 请求用户确认或补充信息 |
| `propose_graph_revision` | 在开发态提出新的 Graph Revision |

Runtime 必须：

- 校验 `command_id` 幂等；
- 校验 `expected_run_revision`；
- 校验动作是否在 `allowed_actions`；
- 校验节点是否 ready 或允许重试/跳过；
- 校验主 Agent 没有越权修改冻结 Graph；
- 持久化命令、决策原因和执行结果。

### 8.3 AgentExecutionEnvelope

主 Agent 或 Runtime 给子 Agent 的任务切片。这里只定义内容，不定义派发机制。

```json
{
  "schema_version": 1,
  "task_id": "task-uuid",
  "project_id": "project-uuid",
  "graph_id": "graph-uuid",
  "graph_revision": 3,
  "run_id": "run-uuid",
  "node_id": "dataset.clean_label",
  "node_run_id": "node-run-uuid",
  "attempt": 1,
  "agent_role": "dataset_agent",
  "objective": "清洗、预打标并生成可训练的 dataset_version",
  "inputs": {
    "raw_data": "artifact://raw-data/batch-2026-10"
  },
  "params": {
    "model_profile": "vision",
    "output_schema": "structured"
  },
  "output_contract": {
    "dataset_version": "schema://avi.dataset/v1"
  },
  "done_when": [
    "所有图片路径可解析",
    "标签符合 schema",
    "训练集和评测集无泄漏",
    "validation.json 校验通过"
  ],
  "constraints": {
    "storage_root": "/target-workspace",
    "sample_limit": 10
  },
  "allowed_capabilities": [
    "data-processing",
    "sandbox"
  ],
  "report_contract": "AgentResultEnvelope@1"
}
```

子 Agent 只接收自己的节点切片，不接收完整 Graph、其他 Agent 会话或无关历史。

### 8.4 AgentResultEnvelope

子 Agent 返回的结构化结果：

```json
{
  "schema_version": 1,
  "task_id": "task-uuid",
  "project_id": "project-uuid",
  "graph_revision": 3,
  "run_id": "run-uuid",
  "node_id": "dataset.clean_label",
  "node_run_id": "node-run-uuid",
  "attempt": 1,
  "status": "succeeded",
  "summary": "生成 1069 条训练数据和 120 条评测数据",
  "outputs": {
    "dataset_version": "artifact://dataset-version-v3"
  },
  "validation": {
    "schema_valid": true,
    "missing_images": 0,
    "train_eval_leak": false
  },
  "attention": {
    "kind": "none",
    "message": ""
  },
  "report_uri": "storage://reports/dataset-clean-label-r1.md",
  "error": null
}
```

状态至少支持：

```text
submitted
waiting_external
succeeded
failed
needs_attention
```

长任务规则：

- 子 Agent 提交 Studio 或 DataFlow 任务后返回 `submitted/waiting_external`；
- Runtime 保存 `external_task_id` 并释放子 Agent；
- Kafka 终态到达后，Runtime 更新 NodeRun；
- 需要解释、校验或最终报告时，再唤醒 Agent 处理。

子 Agent 的 `status=succeeded` 只是结果声明。Runtime 校验 Artifact、Schema、
完成条件和 fingerprint 后，才能把 NodeRun 置为 succeeded。

### 8.5 Attention

`attention.kind` 首期定义：

| kind | 含义 |
|---|---|
| `none` | 无人工或主 Agent 介入 |
| `decision` | 需要业务决策 |
| `data_quality` | 数据质量异常 |
| `rollback` | 建议回退或进入开发分支 |
| `resource` | 资源、配额或环境问题 |
| `user_action` | 必须用户处理 |

Attention 只生成通知或 Incident，不直接修改 GraphRun 状态。

### 8.6 完整 Graph 协同闭环

首个开发里程碑不是只验证一份 Schema，而是让一份完整的 AVI GraphRevision 在开发态
跑完。执行顺序固定为：

```text
create GraphRun
  -> Runtime calculates ready nodes
  -> ManagerDecisionContext
  -> ManagerCommand(start_node)
  -> Runtime creates NodeRun
  -> AgentExecutionEnvelope
  -> stage subagent executes
  -> AgentResultEnvelope
  -> Runtime validates Artifact and completion
  -> NodeRun succeeded
  -> next ready node
  -> GraphRun succeeded
```

首轮关闭门禁使用当前已确认计划中的四个 Stage：

```text
dataset -> inference -> evaluator -> train
```

四个节点分别路由到 `dataset_agent`、`inference_agent`、`evaluator_agent` 和
`train_agent`。这些子 Agent 是 Runtime 内部执行角色，不新增 Chat API，也不与
EchoMind Manager 或 Pyromind Workbench 的独立会话边界冲突。

M4 允许节点使用小样、fixture 或确定性开发 Executor 产生 Artifact，先证明 Graph、
Manager、Runtime、子 Agent 和结果协议可以形成闭环。`contract_probe` 的占位结果
不能作为闭环成功依据。M5-M7 再逐个替换为真实业务 Executor。

M4 验收必须同时满足：

- 完整 Graph 的所有节点都有 NodeRun，不能跳过未执行节点；
- 每次启动都来自 ManagerCommand，并通过 Runtime 的 revision、幂等和权限校验；
- 每个子 Agent 只收到自己的 AgentExecutionEnvelope，不读取完整 Graph 或主会话历史；
- 每个 NodeRun 都绑定输入 Artifact、输出 Artifact、尝试次数和执行证据；
- Runtime 校验成功后 NodeRun 才能成功，主 Agent 和子 Agent 都不能直接写 Run 状态；
- GraphRun 只有全部必要节点成功后才置为 succeeded；
- 失败路径至少验证一次暂停、Manager 决策和重试，并保留完整事件序列；
- Runtime 重启后可以从持久化状态继续同一个 GraphRun；
- 替换任一开发 Executor 时不需要修改 ManagerCommand、AgentResultEnvelope 或
  GraphRevision Schema。

## 9. Graph 编译与发布

### 9.1 开发态

- Agent 可以创建和修改 PlanRevision；
- 节点 Executor 可以是开发实现；
- 可以逐个 NodeRun 启动；
- 小样和全量验证结果保存在 Run 中；
- Graph 可以重新编译，不构成生产版本。

### 9.2 发布门禁

至少满足：

- 所有节点有已知 TaskType 和真实 Executor；
- 所有输入输出 Artifact 可校验；
- Graph 无环；
- 必需的 Evaluation Gate 已定义；
- 最近一次验证 Run 成功；
- 没有 unresolved 参数和缺失 Resource；
- 用户明确确认发布。

### 9.3 生产态

- Deployment 引用不可变 GraphRevision；
- 修改 Graph 必须创建新 Revision；
- 运行中的 GraphRun 始终固定原 Revision；
- 主 Agent 可以决策，但不能修改 Deployment 指向；
- 发布新版本不影响历史 Run。

## 10. AVI Graph 蓝图

首轮协议闭环使用当前计划生成的四阶段 Graph：

```text
dataset
  -> inference
  -> evaluator
  -> train
```

它用于验证 M4 的 Graph 执行和主/子 Agent 协同协议。真实 Executor 接入后，再按
任务粒度演进为以下单轮 Graph：

```text
raw_data
  -> dataset.clean_label
  -> dataset.validate
  -> train.train_run
  -> eval.final
  -> promote
```

节点策略：

| Node | Trigger | Decision | Executor |
|---|---|---|---|
| `raw_data` | manual/event | auto | source |
| `dataset.clean_label` | manual | manager | platform_workflow |
| `dataset.validate` | auto | auto | deterministic |
| `train.train_run` | manual | manager | studio_workflow |
| `eval.final` | auto | auto | evaluation_workflow |
| `promote` | auto | auto | deterministic |

后续再扩展 checkpoint 流：

```text
train.train_run
  -> eval.checkpoint_eval (on_checkpoint)
  -> eval.analysis (all checkpoints terminal)
  -> dataset.relabel
  -> dataset.refine_dataset
  -> next dataset_version
```

一轮 Graph 保持无环；下一轮由 Runtime 的 Iteration 规则创建新 GraphRun 并绑定
上一轮 Artifact。

## 11. 当前实现与目标差距

当前已实现：

- Project/Graph 文件持久化；
- Stage 级节点和边；
- `contract_probe` Executor；
- StageRun 和 JSON Artifact；
- Dataset `/start` 向当前会话投递开发任务；
- 计划确认后更新同一个 Project/Graph。

目标设计尚未实现：

- GraphRevision 不可变版本；
- Task 级节点和端口映射；
- Artifact 类型系统；
- NodeRun attempt 和 external task 绑定；
- Decision Policy 和 Trigger Policy；
- Incident、暂停和 DevelopmentBranch；
- ManagerDecisionContext / ManagerCommand；
- AgentExecutionEnvelope / AgentResultEnvelope；
- Deployment 和新 Run 恢复。

## 12. 实现边界

一步一步的执行顺序以 `docs/echomind-avi-single-turn-plan.md` 的 M0 到 M8 为准。
每个阶段的具体验证步骤见 `docs/echomind-milestone-verification.md`。本节只说明
各项设计分别落在哪个能力边界。

### Graph Schema v1：M1-M2

M1 以现有 `ProjectGraph` 为基线收敛契约，不另造一套平行 Graph：

- 复用现有 Node、Edge、`config` 和拓扑校验；
- 增加 `revision`、`parent_revision` 等版本信息；
- GraphRevision 与运行状态分离；
- 保留当前 ProjectGraph 作为兼容读取结构和前端 Stage 视图；
- 兼容现有 `schema_version=1` 的 `graph.json`；
- 不切换到 Revision 文件持久化。

M1 验收：

- 现有 AVI ProjectGraph 可以无损转换为 GraphRevision 并投影回兼容视图；
- 旧 `graph.json` 可以直接读取；
- 非法 Revision、重复节点、未知边和环会被拒绝；
- 不改变 Dataset 当前开发入口。

M2 在 M1 契约上生成 Task Graph：

- 从现有 `WorkflowPlan` 的 tasks、inputs、outputs 和上下游引用编译 Task Node；
- 引入 Artifact Port 和 Edge Binding；
- 保持现有 Stage 字段用于前端聚合；
- 保存不可变 Revision 历史。

M2 验收：

- 现有 AVI Plan 可以编译成 Task Graph；
- 每条 Edge 都能解析到明确 output/input；
- Graph 校验能拒绝缺失输入、类型不匹配和环；
- GraphRevision 不可修改。

### Agent Contract v1：M4

- 定义 ManagerDecisionContext 和 ManagerCommand；
- 定义 AgentExecutionEnvelope 和 AgentResultEnvelope；
- 定义 Decision Policy 和 Attention；
- 使用当前四阶段 AVI Graph 跑通完整开发态闭环；
- 暂不要求所有业务 Executor 已生产化。

验收：

- Schema 有严格版本和校验；
- 命令和结果都可幂等；
- 主 Agent 不能直接修改状态；
- 子 Agent 结果必须经过 Runtime 校验。
- 四阶段 Graph 全部通过 NodeRun、Artifact 和事件序列完成闭环；
- 失败、重试和 Runtime 恢复至少各验证一次；
- ManagerCommand 和 AgentResultEnvelope 在 M5-M7 替换业务 Executor 时保持不变。

### 生产异常与开发分支：M8

- 增加 Incident 和 DevelopmentBranch；
- 支持暂停 GraphRun；
- 从 Run 快照 Fork 开发分支；
- 修复后创建新 GraphRevision；
- 支持新 Production Run 从起点或显式恢复点启动。

验收：

- 暂停后不再派发新的下游任务；
- 原 GraphRevision、Run 和失败现场不被修改；
- 开发分支有完整父引用；
- 未发布的新 GraphRevision 不能替换生产 Deployment。

### AVI 纵向闭环：M5-M7

- 先接 `dataset.clean_label` 和 `dataset.validate`；
- 再接真实的 `inference` 和 `evaluator`；
- 最后接 `train.train_run` 和 checkpoint；
- 每个阶段都通过 NodeRun、Artifact 和 Runtime 回调验证。

## 13. 首期默认决策

1. 生产态只在阶段边界使用 `manager` 决策；checkpoint、回调和重试由 Runtime
   自动处理；
2. GraphRun 暂停时先停止新调度，已提交的平台任务默认继续，由用户显式决定是否
   取消；
3. DevelopmentBranch 修复后首期只创建新 Production Run，不支持原地恢复；
4. 主 Agent 可以决定启动、重试和跳过；取消平台任务、回退和发布新 Revision 需要
   用户确认；
5. Artifact Schema 首期要求 major 完全一致，minor 只能在显式声明兼容时跨版本读取；
6. TaskType 首期内置 AVI 类型，注册式节点库后置。
