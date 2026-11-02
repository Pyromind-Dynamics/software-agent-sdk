# EchoMind Project 开发态与四节点执行渐进方案

> 状态：实施基线，待评审后从 S1 开始
> 更新时间：2026-11-02
> 首期场景：AVI 单轮，Dataset / Train / Inference / Evaluator 四节点
> 关系：本文替代旧方案中“开发态先引入 GraphRevision，再逐步执行”的部分。
> 两个 Agent 的入口和边界分别见 `docs/echomind-dual-agent-change-design.md`，
> 主/子 Agent 的结构化长周期契约见 `docs/echomind-graph-agent-contract.md`。

## 1. 方案结论

第一期采用轻量模型：

```text
需求澄清
  -> 用户确认计划
  -> 创建 Project
  -> Project 进入 development
  -> Working Graph 可以被 Agent 和用户持续修改
  -> 每个 Node 可以单独启动验证
  -> 当前 Working Graph 可以整体调试
  -> 验证完成后发布
  -> 发布时冻结一份 Published Graph
  -> 生产执行只读这份冻结定义
```

开发阶段不引入用户可见的 GraphRevision、版本确认、版本迁移或每次改图都要生成新
版本的流程。开发态只有一个当前 Working Graph，用户通过对话提出需求，Agent 直接
修改节点、边、配置和需求清单，修改后立即可以在页面上看到。

为了避免“执行到一半 Graph 被改掉”，每次启动 Node 或 Graph 调试时，Runtime 会把
当前 Graph 复制成该次运行的内部快照，并记录摘要。这个快照只用于保证本次运行
一致，不作为用户产品中的版本概念，不要求用户确认，也不会阻止用户继续修改图。

发布是唯一明确的固化点：

- 发布前 Project 为 `development`；
- 发布后 Project 为 `frozen`，页面展示为“已发布”；
- 发布时把当前 Working Graph 固化成一份 Published Graph；
- 首期一个 Project 只发布一次，不支持发布后原地修改；
- 后续如果要多版本发布、回滚和开发分支，再在发布层扩展，不提前污染开发态。

## 2. 这次设计修正了什么

### 2.1 开发态是调试，不是版本管理

开发态需要支持：

- 随时增加、删除节点；
- 修改节点名称、类型、配置、执行角色和输入输出；
- 增加、删除、调整边；
- 修改需求 checklist；
- 单独运行任意节点；
- 反复重跑同一节点；
- 查看每次运行的输入、日志、错误和 Artifact；
- 运行当前整张图进行调试；
- 在一次运行尚未结束时继续修改图，下一次运行使用最新图。

开发态不需要：

- 每次改图生成新 GraphRevision；
- 每次改图要求用户确认一个新版本；
- 维护复杂版本历史；
- 在 UI 中展示 revision 编号；
- 把旧调试 Run 绑定到“过期版本”并创建开发分支。

### 2.2 每个阶段都必须可独立验证

四个业务节点不能等整张 Graph 完成后才能验证。每个节点都需要：

```text
输入选择/绑定
  -> 启动节点验证
  -> 查看 Agent 和平台执行状态
  -> 查看日志、错误、Artifact
  -> 人工检查结果
  -> 重跑或继续下一步
```

单节点验证和整图运行共用 NodeRun、WorkItem、AgentRoleSession、外部提交和 Artifact
机制，但单节点验证不要求所有上游节点已存在。

### 2.3 发布时才固化

发布前允许 Working Graph 不完整。例如可以只有一个 Dataset 节点、临时缺失
Inference 输入，或者刚删除一个节点还没重新连线。只要图的结构本身合法，就允许
保存和继续开发。

发布时必须通过完整校验：

- 所有必需节点存在；
- 所有必需输入都已绑定；
- 所有 Edge 指向存在的 Port；
- 无环；
- 每个节点的配置和 Executor 可解析；
- 每个节点至少有一次与当前配置匹配的成功验证；
- 当前 Working Graph 至少有一次完整调试成功；
- 没有等待中的节点或发布门禁；
- Artifact Schema 和完成条件可校验。

### 2.4 主 Agent 不承担逐节点调度

首期采用轻量控制方式：

- 用户可以在页面点击“验证此节点”或“运行整图”；
- EchoMind Manager 可以通过结构化工具发起相同动作；
- Runtime 自动推进整图中已经满足依赖的节点；
- Manager 只在需要用户确认、节点失败、外部提交不确定或 GraphRun 完成时收到通知；
- Manager 不逐节点下发 start 命令，不处理队列、租约、Kafka 重复消息或心跳。

后续如果确实需要“每个阶段都让主 Agent 决策”，可以通过节点级
`requires_manager_approval` 开关扩展，不改变底层 NodeRun 和调度模型。

## 3. 生命周期

### 3.1 规划前

用户通过 EchoMind Chat 描述目标。Manager 使用 `experiment-planning` Skill 完成需求
澄清，生成 WorkflowPlan。此时没有 Project。

### 3.2 Project 创建

用户确认 PlanProposal 后：

1. 保存 PlanRevision；
2. 创建 Project；
3. 根据计划生成第一版 Working Graph；
4. Project 状态为 `development`；
5. 页面进入开发工作台；
6. 后续对话直接修改 Working Graph，不再要求确认新 PlanRevision 才能改图。

PlanRevision 继续作为“用户最初确认了什么”的规划记录，但开发期间的节点微调不强制
生成新的 PlanRevision。需要追溯需求变化时，由 Graph 修改事件和运行记录承担。

### 3.3 开发态

```text
Project(development)
  -> 修改 Working Graph
  -> 单节点验证 / 重跑
  -> 查看 Artifact
  -> 修改节点配置或边
  -> 再验证
  -> 完整 Graph 调试
  -> 发布
```

开发态允许：

- Graph 临时不完整；
- 节点配置引用尚未生成的 Artifact；
- 任意节点使用显式输入单独验证；
- 从历史 NodeRun 选择 Artifact 作为下一次输入；
- 修改后继续使用旧的 Artifact，但必须显示其来源和 fingerprint；
- 同一节点有多次运行记录；
- 同一 Graph 有多次完整调试记录。

开发态不允许：

- 创建环；
- Edge 指向不存在的节点或 Port；
- 重复 Node ID；
- 使用未知 Node Type 或无法解析的 Executor；
- 直接编辑已经用于完成的 Artifact 内容；
- 在没有发布的情况下声称 Production Run 已经生效。

### 3.4 发布

发布动作：

```text
校验发布条件
  -> 生成 Published Graph 快照
  -> 保存 published_graph_id 和 digest
  -> Project.lifecycle=frozen
  -> 禁止继续修改 Working Graph
  -> 允许创建 Production Run
```

首期发布后不支持原地修改。需要调整流程时：

- 首期可以创建新 Project；
- 后续再增加“从已发布 Project 复制开发副本”或正式版本机制。

### 3.5 生产运行

首期只支持手工创建 Production Run：

- 运行只读取 Published Graph；
- 不依赖 EchoMind Chat 在线；
- Runtime 自动按依赖推进；
- Manager 不参与正常节点推进；
- 异常时 Run 暂停并通知 Manager；
- Agent 可以读取错误并辅助修复，但不能修改当前 Published Graph；
- 修复后首期创建新的 Project/发布流程，后续再增加版本和开发分支。

## 4. 当前实现基线与缺口

### 4.1 已经具备

| 能力 | 当前状态 |
|---|---|
| 两个独立 Chat | Pyromind Workbench 与 EchoMind Manager |
| 通用规划 | `experiment-planning`、PlanProposal、PlanRevision |
| Project 创建 | 用户确认计划后创建文件型 Project/Graph |
| Graph 模型 | 节点、边、拓扑校验、四种节点类型 |
| 持久任务执行 | `experiment_task_runs` |
| 队列与恢复 | `runtime_work_items`、租约、代次、对账 |
| Agent 角色 | Manager 和内部 Eval role |
| 外部平台提交 | 幂等账本、Kafka/查询观察 |
| Artifact | 文件登记、SHA-256、基础读取 |
| 阶段展示壳 | StageRun 和 `contract_probe` |

### 4.2 当前缺口

| 缺口 | 影响 |
|---|---|
| Working Graph 只能从 Plan 生成，没有开发态修改契约 | 用户不能随对话持续改图 |
| 没有节点独立验证 API | 四个节点只能启动壳流程，不能按阶段调试 |
| StageRun 没有关联 Runtime 队列和角色 | 刷新或重启后恢复能力不足 |
| 没有整图调试 Run | 无法验证依赖、Artifact 传递和失败传播 |
| 四个真实 Executor 都未接入 | Dataset、Train、Inference、Evaluator 没有业务闭环 |
| 没有发布快照 | 无法区分可修改开发图和不可修改生产图 |
| 前端主要展示 StageRun | 不能按节点查看配置、输入、输出和多次运行 |

### 4.3 两个可用 ID 的边界

| ID | 含义 |
|---|---|
| `project_id` | 一个实验工作台 |
| `graph_id` | Project 当前 Working Graph 的稳定身份 |
| `graph_run_id` | 某一次节点验证、整图调试或生产运行 |
| `node_id` | Graph 中的节点 |
| `node_run_id` | 某节点的一次执行 |
| `attempt` | 同一节点同一输入下的重试序号，仅用于区分执行记录 |
| `task_id` | 外部平台任务 |
| `artifact_id` | 已验证产物 |
| `conversation_id` | EchoMind Chat 会话 |

`graph_run_id` 在开发态不携带版本语义。它只回答“这次运行使用的是什么快照、产生了
什么结果”。

## 5. 轻量数据模型

### 5.1 Project

```json
{
  "schema_version": 1,
  "project_id": "project-uuid",
  "name": "AVI 图片质检实验",
  "lifecycle": "development",
  "echomind_conversation_id": "conversation-id",
  "owner_user_id": "42",
  "graph_id": "graph-uuid",
  "published_graph_id": null,
  "created_at": "2026-11-02T00:00:00Z"
}
```

生命周期首期只有：

```text
development
  -> frozen
```

`frozen` 在 UI 中显示为“已发布”。

### 5.2 Working Graph

继续使用当前 `graph.json`，保持可修改：

```json
{
  "schema_version": 1,
  "project_id": "project-uuid",
  "graph_id": "graph-uuid",
  "nodes": [
    {
      "id": "dataset",
      "type": "dataset",
      "title": "数据准备",
      "launch_mode": "agent",
      "executor": "dataset_role",
      "config": {}
    }
  ],
  "edges": [
    {
      "source": "dataset",
      "target": "train"
    }
  ],
  "modified_at": "2026-11-02T00:00:00Z"
}
```

Working Graph 不增加：

- revision；
- parent_revision；
- revision_history；
- release_channel；
- deployment 字段。

每次修改只覆盖当前 `graph.json`，并通过事件通知前端刷新。

### 5.3 Execution Snapshot

每次创建 GraphRun 时，在 SQL 中保存：

```json
{
  "graph_run_id": "graph-run-uuid",
  "project_id": "project-uuid",
  "mode": "node_validation",
  "target_node_id": "dataset",
  "graph_snapshot": {},
  "graph_digest": "sha256:...",
  "request_key": "user-request-uuid"
}
```

这份快照用于：

- 保证一次运行不受运行期间 Graph 编辑影响；
- 恢复时知道当时使用了什么拓扑和配置；
- 判断节点配置变化后旧的验证结果是否过期。

快照不是产品版本，不出现在正常开发操作的确认流程中。

### 5.4 GraphRun

首期 `graph_runs` 表建议包含：

| 字段 | 说明 |
|---|---|
| `id` | GraphRun ID |
| `owner` | 用户/Conversation/Profile 摘要 |
| `project_id` | Project |
| `mode` | `node_validation`、`graph_debug`、`production` |
| `target_node_id` | 单节点验证目标，其他模式为 null |
| `graph_snapshot` | 本次运行的 Working Graph / Published Graph 快照 |
| `graph_digest` | 快照摘要 |
| `state` | `queued`、`running`、`waiting_user`、`paused`、`succeeded`、`failed`、`cancelled` |
| `request_key` | 创建幂等键 |
| `active_node_run_id` | 串行整图运行时当前活动节点 |
| `error` | 结构化错误摘要 |
| `created_at`、`started_at`、`finished_at` | 时间 |

首期没有 GraphRunCommand 表。用户或 Manager 的启动、重试、取消动作通过 API 请求键
和 NodeRun/GraphRun 状态保证幂等。

### 5.5 NodeRun

继续扩展 `experiment_task_runs`，不新建平行执行表。新增：

| 字段 | 说明 |
|---|---|
| `graph_run_id` | 所属运行，旧的 standalone task 为 null |
| `node_id` | Graph Node ID |
| `attempt` | 该节点在当前 GraphRun 下的重试序号 |
| `node_config_digest` | 节点配置指纹，用于判断验证是否过期 |
| `input_bindings` | 本次实际输入 Artifact/Resource |
| `output_bindings` | 本次已登记输出 Artifact |
| `waiting_reason` | 等待用户、平台或外部确认的原因 |
| `started_at`、`finished_at` | 节点执行时间 |

`task_id` 在 Graph 模式保存 `node_id` 或稳定的 node execution key，避免破坏现有非空
列；Graph 身份以 `graph_run_id + node_id` 为准。

### 5.6 Artifact

节点之间通过 Artifact 传递。首期 AVI 类型：

| Node | Artifact Type |
|---|---|
| Dataset | `avi.dataset_version` |
| Train | `model.checkpoint` |
| Inference | `avi.prediction_set` |
| Evaluator | `avi.metrics`、`avi.eval_result`、`avi.atif_report` |

每个 Artifact 记录：

- `artifact_id`；
- `project_id`；
- `graph_run_id`；
- `node_run_id`；
- `type`；
- `schema`；
- `uri`；
- `sha256`；
- `metadata`。

### 5.7 Published Graph

发布后保存独立文件或记录：

```text
projects/<project_id>/published-graph.json
```

内容包含：

- Published Graph 快照；
- `published_graph_id`；
- `source_graph_id`；
- `graph_digest`；
- 发布时四个节点的最近成功验证引用；
- 发布时的完整 Graph 调试 Run ID；
- `published_at`；
- `published_by`。

首期只保留一份 Published Graph，不实现版本列表。

## 6. Working Graph 编辑契约

### 6.1 操作

新增统一的 Graph 修改入口：

```text
apply_graph_patch(
  project_id,
  request_id,
  operations[]
)
```

首期操作：

| operation | 作用 |
|---|---|
| `add_node` | 增加节点 |
| `update_node` | 修改标题、executor、launch_mode、config |
| `remove_node` | 删除节点及所有关联边 |
| `add_edge` | 增加边 |
| `remove_edge` | 删除边 |
| `update_requirements` | 修改当前 Project checklist |

建议请求：

```json
{
  "request_id": "request-uuid",
  "operations": [
    {
      "op": "add_node",
      "node": {
        "id": "dataset",
        "type": "dataset",
        "title": "数据准备",
        "launch_mode": "agent",
        "executor": "dataset_role",
        "config": {}
      }
    },
    {
      "op": "add_edge",
      "source": "dataset",
      "target": "train"
    }
  ]
}
```

### 6.2 校验

每次修改立即校验：

- Project 存在且属于当前用户；
- Project 处于 development；
- Node ID 唯一；
- Node Type 已知；
- Edge source/target 存在；
- 无自环；
- 无环。

开发态允许暂时不完整的图，所以立即校验不要求：

- 所有必需输入已绑定；
- 所有节点的 Executor 已经具备真实实现；
- 图可以完整运行；
- 所有下游节点都已连接。

这些完整性条件在启动整图调试和发布时校验。

### 6.3 并发与运行

- 修改使用原子文件替换；
- Worker 或 Agent 不直接写 `graph.json`；
- 正在运行的 GraphRun 不读取最新 `graph.json`，只使用自己的快照；
- Graph 修改在一个 Run 运行期间允许发生；
- 下一次 Node/Graph 运行使用最新 Working Graph；
- 页面明确显示“当前图已修改，旧运行继续使用旧快照”；
- 如果用户修改了某节点配置，该节点之前的成功验证标记为“已过期”，但仍保留历史。

### 6.4 Agent 工具

EchoMind Manager 首期增加：

```text
get_project_graph
apply_graph_patch
get_project_run
start_node_validation
start_graph_debug_run
publish_project
```

这些工具只加入 EchoMind Manager Profile。通用 Tool Registry、Pyromind Workbench
Profile 和领域 Skill 不感知 EchoMind Project。

建议新增一个通用但属于 EchoMind 的 `experiment-graph-development` Skill：

- 指导 Manager 根据用户对话修改当前 Graph；
- 指导何时启动单节点验证；
- 指导如何查看失败和 Artifact；
- 不包含 AVI 专属节点细节；
- 不承载调度器；
- 不修改 `data-processing`、`inference-evaluation` 等通用 Skill。

## 7. 单节点验证

### 7.1 目标

任意阶段都可以在完整 Graph 尚未可用时独立启动和验证：

- Dataset 可以验证数据准备；
- Train 可以消费已有 Dataset Artifact 验证 Workflow；
- Inference 可以消费已有 DatasetVersion + checkpoint 验证预测；
- Evaluator 可以消费已有 predictions + 真值验证评分。

### 7.2 启动请求

```json
{
  "request_id": "request-uuid",
  "node_id": "train",
  "input_bindings": {
    "dataset_version": {
      "artifact_id": "artifact-uuid"
    }
  },
  "config_overrides": {}
}
```

后端：

1. 读取当前 Working Graph；
2. 找到目标节点；
3. 合并显式输入；
4. 创建 `GraphRun(mode=node_validation)`；
5. 创建目标 NodeRun；
6. 写入 WorkItem；
7. 唤醒对应 role；
8. 等待 Agent 或平台终态；
9. 校验 Artifact；
10. 完成 GraphRun。

### 7.3 节点状态

首期 NodeRun 状态：

```text
queued
  -> running
  -> waiting_user
  -> waiting_external
  -> verifying
  -> succeeded

queued/running/waiting_user/waiting_external/verifying
  -> failed

queued/running/waiting_external
  -> cancelled
```

### 7.4 单节点验证不能绕过的东西

- 对应 role 的工具和 Skill 白名单；
- 远端 Sandbox；
- Artifact Schema 校验；
- 外部任务提交幂等；
- Runtime 重启恢复；
- 用户 Cookie/x-cluster 等凭据的请求级传递；
- Project 和 Conversation 归属校验。

单节点验证可以绕过：

- 其他上游节点必须已经成功；
- 完整 Graph 的依赖顺序；
- 后续节点尚未完成；
- 当前 Working Graph 的完整性校验。

### 7.5 验证新鲜度

NodeRun 保存 `node_config_digest`。页面对比当前节点配置：

- digest 相同：显示“已验证”；
- digest 不同：显示“配置已修改，需要重新验证”；
- 输入 Artifact 被替换：显示“输入已变化，需要重新验证”；
- 旧 NodeRun 仍可查看，不删除。

这里只做新鲜度判断，不创建节点版本。

## 8. 整图调试

### 8.1 启动

用户点击“运行整图”或 Manager 调用 `start_graph_debug_run`：

1. 读取当前 Working Graph；
2. 校验结构和必需输入；
3. 创建 `GraphRun(mode=graph_debug)` 并保存快照；
4. 计算 ready Node；
5. Runtime 串行推进；
6. 成功后运行下一个节点；
7. 失败后暂停并通知 Manager/用户。

### 8.2 依赖

首期四节点依赖：

```text
Dataset
  -> Train
  -> Inference（同时需要 Dataset）
  -> Evaluator（同时需要 Dataset + Inference）
```

对应 Edge：

| Source | Target |
|---|---|
| dataset | train |
| dataset | inference |
| train | inference |
| dataset | evaluator |
| inference | evaluator |

首期 Node 的输入仍由 Port 表达。现有单 source/target Edge 可以保留用于展示，
Runtime 使用节点输入绑定判断 readiness。

### 8.3 自动推进

- Runtime 自动选择下一个 ready Node；
- 同时最多一个 active NodeRun；
- 正常成功不要求 Manager 每次确认；
- Dataset 小样需要用户确认时进入 `waiting_user`；
- 平台任务等待时进入 `waiting_external`；
- 失败时 GraphRun paused；
- 用户或 Manager 可以重试失败节点；
- 重试创建新的 NodeRun attempt，不覆盖历史；
- 运行中使用快照，当前图继续可编辑。

### 8.4 图修改与运行的关系

```text
Run A snapshot = graph digest A

用户修改 Working Graph

Run A 继续使用 digest A
下一次 Run B 使用 digest B
```

不要求：

- 修改 Graph 前停止 Run A；
- 将 Run A 标记为旧版本；
- 为 Run B 创建用户可见版本；
- 用户确认 GraphRevision。

## 9. 发布与固化

### 9.1 发布检查

发布按钮只在以下条件满足时可用：

| 条件 | 判断 |
|---|---|
| Project 为 development | 是 |
| Graph 结构合法 | 是 |
| 所有必需 Node 存在 | 是 |
| 所有必需输入绑定完成 | 是 |
| 每个 Node 当前配置至少有一次成功验证 | 是 |
| 最近一次整图调试成功 | Graph digest 与当前 Working Graph 一致 |
| 无运行中 GraphRun | 是 |
| 无待用户确认事项 | 是 |

### 9.2 发布快照

```json
{
  "schema_version": 1,
  "published_graph_id": "published-graph-uuid",
  "project_id": "project-uuid",
  "source_graph_id": "graph-uuid",
  "graph": {},
  "graph_digest": "sha256:...",
  "node_validation_run_ids": {
    "dataset": "node-run-uuid",
    "train": "node-run-uuid",
    "inference": "node-run-uuid",
    "evaluator": "node-run-uuid"
  },
  "graph_debug_run_id": "graph-run-uuid",
  "published_at": "2026-11-02T00:00:00Z",
  "published_by": "42"
}
```

### 9.3 发布后

- Working Graph 不再接受修改；
- 保留所有开发和调试 Run；
- Production Run 只引用 Published Graph；
- 前端显示“已发布”及发布时使用的节点验证/整图运行；
- Chat 可以继续只读查询；
- 首期不允许原地版本更新；
- 后续版本管理作为独立阶段设计。

## 10. 四节点职责

### 10.1 Dataset

输入：

- 原始 Storage 数据；
- 用户的数据处理目标；
- 输出 Schema 和资源约束。

执行：

1. 读取真实数据；
2. 使用 `data-processing` 选择处理范式；
3. 生成 Pipeline；
4. 在远端 Sandbox 跑小样；
5. 等待用户确认；
6. 调用 `df_submit_pipeline`；
7. 等待平台终态；
8. 校验 `processed.jsonl`、`report.json`、split 和 fingerprint。

输出：

- `avi.dataset_version`；
- train/validation/test manifest；
- 输入与输出 fingerprint；
- 数据校验报告。

节点验证通过条件：

- 小样可读且符合预期；
- 全量任务有 task/run/output ID；
- 数据数量和 split 一致；
- 标签和图片路径可解析；
- 无跨 split 泄漏；
- Artifact 可被下游读取。

### 10.2 Train

输入：

- DatasetVersion；
- 用户训练目标；
- 模型和 GPU 资源；
- 训练节点契约。

执行：

1. 使用 `generate-workflow-dsl` 生成 Workflow；
2. 使用 `validate_workflow_dsl` 校验；
3. Runtime 冻结 DSL、code、配置和输入指纹；
4. 通过 Graph-aware submission port 创建训练任务；
5. 等待 Kafka/查询终态；
6. 校验 checkpoint Manifest。

这里不直接复用交互式 `submit_workflow` Tool 的 Copilot 确认语义。Graph 路径复用
底层平台提交能力，由 GraphRun 创建动作和节点验证动作承担授权记录。

输出：

- `model.checkpoint`；
- workflow/task/correlation ID；
- checkpoint 文件清单和 SHA-256；
- 训练日志和失败摘要。

节点验证通过条件：

- DSL 校验通过；
- 平台任务真实创建；
- 任务终态成功；
- checkpoint 文件完整；
- checkpoint 可被 Inference 读取；
- 重启或重复请求不会重复提交。

### 10.3 Inference

输入：

- DatasetVersion 的 validation/test split；
- Train checkpoint；
- evaluation config；
- 模型服务和 GPU 配置。

执行：

1. 准备 dataset config、evaluation config 和模型引用；
2. 调用 `df_submit_evaluation(operation="inference")`；
3. 等待平台任务；
4. 校验 predictions 覆盖率、sample ID、模型指纹和输入指纹。

输出：

- `avi.prediction_set`；
- `predictions.jsonl`；
- inference task/run ID；
- 预测数量和样本覆盖报告。

节点验证通过条件：

- 每个预测可回查 sample；
- 预测数量与输入 split 一致；
- 模型/checkpoint 指纹一致；
- 没有混入训练集；
- 平台成功且 Artifact 校验通过。

### 10.4 Evaluator

输入：

- predictions；
- DatasetVersion 中的真值；
- 冻结的评分规则/脚本；
- 可选 report spec。

执行：

1. 调用 `df_submit_evaluation(operation="score")`；
2. 生成 metrics；
3. 调用 `df_submit_evaluation(operation="report")` 或复用现有 CPU 报告路径；
4. 校验指标、数量和报告内容。

输出：

- `avi.metrics`；
- `avi.eval_result`；
- `avi.atif_report`；
- badcase 和混淆矩阵。

节点验证通过条件：

- 不重新运行模型推理；
- 同一 predictions 重复评分结果一致；
- 总样本数、成功数、错误数与 predictions 一致；
- 报告可读；
- 前端可以加载标准结果。

## 11. Runtime 执行方式

### 11.1 复用现有基础设施

| 现有能力 | 新用途 |
|---|---|
| `runtime_work_items` | 节点验证、图推进、重试和恢复 |
| `agent_role_sessions` | Dataset、Train、Eval role |
| `external_task_submissions` | NodeRun 的平台任务 |
| Kafka Inbox | 平台任务终态 |
| 平台查询 | Kafka 不可用时对账 |
| Artifact 校验 | NodeRun 成功前置条件 |
| Runtime Outbox | 页面和 Manager 通知 |

### 11.2 节点执行流程

```text
GraphRun/NodeRun 创建
  -> WorkItem
  -> role session
  -> Agent 探查和准备
  -> Agent 调用领域 Tool
  -> Runtime 等待外部平台
  -> ArtifactVerifier
  -> NodeRun succeeded
  -> 单节点 Run 结束
     或整图 Run 计算下一个 ready Node
```

### 11.3 Agent 与 Runtime 边界

Agent 负责：

- 理解当前节点目标；
- 探查真实输入；
- 选择领域参数；
- 使用允许的 Skill/Tool；
- 提交平台任务；
- 解释错误；
- 声明输出 Artifact。

Runtime 负责：

- Graph/Node 状态；
- 输入绑定和验证；
- WorkItem 和 lease；
- 外部任务幂等；
- Kafka/查询对账；
- Artifact 存在性、Schema 和 fingerprint；
- 是否允许进入下游节点；
- 发布条件。

Agent 不能直接写：

- GraphRun 状态；
- NodeRun 状态；
- Published Graph；
- 其他节点的输入；
- 其他角色的会话。

## 12. API 与前端

### 12.1 开发期 API

```text
GET  /api/echomind/projects/{project_id}
GET  /api/echomind/projects/{project_id}/graph
POST /api/echomind/projects/{project_id}/graph/actions
POST /api/echomind/projects/{project_id}/nodes/{node_id}/validate
GET  /api/echomind/projects/{project_id}/node-runs
POST /api/echomind/projects/{project_id}/debug-runs
GET  /api/echomind/projects/{project_id}/debug-runs
GET  /api/echomind/projects/{project_id}/runs/{graph_run_id}
POST /api/echomind/projects/{project_id}/runs/{graph_run_id}/cancel
POST /api/echomind/projects/{project_id}/runs/{graph_run_id}/retry-node
```

### 12.2 发布 API

```text
GET  /api/echomind/projects/{project_id}/publish-readiness
POST /api/echomind/projects/{project_id}/publish
GET  /api/echomind/projects/{project_id}/published-graph
POST /api/echomind/projects/{project_id}/production-runs
```

### 12.3 前端

开发态页面需要增加：

- Working Graph 的“开发中”状态；
- 节点独立验证按钮；
- 节点最近一次成功验证和“已过期”状态；
- 节点运行历史；
- 当前输入和输出 Artifact；
- 日志和错误；
- “运行整图”按钮；
- GraphRun 进度；
- 失败节点重试；
- 发布条件清单；
- 发布按钮；
- 发布后只读视图。

前端不展示 GraphRevision、parent revision 或版本差异。它展示的是：

```text
当前图是否可运行
哪些节点已经验证
上次验证是否过期
当前运行走到哪一步
```

### 12.4 兼容旧接口

首期保留：

- `GET /api/echomind/projects/{project_id}`；
- `POST .../nodes/{node_id}/start`；
- `POST .../nodes/{node_id}/run`；
- `GET .../runs`；
- `GET .../artifacts/{artifact_id}`。

兼容策略：

- 旧 StageRun 继续可读；
- 新节点验证 Run 成为主数据源；
- 旧 `/start` 逐步映射到 `node_validation`；
- 迁移完成前，新前端优先读取 NodeRun，旧接口只作为 fallback；
- Workbench Chat 不受影响。

## 13. 渐进式实施阶段

每个阶段单独实现、单独启动验证、单独回归。上一阶段未通过不进入下一阶段。

阶段总览：

| 阶段 | 目标 | 是否真实业务执行 |
|---|---|---|
| S1 | Project 开发态与可变 Graph | 否 |
| S2 | 任意节点独立验证壳 | 否 |
| S3 | Dataset 真实执行 | 是 |
| S4 | Train 真实执行 | 是 |
| S5 | Inference 真实执行 | 是 |
| S6 | Evaluator 真实执行 | 是 |
| S7 | 四节点整图调试 | 是 |
| S8 | 发布冻结与手工生产运行 | 是 |

### 通用启动方式

每个阶段开始验证前：

```sh
./start_inference.sh
cd /Users/zhangpeng/PycharmProjects/minimal-chat-frontend
npm run dev
```

浏览器打开：

```text
http://localhost:5173/?chat=echomind&userId=42
```

需要远端 Sandbox 或平台调用时，在页面认证区配置：

- Cookie 或 Bearer；
- `x-cluster`。

每阶段自动化回归基础命令：

```sh
uv run pytest -q tests/pyromind_runtime tests/agent_server
cd harness-adapter/pi-runtime && npm test
cd /Users/zhangpeng/PycharmProjects/minimal-chat-frontend
npm test
npm run build
npm run lint
```

## 14. S1：Project 开发态与可变 Graph

### 14.1 阶段目标

Project 创建后保持 `development`，用户可以通过 EchoMind Chat 让 Manager 持续修改
当前 Graph。页面刷新后能看到最新图，不需要确认新版本。

### 14.2 实现内容

后端：

- 复用当前 Project/Graph 文件模型；
- 新增 `apply_graph_patch` 服务；
- 新增 graph mutation API；
- 增加 Node 类型、ID、Edge 和环校验；
- 允许开发态图暂时不完整；
- 修改 `graph.json` 后写 ProductEvent；
- Project 仍为 `development`；
- 不新增 GraphRevision 表。

Agent：

- EchoMind Manager 增加 `get_project_graph` 和 `apply_graph_patch`；
- 增加 EchoMind 专属 `experiment-graph-development` Skill；
- Manager 根据用户话术修改节点、边和配置；
- 不把 Graph 修改逻辑放入通用 Skill/Tool。

前端：

- Graph 面板读取最新 Working Graph；
- 显示“开发中”；
- 刷新后保留修改；
- 节点和边变化来自服务端，不由前端自己维护第二份图；
- 增加节点详情和配置展示，支持看到修改结果。

### 14.3 验证内容

自动化：

- 新增 `tests/pyromind_runtime/test_project_graph_editing.py`；
- 覆盖 add/update/remove node、add/remove edge、环、重复 ID、未知节点；
- 覆盖 Project 归属和 frozen 状态拒绝修改；
- 保留现有计划确认和 Project 创建测试。

手工：

1. 启动后端和前端；
2. 打开 EchoMind 页面；
3. 生成计划并确认；
4. 确认页面出现 Project 和 Graph；
5. 在 Chat 中输入“把 Evaluator 节点改成检测报告评审”；
6. 检查节点标题或配置发生变化；
7. 输入“删除 Inference 再重新加回来”；
8. 检查节点和关联边即时更新；
9. 刷新页面；
10. 检查 Graph 与刷新前一致；
11. 再新建一个 Workbench 会话，确认原 Workbench 行为不受影响。

通过标准：

- 不需要确认 GraphRevision；
- 不需要创建新 Project；
- Graph 修改可刷新恢复；
- 修改期间没有主动运行；
- 环等非法结构被拒绝；
- 旧 Plan 和 Project 创建测试通过。

### 14.4 回归影响

- 原有计划确认只创建 Project/Graph，不自动执行；
- 原 Project 读取接口保持兼容；
- Workbench Chat 和 Profile 不改；
- 旧 `graph.json` 可读取；
- 未启用的部署不暴露 Graph 修改按钮。

## 15. S2：任意节点独立验证壳

### 15.1 阶段目标

四个节点都可以单独点击“验证”，生成持久 NodeRun、状态、日志和 Artifact。此阶段
只验证运行链路，不声称完成真实 Dataset/Train/Inference/Evaluator 业务。

### 15.2 实现内容

S2 采用轻量实现，先复用现有 `StageRun` 持久化壳，不提前引入
`GraphRun`、`runtime_work_items` 或新的节点执行数据库。

后端：

- 复用 `projects/<project_id>/runs/*.json` 保存节点验证 Run；
- 扩展 `StageRun`：
  - `mode=node_validation`；
  - `request_id` 幂等键；
  - `node_config_digest` 配置指纹；
- 四个节点都可通过现有 `/run` 入口启动验证壳，不再为 Agent 节点单独拒绝；
- Dataset 的 Agent 开发入口继续由 `/start` 投递会话，和验证壳保持两个动作；
- 使用现有 `contract_probe` executor；
- 生成 `node_validation_report` Artifact；
- executor 失败会落成 `failed` StageRun；
- 相同 `request_id` 返回原 Run，不重复执行；同 ID 换节点或换配置返回冲突。

Agent：

- S2 不新增 Manager 工具或领域 Skill；
- 节点验证由前端直接调用 Project API；
- 真实节点执行和 Manager 参与留到后续阶段。

前端：

- 每个节点显示“验证”按钮；
- 展示最近一次验证状态；
- 展示验证历史、时间、错误和 `node_validation_report`；
- 配置修改后显示“验证已过期”；
- 页面刷新后从 `/runs` 恢复状态。

### 15.3 验证内容

自动化：

- 扩展 `tests/agent_server/test_echomind_stage_runs.py`；
- 四个节点都能创建验证 Run；
- 相同 request ID 不创建重复 Run；
- executor 失败状态可持久化；
- 配置指纹前后端一致；
- Artifact 内容、SHA-256 和读取接口正确；
- 旧 StageRun 字段仍可读取。

手工：

1. 打开已有 Project；
2. 对 Dataset 节点点击“验证”；
3. 观察状态从 running 到 succeeded；
4. 打开生成的 `node_validation_report`；
5. 对剩余三个节点重复；
6. 修改 Train 节点配置；
7. 检查 Train 显示“验证已过期”，其他节点不受影响；
8. 重新验证 Train；
9. 刷新页面，检查验证状态和历史仍然存在。

通过标准：

- 四节点都能独立启动；
- 每个节点都有可查询的配置指纹、状态、时间和 Artifact；
- 修改一个节点只让该节点的旧验证过期；
- 刷新后状态不丢失，相同 request ID 不重复执行；
- 此阶段结果只用于验证执行壳。

### 15.4 回归影响

- 旧 StageRun 仍可读取；
- `contract_probe` 是 S2 的唯一节点执行器；
- 现有 WorkQueue 和 ExperimentExecution 保持独立，不做改造；
- 没有新增 GraphRevision 或版本确认。
- `GraphRun`、取消、进程中断恢复和真实节点执行在后续阶段补充。

## 16. S3：Dataset 真实执行

### 16.1 阶段目标

Dataset 节点使用 `data-processing` 和远端 Sandbox 完成真实数据准备，并产出可被
其他节点使用的 DatasetVersion。

### 16.2 实现内容

- 新增 Dataset role；
- role 加载 `data-processing`；
- 绑定 `df_submit_pipeline`、`df_check_progress`、Storage 和 Sandbox 工具；
- NodeRun 支持 `waiting_user`；
- 小样结果和用户确认写回 NodeRun；
- 全量提交关联 `external_task_submissions`；
- Kafka/查询终态驱动 NodeRun；
- ArtifactVerifier 校验数据、报告、split 和 fingerprint；
- 生成 `avi.dataset_version`。

### 16.3 验证内容

自动化：

- 小样前不提交全量；
- 用户拒绝后 NodeRun cancelled；
- 用户确认后只提交一次；
- 提交响应不确定时不自动重提；
- Kafka 重放幂等；
- 缺文件、数量不一致、schema 错误导致 failed；
- 通用 data-processing 测试无 EchoMind/AVI 分支。

手工：

1. 为 Dataset 节点选择真实 Storage 输入；
2. 点击节点验证；
3. 检查 Manager/Chat 收到 Dataset role 的小样结果；
4. 检查 NodeRun 状态为 waiting_user；
5. 确认小样；
6. 检查平台任务真实提交；
7. 检查页面显示 platform task ID、output dir；
8. 等待终态；
9. 查看 DatasetVersion Artifact；
10. 在 Train 节点验证时选择该 Artifact；
11. 重启后端，确认不会重复提交。

通过标准：

- 真实数据可读；
- 小样、确认、全量在同一 NodeRun 链路；
- Artifact 可被下游选择；
- 重启、重复事件和不确定提交不会重复执行。

### 16.4 回归影响

- 其他三个节点仍使用 development stub；
- 原 Dataset `/start` 在兼容模式下可继续投递会话命令；
- 新路径验证通过后再切换前端主按钮；
- Workbench 通用 data-processing 行为不变。

## 17. S4：Train 真实执行

### 17.1 阶段目标

Train 节点消费 DatasetVersion，生成并校验 Workflow，创建真实训练任务，产出完整
checkpoint Artifact。

### 17.2 实现内容

- 新增 Train role；
- 加载 `generate-workflow-dsl`；
- 生成并调用 `validate_workflow_dsl`；
- 保存 workflow/code/config fingerprint；
- 新增 Graph-aware workflow submission port；
- 外部提交账本关联 NodeRun；
- Kafka/查询更新状态；
- checkpoint Manifest 校验；
- 失败错误结构化。

### 17.3 验证内容

自动化：

- 无 Dataset Artifact 时拒绝启动；
- DSL 校验失败时不提交；
- 重复 request ID 不重复训练；
- 平台明确失败保存失败；
- 提交不确定不自动重提；
- checkpoint 缺文件不成功；
- Kafka 重放不重复。

手工：

1. 先完成 Dataset S3 验证；
2. 在 Train 节点选择 DatasetVersion；
3. 启动 Train 节点验证；
4. 在页面查看生成的 Workflow 和校验结果；
5. 提交一个小规模训练任务；
6. 查看 task ID 和 GraphRun/NodeRun 关联；
7. 等待平台终态；
8. 查看 checkpoint Artifact 和文件清单；
9. 重启后端，确认没有重复训练；
10. 人为修改 Workflow 配置后确认旧验证过期。

通过标准：

- Train 只消费选定 DatasetVersion；
- Workflow 先校验后提交；
- NodeRun 可追踪平台任务；
- checkpoint 可被 Inference 选择；
- 失败和不确定提交行为正确。

### 17.4 回归影响

- Dataset 节点继续可独立验证；
- 未完成的 Inference/Evaluator 仍使用 stub；
- 交互式 `submit_workflow` 的 Workbench 行为不变；
- Graph 训练提交由独立 port 完成，不污染通用 Tool。

## 18. S5：Inference 真实执行

### 18.1 阶段目标

Inference 节点消费 DatasetVersion 和 checkpoint，提交真实推理任务，生成可验证的
predictions。

### 18.2 实现内容

- Eval role 增加 Inference Node operation；
- 绑定 `df_submit_evaluation(operation="inference")`；
- 输入 DatasetVersion、checkpoint、dataset config、evaluation config；
- 关联 inference task/run；
- 校验 predictions 覆盖率、sample ID 和模型指纹；
- 生成 `avi.prediction_set`。

### 18.3 验证内容

自动化：

- checkpoint 或 Dataset 缺失时拒绝；
- predictions 数量不一致失败；
- sample 无法回查失败；
- 模型版本不一致失败；
- 平台不确定提交不自动重试；
- 相同 request 不重复推理。

手工：

1. 选择已验证的 DatasetVersion 和 checkpoint；
2. 启动 Inference 节点验证；
3. 检查配置实际使用的模型和 split；
4. 提交推理任务；
5. 查看 task/run ID；
6. 等待终端状态；
7. 打开 predictions Artifact；
8. 抽查 sample ID 与 Dataset 的对应；
9. 重启后端确认不重复提交。

通过标准：

- predictions 可追踪到模型、输入和平台任务；
- 预测覆盖完整；
- 失败不会伪造结果；
- Evaluator 可以选择该 Artifact。

### 18.4 回归影响

- Dataset 和 Train 单独验证仍通过；
- Evaluator stub 不依赖预测内部格式的额外假设；
- 通用 inference-evaluation 测试保持通过。

## 19. S6：Evaluator 真实执行

### 19.1 阶段目标

Evaluator 节点消费 predictions 和真值，生成确定性指标、评测结果和标准报告。

### 19.2 实现内容

- Eval role 增加 score/report operation；
- 绑定 `df_submit_evaluation(operation="score")`；
- 绑定 report 生成；
- 校验 metrics、预测数量和错误数；
- 生成 `avi.metrics`、`avi.eval_result`、`avi.atif_report`；
- 前端读取并展示指标和报告。

### 19.3 验证内容

自动化：

- 相同 predictions 重复评分结果一致；
- 评分失败不重新运行 Inference；
- metrics 数量与 predictions 不一致时失败；
- report 不可读时不成功；
- Artifact 重放幂等；
- AVI reference 不污染通用评测逻辑。

手工：

1. 选择已验证 predictions 和 DatasetVersion；
2. 启动 Evaluator 节点；
3. 检查使用同一份 predictions，不重新启动模型；
4. 等待评分完成；
5. 查看 metrics 和 report；
6. 使用同一 requests 再运行一次；
7. 比较指标 hash；
8. 打开前端标准报告；
9. 检查 badcase 与 confusion matrix。

通过标准：

- 评分结果确定；
- 报告可读；
- 不重复 GPU 推理；
- 指标和样本覆盖一致；
- 可以作为整图调试的最终节点。

### 19.4 回归影响

- 四个节点现在都能独立真实验证；
- 每个节点仍可反复启动和查看历史；
- 不引入版本；
- 整图调试尚未自动启用，避免一次性引入过多行为。

## 20. S7：四节点整图调试

### 20.1 阶段目标

使用当前 Working Graph 的快照，从 Dataset 开始串行推进完整四节点，验证依赖、
Artifact 绑定、失败暂停和重启恢复。

### 20.2 实现内容

- GraphScheduler `tick()`；
- 当前 Graph 快照；
- readiness 计算；
- 单 active NodeRun；
- 自动推进；
- 失败暂停和重试；
- GraphRun 事件和页面通知；
- Manager 读取失败并决定重试；
- 允许运行期间继续修改 Working Graph。

### 20.3 验证内容

自动化：

- 四节点成功顺序；
- Inference 同时等待 Dataset + Train；
- Evaluator 同时等待 Dataset + Inference；
- 不满足依赖时不启动；
- 同时最多一个活动 NodeRun；
- 失败后下游不启动；
- 重试创建 attempt+1；
- 运行中修改 Graph 不影响当前快照；
- 新 Run 使用最新 Working Graph；
- Runtime 重启后继续；
- 重复启动请求不创建两个 GraphRun。

手工：

1. 确保四个节点均有当前配置的成功单节点验证；
2. 点击“运行整图”；
3. 观察 Dataset 完成后 Train 启动；
4. 观察 Inference 等待 Dataset 和 Train 两个输入；
5. 观察 Evaluator 等待 Inference predictions；
6. 检查每个节点输入输出 Artifact；
7. 在 GraphRun 运行时修改某个节点标题或配置；
8. 确认运行中的 GraphRun 不受影响；
9. 运行结束后再次启动，确认使用最新图；
10. 人为让 Train 失败，确认 Inference/Evaluator 不启动；
11. 重试 Train；
12. 重启后端，确认 GraphRun 可以继续或正确暂停。

通过标准：

- 当前 Working Graph 可以整图调试；
- 节点依赖和 Artifact 正确；
- Graph 修改与运行快照互不干扰；
- 失败、重试、恢复可用；
- 不需要逐节点 Manager 确认。

### 20.4 回归影响

- 单节点验证保持可用；
- 单节点 Run 不参与整图状态；
- 旧 StageRun 只读兼容；
- Workbench 不受影响。

## 21. S8：发布冻结与手工生产运行

### 21.1 阶段目标

将已验证的当前 Graph 固化为 Published Graph，禁止修改，并支持手工创建只读该快照的
生产运行。

### 21.2 实现内容

- 发布 readiness 计算；
- Published Graph 生成；
- Project lifecycle 切到 frozen；
- Graph mutation API 拒绝 frozen；
- Production Run 创建；
- Production Run 使用 Published Graph；
- 前端发布确认和只读页面；
- 首期手工触发，不接 cron/event。

### 21.3 验证内容

自动化：

- 节点验证过期时不能发布；
- 最近整图调试失败时不能发布；
- 有运行中 GraphRun 时不能发布；
- 发布后修改 Graph 被拒绝；
- Production Run 使用发布快照而不是 Working Graph；
- 重复发布请求幂等；
- 发布后的旧调试历史可读。

手工：

1. 完成四节点单节点验证；
2. 完成一次成功整图调试；
3. 修改一个节点但不重新验证，确认发布按钮不可用；
4. 重新验证并完成整图调试；
5. 点击发布；
6. 检查 Project 显示“已发布”；
7. 尝试通过 Chat 修改节点，确认被拒绝；
8. 创建 Production Run；
9. 修改文件中的 Working Graph，确认 Production Run 不读取该修改；
10. 检查 Production Run 自动推进和失败暂停；
11. 刷新页面，检查发布图和运行历史仍存在。

通过标准：

- 发布前可自由调试；
- 发布时只有一个明确固化点；
- 发布后流程不可修改；
- Production Run 可复现发布图；
- 首期不依赖版本列表。

### 21.4 回归影响

- Development 历史保持只读；
- 已发布 Project 的 Chat 只能查询和辅助分析；
- 新 Project 仍可从 development 开始；
- 未来版本管理可以在 Published Graph 层追加，不影响开发态。

## 22. 测试矩阵

| 范围 | 必测 |
|---|---|
| Graph 修改 | add/update/remove、边、环、刷新、并发写 |
| Project 生命周期 | development 可改、frozen 拒绝改 |
| Node validation | 四节点启动、成功、失败、取消、重试、过期 |
| Dataset | 小样、确认、全量、Kafka、Artifact |
| Train | DSL、提交、checkpoint、恢复 |
| Inference | 覆盖、模型指纹、平台任务 |
| Evaluator | 确定性、指标、报告、不重复推理 |
| Graph debug | readiness、依赖、单 active、失败暂停、快照 |
| Publish | readiness、冻结、生产快照 |
| 兼容 | 旧 StageRun、旧 ExperimentExecution、Workbench |

每个阶段至少保留以下证据：

- 自动化测试输出；
- 一次页面操作截图；
- GraphRun/NodeRun ID；
- Platform task ID；
- Artifact URI 和 SHA-256；
- 失败/恢复记录；
- 回归命令结果。

验收证据建议放在：

```text
workspace/acceptance/echomind-project-dev/
  s1-graph-edit.png
  s2-node-runs.json
  s3-dataset-artifact.json
  s4-train-checkpoint.json
  s5-predictions.json
  s6-evaluator-report.json
  s7-graph-run.jsonl
  s8-published-graph.json
  verification.md
```

不得保存 Cookie、Token、Authorization、API key 或签名 URL。

## 23. 分阶段合并策略

建议每个阶段单独提交和验证：

```text
S1：Project graph editing + frontend graph refresh
S2：Node validation shell + frontend controls
S3：Dataset real executor
S4：Train real executor
S5：Inference real executor
S6：Evaluator real executor
S7：Graph debug scheduler
S8：Publish and manual production run
```

约束：

- 每阶段新增 feature flag 或 capability；
- 默认关闭未验证的新路径；
- 不把多个真实 Executor 放在同一个大改动里；
- 每个阶段结束都保留可运行旧路径；
- 每个阶段只在上一阶段验收后开始；
- 失败时关闭当前阶段开关即可回退，不删除历史数据。

建议开关：

```text
ECHOMIND_PROJECT_GRAPH_EDITING=0
ECHOMIND_NODE_VALIDATION=0
ECHOMIND_DATASET_EXECUTOR=0
ECHOMIND_TRAIN_EXECUTOR=0
ECHOMIND_INFERENCE_EXECUTOR=0
ECHOMIND_EVALUATOR_EXECUTOR=0
ECHOMIND_GRAPH_DEBUG_RUN=0
ECHOMIND_PROJECT_PUBLISH=0
```

## 24. 首期实现顺序

下一步只做 S1，不同时做 NodeRun 和真实 Executor。

S1 的交付物：

```text
Project 保持 development
  + Working Graph 可修改
  + Manager 可通过结构化工具改图
  + 页面刷新可见
  + 不引入 GraphRevision
  + 不启动任何业务任务
```

S1 通过后再做 S2，让四个节点都能单独启动和验证。S2 通过后，按 Dataset、Train、
Inference、Evaluator 顺序逐阶段接入真实执行。

这条路径的关键门禁是：

1. S1 确认“开发态真的是可变的”；
2. S2 确认“每个阶段都能单独启动和观察”；
3. S3-S6 确认“每个真实业务节点都能独立验证”；
4. S7 确认“当前图可以整体调试”；
5. S8 确认“发布才固化，生产只读发布快照”。
