# EchoMind 阶段完成目标与验证流程

> 状态：开发验收基线
> 更新时间：2026-10-30
> 主方案：`docs/echomind-avi-single-turn-plan.md`
> Graph 与 Agent 契约：`docs/echomind-graph-agent-contract.md`
> 执行阶段基线：`docs/echomind-four-node-graph-execution-plan.md`
>
> 本文保留较早的里程碑验收设计。当前开发态采用可变 Working Graph，执行顺序以
> 新方案中的 S1-S8 和对应手工验证流程为准。

## 1. 验收原则

每个阶段都必须同时回答四个问题：

1. 完成目标：这一阶段结束后，用户或系统能多做什么？
2. 验证内容：哪些结构、行为和边界必须被检查？
3. 验证流程：按什么顺序执行验证，使用什么输入和故障场景？
4. 通过证据：留下什么可复查的结果，证明验收确实发生？

阶段采用串行门禁。当前阶段未通过，不进入下一阶段。任何临时绕过必须在方案中
明确记录，不能通过修改共用 Skill、Tool 或框架来掩盖链路差异。

统一验证顺序：

```text
静态检查
  -> 契约单元测试
  -> 正常路径集成测试
  -> 失败、幂等和恢复测试
  -> 旧链路回归
  -> 人工可观察结果检查
  -> 保存验收证据
```

## 2. M0：当前基线

### 完成目标

确认当前双 Agent、计划确认、Project/Graph、StageRun、Artifact 和前端能力可以
作为后续开发基线。

### 验证内容

- Pyromind Workbench 与 EchoMind Manager 使用独立入口和独立会话；
- 计划确认前没有 Project，确认后创建或更新同一个 Project/Graph；
- 四个 Stage 节点可以独立启动或运行；
- Executor `/run` 可以产生 Contract Probe Run 和 Artifact；
- Dataset `/start` 只投递 EchoMind 会话命令，不伪装成已创建 StageRun；
- 前端能够展示计划、Graph、Run 和 Artifact。

### 验证流程

1. 运行现有后端相关测试和前端测试。
2. 创建 Workbench 会话和 EchoMind 会话，确认二者不能互相读取或续接。
3. 在 EchoMind 中生成 AVI 计划，确认前检查 Project 不存在。
4. 确认计划，检查 Project ID、Graph ID 和节点边。
5. 确认后检查只创建 Project/Graph，不自动创建 GraphRun 或启动节点。
6. 分别调用 Inference、Evaluator、Train 的 `/run`，检查 Run 和 Artifact。
7. 调用 Dataset `/start`，确认返回 command ID，且不会伪造 StageRun。
8. 刷新前端，检查计划历史、Project、Graph 和 Run 仍可读取。

### 通过证据

- 后端和前端测试结果；
- 一次计划确认前后的 Project 查询结果；
- 四阶段 Graph JSON；
- Contract Probe Run 和 Artifact；
- Dataset `/start` 的 command ID 和会话事件。

## 3. M1：Graph 契约收敛

### 完成目标

以当前 `ProjectGraph` 为基线形成版本化 `GraphRevision` 契约，并能与旧 Graph
双向转换，同时不改变现有接口行为。

### 验证内容

- GraphRevision 包含 revision、parent revision 和完整节点、边信息；
- Graph 定义与节点运行状态分离；
- 旧 `schema_version=1` 的 `graph.json` 可以直接读取；
- 当前 ProjectGraph 可以无损转换为 GraphRevision 并投影回去；
- 重复节点、未知边、自环、环和非法 Revision 会被拒绝。

### 验证流程

1. 使用当前 AVI ProjectGraph 作为兼容 fixture。
2. 执行 `ProjectGraph -> GraphRevision -> ProjectGraph` round-trip。
3. 比较节点 ID、类型、配置、边和 lifecycle，要求无信息丢失。
4. 加载旧 `graph.json`，确认无需数据迁移。
5. 分别构造非法 Revision、重复节点、未知边和环，确认校验失败。
6. 运行现有 EchoMind、Workbench 和 Project API 回归测试。
7. 运行 `git diff --check` 和对应文件的 pre-commit。

### 通过证据

- round-trip 前后 JSON 或等价性测试；
- 旧 Graph fixture 读取测试；
- 非法图校验测试；
- 现有接口无行为变化的回归测试。

## 4. M2：GraphRevision 编译

### 完成目标

把已确认 `WorkflowPlan` 编译成 Task 级、不可变的 GraphRevision，同时保留当前
Stage Graph 作为前端兼容投影。

### 验证内容

- Task、inputs、outputs、resources 和上下游引用可以生成明确节点与端口；
- 每条 Edge 都能解析到合法 output/input；
- 同一 PlanRevision 重复编译结果确定；
- Plan 修订产生新 Revision，不覆盖历史；
- 缺失输入、类型不匹配、环和未解析 Executor 会被拒绝；
- 旧 ProjectGraph 接口仍可使用。

### 验证流程

1. 使用完整 AVI PlanRevision 作为输入。
2. 连续编译两次，比较 GraphRevision 内容和确定性的 revision fingerprint。
3. 修改计划后再次确认，检查产生新 Revision 和 parent revision。
4. 检查 revision 历史，确认旧版本仍可读取。
5. 构造缺输入、端口类型不匹配、环和未知 Executor，确认被拒绝。
6. 从同一 GraphRevision 生成 Stage 投影，检查前端所需字段不变。
7. 运行现有 Project/Plan API 回归测试。

### 通过证据

- 两份可比较的 GraphRevision；
- revision 历史和 parent 引用；
- 端口绑定与 Edge 解析结果；
- 非法 Graph 的明确错误；
- 兼容 Stage Graph 投影。

## 5. M3：Durable NodeRun

### 完成目标

把 Dataset `/start` 从“只投递会话命令”升级为可查询、可恢复的 GraphRun/NodeRun。

### 验证内容

- `/start` 在分发命令前创建 GraphRun 和 NodeRun；
- 计划确认只创建或更新 Graph 并解锁开发工具，不自动执行 Graph；
- 响应返回真实 run ID、node run ID 和 command ID；
- 失败分发不会留下永久 running；
- Runtime 或服务重启后状态仍存在；
- 重复启动命令不会产生重复 NodeRun；
- 前端优先使用 NodeRun 状态展示节点。

### 验证流程

1. 启动 Dataset 节点，记录 run ID、node run ID 和 command ID。
2. 查询 GraphRun 和 NodeRun，确认状态、输入引用和关联命令。
3. 重复发送相同 command ID，确认返回原结果而不是创建新 NodeRun。
4. 模拟命令分发失败，确认 NodeRun 进入 failed。
5. 在一半状态时重启 Runtime，确认可以从持久化状态恢复。
6. 刷新前端，确认节点状态来自 NodeRun。
7. 运行旧 StageRun 读取兼容测试。

### 通过证据

- `/start` 响应；
- GraphRun 和 NodeRun 持久化 JSON；
- 重复命令和分发失败测试；
- Runtime 重启恢复记录；
- 前端节点状态截图或测试结果。

## 6. M4：完整 Graph 的 Agent 协同闭环

### 完成目标

使用完整 `dataset -> inference -> evaluator -> train` Graph，让 EchoMind Manager、
Runtime 和节点子 Agent 按照结构化协议完成一次开发态协议闭环。

本阶段允许使用小样、fixture 或确定性开发 Executor，但必须产生真实 Artifact，
不能用 `contract_probe` 占位结果证明成功。

### 验证内容

- 所有节点都有 NodeRun，不能跳过节点；
- 每次节点启动都由 ManagerCommand 驱动并通过 Runtime 校验；
- 每个子 Agent 只收到自己的 AgentExecutionEnvelope；
- 子 Agent 返回 AgentResultEnvelope，不能直接写 Graph 或 Run；
- Runtime 校验 Artifact、Schema、完成条件和幂等键；
- GraphRun 仅在全部必要节点成功后成功；
- 失败、暂停、重试和 Runtime 恢复路径可用。

### 验证流程

1. 从已确认 AVI Plan 创建 GraphRevision，并创建开发态 GraphRun。
2. Runtime 计算首个 ready 节点并生成 ManagerDecisionContext。
3. Manager 返回 `start_node` ManagerCommand。
4. Runtime 创建 NodeRun，生成节点专属 AgentExecutionEnvelope。
5. 对应 `dataset_agent` 执行并返回 AgentResultEnvelope。
6. Runtime 校验输出 Artifact 后推进 `inference_agent`。
7. 按同样流程依次完成 `evaluator_agent` 和 `train_agent`。
8. 检查完整事件序列、四个 NodeRun 和所有输入输出 Artifact。
9. 人为让一个节点失败，确认 GraphRun 暂停，Manager 决定重试后继续。
10. 在运行中重启 Runtime，确认 GraphRun 不丢状态并继续完成。
11. 重放相同 command 和 result，确认不产生重复 NodeRun 或 Artifact。

### 通过证据

- 一份完整的 GraphRun 事件序列；
- 四个 NodeRun 记录；
- ManagerCommand 和 AgentResultEnvelope 样本；
- 节点输入输出 Artifact；
- 失败、重试和 Runtime 恢复记录；
- 证明主/子 Agent 均未直接修改状态。

## 7. M5：Dataset 真实执行器

### 完成目标

把 Dataset 开发 Executor 替换为通用 `data-processing` 和平台 DataFlow，产出可被
下游使用的 DatasetVersion。

### 验证内容

- Agent 读取真实 Storage 数据并生成匹配的 Pipeline；
- 小样、用户确认、全量和报告位于同一个 NodeRun 链路；
- `df_submit_pipeline` 的 task、run、output 和 fingerprint 可追踪；
- Kafka 终态或等价测试回调可以更新 NodeRun；
- 会话断开和 Runtime 重启后不重复提交任务；
- 通用 Skill/Tool 不出现 EchoMind 或 AVI 专属分支。

### 验证流程

1. 使用小规模 AVI 数据启动 Dataset 节点。
2. 检查 Agent 生成的 Pipeline 和真实输入探查结果。
3. 运行小样并检查样本输出、标签和图片角色。
4. 用户确认后提交全量 DataFlow 任务。
5. 记录 task ID、run ID、output 和 fingerprint。
6. 模拟 Kafka 终态并检查 NodeRun 和 Artifact。
7. 在等待任务期间重启 Runtime，确认不会重复提交。
8. 运行通用 data-processing 回归测试，确认没有 EchoMind/AVI 分支。

### 通过证据

- Pipeline 和小样结果；
- 全量任务 ID 和时间线；
- Kafka 终态记录；
- `avi.dataset_version` Artifact；
- 重启恢复和幂等测试。

## 8. M6：Inference 与 Evaluator 真实执行器

### 完成目标

把 M4 的推理和评估开发 Executor 替换为真实执行器，完成
`dataset_version -> inference.predictions -> evaluator.eval_result`。

### 验证内容

- Inference 消费固定 DatasetVersion 和模型资源；
- 预测可以通过 `sample_id` 关联回数据版本；
- Evaluator 只消费 predictions、真值和固定评分规则；
- 同一 predictions 重复评分结果完全一致；
- Inference 失败时 Evaluator 不会伪造成功；
- ATIF/ATIF-like 结果可被前端或标准查看器加载。

### 验证流程

1. 使用已通过的 DatasetVersion 启动 Inference NodeRun。
2. 检查预测数量、sample ID、模型版本和输入 fingerprint。
3. 启动 Evaluator NodeRun，生成指标和报告 Artifact。
4. 使用相同 predictions 再评分一次，比较结果 hash。
5. 人为制造推理缺失或 schema 错误，确认 Evaluator 不执行。
6. 加载 ATIF 报告，检查图片回看、GT/prediction 对照和混淆矩阵。
7. 运行真实 Graph，确认前两个节点可以串联完成。

### 通过证据

- predictions Artifact；
- 模型版本和输入 fingerprint；
- 两次相同评分结果；
- `avi.eval_result` 和 `avi.atif_report`；
- 失败门禁记录。

## 9. M7：Train 真实执行器与业务闭环

### 完成目标

把 Train 开发 Executor 替换为平台 Workflow，完成
`dataset_version -> train.train_run -> checkpoint`，形成 AVI 单轮首个真实业务闭环。

### 验证内容

- Train 只消费已通过的 DatasetVersion；
- Agent 生成并验证 Workflow，但 Runtime 负责任务状态；
- workflow revision、task ID 和 correlation 可追踪；
- Kafka 终态可更新 NodeRun；
- checkpoint 完整后才生成 Artifact；
- Runtime 重启和重复请求不会重复提交训练。

### 验证流程

1. 从 M6 通过的 GraphRun 进入 Train 节点。
2. 检查 Workflow DSL 和平台校验结果。
3. 提交训练任务并记录 workflow revision、task ID 和 correlation。
4. 模拟 Kafka 成功终态，检查 checkpoint Artifact。
5. 模拟失败终态，检查结构化错误和 Manager 决策。
6. 重放相同请求并重启 Runtime，确认不会重复创建训练任务。
7. 从头执行完整 Graph，确认四个节点均使用真实 Executor 且 GraphRun 成功。

### 通过证据

- Workflow 和校验结果；
- 训练 task ID 和 Kafka 事件；
- checkpoint Artifact；
- 重复提交防护记录；
- 一次全真实 GraphRun 的完整事件和产物。

## 10. M8：生产暂停、开发分支与发布

### 完成目标

把已验证 Graph 冻结并发布到生产执行引擎；发生异常时可以暂停、Fork 开发分支、
修复并发布新 Revision。

### 验证内容

- Deployment 固定引用不可变 GraphRevision；
- 生产 GraphRun 不依赖 Chat 在线；
- 异常时停止新调度并生成 Incident；
- DevelopmentBranch 继承原 Revision、Run、输入和 Artifact 引用；
- 修复产生新 GraphRevision 和新 Deployment；
- 旧 GraphRevision、Run 和失败现场不被修改。

### 验证流程

1. 发布已通过的 GraphRevision，创建 Deployment。
2. 使用手工触发创建生产 GraphRun，确认不依赖 EchoMind Chat。
3. 人为制造节点失败，确认 GraphRun 暂停并生成 Incident。
4. 从失败现场 Fork DevelopmentBranch，检查父引用和继承资源。
5. 在开发分支修复并完成验证，创建新 GraphRevision。
6. 发布新 Deployment 并创建新 Production Run。
7. 对比旧 Run 和新 Run，确认历史现场完全保留。
8. 验证未发布的新 Revision 不能替换当前 Deployment。

### 通过证据

- Deployment 和 Production Run；
- Incident 和暂停时间线；
- DevelopmentBranch 父引用；
- 新旧 GraphRevision 和 Run；
- 旧现场不可变和新版本发布记录。

## 11. 两个总门禁

### 协议闭环门禁：M4

证明 EchoMind Manager、Runtime、子 Agent、GraphRevision、NodeRun 和 Artifact
已经就完整 Graph 执行协议达成一致。

### 真实业务闭环门禁：M7

证明同一套 Graph 和 Agent 协议可以驱动真实 Dataset、Inference、Evaluator 和
Train 完成 AVI 单轮业务闭环，而不需要为真实 Executor 修改协议。

只有 M4 和 M7 都通过，才进入 M8 的生产发布能力。
