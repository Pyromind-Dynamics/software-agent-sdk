# EchoMind Runtime 分阶段整合实施方案

> 方案日期：2026-11-11
>
> 当前主线：`feat/echomind_v0`
>
> Runtime 整合分支：`codex/echomind-runtime-consolidation`
>
> 目标态语义参考：`docs/echomind-project-publish-runtime-p1.md`
>
> 本文定义如何从当前主线出发，逐步吸收后台分支中的发布、Graph Runtime、
> Executor 和常驻资源能力。实施顺序、阶段边界和验收标准以本文为准。

## 0. 当前实施状态

| 阶段 | 状态 | 当前验证 |
|---|---|---|
| R0 | 已完成 | 51 个研发态回归测试通过；6742 只读加载且文件哈希不变 |
| R1 | 已完成 | Runtime 契约和 Store 测试通过；6742 读取不迁移 |
| R2 | 已完成 | 6742 preflight 稳定返回阻塞；文件哈希不变 |
| R2.5 | 已完成，待人工验收 | 6742 派生合同草案可审阅；Dataset 双输出；原 Project 只读 |
| R4 | 已完成，待人工验收 | 不可变 Release、active release、publish/reopen 已接通 |
| R3 | 已完成离线版，待人工验收 | ContractProbe 完整跑通 Dataset -> Train -> Evaluator |
| R5 | 已完成，待人工验收 | 6742 真实产物 replay 串通三个节点，原 Project 只读 |
| R6 | 已完成实现，待真实平台验收 | Dataset、Train、Evaluator Runtime 定向回归 23 项通过；三节点最小验证 Project 已生成，真实平台任务链待执行 |
| R7 | 进行中 | RuntimeSlot、Tracer/Serving health、采集模拟和 Graph poller 已通过；前端接入待完成 |

## 1. 背景与结论

后台分支已经实现了 Project 发布、PublishedGraph、GraphRun/StageRun/TaskRun、
Runtime Scheduler、RuntimeSlot、Tracer Acquisition 和 Dataset Runtime Executor。
但该分支是在 `55e4b628` 上继续开发的，当前主线此后已经发生了较大的执行模型变化。

当前主线的事实基线：

- Manager 通过 Pi native subagent 委派 Dataset、Train、Eval 等角色。
- `workflowPLAN.json` 是研发协作文件，不是 Runtime 状态机。
- 平台任务通过当前工具链提交，任务事实记录在 product/external task 快照中。
- 旧的 `ExperimentContext -> TaskRun -> work queue` 执行链已经被删除或弱化。
- 当前 Project/Graph 仍以研发态可编辑结构为主，没有真正的发布态调度器。

后台分支的事实基线：

- 已经有独立的发布快照和 Runtime 控制面。
- Scheduler 消费发布图，不依赖 Manager 决定普通节点顺序。
- Dataset Executor 当前以 Tracer batch 作为固定输入。
- P1 的部分实现仍引用旧主线的 StageRun、ProjectNode 和测试假设。

因此不能整分支 merge。正确策略是：

```text
当前主线保持研发态能力不变
        +
在新 worktree 中单独移植发布态 Runtime 控制面
        +
每个阶段独立提交、独立回归、独立验收
```

本文不恢复旧 ExperimentContext、SQL execution、work queue 或 delegated execution
链路。发布态 Runtime 是当前主线旁边新增的一层，不替换现有研发态 Agent 流程。

## 2. 不可破坏的边界

### 2.1 研发态

研发态继续由 Manager 和角色子 Agent 驱动。

- WorkingGraph 可编辑。
- 用户可以随时调整节点、输入输出、实现和约束。
- 不要求每个中间状态都合法，也不展示发布版本。
- 节点研发运行属于 `agent_development` 或 `node_validation`。
- Runtime Scheduler 不接管研发态对话。

### 2.2 发布态

发布是显式动作。发布后执行图和实现方法不可变。

- Runtime 只消费 PublishedGraph。
- Runtime 不要求 Manager 参与普通节点推进。
- GraphRun 表示一轮执行，StageRun 表示节点执行，TaskRun 表示一次外部任务。
- 每轮变化的输入和产物写入 GraphRun、StageRun 和 Artifact，不回写 WorkingGraph。
- 异常时 GraphRun 可以暂停，由 Manager 解释和协助恢复。
- 如果需要修改流程，先把 Project reopen 到研发态，再重新发布新的 Release。

### 2.3 固化内容

发布固化的是执行方法，不是每次运行的数据路径：

- 固化：拓扑、端口、绑定关系、Executor、实现版本、策略和必要的运行配置。
- 每轮提供：数据路径、批次、基础模型、评测集和其他 Run inputs。
- 每轮产生：Dataset artifact、模型 artifact、评测报告和 promotion decision。

流程运行所需的凭据可以属于固化资产的一部分。本期允许发布内容携带运行所需
凭据，不把凭据提取或脱敏作为阶段准入条件。

### 2.4 身份链

后续统一采用下面的身份关系：

```text
Project
  └── Release (only after publish)
        └── GraphRun (one execution round)
              └── StageRun (one graph node)
                    └── TaskRun (one external task attempt)
```

- `project_id` 是唯一业务身份。
- Release 是内部审计身份，不作为研发阶段的用户版本号。
- GraphRun 必须绑定 `release_id` 和 `release_digest`。
- StageRun 必须绑定 `graph_run_id` 和 node ID。
- TaskRun 必须绑定 `stage_run_id`、provider、幂等键和请求摘要。

## 3. 当前 Golden Case

### 3.1 样例身份

- 对话：`375c0c97aa9b46a1a879effed4f0e904`
- Project：`6742edad2fd1419e9e6c979847463fe2`
- Project 目录：
  `workspace/projects/6742edad-2fd1-419e-9e6c-979847463fe2`

### 3.2 实际 Graph

该样例是 3 节点 Graph：

```text
dataset -> train -> eval
```

当前形态：

- Project lifecycle 为 `development`。
- `active_release_id` 为空。
- Dataset 使用 `launch_mode=agent`。
- 三个节点均使用 `executor=contract_probe`。
- 节点没有 typed ports、implementation binding 或 implementation digest。
- 输入输出依赖 `config.tasks` 中的 task/resource 引用。

### 3.3 已产生的真实产物

Dataset：

- 平台 task：`10229`
- 输入路径：
  `/workspace/datasets/hql_test/pcb_avi_recheck_0823_0828_test5`
- 关键产物：
  `cleaned_train_normalized.jsonl`
  `val_normalized.jsonl`

Train：

- 平台 task：`10237`
- 基模：`/workspace/models/Qwen/Qwen3-VL-2B-Instruct`
- 输出：`/workspace/models/pcb_avi_sft_test5`

Evaluator：

- 平台 task：`10246`
- platform run：`d59bc231-8773-477c-8f7c-bd986a380585`
- 关键产物：评测报告和逐样本结果

### 3.4 样例定位

该 Project 是研发态 golden case 和真实产物 replay case，不是现成的可发布图。

- 不能原地修改为 frozen。
- 不能为了让样例通过发布校验而降低发布标准。
- 第一阶段只验证它仍能被当前主线加载，且读取不产生迁移写盘。
- 后续通过显式 Runtime Contract 和 Release 副本验证发布链路。
- 四节点 Graph 不能由该样例证明。该样例只证明 3 节点闭环。

## 4. 目标对象与端口合同

### 4.1 ProjectNode 扩展

在保留现有字段兼容性的前提下，逐步增加：

```text
ProjectNode
  id
  type
  title
  launch_mode
  executor
  task_type
  inputs: dict[str, ProjectNodeInput]
  outputs: dict[str, ProjectNodeOutput]
  implementation_ref
  implementation_digest
  policies
  config
```

`ProjectNodeInput`：

```text
artifact_type
schema_ref
required
binding
```

`binding` 支持：

- `run_input`：绑定 GraphRun inputs。
- `stage_output`：绑定上游 StageRun 输出端口。
- `runtime_slot`：绑定 Tracer、Serving Inference 等常驻资源。
- `constant`：绑定不可变常量。

旧 WorkingGraph 可以没有这些字段。缺少字段不代表数据损坏，只代表不能发布。

### 4.2 PublishedGraph

PublishedGraph 是不可变发布快照，至少包括：

```text
schema_version
release_id
project_id
source_graph_id
source_graph_digest
execution_profile
runtime_requirements
promotion_policy
nodes
edges
content_sha256
published_at
published_by
```

第一期 Execution Profile 建议保持：

- `echomind.flywheel/v1`
- 恰好一个 Dataset、一个 Train、一个 Evaluator。
- 拓扑固定为 `Dataset -> Train -> Evaluator`。

Inference 在当前 Profile 中是 RuntimeSlot 资源或 Evaluator 内部能力，不是业务节点。
如果后续需要独立 Inference 节点，新增 Profile，不在 v1 中混合两种语义。

### 4.3 GraphRun

```text
GraphRun
  run_id
  project_id
  release_id
  release_digest
  round_no
  request_id
  trigger
  inputs
  status
  outcome
  current_stage_ids
  promotion_result
  failure
  timestamps
```

状态建议：

```text
pending
running
waiting_external
paused
succeeded
failed
canceled
```

同一 Project 同时最多有一个 active GraphRun。重复 request ID 和重复 batch key 必须返回
同一个 GraphRun。

### 4.4 StageRun

现有 StageRun 继续服务研发态节点验证，同时增加发布态字段：

```text
graph_run_id
mode: node_validation | agent_development | published_run
attempt
executor_kind
task_type
implementation_ref
implementation_digest
input_refs
output_refs
task_runs
```

发布态 StageRun 必须有 `graph_run_id`；研发态 StageRun 保持可为空。

### 4.5 TaskRun

```text
TaskRun
  task_run_id
  stage_run_id
  provider
  provider_run_id
  idempotency_key
  request_digest
  status
  timestamps
  error
```

外部提交顺序必须固定：

```text
persist submitting TaskRun
    -> submit provider
    -> persist provider_run_id
    -> wait callback or poll
    -> persist terminal result
```

进程在 submit 后崩溃时，下一次恢复通过 idempotency key 查询 provider，不能直接重复提交。

### 4.6 Runtime Executor

统一 Executor 端口：

```text
prepare(context) -> completed | waiting_external | failed
submit(context, task_run) -> submitted | failed
poll(context, task_run) -> running | completed | failed
cancel(context, task_run) -> canceled | failed
```

强制规则：

- Executor 不修改 PublishedGraph。
- Executor 只能读取 StageRun inputs 和 Project 资源槽。
- Executor 返回 ArtifactRef，不把大对象直接塞进 Run JSON。
- completed 必须返回节点声明的全部 required outputs。
- 失败必须给出 failure category 和可诊断错误。

### 4.7 Dataset Source Provider

后台分支当前的 Dataset Runtime Executor 强制使用 `tracer.batch`。这个假设不能进入
通用 Runtime 合同。

Dataset 输入抽象为：

```text
DatasetSourceProvider
  dataset_source
  prepare_input
  claim
  release
  artifact_mapping
```

第一期至少支持：

- `storage_directory`：读取已有 Storage 目录，对应 6742。
- `tracer_batch`：后台 P1 的 Tracer 输入，放在后续阶段接入。

后续可扩展已有 dataset artifact、外部 URI 和人工上传。Tracer claim/release 是 Provider
能力，不是 Dataset Executor 的固定前置条件。

## 5. 分阶段实施

### R0：隔离基线与回归门禁

目标：

- 建立独立开发树，避免影响当前主线的其他对话和未提交改动。
- 锁定当前研发态行为和 6742 加载行为。

实施：

1. 从当前 HEAD `1f9d93c3` 创建 worktree：

   ```text
   /Users/zhangpeng/PycharmProjects/software-agent-sdk-runtime-consolidation
   branch: codex/echomind-runtime-consolidation
   ```

2. 不修改主工作区的 `start_inference.sh`、untracked `.pr/` 和 Runtime 文档。
3. 建立当前回归命令集合。
4. 增加一个只读 golden-case loader 测试，从 fixture 或只读路径加载 6742 的
   `project.json` 和 `graph.json`。
5. 记录当前研发态事件、任务快照和 Graph 内容，不启动服务。

基础回归：

```bash
uv run pytest \
  tests/agent_server/test_echomind_router.py \
  tests/agent_server/test_echomind_stage_runs.py \
  tests/agent_server/test_pyromind_subagent.py \
  tests/pyromind_runtime/test_preparation.py \
  tests/pyromind_runtime/test_native_preparation.py \
  tests/pyromind_runtime/test_subagent_operations.py
```

验收：

- 当前测试基线通过。
- 6742 可加载。
- 6742 仍为 development，三个节点和原 lifecycle 不变。
- 加载过程不写 project.json 或 graph.json。
- 不启动、不停止当前主线服务。

禁止：

- 不在此阶段移植 Runtime 代码。
- 不对 golden case 执行读写迁移。

### R1：契约、Artifact 和 Run Store

目标：

- 先让发布态数据模型可以独立存在。
- 不改变节点执行行为，不启动 Scheduler。

移植范围：

- 从 `d1fc8af3` 选择性吸收：
  - `ArtifactRef`
  - `RuntimeFailure`
  - `GraphRun`
  - `TaskRun`
  - GraphRun/TaskRun Store
  - StageRun 发布态字段
  - canonical ID 和原子写入 helper
- 从 `ab6ba1c4` 只吸收 PublishedGraph 的纯模型和 digest 生成。
- 新增 ProjectNode typed ports 和 implementation binding 字段。

暂不移植：

- RuntimeSlot
- AcquisitionSimulation
- RuntimeGraphScheduler
- poller 和 bootstrap wiring
- 真实 Executor
- 发布 API 和 reopen API

兼容策略：

- 所有新字段给默认值。
- 旧 Project/Graph 直接解析，不做读时迁移。
- 新字段只在显式发布编译时使用。
- 不修改现有 `graph.json` schema_version。
- StageRun 旧记录缺少新字段时正常加载。

验收：

- 6742 加载后内容与原始 JSON 等价。
- 旧 Project round-trip 不增加新字段之外的业务变化。
- GraphRun 状态不变量测试通过。
- TaskRun provider ID、幂等键和终态校验通过。
- canonical project ID 在带/不带连字符时指向同一目录。
- 当前研发态测试全绿。

阶段提交：

```text
feat(echomind): add runtime contracts and run stores
```

### R2：发布编译器与只读 Readiness

目标：

- 把 WorkingGraph 是否能发布变成可解释的检查。
- 先只读 preflight，不执行发布。

实施：

- 从 `ab6ba1c4` 吸收 PublicationIssue 和校验规则。
- 实现 `compile_working_graph(project) -> PublishedGraphDraft`。
- 编译器负责：
  - requirement 完整性。
  - 节点 launch mode。
  - typed ports 和 binding 解析。
  - implementation ref/digest。
  - profile 拓扑。
  - Promotion policy。
  - Evaluator candidate inference policy。
- 对新旧 Graph 做显式区分：
  - 旧 Graph 可显示“缺少 Runtime Contract”。
  - 不自动猜测实现。
  - 不通过降低校验来兼容旧样例。

API：

```text
GET  /api/echomind/projects/{project_id}/publication/preflight
POST /api/echomind/projects/{project_id}/publication/compile
```

`compile` 只返回候选 Release 和 digest，不写 active release，不改变 lifecycle。

6742 预期问题：

- Dataset 仍是 agent launch mode。
- 三个节点都缺少 typed ports。
- 三个节点都缺少 implementation binding。
- Evaluator 缺少 candidate inference policy。

验收：

- 6742 preflight 返回稳定、完整的问题列表。
- preflight/compile 前后磁盘内容不变。
- 合成合法图返回 ready，并生成稳定 digest。
- 相同输入重复 compile，digest 完全一致。
- 不允许 preflight 隐式补 implementation 或自动改成 executor。

阶段提交：

```text
feat(echomind): add publication compiler and preflight
```

### R2.5：研发配置到 Runtime Contract 的显式升级

目标：

- 将研发态 `graph.config.tasks/resources` 转换为可审阅的 typed Runtime Contract。
- 不在读取或 preflight 时原地修改 Project。
- 用户显式 apply 后生成派生 Project，原 Project 保持只读。

实施：

- 新增 `RuntimeContractUpgradeDraft`。
- 从 task inputs 推导端口 binding：
  - resource 输入转换为 `run_input`。
  - task output 输入转换为 `node_output`。
- 从 task outputs 推导节点输出：
  - Dataset：`train_messages`、`val_messages`。
  - Train：`trained_model`。
  - Evaluator：`eval_report`。
- 为当前已注册的 `contract_probe` 生成稳定 implementation ref/digest。
- 对 Evaluator 缺失的候选推理和 promotion policy 给出显式 assumption，用户
  apply 时一并固化。
- 派生 Project 使用确定性 ID，重复 apply 幂等。

API：

```text
GET  /api/echomind/projects/{project_id}/runtime-contract/draft
POST /api/echomind/projects/{project_id}/runtime-contract/apply
```

验收：

- 6742 可生成 ready draft。
- Dataset 草案包含 `train_messages` 和 `val_messages` 两个输出。
- 草案包含 executor launch、promotion policy 和 candidate inference policy
  的假设说明。
- apply 后原 6742 文件哈希不变。
- 派生 Project 的 preflight 为零阻塞。

阶段提交：

```text
feat(echomind): derive runtime contracts from development graphs
```

### R3：离线 GraphRun Scheduler

目标：

- 在完全不接真实平台任务的情况下验证 Runtime 状态机。

实施：

- 从 `ca232e45` 移植：
  - RuntimeNodeExecutor 协议。
  - RuntimeExecutorRegistry。
  - RuntimeGraphScheduler。
  - Project lease。
  - start/advance/callback/pause/resume/cancel/recover。
  - resolve_stage_inputs。
- 使用 `ContractProbeRuntimeExecutor` 和 fake slow executor。
- 不在应用启动时自动开启生产 poller。
- 测试 app 内可以显式启动 poller。
- 暂不接 `submit_workflow`、evaluation 或 df pipeline。
- Scheduler 只消费 PublishedGraph。

API：

```text
POST /api/echomind/projects/{project_id}/graph-runs
GET  /api/echomind/projects/{project_id}/graph-runs
GET  /api/echomind/projects/{project_id}/graph-runs/{run_id}
POST /api/echomind/projects/{project_id}/graph-runs/{run_id}/advance
POST /api/echomind/projects/{project_id}/graph-runs/{run_id}/callback
POST /api/echomind/projects/{project_id}/graph-runs/{run_id}/pause
POST /api/echomind/projects/{project_id}/graph-runs/{run_id}/resume
POST /api/echomind/projects/{project_id}/graph-runs/{run_id}/cancel
```

核心测试：

- Dataset -> Train -> Evaluator 串行推进。
- 一个 Stage 完成后才解析下游输入。
- 重复 advance 不重复创建 TaskRun。
- 重复 callback 幂等。
- 服务重启后可以 recover 到正确状态。
- 同一 Project 第二个 active GraphRun 被拒绝。
- lease 超时不会并发推进同一 Project。
- cancel 将 active Stage 和 Task 取消。
- fake executor waiting 状态立即返回 HTTP。

验收：

- 所有状态转换有显式 Pydantic 不变量。
- 研发态 Agent 流程测试不回归。
- 不产生任何平台业务任务。
- 不修改 6742 原 Graph。

当前实现：

- 已支持 start/advance/callback/pause/resume/cancel/recover 的离线状态机。
- 第一版不要求 RuntimeSlot，也不启动生产 poller。
- `ContractProbeRuntimeExecutor` 会按节点声明的全部输出端口生成 Artifact。
- 6742 的临时派生副本已经完整跑通 Dataset -> Train -> Evaluator。

阶段提交：

```text
feat(echomind): add offline graph runtime scheduler
```

### R4：发布、冻结与 Reopen

目标：

- 打通 WorkingGraph 到不可变 Release 的正式链路。

实施：

- 从 `ab6ba1c4` 吸收 PublicationStore 和 ProjectPublicationService。
- Release 文件写入后不可覆盖。
- active release 指针只指向已存在且 digest 匹配的 Release。
- 发布成功后：
  - Project lifecycle 改为 frozen。
  - Graph lifecycle 改为 frozen。
  - `active_release_id` 指向新 Release。
- reopen：
  - 只允许 frozen Project。
  - active GraphRun 存在时拒绝。
  - Project/Graph 回到 development。
  - 历史 Release 保留只读。
- 本期不提供用户可见版本列表。

API：

```text
POST /api/echomind/projects/{project_id}/publish
POST /api/echomind/projects/{project_id}/reopen
```

验收：

- 非法图无法发布，一次返回全部问题。
- 合法测试图发布后 Release digest 稳定。
- 修改 Release 文件后加载失败。
- frozen Project 无法修改 Graph 或创建研发态节点。
- active GraphRun 存在时 reopen 返回冲突。
- reopen 后再次发布生成新 Release，旧 Release 不变。
- 6742 原 Project 仍保持 development。

当前实现：

- `PublicationStore` 写入 `published_graphs/<release_id>/graph.json`。
- 已存在 Release 不允许被不同内容覆盖。
- active release 加载时校验 release ID 和 content digest。
- `publish` 成功后 Project/Graph 同时进入 frozen，并写入 active release。
- `reopen` 要求 Project 已 frozen 且不存在 active GraphRun。
- 旧 Release 在 reopen 后保留，后续重新发布生成新 Release。

阶段提交：

```text
feat(echomind): publish and reopen immutable project releases
```

### R5：6742 真实产物 Replay

目标：

- 用已经完成的真实平台产物验证 Runtime 数据流，不重新提交昂贵任务。

实施：

1. 为 6742 建立显式 Runtime Contract fixture：
   - Dataset 输入改为 Storage directory provider。
   - Dataset 输出 `dataset.train` 和 `dataset.validation`。
   - Train 输入 Dataset outputs 和 base model。
   - Train 输出 model/checkpoint。
   - Evaluator 输入 model、validation dataset 和 baseline/policy。
2. 用真实 artifact URI 映射 Run inputs。
3. 使用 artifact replay executor：
   - 不调用平台。
   - 按 Stage 返回预登记的 ArtifactRef。
   - 校验 required port、schema_ref 和 digest。
4. 创建测试用 Release 副本，不冻结用户原始 Project。
5. 运行完整 GraphRun，并保存本地 replay fixture。

验收：

- Dataset Stage 产出真实 train/val URI。
- Train Stage 成功解析 Dataset 的两个输出。
- Train Stage 产出真实 model path。
- Evaluator Stage 解析 model 和 validation dataset。
- Evaluator 产出真实 report 和逐样本结果引用。
- GraphRun 最终为 succeeded 或业务上明确的 no-go，不把 no-go 当成执行失败。
- 整个 replay 不调用 Manager，不提交新平台任务。
- 服务重启后可以从 Run Store 恢复并查看完整轨迹。

当前实现：

- 新增 `ArtifactReplayRuntimeExecutor`，从节点 `config.replay_artifacts` 读取
  每个输出端口对应的真实 URI。
- Scheduler 将固化产物写入 StageRun Artifact，并在下游 `input_refs` 中保留
  `source_uri`。
- 6742 的 replay 链路已经实际验证：
  - Dataset 输入 test5 目录。
  - Dataset 输出 `cleaned_train_normalized.jsonl` 和 `val_normalized.jsonl`。
  - Train 同时解析两个 Dataset 输出和基模路径，输出
    `/workspace/models/pcb_avi_sft_test5`。
  - Evaluator 解析候选模型和验证集，输出 task `10246` 的结果引用。
- 整个 replay 不读取或重新提交真实平台任务，原 6742 文件哈希不变。

阶段提交：

```text
test(echomind): replay golden project runtime artifacts
```

### R5.5：当前已移植的真实执行能力

在进入 Evaluator 和 callback 前，以下能力已经在新 worktree 完成并通过回归：

- Dataset Runtime Executor：
  - 支持 `storage_directory` 和 `tracer_batch` 两类输入。
  - 输出固定为 `train_messages`、`val_messages`。
  - Storage provider 不执行 Tracer claim/release；Tracer provider 才管理批次生命周期。
- Train Runtime Executor：
  - 支持内嵌 `workflow` 或 `workflow_path`，二者必须且只能提供一个。
  - 使用显式 `workflow_bindings` 将 Stage 输入写入 Workflow JSON 路径。
  - 根据 `TaskRun.idempotency_key` 生成稳定任务名，提交前按名称查找并认领。
  - 平台提交使用 `retry_create=false`，结果不确定时依靠稳定任务名恢复。
  - 成功后只产出 `trained_model` Artifact。
- Evaluator Runtime Executor：
  - 与 Train 复用同一套 Workflow Runtime 执行机制。
  - 配置键为 `evaluator_runtime`，支持内嵌 `workflow` 或 `workflow_path`。
  - 使用 `pyromind.studio.evaluation` provider 和 `echomind_evaluator` task kind。
  - 成功后只产出 `eval_report` Artifact。
- Runtime 调度：
  - Dataset 完成后进入 Train。
  - Train 首次推进停在 `waiting_external`。
  - 服务恢复后先 poll running，再 poll succeeded，然后继续 Evaluator。
  - Train 只能提交一次，重复恢复不会创建第二个平台任务。
- Kafka callback：
  - 收到平台 task 状态后，先按 `provider_run_id` 查找发布态 TaskRun。
  - 命中后恢复对应 GraphRun；未命中才进入原 ConversationRuntime 回调。
  - 研发态外部任务和发布态 Graph 任务共用 Kafka 链路，状态机互不接管。
- RuntimeSlot 与采集：
  - RuntimeSlot、Tracer/Serving health 已接入。
  - Tracer ready batch 可以幂等创建 GraphRun。
  - 后台 Graph poller 可以恢复 active GraphRun。

Train 节点配置：

```json
{
  "workflow": {"name": "train", "nodes": []},
  "workflow_bindings": {
    "nodes[0].data.config.train_data_path": "train_dataset",
    "nodes[0].data.config.val_data_path": "validation_dataset"
  },
  "output_uri": "storage://models/{graph_run_id}/candidate",
  "output_node_id": "save_model",
  "output_parameter": "output_path"
}
```

也可以把 `workflow` 替换为 `workflow_path`，但不能同时配置。`output_uri` 是平台
未返回节点产物时的回退值；`output_node_id` 和 `output_parameter` 用于从平台
Workflow 输出节点读取真实模型 URI。

Evaluator 使用相同配置结构，但放在 `evaluator_runtime` 下，输出端口为
`eval_report`，默认输出 URI 回退输入为 `eval_report_uri`。Workflow 内部应完成
Candidate Inference、评测和报告生成；Runtime 只负责平台任务生命周期与标准端口。

凭据规则：

- 当前请求触发提交时，复用用户登录凭据，并按用户和集群加密保存平台 Access Key。
- 后台查询、取消和恢复按 Project 所有者和集群查找已保存的 Key。Storage 走 SDK 接口。
- `workflow_path` 在提交前需要请求凭据准备 Workflow。进程如果在 submit 前崩溃，
  后台恢复只有在内嵌 `workflow` 或提交已成功后才能完成；不能假设服务 Key
  可以重新解析用户工作区路径。

登录用户打开活动 GraphRun 时会自动刷新凭据。凭据失效且用户离线时，GraphRun 保留
原平台任务 ID 并等待下一次授权请求，后台取得更新后的 Key 后继续执行。

Train/Evaluator workflow 采用“固定 Executor + 每轮渲染模板”的方式：

- `workflow_templates` 渲染任意 workflow JSON 字段。
- `command_template + command_binding_path` 生成评测命令。
- `output_uri + output_node_id + output_parameter` 只改写目标输出节点。
- `artifact_uri` 表示下游实际消费的产物，可与训练输出目录不同。

这样输入路径、GraphRun ID 和模型输出目录可以每轮变化，不需要重新发布 Executor。

当前最小真实验收使用：

- 验证 Project：`05783474-87c8-5430-8aba-1928d4e70e66`
- Release：`cd40f826-562f-5bed-98a3-c2b9a15d83a3`
- Dataset manifest：
  `workspace/runtime-validation/pcb-avi-test5-v1/dataset_manifest.json`
- Train 使用平台 task `10237` 的 9 节点 workflow 快照。
- Evaluator 使用 v52 三节点评测 DSL。

验收输出全部放在：

```text
/workspace/models/echomind-runtime-validation/{graph_run_id}/
```

不得复用或覆盖源 Project 的旧训练和评测目录。

### R6：真实节点 Executor

目标：

- 将 replay 替换为当前主线的真实工具调用，但不恢复旧执行器。

实施顺序：

1. Dataset：
   - 首先实现 `StorageDatasetSourceProvider`，直接对应 6742。
   - 将后台 `echomind_dataset_runtime.py` 拆成 Tracer Provider 和 Dataset Executor。
   - claim/release 只在 Tracer Provider 中使用。
2. Train：
   - Runtime Executor 调用当前 `submit_workflow` 能力。
   - 任务提交、task ID、Kafka callback 和产物查询映射到 TaskRun。
   - 不复用旧 ExperimentContext。
3. Evaluator：
   - 复用当前 evaluation pipeline、platform task facts 和 report producer。
   - Candidate Inference 可以是内部 RuntimeSlot 操作，不新增业务节点。
4. External task recovery：
   - TaskRun 持久化 idempotency key。
   - callback 只负责唤醒。
   - poll/recover 是最终一致性路径。
5. 增加单节点真实最小验证入口。

验收：

- Dataset 单节点可以先使用 Storage provider 完成。
- Train 失败可以从 TaskRun 恢复并重试，不重复提交成功任务。
- Eval callback 能把 waiting_external 推进到 completed。
- 平台任务事实在 Project、StageRun 和 TaskRun 三处可关联。
- 一个最小真实图可以完整跑通。
- Runtime 不依赖 Manager 自动推进。

阶段提交建议拆分：

```text
feat(echomind): add storage dataset runtime provider
feat(echomind): add train runtime executor
feat(echomind): add evaluator runtime executor
feat(echomind): recover runtime tasks from callbacks
```

当前进度：

- Dataset 已完成，单节点和调度内测试通过。
- Train 已完成，单节点、恢复、稳定任务名、动态 workflow 模板、运行级输出
  路径和调度闭环测试通过。
- Evaluator 已完成，单节点、命令模板、运行级 merged model/report 路径和调度
  恢复测试通过；Workflow 内仍由评测实现负责 Candidate Inference 和 report
  producer。
- Kafka 消费侧已接入发布态 TaskRun 查找；命中后唤醒 GraphRun，未命中时保持原
  ConversationRuntime 路径。
- 已生成隔离验证 Project `05783474-87c8-5430-8aba-1928d4e70e66` 和
  production Release，尚未执行真实平台 Train/Evaluator 任务，也尚未完成
  Runtime Workbench 轨迹展示。

### R7：RuntimeSlot、Tracer、采集和前端

目标：

- 在发布与真实执行闭环稳定后，再接入常驻资源和自动采集。

实施：

- 从 `f9b6965e` 移植 RuntimeSlot、Tracer/Serving provider 和 health。
- 从 `5a405287` 移植 AcquisitionSimulation 和 ready batch dispatcher。
- 从 `1d291cf7` 保留 TracerBatch provider，作为 DatasetSourceProvider 的一种实现。
- 前端增加：
  - publication preflight。
  - publish/reopen。
  - GraphRun 时间线。
  - StageRun/TaskRun 详情。
  - waiting_external、paused、resume、cancel。
  - 异常降级回 development 的入口。
- Manager 只在 GraphRun 暂停、完成或需要人工解释时收到事件。

验收：

- 从 Project 资源创建或复用 Tracer。
- 模拟采集只通过 Tracer 写入数据。
- ready batch 幂等创建 GraphRun。
- 一个 active GraphRun 完成后才能启动下一批。
- Tracer capacity 满时进入 paused/running，不丢进度。
- 服务重启后 RuntimeSlot、simulation 和 GraphRun 都能恢复。
- 前端可以看到完整节点轨迹，而不是只看到 Manager 对话结果。

阶段提交建议拆分：

```text
feat(echomind): add runtime slots and tracer resources
feat(echomind): add tracer acquisition simulation
feat(echomind): add runtime graph execution UI
```

## 6. P1 Commit 与阶段映射

| 后台 Commit | 内容 | 整合阶段 |
|---|---|---|
| `94cee480` | P1 技术方案 | 文档审阅后合并 |
| `d1fc8af3` | 基础契约和 Store | R1，RuntimeSlot 延后到 R7 |
| `ab6ba1c4` | 发布模型、校验、发布 API | R2 吸收模型和校验，R4 吸收发布/重开 |
| `ca232e45` | GraphRun Scheduler | R3 |
| `f9b6965e` | RuntimeSlot/Tracer | R7 |
| `5a405287` | Acquisition Simulation | R7 |
| `1d291cf7` | Dataset Runtime Executor | R6，先拆 DatasetSourceProvider |

不允许直接 cherry-pick 整组提交。每个 Commit 先按阶段拆文件，再适配当前主线。

## 7. Golden Case 升级策略

6742 的正确升级方式：

1. 保持原 Project 只读。
2. 复制 Project fixture 或创建测试专用副本。
3. 显式补充 Runtime Contract。
4. 运行 preflight。
5. 生成测试 Release。
6. 用真实产物 replay。
7. 通过后再决定是否在 UI 中提供“升级研发态 Project 到可发布合同”的动作。

升级不是自动迁移：

- 不因读取或 preflight 直接修改 `config.tasks` 或 Project 文件。
- `config.tasks` 只用于生成可审阅 draft；必须显式 apply 才产生派生 Project。
- 不把 `contract_probe` 自动升级成真实业务 Executor。
- 不把 3 节点 Graph 擅自扩成 4 节点 Graph。

## 8. 测试矩阵

| 层级 | 覆盖内容 | 每个阶段要求 |
|---|---|---|
| Model | Pydantic 不变量、round-trip、digest | R1 起持续增加 |
| Store | 原子写、canonical ID、旧记录兼容 | R1/R4 |
| Compiler | readiness、缺失端口、binding、profile | R2 |
| Scheduler | 顺序推进、幂等、callback、恢复、lease | R3 |
| Golden Replay | 真实 artifact 依赖和 Run 轨迹 | R5 |
| Executor | prepare/submit/poll/cancel、外部任务关联 | R6 |
| Resource | Tracer 创建/复用、capacity、采集和恢复 | R7 |
| API/UI | 权限、cluster、状态展示、交互动作 | R4/R7 |
| Regression | 当前研发态 native subagent 流程 | 每个阶段 |

每个阶段固定执行：

```bash
uv run pytest <本阶段测试>
uv run pre-commit run --files <本阶段改动文件>
```

每次进入下一阶段之前，都重新执行当前研发态回归集合。

## 9. 当前人工验收流程

1. 对研发态 Project 请求 Runtime Contract draft。
2. 审阅 `assumptions`、Dataset 双输出和 Evaluator policy。
3. 使用 draft 的 `content_sha256` 调用 apply。
4. 对返回的派生 Project 请求 publication preflight 和 compile。
5. 发布派生 Project，确认 lifecycle 为 `frozen` 且 active release 可读取。
6. 创建 GraphRun，输入本轮的数据目录和基模。
7. 查询 GraphRun detail，确认三个 StageRun 的 `input_refs`、`output_refs`
   和 `source_uri` 符合预期。
8. 调用 reopen，确认 Project 回到 development，历史 Release 仍可读取。

6742 验收基准：

- 原 Project 哈希不变。
- 派生 Project preflight 为零阻塞。
- Dataset 两个输出分别指向 `cleaned_train_normalized.jsonl` 和
  `val_normalized.jsonl`。
- Train 同时消费训练集、验证集和基模。
- Evaluator 同时消费候选模型和验证集。
- GraphRun 以 `baseline_retained` 成功结束，不触发平台重新提交。

## 10. 分支、提交与回退

开发树：

```text
/Users/zhangpeng/PycharmProjects/software-agent-sdk-runtime-consolidation
branch: codex/echomind-runtime-consolidation
based on: 1f9d93c3
```

规则：

- 主工作区继续作为用户当前研发和验证环境。
- 整合 worktree 不启动 8000 服务，不占用用户当前进程。
- 每个 R 阶段一个独立 commit。
- 每个阶段验收通过后打本地 tag：

  ```text
  echomind-runtime-r0
  echomind-runtime-r1
  ...
  ```

- 不把多个阶段压成一个提交。
- 不直接 merge P1 分支。
- 不修改用户未提交文件。
- 不提交与当前阶段无关的格式化和重构。

## 11. 阶段门禁

每阶段进入下一阶段前必须满足：

1. 本阶段自动化测试通过。
2. 当前研发态回归通过。
3. 6742 仍能只读加载。
4. 没有隐式数据迁移。
5. 没有重新引入旧 ExperimentContext/work queue 执行链。
6. API、Store 和测试对同一身份链使用一致 ID。
7. 阶段文档和已知限制同步更新。
8. 阶段 commit 和 tag 已建立。

任何一项不满足，不进入下一阶段。

## 12. 一期完成定义

一期完成不是“代码已经合并”，而是以下闭环成立：

```text
研发态 Project 完成 3 节点流程
    -> 显式生成 Runtime Contract
    -> preflight 通过
    -> 发布 immutable Release
    -> 创建 GraphRun
    -> Runtime 按 Dataset -> Train -> Evaluator 推进
    -> 平台任务和回调正确关联
    -> 每节点产物可追溯
    -> GraphRun 成功、no-go 或暂停有明确结果
    -> Manager 可以解释结果并协助 reopen
```

其中：

- 发布前仍按当前研发态 Agent 模式工作。
- 发布后普通调度不依赖 Manager。
- 6742 真实产物能 replay 出完整 Runtime 轨迹。
- 同一真实图至少完成一次由当前工具链执行的最小闭环。
- 当前主线已有研发能力和对话流程不回归。

## 13. 后续扩展

一期稳定后再考虑：

- 独立 Inference 业务节点和 `echomind.flywheel/v2`。
- 并行 DAG、条件分支和循环节点。
- 多 Project 公平调度和配额。
- 数据库 Store 和多后端实例 lease。
- 自动异常恢复。
- 用户可见发布历史和显式回滚。
- 真实业务采集流量接入。
- 多场景 Profile，而不仅限于 PCB AVI。
