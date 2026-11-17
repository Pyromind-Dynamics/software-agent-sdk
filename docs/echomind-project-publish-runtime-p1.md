# EchoMind Project 发布与执行一期方案

> 方案版本：2026-11-09
>
> 本文定义 EchoMind Project 从研发态进入发布态，并由 Runtime 执行有界多轮训练飞轮的闭环。它同时作为前后端实现、联调、测试和阶段验收的共同依据。

## 1. 目标

一期需要跑通下面的完整用户链路：

```text
用户描述目标
    ↓
Manager 引导创建 Project 和 WorkingGraph
    ↓
用户完成 Dataset、Train、Evaluator 的研发与验证
    ↓
Manager 把验证结果整理成节点执行合同
    ↓
用户采用验证通过的执行方案，工作台引导 Manager 准备常驻资源
    ↓
用户确认资源方案，Runtime 搭建或绑定 Serving Inference A 与 Tracer，并检查健康
    ↓
用户发布已绑定健康资源的执行方案
    ↓
用户启动模拟采集
    ↓
Acquisition Simulation 按真实请求路径写入 Tracer，形成 ready batch
    ↓
Batch Dispatcher 自动创建 GraphRun
    ↓
Dataset → Train → Evaluator
    ↓
Runtime 根据评测结果切换 Serving Inference 或保留当前模型
    ↓
Manager 收到本轮结果并向用户解释
```

一期的最终产物不是一段“大脚本”，而是一套可重复执行的发布资产和 Runtime：

- 发布后的 Graph 不可变。
- 每轮数据、模型和任务产物可以变化，但执行方法不变。
- 正常执行由 Runtime 确定性推进，不要求 Manager 每轮重新规划。
- 平台异步任务不会阻塞 HTTP 请求，也不会因服务重启丢失。
- Manager 保留在生产链路中，负责观察、解释异常和协助恢复。

## 2. 范围

### 2.1 一期包含

- 工作图和发布快照的数据模型。
- 发布校验和发布目录。
- Tracer、Serving Inference A、baseline 的长期资源槽。
- Runtime 创建、启动和健康检查 Tracer EchoMind Instance。
- 通过模拟采集流量生成真实 Tracer span/batch 的闭环验证链路。
- Tracer ready batch 自动触发 GraphRun。
- 有界多轮运行：ready batch 排队，当前 GraphRun 完成后自动执行下一批。
- 单轮 GraphRun 和节点级 StageRun。
- 平台异步任务等待、轮询、失败暂停和重启恢复。
- Dataset、Train、Evaluator 三类业务 Executor，以及 Evaluator 内部的 Candidate Inference Provider。
- 自动 Cutover 和 Candidate 清理。
- 前端发布、资源激活、自动多轮、进度和产物展示。
- GraphRun 完成或暂停后通知 Manager。

### 2.2 一期不包含

- 真实业务流量接入和无限期生产 daemon。
- 多 Project 并发调度和跨 Project 配额。
- 通用并行 DAG、条件分支和循环。
- 用户可见的多版本管理和版本回滚。
- 复杂的补偿事务、自动修改发布图和自动修复业务产物。
- 自动退研发态、自动重新规划和自动重新发布。
- Tracer 的自动自愈和异常资源回收。
- Serving Inference 的自动创建、自愈及资源回收。
- 将平台任务执行器扩展成任意脚本市场。

一期仍使用文件持久化，并假设同一个 Project 只由一个后端进程调度。需要多实例调度时再引入数据库租约或独立队列。

## 3. 核心原则

1. **发布冻结方法，不冻结每轮输入。** Executor 实现、节点参数契约和拓扑在发布后不可变；数据路径、模型路径和平台任务 ID 属于每轮 Run 的输入或产物。
2. **GraphRun 就是一轮。** 不额外引入 RoundRun。一个 GraphRun 通过 `round_no` 表示这是第几轮。
3. **StageRun 是节点级主 Run。** 一个业务节点的一次执行对应一个 StageRun；平台重试是 StageRun 下的 TaskRun。
4. **长期资源与每轮节点分离。** Tracer 和 Serving Inference A 是项目资源，不是每轮都重建的节点；Candidate Inference B 由 Evaluator 在轮内临时创建，不形成独立业务节点。
5. **正常执行不依赖 Agent。** Runtime 按发布图和规则推进；Agent 不参与普通节点选择。
6. **异步任务必须可恢复。** 外部任务提交前后都要持久化状态，回调或轮询只负责唤醒推进。
7. **评测不通过不是执行失败。** Candidate 未超过 baseline 时，Evaluator 和 GraphRun 仍然成功，只是 `promotion_decision=no_go`。
8. **密钥只保存引用。** RuntimeSlot、StageRun、Artifact 和日志不得保存明文 access key。
9. **Batch 自动触发幂等。** 一个 `project_id + tracer_resource_id + batch_id` 最多创建一个 GraphRun。

## 4. Demo 映射

Demo 的状态可以映射为下面的模型：

| Demo | EchoMind Runtime |
|---|---|
| `RuntimeState` | `RuntimeSlot` |
| `RoundState` | `GraphRun` |
| Studio Train Task ID | Train StageRun 下的 TaskRun |
| Studio Eval Task ID | Evaluator StageRun 下的 TaskRun |
| Candidate Inference B | Evaluator StageRun 内部创建的临时资源 |
| `cutover` | GraphRun Finalizer |
| `release_batches` | GraphRun Finalizer |
| Agent feeder | Acquisition Simulation |
| `daemon` | 一期由 Batch Dispatcher 驱动有界多轮；无限期生产 daemon 留到二期 |

Demo 中并不存在一个随时变化的全局 `step` 字段，只是按顺序执行。EchoMind 一期使用发布图和显式状态机，但执行语义保持一致：

```text
每个 Round 消费一批 Tracer 数据
    ↓
训练 Candidate
    ↓
在 Candidate 上执行固定评测
    ↓
与当前 Serving Inference 的 baseline 比较
    ↓
通过则切换服务；不通过则保留当前服务
```

## 5. 对象模型

### 5.1 Project

Project 是唯一产品身份，沿用现有 `project_id`。Graph、Run、Artifact 和 RuntimeSlot 都归属于 Project。

```text
Project
  lifecycle: development | frozen
  active_release_id: str | null
```

生命周期：

| 当前状态 | 动作 | 目标状态 | 条件 |
|---|---|---|---|
| `development` | publish | `frozen` | 发布校验全部通过 |
| `frozen` | reopen | `development` | 当前没有 running/waiting/paused 的 GraphRun |
| `frozen` | create GraphRun | `frozen` | RuntimeSlot ready，且没有其他 active GraphRun |

`reopen` 保留历史发布快照和 Run，但只允许使用最新发布快照重新生成 WorkingGraph。前端一期不展示版本列表。

### 5.2 WorkingGraph

开发阶段的 WorkingGraph 允许持续修改，不作为正式执行依据：

- 节点和边可调整。
- 节点状态、验证记录和 Executor 绑定可更新。
- 研发期间的 StageRun 继续使用 `agent_development` 或 `node_validation`。
- Agent 对话和子 Agent 委派仍可发生。

发布前，WorkingGraph 中每个节点必须已经具备可执行绑定。发布 API 不调用模型重新生成实现，只做校验和冻结。

### 5.3 PublishedGraph

PublishedGraph 是发布时生成的不可变快照：

```text
PublishedGraph
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

`release_id` 是内部审计身份，不作为用户需要维护的版本号。旧发布快照保留给历史 GraphRun 使用，但一期不提供版本切换 UI。

### 5.4 RuntimeSlot

RuntimeSlot 保存跨轮复用的服务和 baseline：

```text
RuntimeSlot
  schema_version
  project_id
  status: uninitialized | initializing | ready | degraded | failed
  tracer
    provider
    resource_id
    endpoint
    admin_secret_ref
    proxy_secret_ref
    target_inference_resource_id
    buffer_config
    health
  serving_inference
    provider
    resource_id
    endpoint
    secret_ref
    model_uri
    health
  baseline
    model_uri
    dataset_uri
    metrics
    evaluated_at
  active_release_id
  active_graph_run_id
  updated_at
```

一期 Provider 能力：

- Tracer 支持 `attach_existing` 和 `create_if_missing`，包括创建 EchoMind Instance、绑定 Serving Inference、等待 endpoint、写入 secret ref 和健康检查。
- Serving Inference A 支持 `attach_existing`。
- Candidate Inference B 支持创建、等待和删除，由 Evaluator 内部调用。

一期不自动修复已经损坏的 Tracer，也不自动回收项目级资源。自动自愈和完整资源回收在二期实现。

本地和 CI 可以使用已有 Tracer 做快速测试；最终 Demo 验收必须从项目资源配置出发创建一次 Tracer，并完成健康检查。

### 5.5 AcquisitionSimulation

AcquisitionSimulation 是一期用于验证真实采集链路的受控生产者。它不是业务节点，也不直接生成训练数据：

```text
AcquisitionSimulation
  simulation_id
  project_id
  request_id
  status: pending | feeding | waiting_batch | graph_started | succeeded | paused | failed | canceled
  source
    train_uri
    val_uri
    image_root
  requested_samples
  completed_samples
  failed_samples
  requested_rounds
  completed_rounds
  tracer_resource_id
  batch_ids
  graph_run_ids
  error
  created_at
  finished_at
```

它只通过 Tracer 的公开数据路径产生数据：

```text
读取模拟业务样本
    ↓
POST /v1/chat/completions
    ↓
POST /end_task
    ↓
Tracer 写入 span 并形成 batch
```

模拟样本可以来自用户 Storage，也可以来自测试数据集。样本中的 GT 只用于计算 reward，不能绕过 Tracer 直接写入 Dataset。

### 5.6 GraphRun

GraphRun 表示一轮已发布 Graph 的执行：

```text
GraphRun
  schema_version
  run_id
  project_id
  release_id
  release_digest
  round_no
  request_id
  trigger: manual | tracer_batch
  source_batch_id
  status
  outcome
  inputs
  current_stage_ids
  promotion_result
  failure
  created_at
  started_at
  finished_at
```

状态：

| 状态 | 含义 |
|---|---|
| `pending` | 已创建，尚未开始调度 |
| `running` | 本地正在执行 Executor 或推进节点 |
| `waiting_external` | 等待至少一个平台异步任务 |
| `paused` | 因异常暂停，等待重试、取消或人工处理 |
| `succeeded` | 全部节点成功，Finalizer 已完成 |
| `failed` | 明确终止，无法继续 |
| `canceled` | 用户取消 |

结果：

| outcome | 含义 |
|---|---|
| `promoted` | Candidate 达标并成为新的 Serving Inference |
| `baseline_retained` | 评测完成但 Candidate 未达标，保留当前服务 |
| `canceled` | 本轮被用户取消 |
| `failed` | 本轮因执行异常终止 |

### 5.7 StageRun

复用现有 StageRun 作为节点级主 Run，并扩展到发布态执行：

```text
StageRun
  run_id
  graph_run_id
  project_id
  node_id
  node_type
  mode: node_validation | agent_development | published_run
  attempt
  status
  task_type
  executor_kind
  implementation_ref
  implementation_digest
  input_refs
  output_refs
  task_runs
  artifacts
  error
```

发布态状态：

```text
pending → running → waiting_external → succeeded
                    ↘ paused → running
                    ↘ failed
```

研发态继续保留 `clarifying` 和 `waiting_user`，发布态 Runtime 不产生这两个状态。

### 5.8 TaskRun

TaskRun 表示 StageRun 下的一次平台提交：

```text
TaskRun
  task_run_id
  stage_run_id
  provider
  provider_run_id
  idempotency_key
  status
  request_digest
  submitted_at
  last_polled_at
  error
```

同一个 StageRun 可以有多个 TaskRun，用于平台重试；同一 TaskRun 不允许重复创建外部任务。

### 5.9 Artifact

节点输入输出都使用引用，不把大文件和平台结果直接塞进 StageRun：

```text
ArtifactRef
  artifact_type
  uri
  content_sha256
  metadata
```

一期主要产物类型：

- `dataset.supervised`
- `model.adapter`
- `model.merged`
- `inference.endpoint`（Candidate 临时资源）
- `evaluation.report`
- `promotion.decision`

## 6. 节点合同

### 6.1 节点结构

发布态节点使用统一合同：

```json
{
  "id": "dataset",
  "type": "dataset",
  "task_type": "dataset.prepare",
  "execution_role": "dataset",
  "launch_mode": "executor",
  "inputs": {
    "source_batch": {
      "artifact_type": "tracer.batch",
      "schema_ref": "echomind://schemas/tracer-batch/v1",
      "required": true,
      "binding": {
        "kind": "run_input",
        "name": "tracer_batch",
        "node_id": null,
        "port": null,
        "value": null
      }
    }
  },
  "outputs": {
    "dataset": {
      "artifact_type": "dataset.supervised",
      "schema_ref": "echomind://schemas/supervised-dataset/v1",
      "required": true
    }
  },
  "executor": {
    "kind": "dataset.pipeline",
    "implementation_ref": "storage://.../pipeline",
    "implementation_digest": "sha256:...",
    "config": {},
    "runtime": {
      "image": null,
      "entrypoint": null
    }
  },
  "config": {},
  "policies": {}
}
```

发布后不允许修改 `executor`、输入输出端口、绑定关系、policies 和拓扑。每轮变化的只有 Run inputs 和上游节点产物。

### 6.2 一期执行 Profile

发布校验分为两层。

所有 Graph 通用校验：

- 节点 ID 唯一。
- 边只引用存在的节点和端口。
- Graph 无环。
- 每个 required 输入都有 binding。
- 每个输出端口唯一。
- 每个节点有已知 Executor。
- 发布节点必须使用 `launch_mode=executor`。
- 所有 requirement 为 `confirmed`。
- Runtime requirements 合法。

一期执行 Profile `echomind.flywheel/v1` 额外要求：

- 恰好一个 Dataset。
- 恰好一个 Train。
- 恰好一个 Evaluator。
- 拓扑固定为 Dataset → Train → Evaluator。
- Evaluator 必须声明 baseline 输入。
- Evaluator 必须声明 Candidate Inference 创建、健康检查和清理策略。

通用数据模型保留扩展空间，但一期 Scheduler 只接受这个 Profile。

旧 WorkingGraph 中如果存在 `inference` 类型的业务节点，发布前必须移除。Serving Inference 属于 RuntimeSlot，Candidate Inference 属于 Evaluator 的内部运行时操作，两者都不作为业务节点进入发布图。

### 6.3 各节点输入输出

| 节点 | 主要输入 | 主要输出 | 外部任务 |
|---|---|---|---|
| Dataset | Tracer batch 或 Run 指定 Storage URI | supervised dataset refs | 可选 |
| Train | Dataset 产物、起始模型、训练参数 | adapter 或 merged model refs | 是 |
| Evaluator | Train 模型、Serving A 信息、baseline、固定评测集 | Candidate B 内部资源、metrics、report、go/no_go | 是 |

当前验证图使用的是本地 Dataset materialization 路径：Dataset Stage 校验
manifest、解析 `train_uri`/`val_uri` 并写出 Dataset artifacts，不创建平台任务。
因此 Dataset 没有 platform task ID 不是失败，也不应与 Train/Evaluator 的
`waiting_external` 混为一谈。若后续要求 Dataset 本身在平台执行并产生产物，
需要新增独立 Dataset workflow backend 和 submit/poll 生命周期。

Dataset 若 claim 了 Tracer batch，GraphRun Finalizer 负责 release。release 必须幂等，失败时记录但不覆盖业务节点的成功状态。

### 6.4 Evaluator 内部的 Candidate Inference

Evaluator 负责下面的资源生命周期：

```text
读取 Train 产出的候选模型
    ↓
调用 Inference Provider 创建 Candidate B
    ↓
等待 Candidate B healthy
    ↓
在固定评测集上比较 Candidate B 与 Serving A
    ↓
输出 metrics、report 和 promotion_decision
```

Candidate B 的创建、等待和清理记录在 Evaluator StageRun 的 TaskRun 中，不创建单独的 Inference StageRun。GraphRun Finalizer 根据 Evaluator 的 decision 决定是否将 B 提升为新的 Serving A。

### 6.5 Runtime 模板与产物绑定

发布固化的是 Executor 实现和节点合同，不是把某一轮的数据路径、平台 task ID
或模型输出目录写死。Train 和 Evaluator 的平台 workflow 是可冻结的执行模板，
每轮由 GraphRun 输入和运行身份渲染后再提交。

支持的模板字段：

| 字段 | 作用 |
|---|---|
| `workflow` / `workflow_path` | 内嵌或从文件加载平台 workflow，二选一 |
| `workflow_bindings` | 把节点端口绑定到 workflow JSON 的指定路径 |
| `workflow_templates` | 对任意 workflow JSON 字符串字段做占位符渲染 |
| `command_template` | 渲染节点命令的便捷字段 |
| `command_binding_path` | `command_template` 最终写入的 workflow JSON 路径 |
| `output_uri` | 本轮平台任务的主输出路径 |
| `output_node_id` + `output_parameter` | 只改写指定平台节点的输出参数 |
| `artifact_uri` | 下游实际消费的产物路径，可与 `output_uri` 不同 |

可用占位符：

```text
project_id
release_id
graph_run_id
stage_run_id
node_id
output_uri
<port_name>
```

`<port_name>` 来自当前 StageRun 的输入。例如 Evaluator 可以使用
`{candidate_model}`、`{validation_dataset}` 或 `{graph_run_id}` 生成平台命令。
未知占位符、越界 JSON 路径和不存在的目标字段必须在提交平台任务前失败，
不能把半渲染的 workflow 提交出去。

`output_uri` 和 `artifact_uri` 的职责不同：

- `output_uri` 是本轮任务的执行目录或平台节点输出参数。
- `artifact_uri` 是写入下游节点的实际产物引用。
- Train 可以输出到
  `/workspace/models/.../{graph_run_id}/candidate`，同时把
  `.../candidate/checkpoint-25` 作为下游消费的模型 Artifact。
- Evaluator 可以让 merged model、命令参数和报告目录全部按
  `graph_run_id` 隔离，不能覆盖历史轮次。

输入源变化不要求修改 Executor 代码。发布快照固化 binding 的语义，每轮只替换
GraphRun inputs；路径来自上一节点时，由 StageRun 输出 Artifact 自动绑定。

## 7. 发布流程

发布是一个显式用户动作：

```text
POST /projects/{project_id}/publish
```

步骤：

1. 获取 Project 锁并确认 lifecycle 为 `development`。
2. 加载 WorkingGraph 和全部 requirement。
3. 执行通用 Graph 校验。
4. 执行 `echomind.flywheel/v1` Profile 校验。
5. 校验 Executor implementation digest，防止 WorkGraph 指向已经变化的实现。
6. 生成 PublishedGraph，计算 canonical JSON digest。
   正式发布要求 Tracer 和 Serving Inference 均已通过健康检查，且资源绑定的候选版本与本次发布一致。
7. 写入不可变 release 目录。
8. 原子更新 `active_release.json`。
9. 将 Project 和 WorkingGraph 标记为 `frozen`。
10. 返回发布摘要。

发布 API 不自动提交平台任务，也不自动创建 Tracer 或 Serving Inference。

发布失败必须一次返回全部可修复问题，前端不展示部分成功。

## 8. Runtime 调度

### 8.1 服务组成

```text
EchoMind Project API
    ├── PublicationStore
    ├── RuntimeSlotStore
    ├── AcquisitionSimulationRunner
    ├── GraphRunStore
    ├── StageExecutorRegistry
    └── GraphRunScheduler
            ├── poller
            ├── callback receiver
            └── startup recovery
```

API 请求只创建任务或读取状态，不等待训练、推理或评测完成。

### 8.2 创建 Tracer

Tracer 由 RuntimeSlot Provider 通过平台 EchoMind API 创建，配置来源包括：

```text
Project identity
  project_id
  conversation_id
  cluster

Tracer runtime config
  api_url = Serving Inference A endpoint
  access_key_ref = Serving Inference A secret
  origin_model
  trajectory_buffer_size
  time_per_round
  training_save_path
  resource config
```

创建步骤：

1. 使用稳定的项目级资源标签和 request_id 查询是否已有 Tracer。
2. 不存在时创建 EchoMind Instance。
3. 等待平台状态可服务。
4. 调用 Tracer `/status` 和 `/v1/models` 完成健康检查。
5. 将 resource_id、endpoint、admin secret ref、proxy secret ref 和配置摘要写入 RuntimeSlot。

重复执行 `ensure` 必须复用已有 Tracer，不能创建第二个同项目实例。创建成功但尚未写入 RuntimeSlot 时，依靠项目级标签查询并认领已有资源。

### 8.3 模拟采集并触发多轮

一期提供受控的 AcquisitionSimulation，用于在不等待真实业务流量的情况下验证完整采集链路。

启动请求：

```text
POST /projects/{project_id}/collection-simulations
{
  "request_id": "...",
  "source": {
    "train_uri": "storage://.../train.jsonl",
    "val_uri": "storage://.../val.jsonl",
    "image_root": "storage://.../images"
  },
  "limit": 20,
  "rounds": 2,
  "splits": ["train", "val"],
  "auto_start_graph_run": true
}
```

`auto_start_graph_run` 一期默认 `true`。只有排查 Tracer 数据质量时允许设为 `false`，正常一期验收必须经过自动触发。

执行步骤：

1. 确认 RuntimeSlot、Tracer 和 Serving Inference A 均 ready。
2. 读取并校验模拟业务样本，样本至少包含 `id`、prompt、图片引用和可计算 reward 的 GT。
3. 使用 `X-Task-ID` 调用 Tracer `/v1/chat/completions`，不能绕过 Tracer 直接调用 Inference。
4. 解析 Serving Inference 的真实响应，并与模拟样本 GT 计算 reward。
5. 使用 Tracer `/end_task` 写入 reward 和 GT。
6. 每批写入前读取 `/status`，遵守 capacity 和 backpressure。
7. 等待 Tracer 自动 seal 出 ready batch，或者使用一期模拟专用 `admin/flush(seal_batch=true)` 完成测试批次边界。
8. 记录 batch ID。
9. 当 `auto_start_graph_run=true` 时，将 ready batch 交给 Batch Dispatcher。
10. `rounds > 1` 时继续写入下一批样本，形成多个 ready batch。

模拟链路的数据路径必须与真实链路一致：

```text
模拟业务样本
    ↓ HTTP
Tracer /v1/chat/completions + /end_task
    ↓
Tracer spans / ready batch
    ↓
Dataset claim + export
    ↓
Train
    ↓
Evaluator
```

AcquisitionSimulation 不能直接创建 `dataset.supervised` Artifact，也不能直接向 Dataset 节点注入训练文件。否则该验证不能证明真实采集链路可用。

模拟请求的幂等键由 `simulation_id + source_sample_id` 组成。重试时已经完成的样本不重复写入，未完成样本继续执行。

AcquisitionSimulation 的 `succeeded` 只表示请求的采集批次均已生成并交给 Batch Dispatcher，不代表对应的 GraphRun 训练和评测成功。每轮 GraphRun 仍独立记录自己的状态和 outcome。

### 8.4 Batch-ready 自动触发

GraphRunScheduler 内的 Batch Dispatcher 负责扫描 Tracer ready batch：

```text
GET /admin/batches
    ↓
筛选 state=ready 且尚未绑定 GraphRun 的 batch
    ↓
按 batch_id 升序排队
    ↓
Project 没有 active GraphRun 时创建下一轮
```

采集和执行是解耦的。Acquisition Simulation 可以继续向 Tracer 写入下一批数据，Batch Dispatcher 在当前 GraphRun 执行期间只负责排队，不并发启动第二个 GraphRun。

触发规则：

- 自动创建 GraphRun 时设置 `trigger=tracer_batch` 和 `source_batch_id`。
- `project_id + tracer_resource_id + batch_id` 必须唯一。
- 当前存在 active GraphRun 时，ready batch 保留在队列中，不提前启动。
- 当前 GraphRun 完成、失败或取消后，Dispatcher 重新检查 queued batch。
- 相同 batch 的重复通知只返回已有 GraphRun。
- 服务重启后重新扫描 ready batch 和已有 GraphRun，不能重复创建。

一期以 poller 为核心触发机制，不依赖 Tracer `WORKFLOW_TRIGGER_SCRIPT` 才能工作。后续可以把 Tracer callback 作为低延迟唤醒信号，但最终仍要通过 Batch Dispatcher 校验和幂等落库。

这使模拟链路具备有界多轮无人值守能力：

```text
Simulation rounds=2
    ↓
batch 1 ready → GraphRun 1 → 完成
    ↓
batch 2 ready → GraphRun 2 → 完成
```

### 8.5 启动一轮

```text
POST /projects/{project_id}/graph-runs
{
  "request_id": "...",
  "inputs": {
    "tracer_batch": "..."
  }
}
```

校验：

- Project 为 `frozen`。
- active release 存在且 digest 匹配。
- RuntimeSlot 为 `ready`。
- 没有同 Project 的 active GraphRun。
- `request_id` 未重复使用。
- Run inputs 满足发布图输入契约。

成功后：

- 保存 GraphRun，初始状态为 `pending`。
- 返回 `202`。
- 调度器获得锁后推进。

### 8.6 推进算法

```text
acquire project lease
load GraphRun and PublishedGraph
if run is terminal: return

resolve all runnable StageRuns
for each runnable node in topological order:
    resolve inputs
    create or resume StageRun
    call executor.start()
    persist result
    if waiting_external: mark GraphRun waiting and stop
    if failed: mark GraphRun paused and stop
    if succeeded: publish output refs

if no node remains:
    run Finalizer
    mark GraphRun terminal

release project lease
```

一期按拓扑顺序串行执行，不并行启动多个节点。

### 8.7 Executor 合同

```python
start(context) -> StartResult
poll(context) -> PollResult
cancel(context) -> CancelResult
```

`StartResult.status`：

- `completed`：同步完成。
- `waiting_external`：外部任务已提交或需要等待。
- `failed`：启动失败。

`PollResult.status`：

- `running`
- `completed`
- `failed`

Executor 不直接修改 GraphRun。所有状态转换通过 Store 的原子写入完成。

### 8.8 外部任务与恢复

平台任务遵循“先记录意图，再提交”：

1. 生成稳定 idempotency key。
2. 写入 TaskRun，状态为 `submitting`。
3. 调用平台。
4. 写入 provider task ID，状态为 `submitted`。
5. StageRun 和 GraphRun 进入 `waiting_external`。
6. 回调或 poller 获取终态。
7. 更新 TaskRun 和 StageRun，再调用 `advance`。

服务重启后：

- 扫描非终态 GraphRun。
- 有 provider task ID 的继续轮询。
- 处于 `submitting` 的 TaskRun 使用原 idempotency key 查询平台，不能直接重新提交。
- 没有 provider task ID 且无法确认是否提交时，进入 paused，由人工确认。

### 8.9 失败与暂停

错误分类：

| 类型 | 处理 |
|---|---|
| `validation` | GraphRun paused，修正输入后重试 |
| `external` | 保存平台错误，允许 StageRun retry |
| `infrastructure` | GraphRun paused，等待恢复 |
| `policy` | GraphRun failed 或 paused，等待用户决定 |
| `canceled` | GraphRun canceled |

StageRun retry 创建新的 TaskRun，不覆盖原失败记录。

重试必须重新检查：

- active release digest 未变化。
- 上游输出 refs 仍存在且 digest 未变化。
- RuntimeSlot 仍 ready。

### 8.10 并发规则

一期约束：

- 一个 Project 同时最多一个 active GraphRun。
- `active` 包括 pending、running、waiting_external、paused。
- 多个 ready batch 按 `batch_id` 升序排队。
- GraphRun 结束后，Batch Dispatcher 自动认领下一个未执行 batch。
- StageRun 和 GraphRun 的推进使用 project lease。
- GraphRun 完成、取消或失败后释放 active 指针。
- 相同 callback 或重复 advance 必须幂等。

### 8.11 凭据与恢复边界

启动或手动推进 GraphRun 时，后端用已验证的用户登录态换取该账号的 Pyromind
Access Key，并按用户和集群加密保存。Project、Release 和 Run 数据只保留业务标识。
Kafka callback 和后台 poller 根据 Project 的所有者和集群读取凭据，查询 Studio，
并通过 SDK Storage 校验该用户的产物。服务器沿用现有加密配置恢复凭据。

登录用户查看活动 GraphRun 时，后端定期刷新保存的 Key。Key 被撤销且用户离线时，
保留当前平台任务 ID 和 `waiting_external` 状态。用户重新登录打开运行后，后台可用
更新后的 Key 继续推进。平台回调上报的状态和 Runtime 核验产物后的状态分别记录。

兼容旧版本已经因凭据缺失写成 `paused` 的 Run 时，重试且输入未变化必须复用
原 `provider_run_id`，不能重复提交平台任务。新 attempt 从原 task 继续 poll。

## 9. Finalizer

Evaluator 成功不等于模型已经上线。GraphRun Finalizer 按下面的幂等事务序列执行：

1. 读取 Evaluator 的 promotion decision。
2. 若 `go=true`：
   - 验证 Candidate B 健康。
   - 将 Serving Inference A 指向 Candidate B 的 job/endpoint/model。
   - 将 Candidate 指标写为新一轮 baseline。
   - 标记 `outcome=promoted`。
3. 若 `go=false`：
   - 保留 A。
   - 释放 Candidate B。
   - 标记 `outcome=baseline_retained`。
4. 释放本轮 Tracer batch。
5. 保存最终报告和 RuntimeSlot。
6. 标记 GraphRun 终态。

Cutover 和 cleanup 必须幂等。若服务和 RuntimeSlot 更新之间存在崩溃窗口，以 RuntimeSlot 中保存的资源身份为准重新 reconcile。

## 10. API

一期新增或扩展以下 API：

| 方法 | 路径 | 用途 |
|---|---|---|
| `POST` | `/api/echomind/projects/{id}/publish` | 发布 WorkingGraph |
| `POST` | `/api/echomind/projects/{id}/reopen` | 没有 active Run 时回到研发态 |
| `GET` | `/api/echomind/projects/{id}/publication` | 查询 active PublishedGraph |
| `GET` | `/api/echomind/projects/{id}/runtime-slot` | 查询长期资源状态 |
| `POST` | `/api/echomind/projects/{id}/runtime-slot/ensure` | 创建或绑定 Tracer，并绑定 Serving Inference A |
| `POST` | `/api/echomind/projects/{id}/runtime-slot/health` | 主动健康检查 |
| `POST` | `/api/echomind/projects/{id}/collection-simulations` | 模拟真实采集并可选自动触发多轮 |
| `GET` | `/api/echomind/projects/{id}/collection-simulations/{simulation_id}` | 查询模拟采集进度和生成的 batch |
| `POST` | `/api/echomind/projects/{id}/graph-runs` | 启动一轮 |
| `GET` | `/api/echomind/projects/{id}/graph-runs` | 查询 Run 列表 |
| `GET` | `/api/echomind/projects/{id}/graph-runs/{run_id}` | 查询 Run 和 StageRun |
| `POST` | `/api/echomind/projects/{id}/graph-runs/{run_id}/pause` | 暂停 |
| `POST` | `/api/echomind/projects/{id}/graph-runs/{run_id}/resume` | 继续 |
| `POST` | `/api/echomind/projects/{id}/graph-runs/{run_id}/retry` | 使用修正后的 inputs 重试失败阶段 |
| `POST` | `/api/echomind/projects/{id}/graph-runs/{run_id}/cancel` | 取消 |

内部 callback 和 `advance` 不要求前端调用。前端通过轮询 Run 详情刷新页面；后续可以替换为 WebSocket/SSE，但不影响 Runtime 状态机。

重试只适用于存在 failed StageRun 的 paused GraphRun。重试会创建新的
StageRun attempt，并保留失败 attempt；不会覆盖历史记录。validation 失败允许同时
提交修正后的 GraphRun inputs。

所有创建型请求携带 `request_id`。相同 `request_id` 返回原资源，不重复执行。

## 11. 存储布局

继续使用 Project 目录，避免一期引入数据库调度器：

```text
projects/<project_id>/
  project.json
  graph.json
  requirements.json
  active_release.json
  published_graphs/
    <release_id>/
      graph.json
      manifest.json
  runtime_slot.json
  collection_simulations/
    <simulation_id>.json
  graph_runs/
    <run_id>/
      graph_run.json
      stage_runs/
        <stage_run_id>.json
      task_runs/
        <task_run_id>.json
      report.json
  artifacts/
    <artifact_id>.json
  locks/
    project.lock
```

要求：

- JSON 写入使用临时文件加原子 replace。
- PublishedGraph 写入完成后不允许原地修改。
- artifact 保存内容和摘要。
- secret_ref 只保存引用，不保存明文。
- 同一 Project 的调度器状态更新必须经过 project lease。

## 12. 前端

开发态继续以聊天为主，发布后增加独立的 Runtime 视图。

发布前：

- 展示 Graph 节点、边、输入输出和 Executor 绑定。
- 展示未确认 requirement 和发布阻断项。
- 采用验证通过的方案后，引导用户在常驻资源卡片发起 Manager 资源方案，审阅并确认搭建。
- 当前方案的 Inference 和 Tracer 探活通过后开放正式发布。
- “发布”需要一次明确确认。

发布后：

- 展示 active release、Tracer、Serving Inference A、baseline 和健康状态。
- 展示“创建/检查 Tracer”“模拟采集并自动执行”和“手动开始一轮”入口。
- Acquisition Simulation 展示样本进度、Tracer batch ID 和自动触发的 GraphRun。
- 展示 ready batch 队列、当前 GraphRun 和下一轮等待状态。
- 展示 GraphRun 状态、当前节点、round number 和开始时间。
- 绑定研发态运行配置时展示 validation Project 来源，避免把来源项目和
  当前研发 Project 混为同一个身份。
- 活跃 GraphRun 在前端定时刷新，后端轮询推进的外部任务状态不需要用户重复
  点击才能显示。
- 每个 StageRun 展示：
  - 状态。
  - 平台 task ID。
  - attempt。
  - input/output refs。
  - 报告和错误。
- 等待平台时显示 `等待平台任务`，不显示为本地卡死。
- Candidate no-go 显示为“本轮完成，保留当前模型”，不显示为失败。
- 失败时提供重试、暂停、取消和重新打开研发态入口。
- Manager 对话展示整轮摘要和异常说明，不参与普通节点推进。

现有 `ProjectGraphPanel` 可以继续作为基础，但需要从“节点验证历史列表”升级为“GraphRun 时间线 + Runtime 状态面板”。

## 13. Manager 集成

发布前 Manager 可以：

- 帮助用户完成节点研发。
- 检查数据和平台产物。
- 整理 Executor 绑定。
- 解释发布校验失败。

发布后 Manager 可以：

- 接收 GraphRun 完成或 paused 事件。
- 用自然语言解释结果和平台错误。
- 建议重试、取消或回到研发态。
- 根据用户明确指令执行重试或暂停。

发布后 Manager 不可以：

- 在正常轮次中临时选择下一节点。
- 修改 PublishedGraph。
- 用未验证的新脚本替换发布 Executor。
- 根据自然语言描述伪造 task ID、指标或产物。

如果用户要改变流程，必须：

```text
reopen → 修改 WorkingGraph → 重新验证 → 再次 publish
```

## 14. 安全

- Tracer access key、Inference access key 和平台凭证使用 secret ref。
- API 响应和前端状态不得返回明文 secret。
- TaskRun 保存 request digest，不保存完整敏感请求。
- Artifact 可以保存业务结果，但不能保存凭证。
- 发布和 Run API 同时校验 owner、cluster 和 session。
- 平台 callback 必须校验 provider task ID、Project、Run 和签名或内部 API key。
- 所有外部输入在进入 Executor 前校验 artifact type 和 digest。

## 15. 实施阶段

### P1.0 契约与持久化

改动：

- 定义 PublishedGraph、RuntimeSlot、AcquisitionSimulation、GraphRun、TaskRun。
- 扩展 StageRun 状态和字段。
- 统一前后端 `ProjectNode` 与执行状态合同。
- 增加 Project 目录 store 和原子写入测试。

验收：

- 模型 round-trip 和状态不变量测试通过。
- 现有研发态 Project 创建、Dataset start 和 node run 不回归。
- 前端 contract 测试通过。

### P1.1 发布与回到研发

改动：

- 增加 publish/reopen API。
- 增加通用校验和 `echomind.flywheel/v1` Profile 校验。
- 写入不可变发布快照和 active release 指针。

验收：

- 非法拓扑、缺失绑定、Agent launch mode、未知 Executor 均被阻止。
- 合法 Project 发布后 lifecycle 为 frozen。
- 有 active GraphRun 时拒绝 reopen。
- 发布快照在文件层不可变。

### P1.2 RuntimeSlot 与 Tracer

改动：

- 增加 RuntimeSlot 模型、Store、Provider。
- Tracer `ensure` 支持 attach existing 和 create if missing。
- 增加 Tracer、Serving Inference 和 baseline 健康检查。

验收：

- 能从 Project 配置创建 Tracer EchoMind Instance 并等待服务健康。
- 重复 ensure 复用已有 Tracer，不创建第二个实例。
- 能绑定测试 Serving Inference A。
- unhealthy 或缺失资源时不能启动 GraphRun。
- secret 不出现在 API、日志和持久化快照中。
- 服务重启后仍能读取资源槽。

### P1.3 GraphRun Scheduler

改动：

- 增加 GraphRun/StageRun/TaskRun 状态机。
- 增加拓扑推进、Batch Dispatcher、project lease、poller、callback 和 startup recovery。
- 用 fake slow executor 验证等待和恢复。

验收：

- 节点完成后自动进入下一节点。
- 平台等待时 HTTP 立即返回。
- 重复 callback、重复 advance 不产生重复任务。
- 同一 Tracer batch 只创建一个 GraphRun。
- 多个 ready batch 在一个 active GraphRun 结束后按顺序自动执行。
- 服务重启后恢复到等待或继续状态。
- 同 Project 第二个 active Run 被拒绝。

### P1.4 Tracer 模拟采集

改动：

- 增加 AcquisitionSimulation 模型、Store 和 Runner。
- 通过 Tracer `/v1/chat/completions` 和 `/end_task` 写入模拟业务样本。
- 遵守 `/status` capacity 和 backpressure。
- 支持有界 `rounds`，连续形成多个 ready batch。
- 将 ready batch 交给 Batch Dispatcher 自动创建 GraphRun。

验收：

- 模拟样本经过 Tracer 后生成真实 span 和 ready batch。
- 不直接生成 Dataset Artifact，不绕过 Tracer 写训练文件。
- 重复 simulation 和样本 ID 不重复写入。
- `auto_start_graph_run=true` 时能生成 GraphRun 并带上 batch ID。
- 两批模拟数据可以触发两个顺序执行的 GraphRun，中途不需要用户再次点击。
- Tracer 不可用或 capacity 满时，模拟任务保持 running 或 paused，不丢进度。

### P1.5 三类 Executor

改动：

- Dataset Executor：处理输入、生成训练数据、管理 batch claim。
- Train Executor：按每轮输入渲染平台工作流、提交任务并等待模型产物。
- Evaluator Executor：渲染 Candidate Inference 和评测 workflow，调用
  Candidate Inference Provider 创建 B，等待服务可用，执行 A/B 评测并输出
  指标和 decision。
- Finalizer：cutover、cleanup、baseline 更新和 batch release。

验收：

- CI 使用 fake provider 覆盖成功、no-go、平台失败、重复提交和重启。
- 动态 `output_uri`、`workflow_templates` 和 `command_template` 有定向单测。
- Train 的 `output_uri` 与下游 `artifact_uri` 可以不同。
- Staging 使用真实小规模数据和平台任务完成一次三节点 Run。
- Candidate no-go 时 A 保持服务，B 被释放。
- Candidate go 时 A 切换到 B，RuntimeSlot 和 baseline 更新。

### P1.6 前端与 Manager 事件

改动：

- 挂载并升级 ProjectGraphPanel。
- 增加发布、创建/检查 Tracer、模拟采集、自动触发多轮、手动开始一轮、暂停、继续、取消和重试。
- 增加 GraphRun 时间线和 RuntimeSlot 状态。
- GraphRun 完成或暂停后向 Project 对话发送 Manager 可消费事件。

验收：

- 用户不依赖 Manager 对话即可完成发布、模拟采集和有界多轮自动执行。
- 等待平台、失败、no-go 和 promoted 四种状态展示正确。
- 服务重启后页面能恢复正确 Run 状态。
- Manager 能收到完成/暂停事件并给出基于真实产物的解释。

## 16. 验收

### 16.1 主链路：从研发 Project 发布三节点

用户主链路必须在原始研发 Project 上完成，不要求用户寻找或手工打开
Validation Project：

1. 用户在研发 Project 的 WorkingGraph 上完成 Dataset、Train、Evaluator
   开发。
2. 打开“发布与执行”。如果节点仍是 `contract_probe`，Runtime Contract 会显示
   当前为契约探测实现。
3. 点击“绑定运行配置并生成副本”。系统按
   `conversation_id + cluster + owner` 找到唯一的
   `echomind.runtime-validation/v1` 验证实现，将真实 Executor、节点配置和
   产物契约写回当前 development Project 的 WorkingGraph。绑定成功后，运行输入
   必须重新采用新 Runtime Contract 的默认值，不能保留绑定前的旧 Dataset 路径。
4. 系统重新生成 Runtime Contract，并用 production 模式检查、编译和发布。
5. 发布副本冻结后，可以创建多轮 GraphRun。后续 Run 只读取发布副本，不再修改
   WorkingGraph。

发布副本必须显示：

```text
publication_mode: production
```

如果 Runtime Contract 仍包含 `contract_probe` 或 `artifact_replay`，production
preflight 必须阻塞，不能生成“看似可执行”的发布副本。

### 16.2 当前三节点最小验收夹具

使用 `scripts/echomind_prepare_runtime_validation.py` 从 Golden Case 生成一次性
验证 Project，用于 Executor 本身的隔离测试。它是自动化测试夹具，不是用户主链路：

- 源 Project：`6742edad-2fd1-419e-9e6c-979847463fe2`
- 验证 Project：`05783474-87c8-5430-8aba-1928d4e70e66`
- Release：`cd40f826-562f-5bed-98a3-c2b9a15d83a3`
- Release digest：
  `sha256:82a1a534523366fa2e183ed676ae90e18e4732f69f3240de1305c6012d29f8b4`
- Dataset manifest：
  `workspace/runtime-validation/pcb-avi-test5-v1/dataset_manifest.json`

验证图固定为：

```text
dataset_runtime -> train_runtime -> evaluator_runtime
```

验证步骤：

1. 运行脚本 dry-run，确认合同和 production preflight 通过。
2. 加载验证 Project，确认源 Project 的 `project.json` 和 `graph.json`
   哈希未变化。
3. 在 Runtime Workbench 使用 Dataset manifest 的本地绝对路径和
   `/workspace/models/Qwen/Qwen3-VL-2B-Instruct` 创建 GraphRun。
   Dataset 输入应是 materialization manifest，而不是原始数据集目录。
   如果输入校验失败，Run 进入 paused，可修正上方输入并使用“使用上方输入重试”，
   无需取消后重新创建整轮。
4. Dataset 完成后，Train 应提交内嵌的真实平台 workflow，状态进入
   `waiting_external`，并显示 provider task ID。
5. Train 成功后，候选模型路径必须包含本次 `graph_run_id`，下游读取
   `checkpoint-25`，不能复用旧路径。
6. Evaluator 提交评测 workflow，merged model、命令参数和报告目录也必须包含
   本次 `graph_run_id`。
7. 每次终态更新后再次调用 `/advance`，最终 GraphRun 应为 `succeeded`。
8. 检查历史 Run 的输入、TaskRun、Artifact 和输出目录互不覆盖。

通过标准：

- 三个 StageRun 按 Dataset -> Train -> Evaluator 推进。
- 平台等待期间 HTTP 请求不悬挂。
- 重复 advance 或 callback 不重复提交平台任务。
- GraphRun 的每个外部任务都能通过 provider task ID 恢复。
- 源 Project、旧输出路径和历史 Release 未被覆盖。

### 16.3 一期最终验收

一期最终验收使用一个完整 Project：

1. 用户通过 Manager 创建 Project。
2. 完成三节点研发，并确认 Dataset、Train、Evaluator 的执行绑定。
3. 采用通过的执行方案，确认 Manager 提议的资源配置。Runtime 搭建或绑定 Serving Inference A 与 Tracer，健康检查通过。
4. 点击发布，系统激活与资源绑定一致的 PublishedGraph。
5. 启动 `rounds=2` 的模拟采集。
6. Acquisition Simulation 按真实 HTTP 路径向 Tracer 写入模拟业务样本。
7. Tracer 依次 seal 出两个 ready batch，Batch Dispatcher 自动创建第一个 GraphRun。
8. Dataset claim 该 batch，导出训练数据并生成 Dataset Artifact。
9. Train 提交平台任务，前端显示 task ID 和 `waiting_external`。
10. 训练完成后自动进入 Evaluator。
11. Evaluator 创建 Candidate Inference B，并在其就绪后执行 A/B 评测。
12. 页面显示 A、B 指标和 `go/no_go`。
13. go 时 A 切换为 B；no-go 时保留 A 并释放 B。
14. Tracer batch 被释放，GraphRun 进入终态。
15. 第一个 GraphRun 完成后，Batch Dispatcher 自动启动第二个 GraphRun。
16. 中途重启后端不会丢失 Run、batch 队列或模拟采集进度，也不会重复提交已确认的外部任务。
17. Manager 收到每轮摘要，并能解释执行结果。

通过标准：

- 不需要 Manager 在每轮中决定下一节点。
- 不需要人工修改已发布文件。
- Dataset 的输入必须来自 Tracer batch，不能直接来自模拟样本目录。
- 平台等待期间没有长 HTTP 请求。
- 重复请求不产生重复训练或推理任务。
- 所有结果都来自平台或真实 Artifact，不由 Agent 编造。

## 17. 测试策略

### 17.1 单元测试

- Graph 拓扑和端口校验。
- 发布快照 digest。
- RuntimeSlot 和 secret redaction。
- Tracer create/ensure 的幂等性。
- Acquisition simulation 的样本去重和 batch 绑定。
- Batch Dispatcher 的 batch 去重、排队和触发顺序。
- GraphRun/StageRun/TaskRun 状态转换。
- idempotency key 和 request_id 去重。
- promotion decision 与 outcome。

### 17.2 服务测试

- fast fake executor 路径。
- slow fake external task 路径。
- callback 重复和乱序。
- Tracer capacity/backpressure 和恢复。
- Simulation 中途重启后的样本进度恢复。
- 两个 ready batch 在前一轮结束后自动触发下一轮。
- poller 超时和退避。
- 服务重启恢复。
- project lease 冲突。

### 17.3 平台集成测试

- attach existing RuntimeSlot。
- 从零创建 Tracer，并完成 `/status` 和 `/v1/models` 健康检查。
- 模拟采集生成真实 span 和 ready batch。
- 小样 Dataset。
- 最小 Train workflow。
- Candidate Inference 创建和清理。
- Evaluator 报告 A/B 比较。

平台集成测试使用独立测试 Project，不复用用户生产 Project。

### 17.4 前端测试

- ProjectNode、PublishedGraph、GraphRun 和 StageRun 类型契约。
- 发布阻断项展示。
- waiting_external、paused、no-go 和 promoted 状态。
- 重试和取消操作。

## 18. 风险与后续

| 风险 | 一期处理 |
|---|---|
| 平台 callback 不稳定 | Batch Dispatcher 以 poller 为主，callback 为加速通道 |
| JSON store 只适合单进程 | 一期限制一个后端调度进程，二期迁移数据库租约 |
| 平台任务重复提交 | 提交前持久化 intent，使用稳定 idempotency key |
| Tracer 重复创建 | 项目级标签、稳定 request_id、创建后认领 |
| 模拟数据绕过真实链路 | Simulation 只允许写入 Tracer，禁止直接生成 Dataset Artifact |
| Candidate 泄漏占 GPU | Finalizer 和取消流程统一 cleanup |
| 输入路径变化 | 使用 Run inputs 和 Artifact 绑定，不写死 Executor 内部 |
| no-go 被误判为失败 | 使用独立 promotion decision 和 outcome |
| Agent 越权修改发布链 | PublishedGraph 不可变，Manager 只能走 reopen |

二期再考虑：

- 接入真实业务流量生产者，复用一期 Batch Dispatcher 和 GraphRun 触发链路。
- 无限期生产 daemon、运行节流和配额。
- 并行 DAG 和条件节点。
- Serving Inference 自动创建、Tracer 自愈和完整资源回收。
- 发布历史和显式版本切换。
- 多后端实例调度。
- 自动异常恢复和更完整的补偿事务。
