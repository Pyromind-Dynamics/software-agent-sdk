# EchoMind Graph 与 Executor 实现基线

> 状态：实施基线，后续代码和验收以此文档为准
> 更新时间：2026-11-05
> 当前进度：远端 Preparation、Project、TaskCorrelation 已并入；本地 Working Graph、
> StageRun、多 TaskRun 关联和四节点 Agent-led 契约已存在；下一步补齐其余
> DelegatedRole，先完成 Dataset 的研发态纵向闭环
> 范围：Project Working Graph、通用 Node、Node Executor、NodeRun、Artifact、
> Dataset 到 Evaluator 的首期单轮闭环
>
> 本文是 EchoMind 执行层的唯一实现基线。旧文档中的节点枚举、四节点拓扑、
> GraphRevision、长期资源和生产闭环方案，与本文冲突时以本文为准：
>
> - Agent 入口和 Profile 边界见 `docs/echomind-dual-agent-change-design.md`
> - 产品与规划背景见 `docs/echomind-avi-single-turn-plan.md`
> - 生产态、故障恢复和多轮闭环仍见 `docs/echomind-graph-agent-contract.md`，
>   但不作为首期实现范围

## 接缝基线（最高优先级）

远端代码和本地实现当前存在三套相邻但尚未闭合的对象：

```text
远端 Preparation.create_project
  -> SQL Project + 首个 TaskRun

本地旧 experiment-plan confirm
  -> 文件 Project + WorkingGraph

本地节点 /start
  -> StageRun
```

新 EchoMind Project 不能继续同时走这三条路径。从 S4 开始，统一为：

```text
Conversation
  -> Preparation / Checklist
  -> 用户确认 WorkflowPlan
  -> create_project
  -> SQL Project（唯一 Project ID）
  -> WorkingGraph（project_id 与 SQL Project 相同）
  -> 用户启动某个 Node
  -> StageRun（节点级主运行）
  -> TaskRun（StageRun 下的确定性执行明细）
  -> 平台任务（external_task_id）
  -> callback 回写 TaskRun / StageRun / Graph Node
  -> Artifact
```

身份和事实源固定如下：

| 对象 | 定义 | 权威来源 |
|---|---|---|
| Project ID | 唯一业务身份；跨 Graph、Run、Artifact 关联只认它 | SQL Project |
| graph_id | Working Graph 实例 ID，不是第二个 Project 身份 | 本地 graph.json |
| StageRun | 节点级主运行，表达澄清、开发、验证、等待和完成 | 首期继续使用文件存储 |
| TaskRun | StageRun 内的一次确定性执行或委派；允许 0..N 个 | `experiment_task_runs` |
| external_task_id | Pyromind 平台任务身份 | TaskRun 的提交记录 |
| artifact_id | 节点输出身份 | StageRun / Project Artifact |

开发态统一规则：

1. `graph.project_id == project.project_id`；`graph_id` 只用于图实例，不参与 Project 查找。
2. SQL Project 创建是提交点；SQL 成功后按同一个 `project_id` 幂等物化 WorkingGraph。
3. 创建 Project 只创建 Project 和 WorkingGraph，不自动创建或启动节点 StageRun。
4. 用户在 Manager 对话中明确要求开发某个节点时，Manager 调用
   `open_project_stage` 创建 StageRun；只有需要角色、Workflow 或平台任务执行时
   才创建 TaskRun。节点卡片只展示状态，不提供“开始开发/重新开发”入口。
5. 创建后的节点、边和节点配置以 WorkingGraph 为开发态事实源；Preparation plan 作为创建快照，不再与 WorkingGraph 并行编辑。
6. 所有 Graph、StageRun、TaskRun、Artifact 查询必须带 `project_id`，禁止只靠对话或 `graph_id` 反查 Project。
7. `StageRun.run_id`、`TaskRun.task_run_id` 和 `external_task_id` 必须在 API 和日志中分别命名，不再用泛化的 `run_id` 互相替代。
8. 旧 `POST /nodes/{node_id}/start` 只保留兼容和测试用途，不是 EchoMind 研发态的产品入口；四节点研发验收统一从对话发起。

当前远端 `create_project` 自动创建首个 Eval TaskRun 的行为，只保留给现有模型评测入口和旧会话兼容。新训练飞轮 Project 创建不得自动执行节点。

## 1. 实施结论

首期实现一条单轮实验链路，不实现多轮飞轮、常驻服务生命周期、事件触发器、
Graph 内部循环和生产调度器。

统一执行模型为：

```text
Project
  -> Working Graph
  -> Node
  -> NodeExecutorBinding
  -> NodeRun
  -> Artifact
```

核心原则：

1. Node 是通用业务任务，不是 Dataset、Train 等封闭枚举。
2. 四节点 `Dataset -> Train -> Inference -> Evaluator` 只是首套 Graph 模板。
3. 每个可执行 Node 在发布时必须绑定一个确定的 Executor。
4. Executor 类型可以复用，不要求每个 Node 生成一套独立实现。
5. 开发态允许 Agent 创建脚本、Workflow 和配置；发布时固化实现 Artifact。
6. 运行输入是动态绑定，代码 Artifact 不绑定某一轮固定路径。
7. Runtime 只执行已固化的 Graph 和 Executor，不依赖 Agent 决策。
8. 首期不新增 Service、Gate、Action、Trigger 等节点类型，也不为空能力预埋字段。

## 2. 与 Demo 的关系

Demo 是场景 reference，不是首期 Runtime 的完整实现目标。

Demo 的闭环包含：

```text
Agent feeder
  -> tracer 收集并封批
  -> Dataset 导出和拒采均衡
  -> Train 提交 GRPO Workflow
  -> Evaluator 合并权重、创建候选推理、运行 holdout、决定晋级
  -> Promote 切换线上推理
  -> Release batch
  -> 下一轮
```

其中长期在线的 Inference A、Tracer、路由切换和多轮循环不属于首期。

首期只实现 Demo 中“一轮核心图”的可执行协议：

```text
Dataset -> Train -> Inference -> Evaluator
```

以后扩展 Demo 闭环时，应新增 Deployment、Trigger 和 Project Resource，
不能把循环直接塞进当前 DAG，也不能让 Agent 成为生产调度器。

## 3. 非目标

首期明确不做：

- GraphRevision 和用户可见版本管理
- Graph 内部环路和循环边
- 常驻 Inference Service 和 Tracer Resource 生命周期
- 定时器、事件触发器、批满触发器和多轮自动执行
- 主 Agent 或子 Agent 参与生产态逐步调度
- 跨 Node 的并行执行和复杂条件边
- 多 Project 版本、回滚和开发分支
- 通用组件 DSL、任意 UI 模板生成
- 直接执行 Agent 工作区中的可变脚本

任何实现如果必须先引入以上能力才能完成，应停止并回到方案评审。

## 4. 基本概念

### 4.1 Project

Project 是用户确认计划后创建的工作单元。SQL Project 是唯一业务身份，也是权限、
创建事务和 TaskRun 的权威来源；本地文件只承载开发态 Working Graph 和节点运行数据。

Project 包含：

- 需求 checklist
- 当前 Working Graph
- 与 SQL Project 相同的 `project_id`
- Development NodeRun 历史
- Artifact
- 发布快照

Project 创建前不存在 Project ID、Graph，也不存在 NodeRun。旧 `confirm_experiment_plan`
创建文件型 Project 的路径不再用于新 EchoMind 会话。

### 4.2 Working Graph

开发态只有一张可修改的 Working Graph：

- 可以增加、删除和修改 Node
- 可以增加、删除和修改 Edge
- 可以暂时不完整
- 每次 NodeRun 创建内部 Graph 快照
- 修改 Graph 不生成用户可见版本

发布时才生成不可变的 Published Graph。

### 4.3 Node

Node 表示一个有明确输入、输出和执行责任的业务任务。

Node 的通用语义包括：

- `task_type`：任务语义，自由字符串，例如 `dataset.prepare`
- `execution_role`：研发态委派角色；决定该节点交给哪个子 Agent 执行
- `inputs`：输入端口和绑定
- `outputs`：输出端口和 Artifact Schema
- `executor`：如何执行
- `config`：业务参数
- `policies`：超时、重试和确认策略

Runtime 不得根据 `type == "dataset"` 或类似领域枚举决定执行逻辑。

### 4.4 NodeExecutor

NodeExecutor 是 Runtime 的确定性执行适配器。

它负责：

- 校验 NodeRun 输入
- 启动底层脚本、Workflow 或平台任务
- 返回外部任务引用
- 对账任务状态
- 校验输出 Artifact
- 返回结构化结果

NodeExecutor 不负责：

- 修改 Graph
- 决定下一节点
- 选择业务 Skill
- 生成自然语言计划
- 直接维护 NodeRun 状态机

### 4.5 NodeRun

NodeRun 表示 Node 的一次具体执行。

当前代码使用 `StageRun` 文件模型。首期不为了改名重写存储：

- 保留 `StageRun` 类型和 `/runs` API 兼容
- 内部按通用 NodeRun 语义演进
- 新增字段必须有默认值，旧 Run 仍可读取
- Runtime 和前端统一称其为 Node Run

StageRun 与现有 `experiment_task_runs` 的关系固定为：

```text
StageRun (节点级主运行)
  -> task_runs: StageTaskRun 0..N (本地关联与最近状态)
  -> SQL TaskRun 0..N (确定性执行明细)
  -> external_task_id 0..1 (该 TaskRun 对应的平台任务)
```

首期闭环通常只有一个主要 TaskRun，但协议不能假设节点只能执行一次。澄清问题、
重试、继续全量或重新提交可能产生多个执行明细。

NodeRun 至少记录：

- `run_id`
- `project_id`
- `graph_run_id`
- `node_id`
- `attempt`
- `status`
- `input_refs`
- `output_refs`
- `executor_kind`
- `implementation_ref`
- `implementation_digest`
- `external_task_id`
- `task_run_id`
- `task_runs`
- `config_digest_version`
- `error`
- 创建、开始和结束时间

TaskRun 必须能通过新增的 `stage_run_id` 反向定位 StageRun。旧 TaskRun 的该字段允许
为空。

`StageRun.task_runs` 是本地可读关联索引，不是 SQL TaskRun 的替代事实源：

- 相同 `request_id` 重试返回同一个 TaskRun；
- 不同 `request_id` 可创建小样、全量或重试的后续 TaskRun；
- 每个关联独立记录 `external_task_id`、`output_uri` 和 Artifact；
- `StageRun.task_run_id`、`external_task_id`、`output_uri` 保留为最近一次执行的
  兼容镜像；
- 纯子 Agent 执行成功允许 `running -> succeeded`，不强制存在平台任务；
- 失败 StageRun 可由新的 TaskRun 重新进入 `running`，但成功的 StageRun 仍为终端；
- `waiting_external` 仍要求真实平台任务 ID 和输出目录。

`config_digest_version=1` 严格按升级前的 `type + executor.kind + config` 计算，
用于继续校验历史 StageRun；`config_digest_version=2` 使用完整节点身份（除
`status` 外），其中包含正式 `execution_role`。旧 StageRun 继续按自身版本校验配置，
不会因为升级中断。

### 4.6 Artifact

Artifact 是 Node 之间的正式数据契约。

Artifact 至少包含：

- `artifact_id`
- `project_id`
- `node_id`
- `run_id`
- `kind`
- `schema`
- `uri`
- `content_sha256`
- `metadata`
- `created_at`

首期 Artifact 可以继续使用 Project 文件存储。Runtime 和下游 Node 只能通过
Artifact ID 或解析后的 URI 使用数据，不能依赖”某轮默认目录”。

### 4.7 开发态执行角色

`execution_role` 是正式 Node 字段，不再从 `config.execution_role` 读取。旧 Graph
在读取时继续兼容后者，并与新模板的默认映射保持一致：

| Node type | Default execution_role |
|---|---|
| dataset | dataset |
| train | train |
| inference | inference |
| evaluator | evaluator |

首期四节点在研发态统一使用 `launch_mode="agent"`。这里的 Agent-led 指由
EchoMind Manager 委派内部执行角色来完成节点工作，不增加新的 Chat 入口。研发态
启动与确认以非空 `execution_role` 为兼容判定，因此升级前仍为
`launch_mode="executor"` 的历史 Graph 也无需改写即可继续开发。
`executor` 仍是发布阶段使用的确定性执行绑定，研发态不以 `executor.kind`
决定节点是否可启动。

## 5. 通用数据模型

### 5.1 ProjectNode

目标结构：

```json
{
  "id": "dataset",
  "title": "Dataset",
  "type": "dataset",
  "task_type": "dataset.prepare",
  "launch_mode": "agent",
  "execution_role": "dataset",
  "inputs": {
    "source": {
      "artifact_type": "avi.raw_dataset",
      "required": true,
      "binding": {
        "kind": "run_input",
        "name": "raw_dataset"
      }
    }
  },
  "outputs": {
    "dataset_version": {
      "artifact_type": "avi.dataset_version",
      "schema": "avi.dataset/v1"
    }
  },
  "executor": {
    "kind": "data_pipeline",
    "implementation_ref": "project://implementations/<id>/pipeline.py",
    "implementation_digest": "sha256:<digest>",
    "config": {
      "model_profile": "none",
      "output_schema": "vision",
      "convert_format": "none"
    }
  },
  "config": {
    "capability": "data-processing"
  },
  "policies": {
    "requires_user_confirmation_before_full": true,
    "max_attempts": 1,
    "timeout_seconds": 7200
  }
}
```

说明：

- `type` 保留给 UI 展示和兼容，不参与 Runtime 分支。
- `task_type` 是通用业务语义。
- `executor` 内部统一使用结构化 `ExecutorBinding`。
- API 读取旧字符串 `executor` 时自动映射为兼容 Binding。
- 新代码禁止继续扩展 `dataset | train | inference | evaluator` 枚举。

### 5.2 InputPort

```json
{
  "artifact_type": "avi.dataset_version",
  "schema": "avi.dataset/v1",
  "required": true,
  "binding": {
    "kind": "node_output",
    "node_id": "dataset",
    "port": "dataset_version"
  }
}
```

首期支持的绑定种类：

| binding kind | 含义 |
|---|---|
| `run_input` | 本次 GraphRun 创建时解析的输入 |
| `node_output` | 上游 Node 的命名输出 |
| `literal` | 不可变标量或配置 |

首期不支持任意字符串表达式、脚本求值或可变全局指针。

### 5.3 OutputPort

```json
{
  "artifact_type": "avi.train_checkpoint",
  "schema": "avi.checkpoint/v1",
  "required": true
}
```

Node 成功前，Runtime 必须校验所有 required 输出存在且类型、Schema 和 digest
匹配。

### 5.4 Edge

目标结构：

```json
{
  "source": "dataset",
  "source_port": "dataset_version",
  "target": "train",
  "target_port": "train_dataset"
}
```

兼容规则：

- 旧 Edge 只有 `source` 和 `target`
- 若源节点只有一个输出、目标节点只有一个输入，读取时自动补齐端口
- 无法唯一补齐时拒绝启动 GraphRun

### 5.5 ExecutorBinding

目标结构：

```json
{
  "kind": "data_pipeline",
  "implementation_ref": "project://implementations/<id>/pipeline.py",
  "implementation_digest": "sha256:<digest>",
  "config": {},
  "runtime": {
    "image": null,
    "entrypoint": "pipeline.py <input_path> <output_path>"
  }
}
```

首期只实现以下 Executor 类型：

| kind | 用途 | 可发布 |
|---|---|---|
| `contract_probe` | NodeRun 和执行壳验证 | 否 |
| `agent_development` | Agent 在开发态构建和验证 Node | 否 |
| `data_pipeline` | 提交 DataFlow-compatible Python Pipeline | 是 |

Train、Inference、Evaluator 的真实 Executor 在各自阶段接入，不提前建立空实现。

## 6. Executor 的生命周期

### 6.1 Node 开发时

Node 先有业务契约，再形成 Executor Binding：

```text
Node 输入输出定义
  -> Agent 选择通用执行范式
  -> 创建脚本、Workflow 或配置
  -> 绑定 development executor
  -> 跑小样和校验
  -> 修改并重跑
```

此阶段的 Executor 可以是 `agent_development`，允许 Agent 参与。

### 6.2 Node 验证通过后

Agent 调用结构化 Tool 固化 Node 实现候选：

```text
finalize_node_implementation(
  project_id,
  node_id,
  executor_kind,
  source_path,
  entrypoint,
  input_schema,
  output_schema,
  config
)
```

该动作只允许 EchoMind 开发会话调用，不加入通用 Skill 或 Workflow Tool。

服务端执行：

1. 读取 Agent 生成的代码或 Workflow。
2. 计算 SHA-256。
3. 复制到 Project 不可变实现目录。
4. 校验入口和参数契约。
5. 写入 Node 的 draft Executor Binding。
6. 记录关联的成功 NodeRun。

实现文件不得继续引用 `public_data/` 中的可变文件。

### 6.3 发布时

发布只处理 draft Binding：

```text
检查成功 NodeRun
  -> 复制 Binding 和实现 Artifact 到 Published Graph
  -> 固化 kind、ref、digest、config、entrypoint
  -> 禁止后续修改
```

发布后即使原 `public_data` 文件被修改或删除，Runtime 仍能使用发布快照。

### 6.4 运行时

Runtime 不使用 Agent Tool，直接通过 Executor Registry：

```text
读取 Published Graph 快照
  -> 解析 Node inputs
  -> 按 executor.kind 获取 Executor
  -> 传入 NodeExecutionContext
  -> 持久化 external task ID 或结果
  -> reconcile 终态
  -> 校验并登记 Artifact
```

## 7. Executor 接口

首期接口以现有同步执行壳为起点，但语义必须支持外部异步任务：

```python
class NodeExecutor(Protocol):
    kind: str

    def validate(self, ctx: NodeExecutionContext) -> None: ...

    def start(self, ctx: NodeExecutionContext) -> ExecutorStartResult: ...

    def reconcile(
        self,
        ctx: NodeExecutionContext,
        state: ExecutorRuntimeState,
    ) -> ExecutorReconcileResult: ...

    def finalize(
        self,
        ctx: NodeExecutionContext,
        observation: ExecutorObservation,
    ) -> NodeExecutionResult: ...
```

约定：

- `start` 必须使用 `node_run_id` 作为幂等键。
- `start` 返回 `inline_succeeded`、`waiting_external` 或明确失败。
- `waiting_external` 必须返回稳定外部任务 ID。
- `reconcile` 必须可重复调用。
- 外部任务响应不确定时不得自动重提。
- `finalize` 只有在产物真实存在且校验通过后才能成功。
- Executor 不修改 NodeRun，只返回结构化结果。

开发期可以提供一个 `SingleShotNodeExecutor` 适配器，把当前同步
`ProjectNodeExecutor.run()` 包装为 `start -> inline_succeeded`，避免一次修改过宽。

## 8. 代码实现与运行输入的分离

### 8.1 固化内容

发布时冻结：

- 代码文件或 Workflow
- 代码 digest
- 入口命令
- 运行时镜像或环境标识
- 参数 Schema
- 输入输出 Artifact Schema
- 校验脚本
- 默认业务参数

### 8.2 每次运行变化

每次 GraphRun 动态解析：

- 输入 Dataset/Artifact
- 输出目录
- `project_id`
- `graph_run_id`
- `node_run_id`
- `batch_id`
- 当前模型 checkpoint
- 当前评测集
- 当前基线指标
- 平台任务 ID

代码 Artifact 接收这些值作为参数，不把某轮路径写入代码。

### 8.3 输入变化规则

| 变化 | 是否重做 Executor |
|---|---|
| 输入 URI 变化 | 否 |
| batch_id 或 run_id 变化 | 否 |
| 输出目录变化 | 否 |
| 模型 checkpoint 变化 | 否 |
| 数据 Schema 不兼容 | 是 |
| 业务规则变化 | 是 |
| 入口参数契约变化 | 是 |
| Workflow 拓扑变化 | 是 |

### 8.4 Demo 的实际做法

Demo 已经采用同一模式：

- `round_id` 每轮新生成
- `batch_id` 每轮由 tracer 选择
- `datas/avi-rounds/<round_id>/export/train.jsonl` 每轮变化
- Workflow JSON 模板固定，提交前注入当前数据路径
- Dataset 导出脚本和训练 Workflow 不因路径变化而重新生成

对应实现见：

- `/Users/zhangpeng/PycharmProjects/AVI-test-echomind-demo/miscs/AVI-test-echomind-orch/src/avi_orch/paths.py`
- `/Users/zhangpeng/PycharmProjects/AVI-test-echomind-demo/miscs/AVI-test-echomind-orch/src/avi_orch/runner.py`
- `/Users/zhangpeng/PycharmProjects/AVI-test-echomind-demo/scripts/run_studio_grpo_training.py`

EchoMind 首期直接遵守该规则：

> 固定的是执行定义，动态的是 Artifact 和本次运行参数。

## 9. 开发态流程

### 9.1 Project 创建

```text
用户描述目标
  -> EchoMind Manager 澄清 checklist
  -> 生成可审阅计划
  -> 用户明确确认
  -> create_project 事务创建 SQL Project
  -> 以同一个 Project ID 幂等物化 Working Graph
```

确认计划不自动启动 NodeRun。

SQL 与本地文件不能组成同一个事务，因此约定：

1. SQL Project 提交是唯一创建事实。
2. SQL 成功后，Project materializer 使用已冻结的 Preparation plan 创建 graph.json。
3. Graph 读取和 Project 打开都能识别“Project 已存在但 Graph 缺失”，并按原 Project ID
   补齐，不生成第二个 Project。
4. 重复执行 `create_project`、页面刷新和进程重启都必须得到同一个 Project。

### 9.2 Node 开发

```text
用户点击 Node 开发
  -> 创建 clarifying development NodeRun
  -> 当前 EchoMind 会话收到 project_id / node_id / run_id 和澄清约束
  -> Agent 读取 Graph/StageRun，提出关键问题
  -> 用户确认节点方案
  -> Agent 调用 approve_project_stage_plan
  -> NodeRun clarifying -> running
  -> Agent 使用 Skill/Tool 和远端 Sandbox
  -> 生成或修改 Node 实现
  -> 运行小样
  -> 记录证据
  -> waiting_user 或 failed
```

### 9.3 小样确认

小样必须由真实工具执行并通过正式校验器。

Agent 自己编写的校验脚本只能作为辅助证据，不能替代平台或 Skill 的正式
validator。

### 9.4 继续全量

用户确认后，在同一个 NodeRun 上创建 continuation：

```text
waiting_user
  -> 用户确认
  -> Agent 提交真实平台任务
  -> waiting_external
  -> Kafka/查询终态
  -> 输出校验
  -> succeeded 或 failed
```

重复确认和重复回调必须幂等。

### 9.5 固化实现

全量成功后，Agent 将实现固化为 Node 的 draft Executor Binding。

没有 draft Binding 的 Node 可以在开发态运行，但不能进入发布快照。

## 10. 发布与单轮 Runtime

### 10.1 发布门禁

发布前必须满足：

- Graph 是 DAG
- 所有 required 输入有来源
- 所有 Edge 端口可解析
- 所有 Node 有可发布 Executor Binding
- 每个 Node 有与当前 Binding digest 匹配的成功 NodeRun
- 至少有一次单轮整图调试成功
- 没有等待中的 NodeRun
- 不使用 `contract_probe` 或 `agent_development`

### 10.2 Published Graph

发布生成不可变快照：

- 节点和边
- 端口和 Artifact Schema
- Executor Binding
- 实现 Artifact 和 digest
- Node 参数和策略
- Graph digest

Working Graph 不再修改，Published Graph 不允许原地编辑。

### 10.3 GraphRun

首期只支持手动创建一次单轮 GraphRun：

```text
读取 Published Graph
  -> 固化本次 Graph 快照
  -> 解析 Run Inputs
  -> 计算 ready Node
  -> 串行执行
  -> 写 Artifact
  -> 推进下游
  -> 成功或失败
```

首期同一 GraphRun 最多一个 active NodeRun。

Node 状态至少包括：

```text
pending | ready | running | waiting_external | succeeded | failed | skipped
```

失败默认暂停 GraphRun，不自动重试，不自动修改 Graph。

## 11. 首期四节点职责

### 11.1 Dataset

- 开发态：Agent 使用 `data-processing` 开发 Pipeline
- 小样：远端 Sandbox 中真实执行和正式校验
- 全量：平台 Data Pipeline 任务
- 输出：`dataset_version`，具体 `schema_ref` 由场景 reference 和 Node 配置决定
- 发布 Executor：`data_pipeline`

Dataset 不是固定 AVI 格式转换器，具体 Schema 由场景 reference 和 Node 配置决定。

### 11.2 Train

- 输入：DatasetVersion、模型配置、训练参数
- 执行：平台训练 Workflow
- 输出：`avi.train_checkpoint`
- 发布 Executor：后续接入的平台 Workflow Executor

Train 必须使用发布时固化的 Workflow 模板和参数 Schema，运行时只注入当前数据、
模型和输出路径。

### 11.3 Inference

- 输入：DatasetVersion、checkpoint
- 执行：创建一次运行范围的推理任务
- 输出：`avi.prediction_set`
- 发布 Executor：后续接入的平台 Inference Executor

首期 Inference 不由 Project 长期持有。运行结束或失败后按 Executor 策略清理。

### 11.4 Evaluator

- 输入：DatasetVersion 真值、PredictionSet、评分脚本或 rubric
- 执行：平台 CPU Evaluator 任务
- 输出：`avi.metrics`、`avi.eval_result`
- 发布 Executor：后续接入的平台 Workflow Executor

首期 Evaluator 只产生指标和报告，不做线上切流。

## 12. S3 实施拆分

S1、S2 已完成。S3.0、S3.1、S3.2 已通过开发和页面验收；
S3.3 的不可变实现固化尚未实现。

### 12.1 S3.0：通用 Node 和 Executor 契约

目标：

- 引入自由字符串 `task_type`
- 引入 InputPort、OutputPort、ExecutorBinding
- 保留旧 Node 和旧 Run 兼容读取
- Runtime 和 API 不再按 `type == "dataset"` 分支
- `type` 只用于展示
- 保留 `contract_probe` 作为开发壳

验收：

- 旧 `graph.json` 可读取
- 旧字符串 `executor` 可迁移
- 新旧 Node 均可通过 Graph API 读取
- 修改 Node 的 `task_type`、inputs、outputs、executor 可持久化
- 删除或修改一个 Node 只让相关 Edge 和验证状态失效
- `rg` 检查确认执行层不存在按 Dataset/Train 名称分支

### 12.2 S3.1：Dataset 小样

目标：

- Dataset Node 声明 `task_type=dataset.prepare`
- EchoMind Manager 完成澄清、确认和交棒，不亲自处理数据
- Dataset 子 Agent 读取 `data-processing` Skill，并从 routing table 选择范式
- 场景 reference 只提供建议；AVI 规则不能被固化成通用 Graph 字段
- Dataset 子 Agent 在远端 Sandbox 创建参数化实现，使用真实数据运行小样
- Manager 根据子 Agent 的真实回报记录 `stage_sample_report` 和 `waiting_user`

必须满足：

- 小样和全量使用同一份参数化实现，输入和输出由 Graph 注入
- 适用平台的场景必须使用平台或 Skill 提供的正式校验能力；Agent 自写 validator
  只能作为辅助证据
- 通用服务端只校验已配置的 sample contract，不内置 AVI 图片、标签或 Schema 规则
- 节点配置显式启用 `sample_validation` 时才执行对应校验；默认 Graph 不启用

验收：

- StageRun 状态为 `waiting_user`
- Artifact 类型默认为 `stage_sample_report`，内容来自子 Agent 的真实小样
- AVI reference 如选择正式 validator，可自行报告 formal validation evidence，
  但该字段不是通用 Dataset schema
- 页面刷新后仍可读取样例结果
- 后端重启不留下永久 `running`

Dataset 的实现固化边界：

- 固化的是参数化实现 Artifact，不是本轮固定输入或输出路径；
- 实现可以声明 `command` 或 `platform_pipeline` 执行方式；
- Runtime 在运行时把本次 Graph 输入绑定注入实现；
- 数据路径、round ID、输出目录变化不要求重新生成实现；
- Schema、业务规则或入口参数契约变化时才需要新实现版本。

本轮实机结果：

- Project：`e15d9437cd45420989a1100df0d02f37`
- Graph：`eebb4d66-ff1b-4ce9-bc41-2fa07ba47ad5`
- Dataset StageRun：`e61d677d-6a73-4bb8-b197-66db9de06a0a`
- 小样 3 条，正式 validator 返回 `passed / vision / rows=3`
- `processed.jsonl` SHA-256：
  `b9e75ccd92f98d18e6dd32b1e4b73a2a27fe970e415aaa9c389c5adc1ba956de`
- 页面显示 Dataset 为“验证通过”，Train 仍为“未验证”

### 12.3 S3.2：Dataset 平台全量

目标：

- 用户明确确认后创建 continuation
- Manager 创建新的 Dataset TaskRun，并交棒给 Dataset 子 Agent
- Dataset 子 Agent 提交一次真实平台 Data Pipeline
- 记录 external task ID 和 output URI
- 平台终态后校验真实产物
- 生成 DatasetVersion Artifact

必须满足：

- 小样通过前不能提交全量
- Manager 不直接调用 `df_submit_pipeline`，也不重新实现子 Agent 的数据处理
- 同一 request ID 不重复提交
- 响应不确定不自动重试
- 只有平台终态成功和产物校验通过才进入 succeeded
- 失败保存真实错误和失败阶段

验收：

- 有真实 platform task ID
- 有 output URI
- `processed.jsonl`、validation report 和 schema 校验可读取
- DatasetVersion 可被 Train Node 绑定
- 后端重启不重复提交

本轮实机结果：

- Platform task：`9703`
- Platform run：`6e0cf250-7ddd-488f-9c9d-0ab638c25c74`
- Sandbox：`sb-efd2b1c80934`
- 输出目录：
  `/.pyromind-agent/ff60dda4c70e4000a0d0ebf93ac18173/data_preparation/6e0cf250-7ddd-488f-9c9d-0ab638c25c74`
- `report.json` 顶层 `status=succeeded`
- 输出 3 条，LLM 调用 0，failures 0
- 正式校验通过：`vision / rows=3`
- DatasetVersion Artifact：
  `28db0c3b-f8fc-4a4b-a3a5-71afe23ab1ba`
- 平台产物 SHA-256 与已确认小样一致

### 12.4 S3.3：Dataset 实现固化

目标：

- 把验证通过的 Pipeline 复制为不可变实现 Artifact
- 写入 draft `data_pipeline` Executor Binding
- 记录实现 digest 和成功 NodeRun
- 验证修改原 `public_data` 文件不影响固化管理

验收：

- Binding 指向 Project 不可变路径
- 文件 digest 与 Binding 一致
- 删除或修改原 `public_data/pipeline.py` 后，已固化实现仍可读取
- 未验证通过的 Node 不能生成 draft Binding
- `contract_probe` 和 `agent_development` 不能进入发布快照

### 12.5 本轮运行链路修复

Dataset 验收过程中发现并修复了两个远端 Sandbox 问题：

1. Access key 客户端未复用。
   - `SandboxExecutionManager` 按 conversation 缓存已认证客户端。
   - 缓存键只保存 env、cluster、user ID 和 auth token 的 SHA-256，
     不保存原始 token。
   - pause 后恢复复用客户端；凭据变化、refresh、delete 会清理缓存。
2. Terminal launch 后 observe socket 静默。
   - 远端命令已成功执行并写出 `out.log`、`rc`、`pid`，
     但同一 launch socket 不再返回 watch 输出。
   - observing 阶段增加 5 秒启动探测；探测超时后关闭旧 socket，
     在新 socket 上只重连 `watch.sh`，不重新 launch 命令。

页面实机验证使用最小命令 `terminal true`，未读取或修改数据，未启动任何 Node：

- Execution：`5ad61d4715b54bef8f6f8cd58cd77b4d`
- `exitCode=0`
- `phase=finished`
- `started=true`
- `stopped=true`
- `round_trips=3`
- `watch=648ms`
- 不再出现 `phase=observing` / `execution_timeout`

## 13. 后续阶段

### 13.1 S4：身份与执行链接缝

阶段目标：不增加新业务能力，先保证新 EchoMind Project 只有一套身份和一条可追踪执行链。

实现内容：

- 新会话只通过 Preparation + `create_project` 创建 Project；
- 使用 ProjectView 将 SQL Project 幂等物化为同 ID WorkingGraph；
- 新会话不再调用旧 `confirm_experiment_plan` 创建第二份文件 Project；
- StageRun 增加可选 `task_run_id`；
- `experiment_task_runs` 增加可选 `stage_run_id`；
- TaskCorrelation 增加可选 `project_id`、`stage_run_id` 和 `task_run_id`；
- 新增 EchoMind 专属 `request_project_stage_task`，由它在创建 TaskRun 后绑定
  StageRun；通用 `request_experiment_task` 不承担 EchoMind 生命周期；
- 平台执行更新事件携带 `project_id`、`stage_run_id` 和 `task_run_id`，支持从
  external task 定位到 NodeRun 和 Working Graph Node；
- 保持旧会话、旧文件 Project 和 Workbench 只读兼容；
- 编写 Project、Graph、StageRun、TaskRun 四方关联测试。

验收内容：

- 准备阶段没有 Project ID；
- 用户确认后只产生一个 Project ID；
- `GET Project`、`GET Graph`、StageRun 和 TaskRun 能互相定位；
- 相同 `create_project` command 重复执行返回同一 Project；
- 启动 Dataset 节点只创建一个 StageRun，不创建第二个 Project；
- 平台 callback 能沿 external task -> TaskRun -> StageRun -> Node 回写；
- 重启和重复请求不产生重复 Project、StageRun 或平台任务。

### 13.2 S5：Train

输出：

- `platform_workflow` Executor
- Workflow 模板固化
- 数据、模型和输出参数动态绑定
- checkpoint Artifact 校验

S5 不改变 Node/Executor 协议。

S5 开始前的必要输入：

- 明确基座模型
- 明确训练数据 Artifact 绑定
- 明确训练 Workflow 模板和输出 checkpoint 约定

当前基座模型尚未提供，因此不能创建真实 Train 平台任务。可以继续做
Train Node 的契约、Workflow 模板和参数绑定壳，但不能用占位模型发起全量训练。

### 13.3 S6：Inference

输出：

- 运行范围 Inference 任务
- PredictionSet Artifact
- 失败清理和外部任务幂等

S6 不引入长期服务资源。

### 13.4 S7：Evaluator

输出：

- 确定性评分或平台 Evaluator Workflow
- Metrics 和 EvaluationResult Artifact
- 指标展示

S7 不执行线上切流。

### 13.5 S8：单轮整图调试

输出：

- GraphRun
- ready 计算
- 串行 NodeRun
- Artifact 传递
- 失败传播和暂停
- 运行快照

### 13.6 S9：发布与单轮运行

输出：

- Published Graph
- Executor Binding 冻结
- 手动 GraphRun
- Published Graph 只读
- 运行状态、Artifact 和错误可追溯

## 14. API 实施

### 14.1 首期保留

```text
GET   /api/echomind/projects/{project_id}
GET   /api/echomind/projects/{project_id}/graph
PATCH /api/echomind/projects/{project_id}/graph
POST  /api/echomind/projects/{project_id}/nodes/{node_id}/start
POST  /api/echomind/projects/{project_id}/runs/{run_id}/confirm
POST  /api/echomind/projects/{project_id}/nodes/{node_id}/run
GET   /api/echomind/projects/{project_id}/runs
GET   /api/echomind/projects/{project_id}/artifacts/{artifact_id}
```

### 14.2 S3.0 增加

图补丁操作增加：

```text
update_node:
  task_type
  inputs
  outputs
  executor
  policies
```

旧字段 `type`、`launch_mode` 和字符串 `executor` 继续接受。

### 14.3 S3.3 增加

```text
POST /api/echomind/projects/{project_id}/nodes/{node_id}/implementation
GET  /api/echomind/projects/{project_id}/nodes/{node_id}/implementation
```

该接口只允许 EchoMind 会话使用，不注册为通用 Tool。

### 14.4 S8/S9 增加

```text
POST /api/echomind/projects/{project_id}/graph-runs
GET  /api/echomind/projects/{project_id}/graph-runs/{graph_run_id}
POST /api/echomind/projects/{project_id}/publish
GET  /api/echomind/projects/{project_id}/published-graph
```

## 15. 文件影响

### 15.1 Runtime Domain

`pyromind-runtime/pyromind_runtime/domain/project_graphs.py`

- 通用 Node 字段
- InputPort、OutputPort
- ExecutorBinding
- Edge port
- 旧格式兼容解析

### 15.2 Agent Server

`openhands-agent-server/openhands/agent_server/echomind_stage_executors.py`

- Executor Registry
- Executor 协议和结果模型
- `contract_probe` 兼容适配

`openhands-agent-server/openhands/agent_server/echomind_stage_runs.py`

- NodeRun 兼容字段
- 输入解析和输出登记
- Executor digest
- GraphRun 关联

`openhands-agent-server/openhands/agent_server/echomind_projects.py`

- 不可变实现目录
- 发布快照
- Working Graph 和 Published Graph 分离

`openhands-agent-server/openhands/agent_server/echomind_graph_templates.py`

- 从已确认计划的 stages、tasks 和 resources 生成节点
- 不包含 AVI Schema、脚本或固定验收字段
- 输出通用 Node、Port 和 ExecutorBinding
- Dataset 在节点级 `execution_role=dataset`，`config` 只保留
  `capability=data-processing`

### 15.3 Product Server

`pyromind-agent-server/pyromind_agent_server/api/echomind_projects.py`

- 通用阶段上下文
- 场景规则通过 Skill/reference 获取
- 删除 AVI 专用 `<think>`、图片和 Schema 硬编码
- 只有澄清、交棒、确认和 StageRun continuation，不在 Manager 中执行数据任务

`pyromind-agent-server/pyromind_agent_server/echomind_stage_tools.py`

- 增加实现固化 Tool
- 保持 EchoMind 专属边界

`openhands-agent-server/openhands/agent_server/agent_profiles/echomind_dataset.py`

- Dataset 子角色只加载通用 `data-processing`
- 包含数据处理、平台提交和结果登记工具
- 不包含 Manager 的 Graph/Project 管理工具

### 15.4 Frontend

`/Users/zhangpeng/PycharmProjects/minimal-chat-frontend/src/echomindContract.ts`

- 通用 Node、Port、ExecutorBinding、GraphRun 类型
- 兼容旧字段

`/Users/zhangpeng/PycharmProjects/minimal-chat-frontend/src/components/ProjectGraphPanel.tsx`

- 不再根据 Node type 决定启动行为
- 节点卡片只读展示 Executor、输入输出、Artifact 和 NodeRun
- 节点研发启动只由 Manager 对话触发，节点卡片不提供开始或重新开发按钮

## 16. 防跑偏规则

以下规则视为代码评审硬门禁。

1. 通用 Runtime 不出现 `if node.type == "dataset"`。
2. 通用 Tool、Skill 和 Submission 层不出现 EchoMind、Project、AVI 分支。
3. Execution 层不出现 `public_data/` 作为发布后路径。
4. Executor 不得读取 Agent 可变工作区作为生产实现。
5. Agent 生成代码只能通过固化流程进入 Runtime。
6. NodeRun 成功前必须校验 required Artifact。
7. 小样通过不能只依赖 Agent 自写 validator。
8. 未确认小样不能提交平台全量。
9. Runtime 重放不能重复创建外部任务。
10. Graph 修改不能让已开始的 GraphRun 混用新旧配置。
11. 首期不通过添加 Trigger、Service、Gate 节点解决局部问题。
12. 场景差异只能进入 Skill reference、Node 配置或实现 Artifact。

## 17. 验证矩阵

| 阶段 | 自动化 | 页面验证 | 通过证据 |
|---|---|---|---|
| S3.0 | Graph/Run 兼容测试 | Graph 显示通用字段 | 旧图可读、无领域分支 |
| S3.1 | Executor/Tool 定向测试 | Dataset Node 小样 | waiting_user + sample report |
| S3.2 | 提交幂等测试 | 确认后提交 | task ID + output URI + DatasetVersion |
| S3.3 | digest/不可变测试 | 查看实现信息 | draft Binding + digest |
| S4 | Project/Graph/StageRun/TaskRun 关联测试 | 创建 Project 后打开 Graph | 唯一 Project ID + 唯一关联链 |
| S5 | Workflow 参数绑定测试 | Train NodeRun | checkpoint Artifact |
| S6 | Inference 任务测试 | Inference NodeRun | prediction Artifact |
| S7 | Evaluator 确定性测试 | Evaluator NodeRun | metrics/report |
| S8 | GraphRun 状态机测试 | 整图调试 | 四 Artifact 串联 |
| S9 | 发布快照测试 | Published Graph | 冻结图 + 单轮运行 |

每个阶段完成后：

```sh
uv run pytest -q tests/pyromind_runtime tests/agent_server
uv run pre-commit run --files <changed-files>
git diff --check
```

涉及前端时：

```sh
cd /Users/zhangpeng/PycharmProjects/minimal-chat-frontend
npm test
npm run build
npm run lint
```

所有阶段都必须通过页面实际操作验证，不能只依赖单测或 mock。

## 18. 当前状态

| 项目 | 状态 |
|---|---|
| 双 Agent 独立入口 | 已实现 |
| SQL Project、Preparation 与 TaskCorrelation | 远端已合并 |
| Project 创建与 Working Graph | S1 已有文件实现，待与 SQL Project 收敛 |
| 任意节点验证壳 | S2 已实现 |
| 通用 Node/ExecutorBinding | S3.0 已实现 |
| Dataset 小样正式校验 | S3.1 开发链路已存在，合并后的真实证据待复验 |
| Dataset 平台全量 | S3.2 代码和自动化已存在，合并后的真实平台闭环待复验 |
| Dataset 实现固化 | S3.3 待实现 |
| StageRun / TaskRun 关联 | D0 已实现多 TaskRun 本地关联与旧数据兼容 |
| 四节点 Agent-led 契约 | D1 已实现正式 execution_role、统一 launch_mode |
| Train/Inference/Evaluator | 未实现真实 Executor |
| GraphRun | 未实现 |
| Published Graph | 未实现 |

Dataset 当前结论：

- 开发态数据 Pipeline、小样、正式校验和平台全量均已跑通。
- 页面状态已验证，Dataset 可进入下一节点开发。
- 当前 Graph 中 Dataset 的 Executor 仍是 `contract_probe`，
  尚未固化为不可变 `data_pipeline` 实现 Artifact。
- `contract_probe` 只用于开发态验证，不能作为发布 Executor。

## 19. 下一步

下一步进入 D2：补齐 Train、Inference、Evaluator 的内部 DelegatedRole，
保持 Manager、TaskRun、StageRun 和 Artifact 协议不变。

执行顺序：

1. 为 Train 注册独立角色、提示词、Skill 和 Tool 清单。
2. 为 Inference 注册独立角色、提示词、Skill 和 Tool 清单。
3. 为 Evaluator 注册独立角色，并兼容现有 `eval` 角色名。
4. 用新 TaskRun 验证每个角色都能返回 Artifact 并回写 StageRun。
5. 四个节点单独闭环后，再进入 GraphRun 串行调度。

在基座模型明确前，不创建真实 Train 平台任务。
