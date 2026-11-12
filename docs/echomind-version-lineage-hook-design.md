# EchoMind 实验版本血缘自动登记技术方案

> 日期：2026-11-12  
> 状态：设计建议，尚未实现。代码事实来自当前工作区，不代表线上部署已验证。  
> 参考：《EchoMind 版本血缘登记技术方案（Hook 实现）》。

## 1. 结论与范围

保留参考方案的核心：**一个实验版本一个目录，数据、训练配置、模型引用和评测证据可以一起读懂；事实自动登记，判断由 Agent 交付；不新增 Agent 写类工具。**

但不照搬“七个 Pi Hook 各自写文件”。当前仓库已经有统一的 Product Runtime、提交凭证、产物版本库、异步任务恢复和选模实现。建议：

1. 在现有执行入口增加少量、显式的生命周期回调，由 Runtime 完成登记。
2. Pi extension 只提供上下文和生命周期适配，不实现产品血缘规则。
3. 复用 `Versions`、`PipelineRuns`、`TrainingEvaluations` 和 checkpoint 检查，不再实现另一套快照与选模。
4. 增加实验目录这一“组合视图”，不把现有“单个产物版本”改成“整次实验版本”。
5. 先完成提交、产物发现、评测归档和恢复闭环，再接趋势索引与前端。

本期不重建 Stage 调度器，不修改 AgentGenome 的经验图契约，不引入模型发布、部署或流量切换。

## 2. 当前代码事实

下列路径均相对 `software-agent-sdk/`，用于实现定位。

| 事实 | 代码依据 | 对方案的影响 |
| --- | --- | --- |
| EchoMind 已配置 init、dataset、train、eval 四个原生委派角色，分别限制工具和 Skill | `pyromind-agent-server/pyromind_agent_server/profiles.py` | 延续现有委派，不新增业务调度链 |
| 子 Agent 的任务文件默认在 `public_data/subagents/{session_id}/` | `harness-adapter/pi-runtime/src/subagent-child-extension.ts` | 保留任务工作区；实验归档另设稳定目录 |
| `submit_workflow` 在 Pi host 内调用 Runtime 的 `prepare_workflow_execution()` | `harness-adapter/harness_adapter/pi_adapter/business_tool_host.py:547` | 提交前闸门应放 Runtime，而不是只拦 Pi 工具 |
| 另有 `submit_versioned_workflow()` 提交入口 | `pyromind-runtime/pyromind_runtime/application/conversation_runtime.py:698` | 两条入口必须调用同一登记用例 |
| 提交时已有配置、DSL 快照和不可变 RunSpec；任务关联由对账补齐 | `application/versions.py`、`application/conversation_runtime.py:903` | 复用现有幂等与恢复机制 |
| 产物版本库存储在 SQLite 的 `projects/runs/artifact_versions`，不是派生 MySQL 索引 | `infrastructure/sqlite_version_store.py`、`pyromind-agent-server/.../bootstrap.py` | 不能把现有版本库直接宣布为“可删除的指标索引” |
| 业务 Project 与版本库 Project 是不同模型 | `domain/preparation.py:ProjectView`、`domain/versions.py:VersionProject` | 必须显式绑定，不能混用 project_id |
| 数据、模型路径导入目前标为 `reference_only` | `application/versions.py:49`、`harness_adapter/artifact_snapshots.py` | 有路径不等于冻结了数据或模型内容 |
| DSL 通过 SDK 转为 nodes/edges；实际字段包括 `train_data_path`、`val_data_path`、`output_path` | `harness_adapter/workflow_submission.py`、`.agents/skills/generate-workflow-dsl/references/workflow-contracts.md` | 不能直接套 `train.params.dataset_path/output_dir` |
| `training_analysis(operation="artifacts")` 已解析执行图、节点输出并检查 checkpoint | `.agents/skills/training-analysis/scripts/training_artifacts.py` | 复用真实产物发现，不按命名猜路径 |
| Runtime 的现有轮询查询任务状态，没有现成 checkpoint 目录轮询器 | `application/conversation_runtime.py:402`、`harness_adapter/workflow_submission.py:56` | H4 不是简单挂载现成轮询；P0 先在训练终态发现 |
| 新评测统一使用 `df_submit_evaluation`，支持 inference、score 等操作 | `.agents/skills/inference-evaluation/SKILL.md`、`openhands-tools/.../evaluation_submit.py` | 不新建一套 Eval DSL 拦截链 |
| 已有完整 checkpoint、同 step、可比组、validation-only 选模与独立 test 校验 | `application/checkpoint_selection.py`、`application/test_evaluation.py` | 不按 pass_rate 自行再选一次 best |
| 训练评分已有 `batch_result.json`、`checkpoint_selection.json`、`iteration_context.json` | `application/training_evaluations.py:42` | 实验归档引用现有结果，不重新生产评测报告 |
| 实验数据库配置支持 SQLite 或 MariaDB，已有迁移链 | `pyromind-agent-server/.../plan_database.py`、`plan_migrations/` | 趋势索引接现有数据库配置，不另设连接体系 |

本次已运行版本、集成、选模、提交和架构边界的定向测试，78 项通过。未做真实 Studio、Storage、MariaDB 或在线模型验证。

## 3. 原方案需要修正的地方

### 3.1 Hook 是触发方式，不是产品逻辑归属

本地 Pi 扩展接口有 `session_start`、`before_agent_start`、可阻塞的 `tool_call`、可改写结果的 `tool_result`。当前 host 也已经接入 `pi-subagents` 的完成通知。

但只有 host 看见了“工具调用结束”，不能证明 Studio 训练或评测完成；子 Agent 提交异步任务后结束当前轮次也是正常行为。

因此，生命周期回调放在真实提交、任务终态和结果验收处。Pi 事件名称和原生 session 文件不进入 Runtime。

### 3.2 路径来自结构化执行事实，不只来自源码

训练数据通常通过 `PathJoinNode → DatasetConfigBuilderNode → ModelTrain*` 连线传递；基模也可能来自 Clone/Download。路径未必是训练节点的字面量。

提交前读取已校验的 nodes/edges，解析确定性绑定；执行后用实际任务图与节点输出补全。不能把 AST 表达式文本当实际路径，不能执行 DSL 来“算路径”。

### 3.3 `chmod` 不能提供所需的冻结保证

原路径仍可能被同用户进程恢复写权限、替换目录项或经其他 Storage 入口覆盖。只给原 JSONL 去掉写权限还存在“算完 hash 到训练读取前被改写”的窗口。

正确的执行冻结是：**保存本次输入的确定字节，并让提交给 Studio 的工作流读取冻结副本。** 权限只作辅助提示。

### 3.4 `cp` 不等于创建成功，也不等于新版本身份

普通 `cp -r` 可能合并已有目标目录，并复制父版本的 task_id、best 和状态。

新目录先以排他 `mkdir` 占用；只复制输入文件和待修改 DSL，不复制 Runtime 管理的事实文件。并发同名时重试，禁止覆盖或合并。

### 3.5 基线不仅取决于评测工作流

相同数据和 DSL 不保证相同生成参数、Prompt、评分脚本或基模。不能去掉所有模型信息后拿到任意一条 baseline。

评分可比性与基线缓存命中是两个不同问题，见第 8 节。

## 4. 架构与职责

```text
EchoMind Profile / 场景 Skill
  声明交付规范、目录布局、指标策略及是否启用自动登记
                    |
现有工具 / HTTP / 可复用执行入口
                    |
ConversationRuntime + PipelineRuns
  统一授权、冻结、提交凭证、任务恢复
                    |
ExperimentLineage（新增通用应用用例）
  prepare / observe / reconcile
           |                         |
ArtifactSnapshotPort           实验归档存储端口
现有 Versions / ProductStore     + 可选趋势索引端口
           |                         |
平台适配实现 / 文件存储实现 / SQLAlchemy 实现

Pi extension → Adapter → 通用上下文及 operation 事件
```

`ExperimentLineage` 是名称建议，不是新的执行引擎或可配置 Hook 框架。

| 层 | 职责 |
| --- | --- |
| Runtime domain | 通用实验归档、输入绑定、证据引用和状态模型 |
| Runtime application | 提交前校验与冻结、任务绑定、归档、对账；只访问 ports |
| Runtime ports | 文件读写/提交、产物发现、可选指标索引；复用已有快照端口 |
| infrastructure / Adapter 下层 | 文件系统、Storage、平台任务图与 checkpoint 探查 |
| Server | 配置、鉴权、装配、查询路由；不直接操作 Harness 会话 |
| Pi Adapter | 原生上下文注入、完成通知转换；不判断 AVI 标签或 EchoMind 业务成功 |
| Profile / 场景 Skill | 启用能力、原始数据规则、交付文件与业务指标口径 |

能力按可信配置装配，默认不启用强制实验目录约束。共享实现不出现 `if profile == "echomind"`、按角色名称调度或 AVI/PCB 分支。模型不能通过请求自行扩大能力或指定别人的 Project。

Pyromind 继续使用原流程；EchoMind 的新增约定只由其 Profile 与场景 Skill 启用。现有 `ProductVersionsTool` 保留兼容，但不加入 EchoMind 白名单，也不要求 Agent 手工登记。

## 5. 身份、目录与事实源

### 5.1 三种身份必须分开

| 身份 | 含义 |
| --- | --- |
| `project_id` | 用户创建的业务 Project |
| `artifact_project_id` | 当前 `VersionProject.id`，如 `p1` |
| `experiment_id` / `version` | 新增组合实验身份 / Project 内显示名，如 UUID / `v2` |

映射记录同时保存 `conversation_id`、owner、profile 与 cluster 作用域。身份从已授权 ProductStore/Project 获取，不相信 Agent 写入的 owner 或 project_id。

`v2` 不是现有产物版本 ID。一次实验可引用多个 dataset/config/code/model/evaluation 版本；`parent_experiment_id` 表示探索分支，不替代产物级 parents。

本期一个业务 Project 仍绑定当前 conversation。不增加跨会话共享上下文；将来若做独立 Project 生命周期，另行设计权限与迁移。

### 5.2 目录

模型可见路径：

```text
/workspace/.pyromind-agent/{conversation_id}/
├── public_data/subagents/...            # 保留现有任务工作区
├── versions/...                         # 保留现有通用产物快照
└── projects/{project_id}/
    ├── raw/{source_id}/...              # 场景约定：项目内共享原图
    └── experiments/
        ├── v0/                         # 基线实验，无训练
        └── v1/
            ├── draft.json              # Agent：父版本声明、输入角色与来源说明
            ├── train.jsonl
            ├── eval.jsonl              # 名称可保留，角色明确为 validation
            ├── test.jsonl              # 可选，独立 test
            ├── WORKFLOW.py             # 提交时输入源码快照
            ├── submitted_workflow.json # 真正提交的冻结执行图
            ├── inputs.json             # 冻结路径、hash、样本数、媒体解析契约
            ├── meta.json               # Runtime 生成的最新可读视图
            ├── checkpoint_path.json    # Runtime 汇总实际节点产物
            ├── eval_result.json        # Runtime 汇总已有评测与选模结果
            ├── base_eval_result.json   # 可选：基线证据或复用引用
            ├── eval/                   # 现有报告/预测等产物的稳定引用
            ├── notes/                  # Agent 的解释与结论
            └── manifests/r000001.json  # Runtime 提交的自描述归档 revision
```

使用 `experiments/` 避免与已有 `versions/` 的产物编号混淆。若产品坚持 `projects/.../versions/vN`，可以改展示布局，但不能复用已有产物版本命名空间。

Storage API 路径通过现有 `storage_paths.py` 转换，去掉 `/workspace`；不得持久化 `/target-workspace`、沙箱宿主机路径或签名 URL。

“原图一份”限定为当前授权 Project 内共享，不做跨用户全局去重。不强制复制所有现有图片到 raw：已有 Storage 数据可以引用，准确标注完整性与可变性。

JSONL 的相对媒体路径必须携带明确解析基准。当前评测默认相对 JSONL 所在目录，训练解析取决于加载器，**不能只在 Skill 中规定“相对 raw 根”就认为生效**。归档输入可保留相对 raw 的场景格式，执行副本按真实数据配置转换；验证训练和评测消费的是同一媒体。原始数据不被静默改写。

### 5.3 唯一事实源的边界

- 现有 ProductStore、External Task Registry、Versions 仍负责执行凭证、授权与恢复，不能删除。
- 新增实验血缘的已提交 `manifests/rN.json` 是实验归档事实源；包含数据、执行图、task/run、产物和评测引用，可离线解释实验。
- `meta.json` 等是同一 revision 的便捷视图；趋势数据库是可重建派生索引。
- `notes/` 是 Agent 判断，不得覆盖 Runtime 的 task_id、hash、状态、指标。

“删除索引后可重建”只指新增趋势索引，不代表删除现有版本数据库后原有执行恢复也能重建。

## 6. 最小数据契约

以下为新增归档示例，不是现有公开协议。字段经 Pydantic schema 校验，未知旧版本不得静默当作成功解析。

```json
{
  "schema_version": 1,
  "revision": 3,
  "experiment_id": "UUID",
  "version": "v1",
  "project_id": "business-project-id",
  "artifact_project_id": "p1",
  "conversation_id": "conversation-id",
  "parent_experiment_id": null,
  "status": "training",
  "registration_status": "complete",
  "inputs": {
    "train": {
      "artifact_version_id": "v12",
      "storage_ref": "/.pyromind-agent/.../frozen/train.jsonl",
      "sha256": "...",
      "count": 1200,
      "media_root": "/datasets/raw",
      "snapshot_scope": "manifest_only"
    }
  },
  "training": {
    "request_id": "...",
    "version_run_id": "r3",
    "task_id": "...",
    "workflow_sha256": "...",
    "submitted_graph_sha256": "..."
  },
  "evidence": [],
  "issues": []
}
```

约束：

- `scope=full/labeled_subset` 由 Dataset 声明并由来源清单检查，不能仅凭 JSONL 条数推断。
- `data_hash` 分别记录 train/validation/test 的原始字节 hash；组合 hash 包含角色和媒体解析配置。
- `snapshot_scope=manifest_only` 不宣称媒体已冻结；媒体内容完整快照必须有可验证的内容摘要或不可变存储证明。
- diff 优先用稳定 sample_id；多图、多轮样本不能只用图片路径作主键。没有稳定键时只给内容差异和 `diff_quality`，不编造 relabeled 数。
- checkpoint 唯一键为 `(task_id, training_node_id, step, kind)`，不是只有 step。
- 一个实验允许多个训练节点，例如 SFT → GRPO；checkpoint 文件采用 `training_nodes[]`，展示 best 必须指明节点、评测组和 selection_run。

状态分两类：业务过程 `draft/frozen/submitting/training/trained/evaluating/done/failed/cancelled/uncertain`；登记过程 `pending/complete/needs_reconcile`。`done` 表示本次要求的证据已交齐，不表示业务达标或可上线。train-only 任务可在产物验收后结束，不强制运行评测。

## 7. 自动登记时序

### H1：上下文摘要

激活/恢复会话以及下一次 Agent turn 前，通过通用上下文提供能力读取已授权实验归档。Adapter 映射到 `before_agent_start`，不让 Runtime 依赖 Pi 事件。

摘要控制在约 30 行：版本、parent、状态、数据数、任务、选模结论、证据缺口。多版本时展示最近版本、当前分支与相关基线，其余由只读查询获取。只读摘要不触发全量 hash 或 SQL 写入。

### H2：提交前冻结

在 `prepare_workflow_execution()` 和 `submit_versioned_workflow()` 的真实平台 submit 之前调用同一 `prepare` 用例：

1. 校验 Project/Conversation/Profile/cluster 绑定，读取可信能力配置。
2. 从已验证执行图追踪数据输入、训练节点、输出和模型来源。仅支持已知节点契约与确定性路径绑定。
3. 用解析出的 train/validation 文件定位实验目录，验证 `draft.json` 的 parent；不从 Agent 自述路径登记。
4. 复制 JSONL 确定字节，计算 hash、计数与 diff；校验媒体引用。冻结失败即不提交。
5. 保存原始 DSL 和实际执行图，重写执行图的数据绑定使其读取冻结副本；沿用节点/边映射，不做全文件任意字符串替换。
6. 重新校验重写后的图，确认没有 `version://`、表达式占位或未解析路径残留，保存提交凭证。

冻结文件通过现有 `ArtifactSnapshotPort` 保存并校验，但不能直接调用当前会把 dataset 降为 reference_only 的 `Versions.save()` 然后宣布已冻结。实现时补充“清单确定字节快照”的通用内部能力，仍显式声明未覆盖的媒体依赖，不扩大旧 `ready` 含义。

原有源码与重写后的图可能路径不同，因此两者同时保存；`WORKFLOW.py` 不能被宣传为未经说明就可直接复跑的最终执行图。

同一 request 重放复用凭证。已冻结实验换数据、换参数或换 DSL，创建新实验；提交结果 uncertain 不得自动换版本再次提交。明确恢复同一冻结运行由现有 resume/执行修订机制处理，不覆盖旧尝试。

复杂动态数据来源在 P0 明确拒绝严格登记，不猜路径。Pyromind 未启用该能力的旧工作流不受实验目录限制。

### H3：提交后绑定

平台返回 `task_id` 后，先落现有 WorkflowExecutionRequest/External Task Registry 凭证，再登记实验引用、VersionRun 和任务，提交归档 revision。

若 task 已创建但归档失败，返回已知 task_id 与 `needs_reconcile`，不把它伪装成“未提交”。checkpoint 目录此时只能标 `declared/unresolved`，不能当作实际存在的模型。

### H4：训练产物发现

P0 以训练终态回调和会话恢复对账为触发，新增产物发现端口实现复用 `resolve_artifacts()/discover()/inspect_checkpoint()` 的逻辑。

这些函数当前位于 Skill 脚本，不能由 Runtime 反向导入。需先提取为可分发的中立辅助模块，原工具和新端口共用；平台访问实现位于 Adapter 下层，Runtime 只消费通用事实模型。

从实际执行图取得 training_node_id 与输出路径，再检查 `checkpoint-n`、权重/分片、LoRA adapter、基模和 trainer_state。不完整 checkpoint 保留 issues，禁止进入选模。

训练中增量发现不是 P0 前提。确有体验需求时，在已有任务维护循环增加受限探查：仅活跃训练、限频、按节点去重，不能给每个 checkpoint 启一个 Agent 或新轮询服务。

### H5：评测提交前绑定与缓存

接 `df_submit_evaluation` 的产品执行链和 `PipelineRuns`，不监听 Agent 是否写了一个 Eval DSL：

- inference 从冻结的 model_source、model_path、数据与配置绑定版本。
- score 从 source_manifest 的 task_id、training_node_id、checkpoint 和预测摘要绑定版本。
- profile/report 不冒充模型评测或改变 best。
- 找不到模型归属时保留可验证的外部模型引用；仅当当前任务要求必须属于指定实验时阻塞，不按路径前缀随意归属。

基线缓存命中应返回原 run_id、证据、指标和 `reused=true`，不创建伪 Studio task。该语义在工具的冻结/预留路径内完成，不能只在 PostToolUse 改一句文本。

### H6：评测终态验收

平台 succeeded 后，读取并核对冻结来源、hash、实际覆盖数量、评分错误与 schema。沿用 `TrainingEvaluations.finalize()` 及 `select_checkpoint()`。

`eval_result.json` 保存每次评测运行的引用与摘要；best 引用 `checkpoint_selection.json` 的真实结论。已提交多个评测批次时以 `evaluation_run_id` 分开，不能只按 step 覆盖。

有独立 test 时保留现有“先 validation 选模，再对选中模型做 test”的约束。test 只报告最终效果，不进入 best 排序。

归档提交后同步趋势索引；报告渲染失败但评分有效时保留指标与问题，不丢弃评测事实。

### H7：交付验收

不使用不存在的通用 `SubagentStop` 名称，也不把 `agent_end` 当成功。现有 Adapter 会将原生完成通知转换成通用 operation 事件。

交付检查必须同时满足：目标 operation 已结束、绑定的平台任务终态可验证、所需产物已经可读。提交任务后正常结束一轮不触发“缺报告打回”。

交付文件名和 schema 由可信 Profile/场景契约声明，检查器只按契约检查，不包含 `if role == "dataset"` 等分支。

缺交付时记录 `delivery_incomplete` 和具体问题，并通过现有通用通知通道反馈。补交沿用原子 Agent 的原生 resume，保留原任务，不重跑训练/推理。先完成可见的验收失败与人工/Manager 修复；自动补交重试须限次，不能建立新的业务调度器。

H7 只能验证已知证据，无法保证“绕过工具手工创建的 Studio 任务也不会漏”。此类任务仅可在拿到可核验 task_id 和归属后导入，不能靠检查 notes 补造血缘。

## 8. 可比性与基线

已有预测评分的 `comparison_group` 包含真实样本与 GT、scoring_spec、评分脚本、generation 的摘要，见 `evaluate_predictions.py:314`。新方案复用这一结果，不只对“删除模型路径后的 DSL”算 hash。

### 两个键

| 键 | 用途 | 必须考虑 |
| --- | --- | --- |
| `evaluation_protocol_hash` | 指标能否直接比较 | 数据角色、样本/GT、媒体身份、字段映射、Prompt/预处理、生成参数、评分规则/脚本及执行协议版本 |
| `baseline_cache_key` | 能否复用某次基线结果 | 上述口径 + 基模身份/修订/内容摘要 + 推理运行环境的相关版本 |

优先沿用已有 comparison_group；缺少其中关键来源时记录 `comparability=unverified`，不把未知配置补成默认值。跨 inference 与 saved-predictions 的共同口径需要显式验证，不能仅因都叫 pass_rate 就视为相同。

基线查找限制在已授权作用域。已存在预测时评分口径改变，优先重跑 CPU 评分；只有预测输入或生成条件改变才需要新的推理。

`v0` 是最初基线的便捷位置，不是任意评测口径的永久基线。新 validation/test、基模或口径变化时，在当前实验保存新的基线证据。baseline 必须指明 `model_identity`、`evaluation_run_id`、protocol hash 与 dataset_role。

`eval_result.json` 的 best 是“某节点、某验证口径、某 selection policy 下的 best”；验收阈值结论单独记录。不能用未验证结果、训练 loss 或测试集表现替代选模证据。

## 9. 写入、一致性与恢复

### 9.1 文件提交

一个逻辑写入者：Runtime 登记用例。训练、评测等回调即使都属于 Runtime，仍可能并发，必须通过实验级 revision/CAS 序列化，不能认为“只有一个模块写”就没有竞争。

默认挂载文件系统实现采用锁内读改写、同文件系统临时文件、flush/fsync、rename，最后提交自描述 `manifests/rN.json`；该文件只引用已写好且摘要匹配的对象。可读视图损坏时从最新有效 revision 重建。

远程 Storage 不能默认有 rename/CAS。当前 `EvaluationStorage` 只提供 read/write/list，没有原子发布承诺。对应实现需写不可变 revision 对象并校验完整性；并发分配必须靠经过验证的共享锁或持久化 CAS，而不是“list 最大版本号加一”。P0 先支持能力已验证的挂载后端，不宣称所有对象存储都能直接保证同样语义。

### 9.2 必须阻塞与可补偿

| 情况 | 处理 |
| --- | --- |
| 越权、跨会话/Profile、路径越界、父版本不合法 | 阻塞 |
| 必需数据缺失、冻结失败、执行图绑定不确定、冻结后输入改变 | 提交前阻塞 |
| 同一实验以不同训练输入或配置再次提交 | 要求新实验 |
| Studio 超时但任务是否已创建未知 | uncertain，禁止自动重提 |
| 已知 task_id 后写归档失败 | 保留任务，needs_reconcile，重放登记 |
| 产物暂不可读、checkpoint 不完整 | 保留缺口，延迟重查，不选 best |
| 指标/来源验收失败 | 不进入有效对比，不宣称 done |
| notes 缺失 | 交付未完成，不改写已确认的平台事实 |
| 趋势索引失败 | 不影响平台任务与归档，稍后补写 |

“只有两种错误阻塞，其余只记日志”不能满足数据冻结与隔离要求。

### 9.3 对账

复用 `recover_version_metadata()`、`reconcile_versions()` 与现有终态处理，增加实验归档对账；不要依赖 SessionStart 单一触发点。

对账读取持久化 request、VersionRun、任务关联和已提交归档 revision，补 task 绑定、产物、评分引用和索引。注册幂等键分别使用 request_id、task/node/step、evaluation_run_id、manifest revision。

当前启动恢复不带用户请求凭据，不能承诺启动时就能重新访问平台 Storage。先做本地凭证修复；需要远端读取的工作在有效 RequestContext 到达后补齐。认证失效与“文件不存在”必须区分，Cookie/Token/Authorization 不落盘。

历史回调不得把新状态退回旧状态；同一事件重放不得重复登记。新版本分叉只复制输入，不继承父实验的平台状态。raw 的追加与清理依赖存储策略和引用检查，不宣称 Hook 能阻止所有外部删除。

## 10. 趋势索引与查询

新增两个派生表即可，名称建议 `experiment_version_index`、`experiment_metric_index`，避免与已有表混淆。接现有 SQLAlchemy engine 与 Alembic 迁移，SQLite/MariaDB 共用模型。

| 表 | 关键字段 |
| --- | --- |
| experiment_version_index | owner_scope、project_id、conversation_id、experiment_id、display_version、parent_id、dir_ref、status、manifest_revision、created_at |
| experiment_metric_index | experiment_id、evaluation_run_id、task_id、training_node_id、model_identity、step、dataset_role、kind、protocol_hash、metric、value、selected、evaluated_at |

指标唯一键至少包含 experiment、evaluation_run、metric；selection 额外关联 selection_run/policy，不能用全局 is_best 表示所有评价口径。

基线模型身份不能省略。一个实验多节点同 step、同一 checkpoint 多次评分、重跑评分脚本等情况不能互相覆盖。可变状态更新以 manifest_revision 防止旧回放覆盖新索引。

趋势查询按已授权 Project、dataset_role、protocol hash、metric 分组；checkpoint 趋势取明确选模批次的结果，baseline 取相同口径与指定模型。用户可切换评测组，不把不可比指标连线。

索引可以删除并从已提交实验 manifests 重建；无 SQL 时 CLI 读取归档，DuckDB 可作分析辅助，但不承担在线授权。

只读查询建议：

- `.../conversations/{conversation_id}/experiments`
- `.../conversations/{conversation_id}/experiments/{experiment_id}`
- `.../conversations/{conversation_id}/experiment-metrics`

路由进入 `ConversationRuntime` 后统一授权。CLI 为只读，复用同一查询/解析实现；不让 bash 直连 MySQL 或自行读取其他会话。前端只渲染产品 DTO，不扫描文件、不解析 DSL、不判断 best。

P0/P1 不新增公开 SSE 事件类型。后续需实时趋势时扩展通用 capabilities、snapshot 和刷新事件，并测试旧客户端兼容；不直接向 API 暴露 `version_frozen`、Pi Hook 名称或原生子 Agent 事件。

## 11. 文件与改动范围

下列新增路径是建议，落地时按现有包组织细化，不作为已存在能力。

| 位置 | 改动 |
| --- | --- |
| `pyromind-runtime/.../domain/experiment_lineage.py` | 新增实验归档、绑定、证据状态与 schema |
| `pyromind-runtime/.../application/experiment_lineage.py` | 新增 prepare/observe/reconcile 用例，不新增调度器 |
| `pyromind-runtime/.../ports/experiment_lineage.py` | 新增归档、产物探查、可选趋势索引契约 |
| `pyromind-runtime/.../infrastructure/` | 文件归档与 SQL 指标索引实现 |
| `application/conversation_runtime.py` | 两条提交入口与终态/恢复调用同一用例 |
| `application/versions.py`、`ports/versions.py` | 增加清单字节冻结能力，保留 reference_only 与现有恢复语义 |
| `application/pipeline_runs.py`、`training_evaluations.py` | 绑定真实评测运行、复用现有 provenance 与 selection |
| `harness-adapter/.../workflow_submission.py` 及平台辅助模块 | 结构化绑定解析与实际产物探查，复用现有算法 |
| `harness-adapter/pi-runtime/` | 通用摘要/交付通知桥接，继续用已有 pi-subagents |
| Server 的 Profile、bootstrap、迁移与查询路由 | 配置声明、能力装配、授权入口和可选索引 |
| EchoMind 场景 Skill 与角色提示 | 草稿、目录及 notes 交付规则，精简修改 |

共享的 data-processing、workflow-execution、inference-evaluation Skill 不追加 AVI 特判。新增场景 Skill 放在 `.agents/skills/`，由 EchoMind 相应角色白名单启用；需要共用的参数契约仍保持中立。

不修改 AgentGenome 的捕获、蒸馏与脚本协议，不把业务实验版本 ID 变成 Genome graph revision。将来固化流程时只引用本方案的输入与产物证据。

## 12. 分期与验收

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| P0：一轮训练可追溯 | Project 绑定、场景目录契约、确定性路径解析、输入冻结、提交凭证、训练终态产物发现、归档恢复 | 一个真实训练版本可回答数据/参数/task/模型位置；Studio 实际读取冻结副本；不依赖 MySQL |
| P1：一轮评测可比较 | score/inference 绑定、协议摘要、基线证据复用、现有选模结果归档、交付验收 | validation 同口径比较；test 不选模；同基模和口径复用证据，不重复推理 |
| P2：查询与体验 | 可重建索引、只读 API/CLI、摘要、趋势展示；按需增量 checkpoint 发现 | 重建后指标一致；旧 Pyromind/客户端/历史会话不受影响 |

P0 不要求实时发现每一个 checkpoint，不要求所有旧实验迁移，不要求目录之外的所有历史数据立即强制归档。

### 必测场景

1. Profile 白名单、跨 Profile 拒绝、Conversation/Project/cluster 隔离；Pyromind 旧流程回归。
2. 字面量、PathJoin、Builder 连线、多个训练节点、无法解析的动态输出。
3. 冻结后原数据被改、复制时文件变化、媒体缺失、相对路径基准不一致。
4. 并发创建同目录、并发回调、CAS 冲突、写一半崩溃、旧 revision 重放。
5. Studio 已创建但归档失败、提交超时 uncertain、重复回调、缺凭据恢复。
6. 不完整权重、分片缺失、LoRA 缺基模、同 step 不同节点、预测无对应模型。
7. 评分脚本/GT/generation/基模变化导致比较组或缓存键变化；test 永不参与选模。
8. 子 Agent 提交后正常结束不误触验收失败；缺交付能补交但不重复平台任务。
9. 无数据库仍可读归档，索引故障可补写，索引删除后可重建。
10. 旧 schema、旧 request、旧产物版本和无实验目录的历史会话继续可读恢复。

跨包实现必须补架构测试，并运行 `uv run pytest tests/pyromind_runtime`。Pi 适配改动另跑 pi-runtime 的类型检查和相关原生子 Agent 集成测试；真实平台验收独立记录，不能以 mock 测试替代。

## 13. 仍需真实环境确认

代码已能回答原方案里的工具名、字段、Pi 阻塞能力、子 Agent 完成来源与常见产物命名。剩余重点不是再问 Agent 工具叫什么，而是验证部署事实：

- 当前部署 SDK 的 DSL converter、节点输入和返回图是否与工作区一致。
- 目标训练节点输出、checkpoint 保留策略及大文件/分片写完的可验证信号。
- 挂载 Storage 的锁、rename、权限和多进程语义，原始媒体能否提供不可变修订。
- 多模态训练加载器的真实相对路径基准，及与现有 inference 数据配置的一致性。
- 项目默认主指标、阈值和基模修订是否已有明确产品来源；缺失时只报告事实，不默认上线标准。

最终建议：**先把现有真实执行链补成可归档、可验证、可恢复的实验记录，再让 Hook 自动触发这些能力；不以“Hook 已运行”替代“版本血缘已经可信”。**
