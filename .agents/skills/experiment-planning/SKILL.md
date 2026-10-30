---
name: experiment-planning
description: >-
  EchoMind 实验规划入口。用于澄清目标、数据、模型、训练、评估和交付约束，
  选择匹配的场景 reference，生成可确认的 WorkflowPlan，并在确认后创建
  Dataset、Inference、Evaluator、Train 等阶段组成的 Project Graph。
---

# EchoMind 实验规划

本 Skill 负责把用户目标转成可审阅、可执行的实验计划。场景业务规则放在
`references/` 中按需读取；入口只维护通用的澄清、计划、Graph 和开发态边界。

## 场景路由

先从请求中识别业务场景、输入形态、模型任务和期望产物，只读取匹配的 reference：

| 场景特征 | reference |
|---|---|
| PCB AVI/AOI 图片质检、单轮 SFT/DPO、缺陷图/差分图/GT 图 | [AVI 单轮实验](references/avi-single-turn.md) |

当前只提供 AVI reference。没有匹配项时，按通用计划契约收集信息，并把尚未确认的
业务规则写入 `unresolved`；不要借用 AVI 的标签、指标或图片角色作为默认值。

## 通用澄清

只询问缺失且会影响计划正确性的内容，优先合并成一次提问：

- 实验要解决的问题、验收目标和本轮交付物；
- 数据位置、实际样本结构、版本、split、隔离和泄漏检查；
- 基模或已有模型、训练方式及资源约束；
- 固定评测集、主指标、补充指标和门禁规则；
- 各阶段输入输出、预期 Artifact、失败重试和人工确认点；
- 哪些内容已确认，哪些仍需用户检查真实数据后决定。

检查 Storage 时，用 `terminal` 查看目录和文件元信息，再用 `read` 打开具体文件。
`read` 不负责列目录；沙箱中的 Storage 使用 `/target-workspace/...` 绝对路径，
不要使用 `/datasets` 等平台逻辑路径。确认计划前只能用 `terminal` 做只读探查，
不要修改文件、提交平台任务或创建外部资源。

## 计划契约

创建计划时：

- `skill` 固定为 `experiment-planning`，用于记录规划入口，不用于选择后端模板。
- `name` 是简短项目名，`goal` 说明问题和验收目标。
- `stages`、`edges`、任务和资源以当前场景 reference 为准；没有 reference 依据的
  内容必须标为待确认。
- 阶段 ID、边和任务输出必须形成无环图；阶段顺序必须表达真实依赖，而不是展示顺序。
- 已知数据、模型和产物写入 `resources` 或任务描述；未知 URI、模型、指标和结果不
  要猜测。
- 用户未明确接受 DPO 的 chosen/rejected 构造方式时，不要自行选择 DPO。
- 计划中不得填写 `project_id`、`graph_id`、Run ID、Task ID、凭据或签名 URL。

首期渲染器支持 Dataset、Inference、Evaluator、Train 四类节点，阶段 `name` 必须能让
平台识别对应类型。每个阶段可独立启动和验证；尚未接入真实执行器的节点必须保持
`contract_probe` 语义，不能把探针成功描述成业务执行成功。

## 计划工具

1. 首次用 `create_experiment_plan` 提交完整计划。
2. 修改前先用 `get_experiment_plan` 读取当前版本和待确认提案。
3. 用 `revise_experiment_plan` 提交完整新计划、`expected_revision` 和修改摘要。
4. 修改待确认草案时同时提交它的 `source_proposal_id`；未变化的阶段和任务 ID 保持不变。
5. 只有用户在计划面板明确确认后，平台才创建或更新 Project/Graph。聊天中的“确认”
   不替代面板确认动作。

首次确认时，Project ID 来自服务端 Plan Revision。后续确认保留同一个 Project ID，
Graph 按新 Revision 更新。不要自行创建第二个 Project。

## Graph 生成边界

平台只从已确认 `WorkflowPlan` 的 `stages`、`edges`、`resources` 和 `tasks` 生成 Graph：

- 场景 reference 提供业务拓扑，不直接写 `project.json` 或 `graph.json`；
- Dataset 阶段绑定通用 `data-processing` 能力并采用 Agent 开发模式；
- Evaluator 使用标准结果格式承载指标和逐样本对照；
- 其余阶段首期可以使用确定性执行器契约壳；
- 不读取第二份 requirements 文件，不维护场景专属 Project 创建接口。

不要猜测服务端生成的 Graph ID，也不要在通用 Tool 或共享 Skill 中加入 EchoMind、
AVI 或某个业务场景的分支。

## 开发态与发布态

计划确认后，同一个 EchoMind Manager 会话进入开发态并获得完整平台能力。此时可以：

- 使用远端 Sandbox 探查真实 Storage 数据；
- 使用 `data-processing` 设计 Pipeline、跑小样并提交平台任务；
- 修改计划和 Graph 草案，并再次通过计划面板确认 Revision；
- 记录错误、产物和验证结果，供后续阶段复用。

小样、人工确认和开发期调试属于开发态。已验证并固化的 Graph 由平台执行引擎运行，
不应要求 Agent 长期盯着任务或实时编排。失败时 Agent 可以作为独立任务读取错误并
辅助修复，但不成为生产态调度器。
