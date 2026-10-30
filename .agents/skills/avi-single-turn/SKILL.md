---
name: avi-single-turn
description: >-
  PCB AVI 图片质检单轮 SFT/DPO 场景入口。用于澄清数据、推理、评估与训练目标，
  生成可确认的实验计划，并在确认后创建 Dataset -> Inference -> Evaluator ->
  Train 四阶段 Project Graph。
---

# PCB AVI 单轮训练

EchoMind Manager 用本 Skill 完成 AVI 单轮训练的需求澄清和实验计划。计划确认后，
平台立即用同一个 Project ID 创建 Project 和 Graph；当前会话随后进入开发态，
取得完整平台开发能力。不要直接写 Project 或 Graph 状态文件。

## 计划契约

创建计划时：

- `skill` 固定为 `avi-single-turn`。
- `name` 是简短项目名，`goal` 说明要解决的问题和验收目标。
- 阶段 ID 固定为 `dataset`、`inference`、`evaluator`、`train`，且全部启用。
- 阶段关系固定为：

```text
dataset -> inference -> evaluator -> train
```

- 数据位置、模型、训练方式和评测口径等已知信息写入 `resources` 或任务描述。
- 尚未确认的信息写入 `unresolved`；不要编造 URI、模型、指标或运行结果。
- 不要在计划中填写 `project_id`、`graph_id`、Run ID、Task ID、凭据或签名 URL。

推荐阶段职责：

| 阶段 | 职责 | 典型产物 |
|---|---|---|
| Dataset | 用通用 data-processing 能力探查、清洗、转换并验证真实 Storage 数据 | 可复现的数据处理 Pipeline 与验证产物 |
| Inference | 用选定模型对固定评测集推理 | 预测结果与推理记录 |
| Evaluator | 按统一口径计算指标并生成报告 | ATIF/标准评估产物 |
| Train | 基于已验证的数据和评测门禁执行 SFT 或 DPO | checkpoint 与训练报告 |

Dataset 阶段只声明平台能力，不复制 `data-processing` Skill 的 SOP，也不把
AVI 字段规则写进通用数据处理实现。开发态由当前 EchoMind Manager 使用
`data-processing` Skill 完成小样、确认和全量验证；Graph 固化后则由执行器按冻结规格运行。

## 需求澄清

先读取 [需求清单](references/requirements.md)，再结合
[AVI 数据契约](references/data-contract.md) 和
[评测口径](references/evaluation.md) 判断信息是否足够。

至少确认：

- 数据位于哪个 Storage 路径，真实数据形态是否可检查；
- 缺陷图、差分图、GT 图等图片角色和标签语义；
- 推理基模、训练方式（SFT 或 DPO）和可用算力约束；
- 固定评测集、主指标、门禁和预期产物；
- 数据划分、批次隔离以及是否存在泄漏风险。

只询问缺失且会影响本阶段的项，合并成一次简洁提问。用户未明确接受 DPO 的
chosen/rejected 构造方式时，不要自行选择 DPO。需求未齐时可以保存带 `unresolved`
的计划草案，但必须明确告诉用户哪些信息仍待补充。

## 计划工具

1. 首次用 `create_experiment_plan` 提交完整计划。
2. 修改前先用 `get_experiment_plan` 读取当前版本和待确认提案。
3. 用 `revise_experiment_plan` 提交完整新计划、`expected_revision` 和修改摘要。
4. 修改待确认草案时同时提交它的 `source_proposal_id`；未变化的阶段和任务 ID 保持不变。
5. 只有用户在计划面板明确确认后，平台才创建或更新 Project/Graph。聊天中的“确认”
   不替代面板确认动作。

确认第一版时，Project ID 来自服务端计划 Revision。确认后续版本时，同一个 Project ID
保留，Graph 按新 Revision 更新。不要自行创建第二个 Project。

## Graph 生成边界

平台从已确认 `WorkflowPlan` 生成 Graph：

- 节点顺序和 ID 来自四个已启用阶段；
- 节点配置来自阶段的 `resources` 和 `tasks`，不读取另一份 requirements 文件；
- `dataset` 节点绑定 `data-processing` 能力并采用 Agent 开发模式；
- 其余阶段首期使用确定性执行器契约壳，不代表业务执行已经实现；
- 阶段可独立启动和验证，但未接入的执行器必须明确显示为壳或尚未执行。

不要直接写 `project.json`、`graph.json`、节点或边，也不要猜测服务端生成的 Graph ID。

## 开发态边界

计划确认后，同一个 EchoMind Manager 会话获得完整平台工具和通用 Skill。此时可以：

- 使用远端 Sandbox 探查真实 Storage 数据；
- 使用 `data-processing` 设计 Pipeline、跑小样并提交平台任务；
- 修改计划和 Graph 草案，并再次通过计划面板确认 Revision；
- 记录错误、产物和验证结果，供后续阶段复用。

不要修改 Pyromind Agent 的会话或工作流，不要成为生产调度器，也不要在通用 Skill/Tool
中加入 EchoMind 或 AVI 专属分支。发布态执行由平台执行引擎负责。
