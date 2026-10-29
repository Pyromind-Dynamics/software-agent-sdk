---
name: avi-single-turn
description: >-
  PCB AVI 图片质检单轮 SFT/DPO 场景入口。用于设计实验计划，或澄清数据、
  推理、评估与训练需求后生成可确认的 Project 创建请求，并套用
  Dataset -> Inference -> Evaluator -> Train 四阶段 Graph 模板。
---

# PCB AVI 单轮训练

这个 Skill 同时覆盖两类开发态工作：

- **计划模式**：使用实验计划工具生成可确认的阶段/任务草案。
- **Project 模式**：补齐需求清单，生成 `project_request.json`，等用户显式确认后由平台创建 Project 和 Graph。

两种模式共享 AVI 业务知识，不代表两个 Agent，也不共享会话上下文。以当前会话实际提供的工具为准，不要在一个模式下伪造另一个模式的结果。

## 计划模式

计划、待确认提案和历史版本通过实验计划工具读取；持久化由 Runtime 负责。

默认阶段图：`Init → Data → Train → Eval`。另保留 Inference、Tracer 两个未启用的占位阶段（`placeholder: true, enabled: false`）。

| 阶段 | 计划中的职责 | 典型输入与预期产物 |
|---|---|---|
| Init | 使用基模跑 bench，建立比较基线 | 基模、固定评测集 → baseline 指标与报告 |
| Data | 检查数据，设计清洗或预打标方案 | 原始数据 → 供训练使用的数据版本 |
| Train | 配置并训练模型 | 数据版本、基模 → checkpoint 与训练报告 |
| Eval | 按相同口径评测训练结果并与 baseline 比较 | checkpoint、固定评测集、baseline → 比较结论与改进建议 |

Init 是图中第一个独立阶段，后续连接 Data；后续实现中可复用 Eval 子 Agent 的评测能力。阶段表示业务工作，任务表示阶段内的一项活，阶段不等于 Agent。

用户可以只做评测或启用其他阶段组合。以用户目标调整阶段启用情况和连线，无需收集未启用阶段的训练参数。Inference、Tracer 当前仍保持占位。运行中的跳过、重试、回退和输入失效处理属于后续能力。

## 计划内容

明确实验目标、已有模型和数据、评价口径、预期结果与约束。未知的资源或执行参数写入 `unresolved`，不编造 URI，也不因为暂缺训练参数而阻止保存评测计划。

- `resources` 登记所需外部资源：`id, name, uri, description`；尚未确定的资源也登记占位项，`uri: null`，缺失信息记入 `unresolved`，任务输入仍通过 `resource_id` 引用，不能只填名称。固定评测集用于后续 baseline 与训练后评测的同口径比较。
- `stages` 使用稳定的局部 ID，填写 `name, description, enabled, placeholder`。
- `tasks` 填写稳定局部 ID、`stage_id`、自由名称、说明、输入、预期输出和完成条件声明。推荐任务名如“基模评测”“检查标注完整性”“小样训练验证”，无固定业务 task_type。
- 外部输入写作 `{name, resource_id}`；上游预期产物写作 `{name, task_id, output_name}`，对应上游 `outputs` 中的名称。产物尚未生成也可用于规划。
- `outputs` 使用 `{name, description}`；`completion_conditions` 使用 `{kind, description}` 描述完成依据，如报告产出、外部任务终态。本步只保存声明，不判断完成。
- `edges` 使用阶段 ID 的 `{source, target}`，表示阶段先后引导。任务输入声明表达具体参数来源；图和输入依赖都不能形成环。

首次用 `create_experiment_plan` 生成完整草稿。修改前用 `get_experiment_plan` 获取最新版本；用 `revise_experiment_plan` 提交完整新计划、`expected_revision` 和修改摘要。存在待确认草稿时同时提交它的 `source_proposal_id`。未改动的阶段、任务 ID 保持稳定。

前端确认后才创建项目或新版本。草稿可展示图，不能宣称已经执行任何阶段，也不能把用户在聊天中说“确认”当作数据库提交成功。

## Project 模式

先读取 [需求清单](references/requirements.md)，逐项判断是否已明确。只询问缺失项，
把相关问题合并成一次简洁提问；不得根据常见默认值替用户确认。

以下 requirement key 必须全部存在、`status=confirmed` 且 `value` 非空：

```text
dataset_uri
dataset_format
image_roles
label_schema
split_policy
inference_model
train_mode
evaluation_rule
artifact_goal
```

`train_mode` 只允许 `sft` 或 `dpo`。只有用户明确接受 DPO 的 chosen/rejected
构造方式时才选择 `dpo`；否则先确认是否能做 SFT。

数据字段、图片角色和标签语义读取
[AVI 数据契约](references/data-contract.md)。评测输出和主指标读取
[评测口径](references/evaluation.md)。不要在这个阶段清洗数据、运行模型或提交训练。

需求齐备后，把请求写到当前会话的：

```text
public_data/project_request.json
```

文件必须符合：

```json
{
  "schema_version": 1,
  "name": "PCB AVI 单轮 SFT",
  "description": "基于缺陷图、差分图和 GT 图训练 AVI 质检模型",
  "skill": "avi-single-turn",
  "requirements": [
    {
      "key": "dataset_uri",
      "label": "数据集位置",
      "value": "datasets/avi/train.jsonl",
      "status": "confirmed"
    }
  ]
}
```

`requirements` 必须包含上面的全部 key。文件只描述需求，不包含 Graph、节点、
`project_id`、`graph_id`、Run ID、本机路径、凭据或签名 URL。

写完后向用户汇总数据、模型、训练方式和评估门禁，并请用户明确确认创建 Project。
用户确认前不得声称 Project 或 Graph 已存在。

## Graph 模板

Project 创建成功后，平台读取
[Graph 模板](references/graph-template.md)，生成固定四节点：

```text
dataset -> inference -> evaluator -> train
```

当前节点只注册 `contract_probe` 执行器。它可以生成一条 Run 和验证执行契约的
`stage_contract_probe` Artifact，但不执行真实数据、推理、评估或训练。可以说明执行
链路已打通，但不得声称某个阶段已经产出业务结果。具体阶段执行器接入后，再由平台状态
接口报告 `running`、`succeeded` 或 `failed`。

Dataset 阶段通过服务端提供的 Project/Stage 上下文启动后，必须读取并遵循通用的
`data-processing` Skill：检查真实 Storage 数据、选择数据处理范式、编写 Pipeline、
先跑小样并等待用户确认，再提交全量验证任务。Graph 只保存阶段、依赖和候选执行规格，
不在本 Skill 中复制通用数据处理 SOP。

## 交付边界

- 不复用或修改 Pyromind Agent 的对话上下文。
- 不在通用 data-processing、sandbox 或 workflow Skill 中加入 AVI 分支。
- 不把 demo 的在线 feeder、Kafka、推理热切换或轮询逻辑塞进本阶段。
- 用户未确认 Project 时，停留在需求和方案讨论。
- Graph、Project 和后续 Stage Run 的 ID 只能由服务端生成。
