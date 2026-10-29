---
name: avi-single-turn
description: >-
  PCB AVI 图片质检单轮 SFT/DPO 场景入口。用于根据缺陷图、差分图和 GT 图，
  澄清数据、推理、评估与训练需求，生成可确认的 Project 创建请求，并套用
  Dataset -> Inference -> Evaluator -> Train 四阶段 Graph 模板。
---

# PCB AVI 单轮训练

这个 Skill 负责把用户的 AVI 目标整理成可创建 Project 的完整需求，不直接执行
节点任务。Project 创建后，平台按本 Skill 的模板生成
`Dataset -> Inference -> Evaluator -> Train` 四节点 DAG。

## 何时使用

- 用户要搭建 PCB AVI/PCB 图片质检的 SFT、DPO、评测或迭代流程。
- 用户给出缺陷图、差分图、GT 图、pass/fail/defect 标签或相关训练数据。
- 用户要把现有 AVI 脚本、demo 或 Studio 工作流转成 EchoMind Project。

## 澄清门禁

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

## Project 创建请求

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

当前节点只注册 `shell` 执行器。可以说明四阶段已进入本地 Graph，但不得声称某个阶段
已经真实运行、产生模型或完成训练。具体阶段执行器接入后，再由平台状态接口报告
`running`、`succeeded` 或 `failed`。

## 交付边界

- 不复用或修改 Pyromind Agent 的对话上下文。
- 不在通用 data-processing、sandbox 或 workflow Skill 中加入 AVI 分支。
- 不把 demo 的在线 feeder、Kafka、推理热切换或轮询逻辑塞进本阶段。
- 用户未确认 Project 时，停留在需求和方案讨论。
- Graph、Project 和后续 Stage Run 的 ID 只能由服务端生成。
