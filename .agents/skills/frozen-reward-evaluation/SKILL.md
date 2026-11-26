---
name: frozen-reward-evaluation
description: 已激活的 rubric-RL Project 使用冻结 Reward 执行 Init 基线、候选重评分与独立测试。
---

通过 `get_project_reward` 核对冻结版本。按 [inference-evaluation](../inference-evaluation/SKILL.md)
保留预测、评分、报告与冻结版本的准备契约。SDK 解释器已部署轻量准备函数：
使用 `pyromind_common.platform.evaluation_core.EvaluationOptions` 校验业务参数，
将已授权的 Reward 快照作为 `frozen_reward` 传给
`pyromind_common.platform.evaluation.stage_evaluation`，该步骤只冻结输入和 DSL，不发 HTTP。
把 `staged.workflow_path` 转成工作区相对路径，与 `staged.name` 一起交给保留的
`validate_workflow_dsl`；从工具 content 取 `validated_workflow_path`，用 `Path` 读取
已保存的实际图，以当前 Project task/attempt 关联经普通 `submit_workflow` 函数调用 SDK 创建一次。
出现 `artifact_unavailable` 时即使 valid=true 也停止，不重构或重写图绕过。
SDK Key 不能被假定为校验 Cookie 的
替代身份；只有本来就持有独立已鉴权 validator 的调用方才用 `prepare_evaluation` 包装。
该路径不调用旧评测提交 Tool；worker 校验包摘要后调用 `reward.py`。

- Init 基线用 `operation="inference"`。评测配置写 `mode: "agent_rubric"`、模型来源和生成设置，省略 `rubrics`。Dataset Mapping 对应真实的输入和稳定 case ID，可以省略 `reference_field`。推理使用冻结任务提示和输入消息。
- Eval 评测指定训练产物时，用 `operation="inference"` 加载该产物，在计划指定的数据集上按冻结 Reward 评分。对已有训练预测评分选模时用 `operation="score"`，提供源清单和预测文件摘要，省略 `scoring_spec_path`、`scoring_script_path`。选模只使用验证集。
- 独立测试用 `operation="inference"`，传 `selection_run_id`。从已授权 Project 的版本、
  训练产物和选模报告取得证据，构造 `evaluation_core.ValidationSelection` 作为准备函数的
  `selection`；其 train/validation/test 为 `FrozenDataset(path=..., sha256=...)`，
  model 为准确 checkpoint-n 的 `ValidationCheckpoint(path=..., kind=..., base_model=...)`，
  LoRA 必须保留验证时的基模。selection 的 `frozen_reward` 与准备函数传入的 Reward 快照一致。
- 准备函数校验选定模型、测试来源、与 train/validation 的路径及摘要差异，最终 test 不设
  `generation.limit`。证据冻结为 validation-selection.json，resume 读取原文件，不重新选模。
  这些一致性检查不证明 Project 归属，不以自报 task_id 替代授权证据。
- 交付逐条分数及其聚合，核对评测产物中的 `reward_manifest_sha256` 与 Project 版本一致。缺文件或摘要不一致时报告执行错误。

SDK 客户端设 `max_retries=0`，提交结果未确认时等回调或核对平台，不重复创建。
保存真实 task_id、run_id 和 staging.storage；Project 元数据绑定失败只补绑定，
报告、指标及父 checkpoint 按现有 Project 入口记录，普通准备函数不自动建立业务记录。
