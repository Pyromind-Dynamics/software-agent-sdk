---
name: frozen-reward-evaluation
description: 已激活的 rubric-RL Project 使用冻结 Reward 执行 Init 基线、候选重评分与独立测试。
---

通过 `get_project_reward` 核对冻结版本。评测仍走 `df_submit_evaluation`，提交端绑定该版本，worker 校验包摘要后调用 `reward.py`。

- Init 基线用 `operation="inference"`。评测配置写 `mode: "agent_rubric"`、模型来源和生成设置，省略 `rubrics`。Dataset Mapping 对应真实的输入和稳定 case ID，可以省略 `reference_field`。推理使用冻结任务提示和输入消息。
- Eval 评测指定训练产物时，用 `operation="inference"` 加载该产物，在计划指定的数据集上按冻结 Reward 评分。对已有训练预测评分选模时用 `operation="score"`，提供源清单和预测文件摘要，省略 `scoring_spec_path`、`scoring_script_path`。选模只使用验证集。
- 独立测试用 `operation="inference"`，传 `selection_run_id`，使用被冻结的测试来源和验证集选定的 checkpoint。
- 交付逐条分数及其聚合，核对评测产物中的 `reward_manifest_sha256` 与 Project 版本一致。缺文件或摘要不一致时报告执行错误。
