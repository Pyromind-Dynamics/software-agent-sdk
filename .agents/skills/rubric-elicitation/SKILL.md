---
name: rubric-elicitation
description: 在创建 Project 前，为需要人工定义质量标准的 RL 任务确认 checklist、校准 case 并冻结 Reward 包。
---

# Rubric 沟通

结合用户目标和数据选训练范式。完整监督轨迹通常适合 SFT，有可程序验证的标准答案时走规则 RL。需要根据多维质量标准判断输出时，用问答卡确认是否走 rubric-RL。尊重用户明确指定的范式。

## 定义和校准

- 确认任务描述、输入输出、好坏示例、不可接受的错误、关键术语、基座和目标数据集。
- 和用户选 3–5 条典型、边界及失败 case，标明校准和独立测试用途。优先导入基座预测。生成端只用任务描述和选定图片，case 说明只给评分端。
- 用 checklist 把质量标准写成可独立判断的条目，确认默认等权、例外权重、硬约束和总分规则。规则能准确判断的条目用 code，需要语义判断的条目用 LLM。只调用有宿主授权的模型，记录固定模型版本和判分 schema。
- 实际运行候选 `reward.py` 评分。用 `ask_user` 展示 case、模型输出、逐条得分及理由、总分，接受认可或文字纠正。记录 case、输入和产物摘要、逐条判分、原始反馈及后续处理。改 checklist 复用预测，改任务描述重生成，补 case 增量跑。修订保留旧记录，重起草清空本轮校准历史。独立测试集只用于最终评估。收到三条反馈后可以提示用户已具备冻结条件，是否结束仍由用户决定。

在 `public_data/rubric_work/` 保存草稿、case 索引、版本和反馈。生成复用键包含输入、任务描述、模型版本和生成配置，评分复用再加输出和评分包摘要。凭据、模型地址和签名 URL 由运行环境提供。

## 冻结和计划

候选子目录内准备：

- `task_prompt.md`。
- `rubric.json`，形如 `{"items":[{"criterion":"…","weight":1,"hard_constraint":false}],"aggregation":"…"}`。草稿只用示例中的字段，省略 ID，由 Runtime 根据条目文字生成。
- `reward.py`，提供 `parse_record(record)`、`score_case(case, response)` 和批量入口 `studio_reward`。`studio_reward` 按输入顺序批量调用同一个 `score_case`，预览须核对两者的逐条分数。输出不合规可记低分，缺输入、配置或运行失败应抛错。`parse_record` 返回字典，推理输入可没有 `response`。评分返回 0～1 的数值，或含 `overall_score` 和可选 `passed` 的字典。附件依赖仅使用包内文件。预览通过直接读取和执行源码运行，避免在冻结目录生成缓存。
- `calibration_review.json`，数组元素 `{"case_id":"…","score":0.8,"summary":"结果与用户反馈"}`。只收录真实校准证据。
- LLM 或混合判分需要 `calibration.json`，包含校准对话示例。模型与密钥通过环境配置，按固定 schema 验证响应，失败应明确报错。

调用 `request_reward_bundle_review`，传相对候选目录、`judge_kind`（`code`、`llm` 或 `hybrid`）和运行要求。Runtime 发布多文件快照并用现有问答卡展示内容。根据用户意见或工具报告的 JSON 字段修正候选文件。排查范围是候选目录。拿到 `approval_id` 后，用 `set_preparation_plan` 提交 `training_method="rubric_rl"` 和 `reward_approval_id`。Project 绑定同一冻结版本，计划文件无需复制包引用。
