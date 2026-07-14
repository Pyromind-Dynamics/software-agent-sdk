# 平台契约覆盖层

本文件只记录运行日志和平台校验器已确认、但同步知识库可能尚未更新的契约。冲突时遵循：
实时 `validate_workflow_dsl` 结果 > 本文件 > 同步知识库与示例。

- 训练与 LoRA 合并节点的 `gpu_product`：`NVIDIA-H200`、
  `NVIDIA-H100-80GB-HBM3`；默认示例使用后者。不要从推理节点的更宽枚举反推训练节点。
- `MetricsConfigBuilderNode.entry` 的内置枚举：`compute_gsm8k`、`compute_accuracy`、
  `compute_bleu`、`compute_rouge_l`，均为裸函数名。自定义 Metrics 才使用 `<py路径>:<函数名>`。
- `RewardItemBuilderNode.entry` 的系统枚举：`geometry_vqa_thinking_reward`、
  `geometry_vqa_answer_reward`，均为裸函数名。自定义 Reward 才使用 `<绝对路径>:<函数名>`。
- `WandbConfigBuilderNode.wandb_api_key` 填 Secret 名（如 `MY_WANDB_KEY`），不是明文 API Key；
  禁止把密钥值写入 DSL。
