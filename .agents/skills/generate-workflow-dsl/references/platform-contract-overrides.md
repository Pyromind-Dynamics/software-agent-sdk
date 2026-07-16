# 易漂移平台契约

只记录同步节点资料之外的产品策略或已验证差异。冲突优先级：

`validate_workflow_dsl` 实时结果 > 本文件 > `node-reference.md` > 模板值。

- 训练与 LoRA Merge 的 `gpu_product` 当前允许 `NVIDIA-H200`、
  `NVIDIA-H100-80GB-HBM3`；默认使用后者。不要从 VLLM 的枚举反推训练枚举。
- 产品默认推荐 Clone 基模：`Qwen/Qwen3-0.6B`、`Qwen/Qwen3-1.7B`、
  `Qwen/Qwen3-4B`、`Qwen/Qwen3-VL-2B-Instruct`、`Qwen/Qwen3-VL-4B-Instruct`。
  列表是推荐策略，不是动态平台白名单；其他模型或 Clone 校验失败时用 Download Model。
- 内置 Metric entry 仅填裸函数名：`compute_gsm8k`、`compute_accuracy`、
  `compute_bleu`、`compute_rouge_l`。
- 内置 Reward entry 仅填裸函数名：`geometry_vqa_thinking_reward`、
  `geometry_vqa_answer_reward`。
- 自定义 Metric/Reward 必须使用上传工具返回的 `<storage_path>:<function>`；不得使用
  Agent 本地文件路径或预先猜测 `/workspace/...`。
- `WandbConfigBuilderNode.wandb_api_key` 填 Secret 名，如 `MY_WANDB_KEY`，不得填真实值。
- Storage 相对路径先由 `PathJoinNode` 加 `/workspace/`，再由 `LoadDataset` 验证并透传。
