# 节点契约速查

这里只记录生成训练/评测工作流所需的参数和端口。先按本表生成，再以
`platform-contract-overrides.md` 和 `validate_workflow_dsl` 纠正漂移契约。

## 数据、模型与路径

| NodeType | 必填输入 | 输出 |
|---|---|---|
| PathJoinNode | base_path, subpath | joined_path |
| LoadDataset | source_dir | dataset_path |
| CloneAndCacheDataset | dataset, target_path | dataset_path |
| DownloadAndCacheDataset | dataset_name, cache_dir, download_source | dataset_path |
| CloneAndCacheModel | model, target_path | model_path |
| DownloadAndCacheModel | modelname, cache_dir, download_source | model_path |

`download_source` 只允许 `huggingface`、`modelscope`。Storage 路径先用
`PathJoinNode(base_path="/workspace/", subpath=<relative>)`，再传给 `LoadDataset`。

默认推荐基模：

- 文本：`Qwen/Qwen3-0.6B`、`Qwen/Qwen3-1.7B`、`Qwen/Qwen3-4B`
- 多模态：`Qwen/Qwen3-VL-2B-Instruct`、`Qwen/Qwen3-VL-4B-Instruct`

推荐列表不是平台动态枚举的替代品。用户指定其他开源模型时用 `DownloadAndCacheModel`；若
Clone 对推荐模型校验失败，也改用 Download，不要换成用户没选的模型。

## 配置 Builder

| NodeType | 必填输入 | 常用可选输入 | 输出 |
|---|---|---|---|
| DatasetConfigBuilderTextNode | — | system_prompt_field, user_prompt_field, assistant_response_field, rejected_field | dataset_kind_config |
| DatasetConfigBuilderMessageNode | messages_field | rejected_field | dataset_kind_config |
| DatasetConfigBuilderVisionNode | image_field | system_prompt_field, user_prompt_field, assistant_response_field, rejected_field | dataset_kind_config |
| DatasetExtraConfigBuilderNode | — | train_max_samples, val_max_samples, sft/dpo/grpo_collator_entry, max_seq_length | dataset_extra_config |
| DatasetConfigBuilderNode | train_data_path | val_data_path, dataset_kind_config, dataset_extra_config | dataset_config |
| ModelConfigBuilderNode | model_path | model_type | model_config |
| LoraConfigBuilderNode | — | lora_rank, lora_dropout, target_modules, exclude_modules | lora_config |
| TrainingConfigBuilderNode | — | num_epochs, batch_size, grad_accum, learning_rate, lr_scheduler_type, logging_steps, save_steps, save_total_limit, eval_steps, seed, resume_from_checkpoint, max_grad_norm | training_config |
| AccelerateConfigBuilderNode | — | zero_stage | accelerate_config |
| WandbConfigBuilderNode | wandb_api_key, wandb_project | wandb_name | wandb_config |
| GRPOTrainingExtraConfigBuilderNode | — | max_steps, num_generations, max_prompt_length, max_completion_length, temperature, enable_chord, enable_hint | grpo_extra_config |

关键默认值：

- Message `messages_field="messages"`；Text/Vision 的 assistant 默认 `gt`，rejected 默认
  `rejected_answer`；必须以 preview 的真实字段为准。
- Dataset Extra：`max_seq_length=4096`；SFT/DPO/GRPO collator 分别为
  `train.sft_collator:make_collate_fn`、`train.dpo_collator:make_collate_fn`、
  `train.data.default_vision_grpo_collate:create_grpo_collate_fn`。
- `model_type`：`auto`、`qwen3vl`、`qwen3.5`。
- 单卡 LoRA 显式使用 `zero_stage=0`。
- scheduler：`linear`、`cosine`、`cosine_with_restarts`、`polynomial`、`constant`、
  `constant_with_warmup`。

## Reward 与 Metrics

| NodeType | 必填输入 | 可选输入 | 输出 |
|---|---|---|---|
| MetricsConfigBuilderNode | entry, name | — | metrics_config |
| MetricsConfigBuilderCustomNode | entry, name | — | metrics_config |
| RewardItemBuilderNode | entry, name | kwargs, weight | reward_item |
| RewardItemBuilderCustomNode | entry, name | kwargs, weight | reward_item |
| RewardConfigBuilderNode | — | reward_item_1...reward_item_5, normalize | reward_config |

- 内置 Metric：`compute_gsm8k`、`compute_accuracy`、`compute_bleu`、`compute_rouge_l`。
- 内置 Reward：`geometry_vqa_thinking_reward`、`geometry_vqa_answer_reward`。
- 内置项填裸函数名；其他入口使用 Custom 节点和 `<storage_path>:<function>`。

## 训练、合并、推理与评测

| NodeType | 必填输入 | 常用可选输入 | 输出 |
|---|---|---|---|
| ModelTrainSFTNode | output_path, dataset_config, training_config, model_config, accelerate_config, gpu_count, gpu_product | lora_config, wandb_config, thinking_as_input_ratio | model_output_path |
| ModelTrainDPONode | 同 SFT | lora_config, wandb_config, thinking_as_input_ratio | model_output_path |
| ModelTrainGRPONode | output_path, dataset_config, training_config, model_config, reward_config, accelerate_config, gpu_count, gpu_product | grpo_extra_config, lora_config, wandb_config, thinking_as_input_ratio | model_output_path |
| ModelMergeLoraNode | lora_path, output_path, model_path, gpu_count, gpu_product | model_type | merged_model_path |
| VLLMInference | model_path, port, gpu_count, gpu_product | environment, max_model_len | endpoint |
| ModelEvalApiNode | endpoint, output_path, dataset_config, metrics_config | endpoint_api_key, endpoint_model, max_samples, max_tokens, temperature | benchmark_output_path |

训练和推理 `gpu_count` 为 1～8；Merge 固定 1。GPU 枚举按覆盖层选，不从推理节点反推训练
节点。`WandbConfigBuilderNode.wandb_api_key` 填 Secret 名，不填 Secret 值。

## 其他边界

- `workflow.py` 是声明式 Python DSL，不能按普通 Python 本地执行。
- 当前生成能力只覆盖 Benchmark、SFT、DPO、GRPO、Merge、Inference/Eval；不凭空生成 OPD。
- DatasetValidator、DatasetToJsonl 和 DataPreprocess 系列不是训练生成默认阶段；格式不合规时
  停止并要求用户重新提供数据，而不是自动插入清洗节点。
