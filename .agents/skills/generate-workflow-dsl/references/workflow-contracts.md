# 工作流阶段与节点契约

本文件是阶段拓扑、节点签名、端口和平台覆盖项的单一事实源。先按这里生成，再以
`validate_workflow_dsl` 的实时结果纠正漂移契约。不要为未列出的组合方式猜字段或连线。

章节：阶段目录｜数据与模型入口｜数据与模型配置｜训练配置与阶段｜Reward 与 Metric｜
Merge、推理与评测｜平台边界。

## 阶段目录

| 目标 | 最小拓扑 | 最终产物 |
|---|---|---|
| Benchmark | 数据配置 → 模型入口 → VLLM → Metric → Eval | `benchmark_output_path` |
| SFT | 数据配置 → 模型入口 → SFT | `model_output_path` |
| DPO | 偏好数据配置 → 模型入口 → DPO | `model_output_path` |
| GRPO | 可验证数据配置 → 模型入口 → Reward → GRPO | `model_output_path` |
| LoRA 训后评测 | 训练 → Merge → VLLM → Metric → Eval | `benchmark_output_path` |
| Full 训后评测 | 训练 → VLLM → Metric → Eval | `benchmark_output_path` |

默认训练选 SFT；有 chosen/rejected 时选 DPO；有可程序化验证答案或 reward 时选 GRPO。
明确要求 Benchmark 时不添加训练节点。训练前后比较必须复用同一数据切分、字段映射、Metric
和 `max_samples`。

## 数据与模型入口

数据来源选择和 Storage 路径规范化查 `data-routing.md`；本节只定义节点签名。

| NodeType | 必填输入 | 输出 |
|---|---|---|
| PathJoinNode | base_path, subpath | `joined_path` |
| LoadDataset | source_dir | `dataset_path` |
| CloneAndCacheDataset | dataset, target_path（默认 `/workspace/datasets/`） | `dataset_path` |
| DownloadAndCacheDataset | dataset_name, cache_dir, download_source | `dataset_path` |
| CloneAndCacheModel | model, target_path（使用 `/workspace/models/`） | `model_path` |
| DownloadAndCacheModel | modelname, cache_dir, download_source | `model_path` |

`download_source` 只允许 `huggingface`、`modelscope`。主 Skill 列出的推荐模型走 Clone；其他
开源模型走 Download。两者都必须通过 `model_path` 连接下游，禁止重复写死缓存路径。

## 数据与模型配置

| NodeType | 必填输入 | 常用可选输入 | 输出 |
|---|---|---|---|
| DatasetConfigBuilderTextNode | — | system_prompt_field, user_prompt_field, assistant_response_field, rejected_field | `dataset_kind_config` |
| DatasetConfigBuilderMessageNode | messages_field | rejected_field | `dataset_kind_config` |
| DatasetConfigBuilderVisionNode | image_field | system_prompt_field, user_prompt_field, assistant_response_field, rejected_field | `dataset_kind_config` |
| DatasetExtraConfigBuilderNode | — | train_max_samples, val_max_samples, sft/dpo/grpo_collator_entry, max_seq_length | `dataset_extra_config` |
| DatasetConfigBuilderNode | train_data_path | val_data_path, dataset_kind_config, dataset_extra_config | `dataset_config` |
| ModelConfigBuilderNode | model_path | model_type | `model_config` |

Message 默认字段为 `messages`；Text/Vision 的 assistant 默认 `gt`，rejected 默认
`rejected_answer`，但必须以 preview 的真实字段为准。`model_type` 只用 `auto`、`qwen3vl`、
`qwen3.5`。

## 训练配置与阶段

| Builder | 常用输入 | 输出 |
|---|---|---|
| LoraConfigBuilderNode | lora_rank, lora_dropout, target_modules, exclude_modules | `lora_config` |
| TrainingConfigBuilderNode | num_epochs, batch_size, grad_accum, learning_rate, lr_scheduler_type, logging/save/eval steps, seed, max_grad_norm | `training_config` |
| AccelerateConfigBuilderNode | zero_stage | `accelerate_config` |
| WandbConfigBuilderNode | wandb_api_key, wandb_project, wandb_name | `wandb_config` |
| GRPOTrainingExtraConfigBuilderNode | max_steps, num_generations, prompt/completion length, temperature, enable_chord, enable_hint | `grpo_extra_config` |

| 训练节点 | 必填输入 | 常用可选输入 | 输出 |
|---|---|---|---|
| ModelTrainSFTNode | output_path, dataset_config, training_config, model_config, accelerate_config, gpu_count, gpu_product | lora_config, wandb_config, thinking_as_input_ratio | `model_output_path` |
| ModelTrainDPONode | 同 SFT | lora_config, wandb_config, thinking_as_input_ratio | `model_output_path` |
| ModelTrainGRPONode | SFT 必填项 + reward_config | grpo_extra_config, lora_config, wandb_config, thinking_as_input_ratio | `model_output_path` |

单卡 LoRA 显式使用 `zero_stage=0`。scheduler 只用 `linear`、`cosine`、
`cosine_with_restarts`、`polynomial`、`constant`、`constant_with_warmup`。数值参数整组决策查
`parameter-decision.md`。

## Reward 与 Metric

| NodeType | 必填输入 | 常用可选输入 | 输出 |
|---|---|---|---|
| MetricsConfigBuilderNode | entry, name | — | `metrics_config` |
| MetricsConfigBuilderCustomNode | entry, name | — | `metrics_config` |
| RewardItemBuilderNode | entry, name | kwargs, weight | `reward_item` |
| RewardItemBuilderCustomNode | entry, name | kwargs, weight | `reward_item` |
| RewardConfigBuilderNode | — | reward_item_1...reward_item_5, normalize | `reward_config` |

内置 Metric 的 `entry` 填裸函数名：数学答案用 `compute_gsm8k`，精确匹配用
`compute_accuracy`，翻译或受约束生成用 `compute_bleu`，摘要或长文本生成用
`compute_rouge_l`。每个 Metrics Builder 只输出一个 `metrics_config`，Eval 只接一个该端口；
未声明的多指标合并方式不得猜测。

内置 Reward 仅有 `geometry_vqa_thinking_reward`、`geometry_vqa_answer_reward`。其他业务指标或
Reward 按 `custom-python-assets.md` 生成、上传并回填 Custom 节点。

## Merge、推理与评测

| NodeType | 必填输入 | 常用可选输入 | 输出 |
|---|---|---|---|
| ModelMergeLoraNode | lora_path, output_path, model_path, gpu_count, gpu_product | model_type | `merged_model_path` |
| VLLMInference | model_path, port, gpu_count, gpu_product | environment, max_model_len | `endpoint` |
| ModelEvalApiNode | endpoint, output_path, dataset_config, metrics_config | endpoint_api_key, endpoint_model, max_samples, max_tokens, temperature | `benchmark_output_path` |

| 场景 | 必须绑定的模型路径 |
|---|---|
| LoRA Merge | `lora_path=training.model_output_path`，`model_path=base_model.model_path` |
| SFT → Merge → GRPO | 为 `merge.merged_model_path` 新建 Model Config，再传给 GRPO |
| 基模 Benchmark | `VLLMInference.model_path=base_model.model_path` |
| LoRA 训后评测 | `VLLMInference.model_path=merge.merged_model_path` |
| Full 训后评测 | `VLLMInference.model_path=training.model_output_path` |

`ModelEvalApiNode.endpoint` 必须绑定 `vllm.endpoint`。VLLM 默认 `port=3000`；Eval 默认
`max_samples=0`（全部）、`max_tokens=256`、`temperature=0.01`，用户已有有效值时不得覆盖。

## 平台边界

- 训练和 LoRA Merge 的 `gpu_product` 当前允许 `NVIDIA-H200`、
  `NVIDIA-H100-80GB-HBM3`，默认后者；Merge 固定单卡。
- VLLM 在 `us-west-1` 使用 `NVIDIA-H100-NVL` 或 `NVIDIA-L40S`，在 `us-west-2` 使用
  `NVIDIA-H200` 或 `NVIDIA-H100-80GB-HBM3`；不要从推理枚举反推训练枚举。
- 训练和推理 `gpu_count` 为 1～8。
- `WandbConfigBuilderNode.wandb_api_key` 只填 Secret 名，不填 Secret 值。
- DatasetValidator、DatasetToJsonl、DataPreprocess 不是训练生成默认阶段；格式不合规时停止，
  不自动插入清洗节点。
