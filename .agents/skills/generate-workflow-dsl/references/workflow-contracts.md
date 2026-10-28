# 工作流阶段与节点契约

本文件覆盖常用训练与评测阶段的拓扑和节点契约。未列出的节点先从 `knowledge/index.md`
定位并读取原文，再以 `validate_workflow_dsl` 的实时结果纠正漂移契约；不要猜字段或连线。

下文 Benchmark、训后评测及 Metric/Eval 契约仅供检查已有工作流。
新的独立模型效果评测统一交给 inference-evaluation，不按这些旧拓扑生成评测阶段。

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

训练阶段按主 Skill 中的数据与训练决策选择；本文件提供对应拓扑。
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

`download_source` 只允许 `huggingface`、`modelscope`。
`Qwen/Qwen3-0.6B`、`Qwen/Qwen3-1.7B`、`Qwen/Qwen3-4B`、
`Qwen/Qwen3-VL-2B-Instruct`、`Qwen/Qwen3-VL-4B-Instruct` 使用 Clone，
`target_path="/workspace/models/"`。其他开源模型默认 Download，
`cache_dir="/workspace/models/<org>/<model>"`、`download_source="huggingface"`，用户指定来源时遵从。
`qwen3.5-2b` 对应 `Qwen/Qwen3.5-2B`，缓存目录为 `/workspace/models/Qwen/Qwen3.5-2B`。
两者都通过输出 `model_path` 连接下游，不重复写死缓存路径。

已确认的文件可用 `PathJoinNode.joined_path` 直接连接数据配置；仅需目录检查时使用
`LoadDataset`，其 `source_dir` 接目录，具体文件从 `dataset_path` 拼接。路径示例见 `data-routing.md`。

## 数据与模型配置

| NodeType | 必填输入 | 常用可选输入 | 输出 |
|---|---|---|---|
| DatasetConfigBuilderTextNode | — | system_prompt_field, user_prompt_field, assistant_response_field, rejected_field | `dataset_kind_config` |
| DatasetConfigBuilderMessageNode | messages_field | rejected_field | `dataset_kind_config` |
| DatasetConfigBuilderVisionNode | image_field | system_prompt_field, user_prompt_field, assistant_response_field, rejected_field | `dataset_kind_config` |
| DatasetExtraConfigBuilderNode | — | train_max_samples, val_max_samples, sft/dpo/grpo_collator_entry, max_seq_length | `dataset_extra_config` |
| DatasetConfigBuilderNode | train_data_path | val_data_path, dataset_kind_config, dataset_extra_config | `dataset_config` |
| ModelConfigBuilderNode | model_path | model_type | `model_config` |

按实际数据选择 Builder，不要求统一转换为 messages：

| 数据形态 | 配置 |
|---|---|
| 文本输入与监督答案分列 | Text，映射实际 system/user/assistant 字段 |
| 已有对话数组 | Message，`messages_field` 指向实际对话列 |
| 图片与文本分列 | Vision，映射图片列和文本列 |
| 偏好对 | 对应 Builder 的 assistant/chosen 与 `rejected_field`；训练阶段为 DPO |
| prompt 加可验证信号 | 对应 Builder 加 Reward，按主 Skill 选择训练阶段 |

Text 的 assistant 默认 `gt`，rejected 默认 `rejected_answer`；Message 的 `messages_field` 必填，
rejected 默认 `rejected_messages`；Vision 的 `image_field` 必填。字段名以真实样本或已有有效映射为准。
已有 messages 保留原格式，媒体内容按实际节点契约配置，不套用平铺字段默认值。

平铺 Vision 示例（图片路径以训练加载器的解析基准为准）：

```json
{"images":["images/a.png"],"system_prompt":"你是图像质检专家。","user_prompt":"判断图中缺陷。","gt":"存在划痕。"}
```

```python
dataset_kind = DatasetConfigBuilderVisionNode(
    id="6",
    image_field="images",
    system_prompt_field="system_prompt",
    user_prompt_field="user_prompt",
    assistant_response_field="gt",
)
```

`model_type` 只用 `auto`、`qwen3vl`、`qwen3.5`。`max_seq_length` 优先遵从用户要求与已有有效配置，
其余结合模型、数据和资源选择。需要评估截断或容量时，多模态长度包括媒体展开和模板 token；
估算应说明依据，不能把字符数当作实测 token 数。生成 DSL 不要求先运行实际 processor/collator。

## 训练配置与阶段

| Builder | 常用输入 | 输出 |
|---|---|---|
| LoraConfigBuilderNode | lora_rank, lora_dropout, target_modules, exclude_modules | `lora_config` |
| TrainingConfigBuilderNode | num_epochs, batch_size, grad_accum, learning_rate, lr_scheduler_type, logging/save/eval steps, save_total_limit, seed, max_grad_norm | `training_config` |
| AccelerateConfigBuilderNode | zero_stage | `accelerate_config` |
| WandbConfigBuilderNode | wandb_api_key, wandb_project, wandb_name | `wandb_config` |
| GRPOTrainingExtraConfigBuilderNode | max_steps, num_generations, max_prompt_length, max_completion_length, temperature, enable_chord, enable_hint | `grpo_extra_config` |

`TrainingConfigBuilderNode.save_total_limit` 显式填写。逐 checkpoint 选模时，结合训练数据量、
epoch、有效 batch 或明确的 `max_steps` 估算总优化步数，按
`ceil(total_steps / save_steps) + 1` 预留保留数量，避免平台默认 `3` 清理候选。
无法估算且未指定保留策略时使用 `300`，说明超过保留数量仍可能丢失候选。
逐 checkpoint 独立评测只依赖已保存模型，`save_steps` 无需与 `eval_steps` 对齐。
保留数量应覆盖本轮所有候选；用户明确指定较小保留数量时说明可能产生覆盖缺口。

SFT/DPO/GRPO 的 `wandb_config` 平台当前按必填校验：必须连接 `WandbConfigBuilderNode.wandb_config`。
`WandbConfigBuilderNode.wandb_api_key` 只填 Secret 名，不填明文 Key；`wandb_project` 必填，
`wandb_name` 可选。`accelerate_config` 同样按必填连接，缺省会被校验拦截。

`GRPOTrainingExtraConfigBuilderNode` 平台默认：`max_steps=200`、`num_generations=8`、
`temperature=0.7`、`max_prompt_length`/`max_completion_length=2048`；用户未指定时可不写这些
端口。`max_prompt_length + max_completion_length` 不应超过 `max_seq_length`。

| 训练节点 | 必填输入 | 常用可选输入 | 输出 |
|---|---|---|---|
| ModelTrainSFTNode | output_path, dataset_config, training_config, model_config, accelerate_config, wandb_config, gpu_count, gpu_product | lora_config, thinking_as_input_ratio | `model_output_path` |
| ModelTrainDPONode | 同 SFT | lora_config, thinking_as_input_ratio | `model_output_path` |
| ModelTrainGRPONode | SFT 必填项 + reward_config | grpo_extra_config, lora_config, thinking_as_input_ratio | `model_output_path` |

单卡 LoRA 显式使用 `zero_stage=0`。scheduler 只用 `linear`、`cosine`、
`cosine_with_restarts`、`polynomial`、`constant`、`constant_with_warmup`。

## Reward 与 Metric

| NodeType | 必填输入 | 常用可选输入 | 输出 |
|---|---|---|---|
| MetricsConfigBuilderNode | entry, name | — | `metrics_config` |
| MetricsConfigBuilderCustomNode | entry, name | — | `metrics_config` |
| RewardItemBuilderNode | entry, name | kwargs, weight | `reward_item` |
| RewardItemBuilderCustomNode | entry, name | kwargs, weight | `reward_item` |
| RewardConfigBuilderNode | — | reward_item_1...reward_item_5, normalize | `reward_config` |

内置 Metric 的 `entry` 填 `module.py:function_name` 形式：数学答案用
`examples/eval_metrics_common.py:compute_gsm8k`，精确匹配用
`examples/eval_metrics_common.py:compute_accuracy`，翻译或受约束生成用
`examples/eval_metrics_common.py:compute_bleu`，摘要或长文本生成用
`examples/eval_metrics_common.py:compute_rouge_l`。每个 Metrics Builder 只输出一个
`metrics_config`，Eval 只接一个该端口；未声明的多指标合并方式不得猜测。

内置 Reward 仅有 `examples/geometry_vqa/reward.py:geometry_vqa_thinking_reward`、
`examples/geometry_vqa/reward.py:geometry_vqa_answer_reward`。其他业务指标或
Reward 按 `custom-python-assets.md` 生成、上传并回填 Custom 节点。

## Merge、推理与评测

| NodeType | 必填输入 | 常用可选输入 | 输出 |
|---|---|---|---|
| ModelMergeLoraNode | lora_path, output_path, model_path, gpu_count, gpu_product | model_type | `merged_model_path` |
| VLLMInference | model_path, port, gpu_count, gpu_product | environment, max_model_len | `endpoint` |
| ModelEvalApiNode | endpoint, endpoint_api_key, endpoint_model, output_path, dataset_config, metrics_config | max_samples, max_tokens, temperature | `benchmark_output_path` |

| 场景 | 必须绑定的模型路径 |
|---|---|
| LoRA Merge | `lora_path=training.model_output_path`，`model_path=base_model.model_path` |
| SFT → Merge → GRPO | 为 `merge.merged_model_path` 新建 Model Config，再传给 GRPO |
| 基模 Benchmark | `VLLMInference.model_path=base_model.model_path` |
| LoRA 训后评测 | `VLLMInference.model_path=merge.merged_model_path` |
| Full 训后评测 | `VLLMInference.model_path=training.model_output_path` |

`ModelEvalApiNode.endpoint` 必须绑定 `vllm.endpoint`；`endpoint_api_key`、`endpoint_model`
必填，默认分别为 `empty`、`default`（对接自建 VLLM 端点时通常保持默认）。VLLM 默认
`port=3000`；Eval 默认 `max_samples=0`（全部）、`max_tokens=256`、`temperature=0.01`，
用户已有有效值时不得覆盖。

## 平台边界

- 训练节点（SFT/DPO/GRPO）的 `accelerate_config`、`gpu_count`、`gpu_product` 端口必填。训练
  `gpu_product` 此前平台实测仅放行 `NVIDIA-H100-NVL`、`NVIDIA-L40S`，节点文档现统一标注
  `NVIDIA-H200`、`NVIDIA-H100-80GB-HBM3`；两者冲突时以 `validate_workflow_dsl` 实测为准。
- Merge LoRA 的 `model_path`、`gpu_count`、`gpu_product` 同为必填；`gpu_product` 文档枚举同推理
  （`NVIDIA-H200`、`NVIDIA-H100-80GB-HBM3`），默认单卡合并。
- 推理（VLLM/Inference）`gpu_product` 枚举为 `NVIDIA-H200`、`NVIDIA-H100-80GB-HBM3`，以部署
  环境实际可用显卡为准；不要从推理枚举反推训练枚举。
- 训练和推理 `gpu_count` 为 1～8（推理节点端口标 `gpu_count`、`gpu_product` 均必填）。
- `WandbConfigBuilderNode.wandb_api_key` 只填 Secret 名，不填 Secret 值。
- DatasetValidator、DatasetToJsonl、DataPreprocess 不是训练生成默认阶段；格式不合规时停止，
  不自动插入清洗节点。
- GPU 枚举随平台迭代漂移；生成后按 `validate_workflow_dsl` 的 `ENUM_VALUE_INVALID` 修正当前 DSL。
