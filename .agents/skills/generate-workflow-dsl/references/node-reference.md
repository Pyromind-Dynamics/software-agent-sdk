# 节点速查与常用参数

本文件按 `.cursor/project_files/node.json` 的 56 个启用节点更新，只保留生成训练、评测与
推理工作流时需要高频读取的契约。对已列出的常见节点，这里的必填/可选输入、输出端口、
枚举、默认值和常用拓扑足以生成第一版 DSL，不要再逐个读取节点文档。只有节点未收录、
所需字段/类型/范围缺失，或校验结果指出契约冲突时，才定向读取
`knowledge/nodes/<NodeType>/<NodeType>.md` 或
[platform-contract-overrides.md](platform-contract-overrides.md)。

`node.json` 中所有节点均为 `is_enabled=true`、`experimental=false`、`deprecated=false`。
DSL 还必须为每个节点提供唯一 `id`，这里不在每张表中重复列出。

## 节点契约速查

`—` 表示该组没有字段。

### 数据与模型资源

| NodeType | 必填输入 | 可选输入 | 输出端口 |
|----------|----------|----------|----------|
| CloneAndCacheDataset | dataset, target_path | — | dataset_path |
| DownloadAndCacheDataset | dataset_name, cache_dir, download_source | — | dataset_path |
| LoadDataset | source_dir | — | dataset_path |
| CloneAndCacheModel | model, target_path | — | model_path |
| DownloadAndCacheModel | modelname, cache_dir, download_source | — | model_path |
| LoadModel | model_path | — | model |
| SaveModel | model, save_path, model_name, save_format | — | save_path |
| PathJoinNode | base_path, subpath | environment | joined_path |

### 配置构建

| NodeType | 必填输入 | 可选输入 | 输出端口 |
|----------|----------|----------|----------|
| DatasetConfigBuilderTextNode | — | system_prompt_field, user_prompt_field, assistant_response_field, rejected_field | dataset_kind_config |
| DatasetConfigBuilderMessageNode | messages_field | rejected_field | dataset_kind_config |
| DatasetConfigBuilderVisionNode | image_field | system_prompt_field, user_prompt_field, assistant_response_field, rejected_field | dataset_kind_config |
| DatasetConfigBuilderNode | train_data_path | val_data_path, dataset_kind_config, dataset_extra_config | dataset_config |
| DatasetExtraConfigBuilderNode | — | train_max_samples, val_max_samples, sft_collator_entry, dpo_collator_entry, grpo_collator_entry, max_seq_length | dataset_extra_config |
| ModelConfigBuilderNode | model_path | model_type | model_config |
| LoraConfigBuilderNode | — | lora_rank, lora_dropout, target_modules, exclude_modules | lora_config |
| TrainingConfigBuilderNode | — | num_epochs, batch_size, grad_accum, learning_rate, lr_scheduler_type, logging_steps, save_steps, save_total_limit, eval_steps, seed, resume_from_checkpoint, max_grad_norm | training_config |
| AccelerateConfigBuilderNode | — | zero_stage | accelerate_config |
| WandbConfigBuilderNode | wandb_api_key, wandb_project | wandb_name | wandb_config |
| RewardItemBuilderNode | entry, name | kwargs, weight | reward_item |
| RewardItemBuilderCustomNode | entry, name | kwargs, weight | reward_item |
| RewardConfigBuilderNode | — | reward_item_1…reward_item_5, normalize | reward_config |
| GRPOTrainingExtraConfigBuilderNode | — | max_steps, num_generations, max_prompt_length, max_completion_length, temperature, enable_chord, enable_hint | grpo_extra_config |
| MetricsConfigBuilderNode | entry, name | — | metrics_config |
| MetricsConfigBuilderCustomNode | entry, name | — | metrics_config |

### 训练、评测与推理

| NodeType | 必填输入 | 可选输入 | 输出端口 |
|----------|----------|----------|----------|
| ModelTrainSFTNode | output_path, dataset_config, training_config, model_config, accelerate_config, gpu_count, gpu_product | lora_config, wandb_config, thinking_as_input_ratio | model_output_path |
| ModelTrainDPONode | output_path, dataset_config, training_config, model_config, accelerate_config, gpu_count, gpu_product | lora_config, wandb_config, thinking_as_input_ratio | model_output_path |
| ModelTrainGRPONode | output_path, dataset_config, training_config, model_config, reward_config, accelerate_config, gpu_count, gpu_product | grpo_extra_config, lora_config, wandb_config, thinking_as_input_ratio | model_output_path |
| ModelTrainOPDNode | output_path, dataset_config, training_config, model_config, teacher_model_config, accelerate_config, gpu_count, gpu_product | lmbda, beta, temperature, max_new_tokens, seq_kd, thinking_as_input_ratio, wandb_config | model_output_path |
| ModelMergeLoraNode | lora_path, output_path, model_path, gpu_count, gpu_product | model_type | merged_model_path |
| VLLMInference | model_path, port, gpu_count, gpu_product | environment, max_model_len | endpoint |
| Inference | model_path, port, gpu_count, gpu_product | environment | endpoint |
| ModelEvalApiNode | endpoint, endpoint_api_key, endpoint_model, output_path, dataset_config, metrics_config | max_samples, max_tokens, temperature | benchmark_output_path |
| TestLLMNode | endpoint, prompt, max_tokens, temperature | — | result |
| ContentPreview | content, file_format | — | content |
| PreviewFile | file_path | — | file_path |

`ModelTrainOPDNode` 已出现在快照中，但同一快照没有产生 `teacher_model_config` 的 Builder
节点。当前 generate-workflow-dsl 只生成 SFT/DPO/GRPO；不要凭空构造 OPD 配置。

### 校验与数据处理节点

这些节点存在于平台，但不是常规训练工作流骨架：

| NodeType | 用途 | 输出端口 |
|----------|------|----------|
| DatasetValidatorNode | 校验数据配置并生成预览 | metrics_yaml, output_preview_path |
| ModelValidateGRPODataloaderNode | 校验 GRPO dataloader/reward/rollout | metrics_json, output_preview_path |
| ModelValidateRewardNode | 校验奖励组合和聚合分数 | status, scores |
| AccelerateGpuNode | 验证 Accelerate GPU 启动 | gpu_info, execution_result, execution_output, message_output |
| DatasetToJsonlNode | 把数据集转为 JSONL | jsonl_path |
| DataPreprocessLabelNode | 数据标注 | result_output_path |
| DataPreprocessReasoningNode | 推理链生成 | reasoning_path |
| DataPreprocessPromptExpandNode | Prompt 扩写 | augmentation_path |
| DataPreprocessMergeNode | 合并 Prompt、推理与标签 | result_output_path |
| DataPreprocessSplitNode | 划分训练集和验证集 | train_output_path, val_output_path |

本 skill 假定训练数据已清洗，不要把上述数据处理节点自动插入 SFT/DPO/GRPO 工作流；只有
用户明确要求且切换到相应的数据清洗能力时才使用。

## 枚举与默认值

### GPU

平台校验确认训练节点和 `ModelMergeLoraNode` 的 `gpu_product` 允许
`NVIDIA-H200`、`NVIDIA-H100-80GB-HBM3`，常规模板使用后者。推理节点按集群选择：
`us-west-1` 使用 `NVIDIA-H100-NVL`、`NVIDIA-L40S`；`us-west-2` 使用上述 H200/H100-80GB
枚举。训练和推理节点的 `gpu_count` 范围是 1～8；`ModelMergeLoraNode.gpu_count` 固定为 1。

### Clone/Download 数据与模型

`CloneAndCacheDataset.dataset` 和 `CloneAndCacheModel.model` 都带有 `skip_enum=true`；
下面是快照提供的选择器候选，不应把它误写成封闭白名单。

训练工作流常用的 Clone 数据集候选：

- `Writer/omniact`、`openai/gsm8k`
- `pyromind/alpaca-gpt4-llm-demo`、`pyromind/geometry-vqa-vlm-demo`、
  `pyromind/self-cognition`

快照中的其他 Clone 数据集候选：

- `OpenGVLab/ScaleCUA-Data`
- `agibot-world/AgiBotWorld-Alpha-327`、`agibot-world/AgiBotWorld-Alpha-327-Extract`
- `agibot-world/AgiBotWorld-Alpha-CtrlWorld-327`、
  `agibot-world/AgiBotWorld-Alpha-Lerobot-327`、
  `agibot-world/AgiBotWorld-Alpha-Openpi-327`
- `cadene/droid_1.0.1`、`gui-360/gui-excel`、`gui-360/processed_data`
- `henryhe0123/PC-Agent-E`、`ritzzai/GUI-R1`
- `xlangai/aguvis-stage1`、`xlangai/aguvis-stage2`、`zonghanHZH/UGround-V1-8k`

Qwen 训练工作流常用的 Clone 模型候选：

- `Qwen/Qwen3-0.6B`、`Qwen/Qwen3-1.7B`、`Qwen/Qwen3-4B`、`Qwen/Qwen3-8B`
- `Qwen/Qwen3-14B`、`Qwen/Qwen3-32B`、`Qwen/Qwen3-32B-FP8`
- `Qwen/Qwen3-235B-A22B-FP8`
- `Qwen/Qwen3-VL-4B-Instruct`、`Qwen/Qwen3-VL-32B-Instruct`、
  `Qwen/Qwen3-VL-235B-A22B-Instruct`

快照中的其他 Clone 模型候选：

- `openai/clip-vit-base-patch32`、`openpi/big_vision`、`openpi/pi05_base`
- `pretrained/gui-ck900-lora`、`stabilityai/stable-video-diffusion-img2vid`

注意：快照中没有 `Qwen/Qwen3-VL-2B-Instruct`。

Download 节点的参数名不要与 Clone 节点混用：

| NodeType | 名称参数 | 缓存参数 | 来源参数 |
|----------|----------|----------|----------|
| DownloadAndCacheDataset | dataset_name | cache_dir（PATH） | download_source |
| DownloadAndCacheModel | modelname | cache_dir（STRING） | download_source |

两者的 `download_source` 都只允许 `huggingface`、`modelscope`，默认 `huggingface`。

### 配置 Builder

- `DatasetConfigBuilderTextNode` 的字段映射全部可选：`assistant_response_field` 默认 `gt`，
  `rejected_field` 默认 `rejected_answer`，system/user 默认空字符串。
- `DatasetConfigBuilderMessageNode.messages_field` 必填且默认 `messages`；`rejected_field`
  可选，默认 `rejected_messages`。
- `DatasetConfigBuilderVisionNode.image_field` 是唯一必填映射；assistant/rejected 默认分别为
  `gt`、`rejected_answer`，system/user 默认空字符串。
- `DatasetConfigBuilderNode` 只有 `train_data_path` 必填；验证路径、kind 和 extra 配置都可选。
- `DatasetExtraConfigBuilderNode` 的默认值：train/val max samples 均为 0，SFT/DPO/GRPO
  collator 分别为 `train.sft_collator:make_collate_fn`、
  `train.dpo_collator:make_collate_fn`、
  `train.data.default_vision_grpo_collate:create_grpo_collate_fn`，`max_seq_length=4096`。
- `ModelConfigBuilderNode.model_type` 允许 `auto`、`qwen3vl`、`qwen3.5`，默认 `auto`。
- `AccelerateConfigBuilderNode.zero_stage` 可选，默认 2。单卡 LoRA 如按平台覆盖层要求使用 0，
  应显式填写。
- `WandbConfigBuilderNode.wandb_api_key`、`wandb_project` 必填，`wandb_name` 可选。
- `ContentPreview.file_format` 允许 `txt`、`json`、`xml`、`python`，默认 `txt`。

`TrainingConfigBuilderNode` 的快照默认值：

| 参数 | 默认值 | 参数 | 默认值 |
|------|--------|------|--------|
| num_epochs | 2 | batch_size | 2 |
| grad_accum | 2 | learning_rate | 1e-4 |
| lr_scheduler_type | constant | logging_steps | 5 |
| save_steps | 500 | save_total_limit | 3 |
| eval_steps | 500 | seed | 42 |
| resume_from_checkpoint | 空字符串 | max_grad_norm | 2 |

`lr_scheduler_type` 允许 `linear`、`cosine`、`cosine_with_restarts`、`polynomial`、
`constant`、`constant_with_warmup`。自动配参时仍以
[parameter-decision.md](parameter-decision.md) 的决策链为准，不要机械照抄默认值。

### Reward、GRPO 与 Metrics

- `RewardItemBuilderNode.entry` 只允许 `geometry_vqa_thinking_reward`、
  `geometry_vqa_answer_reward`，默认前者；其他 entry 使用 `RewardItemBuilderCustomNode`。
- 两种 RewardItem Builder 都要求 `entry` 和 `name`；`kwargs` 默认空字符串，`weight=1.0`。
  `RewardConfigBuilderNode` 最多组合 5 个 reward item，`normalize=false`。
- `GRPOTrainingExtraConfigBuilderNode` 默认：`max_steps=200`、`num_generations=8`、
  prompt/completion 最大长度均为 2048、`temperature=0.7`、chord/hint 均关闭。
- `MetricsConfigBuilderNode.entry` 允许 `compute_gsm8k`、`compute_accuracy`、
  `compute_bleu`、`compute_rouge_l`；自定义 entry 使用 `MetricsConfigBuilderCustomNode`。

## 常用拓扑片段

### 数据接入

```python
# 用户上传的已清洗数据：先 preview_dataset，再接入返回的 Storage 相对文件路径
uploaded_dataset = PathJoinNode(
    id="1",
    base_path="/workspace/",
    subpath="datasets/my_data/train.jsonl",
)

# Clone 节点直接输出数据集目录，再按实际文件结构拼接训练文件
dataset = CloneAndCacheDataset(
    id="2",
    dataset="pyromind/self-cognition",
    target_path="/workspace/datasets/",
)
train_file = PathJoinNode(
    id="3",
    base_path=dataset.dataset_path,
    subpath="self-cognition.jsonl",
)

# 任意远程数据集使用 Download 节点
downloaded_dataset = DownloadAndCacheDataset(
    id="4",
    dataset_name="pyromind/easyhard-24k",
    cache_dir="/workspace/datasets/pyromind/easyhard-24k",
    download_source="huggingface",
)
```

### 字段映射（三选一）与数据配置

```python
# prompt/response 独立字段
dataset_kind = DatasetConfigBuilderTextNode(
    id="5",
    user_prompt_field="question",
    assistant_response_field="answer",
    # system_prompt_field="system",
    # rejected_field="rejected_answer",  # DPO
)

# messages 对话格式
dataset_kind = DatasetConfigBuilderMessageNode(
    id="5",
    messages_field="messages",
    # rejected_field="rejected_messages",  # DPO
)

# 多模态；只有 image_field 是 schema 必填项
dataset_kind = DatasetConfigBuilderVisionNode(
    id="5",
    image_field="image_path",
    user_prompt_field="question",
    assistant_response_field="answer",
)

dataset_extra = DatasetExtraConfigBuilderNode(
    id="6",
    train_max_samples=0,
    val_max_samples=0,
    sft_collator_entry="train.sft_collator:make_collate_fn",
    dpo_collator_entry="train.dpo_collator:make_collate_fn",
    grpo_collator_entry="train.data.default_vision_grpo_collate:create_grpo_collate_fn",
    max_seq_length=4096,
)
dataset_config = DatasetConfigBuilderNode(
    id="7",
    train_data_path=train_file.joined_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
    dataset_extra_config=dataset_extra.dataset_extra_config,
    # val_data_path=validation_file.joined_path,
)
```

### 模型与训练配置

```python
model = CloneAndCacheModel(
    id="8",
    model="Qwen/Qwen3-4B",
    target_path="/workspace/models/",
)
model_config = ModelConfigBuilderNode(
    id="9",
    model_path=model.model_path,
    model_type="auto",
)
lora_config = LoraConfigBuilderNode(
    id="10",
    lora_rank=8,
    lora_dropout=0.05,
    target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
)
training_config = TrainingConfigBuilderNode(
    id="11",
    learning_rate=1e-4,
    batch_size=2,
    grad_accum=2,
    num_epochs=1,
    save_steps=500,
    save_total_limit=3,
)
accelerate_config = AccelerateConfigBuilderNode(
    id="12",
    zero_stage=0,
)

# 仅启用 WandB 时创建；wandb_api_key 填 Secret 名，不填明文密钥
wandb_config = WandbConfigBuilderNode(
    id="13",
    wandb_api_key="MY_WANDB_KEY",
    wandb_project="studio_training",
    # wandb_name="sft-run",
)
```

### 训练、合并、推理与结果预览

```python
sft_train = ModelTrainSFTNode(
    id="14",
    output_path="/workspace/output/sft/",
    dataset_config=dataset_config.dataset_config,
    training_config=training_config.training_config,
    model_config=model_config.model_config,
    accelerate_config=accelerate_config.accelerate_config,
    lora_config=lora_config.lora_config,
    thinking_as_input_ratio=0,
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
)
merge = ModelMergeLoraNode(
    id="15",
    lora_path=sft_train.model_output_path,
    output_path="/workspace/output/merged/",
    model_path=model.model_path,
    model_type="auto",
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
)
infer = VLLMInference(
    id="16",
    model_path=merge.merged_model_path,
    port=3000,
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
    # max_model_len=4096,
)
test = TestLLMNode(
    id="17",
    endpoint=infer.endpoint,
    prompt="Hello, how are you?",
    max_tokens=100,
    temperature=0.7,
)
preview = ContentPreview(
    id="18",
    content=test.result,
    file_format="txt",
)
```

### 评测（bench）

```python
metrics = MetricsConfigBuilderNode(
    id="19",
    entry="compute_gsm8k",
    name="gsm8k",
)

# 自定义指标改用 MetricsConfigBuilderCustomNode，entry 形如
# "/workspace/script/agent/acc.py:acc_func"。
evaluate = ModelEvalApiNode(
    id="20",
    endpoint=infer.endpoint,
    endpoint_api_key="empty",
    endpoint_model="default",
    output_path="/workspace/outputs/bench",
    dataset_config=dataset_config.dataset_config,
    metrics_config=metrics.metrics_config,
    max_samples=100,
    max_tokens=256,
    temperature=0.01,
)
```
