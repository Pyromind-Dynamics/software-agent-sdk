# 示例工作流模板

只把示例当作拓扑骨架。参数名、输入端口和输出端口以知识库节点 schema 为准；参数值按
“用户要求 > 现有有效值 > 当前数据与资源决策 > 模板值”确定。
示例为可复现性使用平台测试集；用户提供已清洗的 Storage 相对路径时，用
`PathJoinNode(base_path="/workspace/", subpath=<用户路径>)` 替换数据获取段，其余拓扑照用。

## 示例1：SFT（Clone Dataset）

```python
# workflow: SFT Training

dataset = CloneAndCacheDataset(
    id=1,
    dataset="pyromind/self-cognition",
    target_path="/workspace/datasets/",
)
model = CloneAndCacheModel(
    id=2,
    model="Qwen/Qwen3-0.6B",
    target_path="/workspace/models/",
)
train_file = PathJoinNode(
    id=3,
    base_path=dataset.dataset_path,
    subpath="self-cognition.jsonl",
)
dataset_kind = DatasetConfigBuilderTextNode(
    id=4,
    user_prompt_field="user_prompt",
    assistant_response_field="gt",
)
dataset_extra = DatasetExtraConfigBuilderNode(
    id=12,
    max_seq_length=4096,
)
dataset_config = DatasetConfigBuilderNode(
    id=5,
    train_data_path=train_file.joined_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
    dataset_extra_config=dataset_extra.dataset_extra_config,
)
model_config = ModelConfigBuilderNode(
    id=6,
    model_path=model.model_path,
    model_type="auto",
)
lora_config = LoraConfigBuilderNode(
    id=7,
    lora_rank=8,
    lora_dropout=0.05,
    target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
)
training_config = TrainingConfigBuilderNode(
    id=8,
    learning_rate=1e-4,
    batch_size=2,
    grad_accum=2,
    num_epochs=1,
    save_steps=500,
    save_total_limit=3,
)
accelerate_config = AccelerateConfigBuilderNode(id=9, zero_stage=0)
sft_train = ModelTrainSFTNode(
    id=10,
    dataset_config=dataset_config.dataset_config,
    model_config=model_config.model_config,
    lora_config=lora_config.lora_config,
    training_config=training_config.training_config,
    accelerate_config=accelerate_config.accelerate_config,
    output_path="/workspace/output/sft/",
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
    thinking_as_input_ratio=0,
)
merge = ModelMergeLoraNode(
    id=11,
    model_path=model.model_path,
    lora_path=sft_train.model_output_path,
    output_path="/workspace/output/merged/",
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
    model_type="auto",
)
infer = VLLMInference(
    id=13,
    model_path=merge.merged_model_path,
    port=3000,
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
)
test = TestLLMNode(
    id=14,
    endpoint=infer.endpoint,
    prompt="Hello, how are you?",
    max_tokens=100,
    temperature=0.7,
)
preview = ContentPreview(
    id=15,
    content=test.result,
    file_format="txt",
)
```

## 示例2：DPO（Clone Dataset）

```python
# workflow: DPO Training

dataset = CloneAndCacheDataset(
    id=1,
    dataset="pyromind/alpaca-gpt4-llm-demo",
    target_path="/workspace/datasets/",
)
model = CloneAndCacheModel(
    id=2,
    model="Qwen/Qwen3-0.6B",
    target_path="/workspace/models/",
)
train_file = PathJoinNode(
    id=3,
    base_path=dataset.dataset_path,
    subpath="alpaca_gpt4_demo.dpo.jsonl",
)
dataset_kind = DatasetConfigBuilderTextNode(
    id=4,
    system_prompt_field="system_prompt",
    user_prompt_field="user_prompt",
    assistant_response_field="gt",
    rejected_field="rejected_answer",
)
dataset_extra = DatasetExtraConfigBuilderNode(
    id=11,
    max_seq_length=4096,
)
dataset_config = DatasetConfigBuilderNode(
    id=5,
    train_data_path=train_file.joined_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
    dataset_extra_config=dataset_extra.dataset_extra_config,
)
model_config = ModelConfigBuilderNode(
    id=6,
    model_path=model.model_path,
    model_type="auto",
)
lora_config = LoraConfigBuilderNode(id=7)
training_config = TrainingConfigBuilderNode(
    id=8,
    learning_rate=1e-6,
    num_epochs=1,
    lr_scheduler_type="constant",
    max_grad_norm=0.5,
)
accelerate_config = AccelerateConfigBuilderNode(id=9, zero_stage=0)
dpo_train = ModelTrainDPONode(
    id=10,
    dataset_config=dataset_config.dataset_config,
    model_config=model_config.model_config,
    lora_config=lora_config.lora_config,
    training_config=training_config.training_config,
    accelerate_config=accelerate_config.accelerate_config,
    output_path="/workspace/output/dpo/",
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
    thinking_as_input_ratio=0,
)
```

## 示例3：多模态 GRPO（Clone Dataset）

```python
# workflow: GRPO Training

dataset = CloneAndCacheDataset(
    id=1,
    dataset="pyromind/geometry-vqa-vlm-demo",
    target_path="/workspace/datasets/",
)
model = CloneAndCacheModel(
    id=2,
    model="Qwen/Qwen3-VL-4B-Instruct",
    target_path="/workspace/models/",
)
train_file = PathJoinNode(
    id=3,
    base_path=dataset.dataset_path,
    subpath="multimodal-open-r1-test.jsonl",
)
dataset_kind = DatasetConfigBuilderVisionNode(
    id=4,
    system_prompt_field="system_prompt",
    user_prompt_field="user_prompt",
    assistant_response_field="gt",
    image_field="image_path",
)
dataset_extra = DatasetExtraConfigBuilderNode(
    id=15,
    grpo_collator_entry="train.data.default_vision_grpo_collate:create_grpo_collate_fn",
    max_seq_length=4096,
)
dataset_config = DatasetConfigBuilderNode(
    id=5,
    train_data_path=train_file.joined_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
    dataset_extra_config=dataset_extra.dataset_extra_config,
)
model_config = ModelConfigBuilderNode(
    id=6,
    model_path=model.model_path,
    model_type="qwen3vl",
)
lora_config = LoraConfigBuilderNode(
    id=7,
    lora_rank=8,
    lora_dropout=0.05,
    target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
)
training_config = TrainingConfigBuilderNode(
    id=8,
    learning_rate=1e-6,
    batch_size=2,
    grad_accum=2,
    num_epochs=1,
)
reward_item_1 = RewardItemBuilderNode(
    id=9,
    entry="geometry_vqa_thinking_reward",
    name="thinking_tags",
    weight=1.0,
)
reward_item_2 = RewardItemBuilderNode(
    id=10,
    entry="geometry_vqa_answer_reward",
    name="answer_acc",
    weight=1.0,
)
reward_config = RewardConfigBuilderNode(
    id=11,
    reward_item_1=reward_item_1.reward_item,
    reward_item_2=reward_item_2.reward_item,
)
grpo_extra = GRPOTrainingExtraConfigBuilderNode(
    id=12,
    num_generations=4,
    temperature=0.7,
    max_completion_length=200,
    max_prompt_length=20000,
)
accelerate_config = AccelerateConfigBuilderNode(id=13, zero_stage=0)
grpo_train = ModelTrainGRPONode(
    id=14,
    dataset_config=dataset_config.dataset_config,
    model_config=model_config.model_config,
    lora_config=lora_config.lora_config,
    training_config=training_config.training_config,
    accelerate_config=accelerate_config.accelerate_config,
    reward_config=reward_config.reward_config,
    grpo_extra_config=grpo_extra.grpo_extra_config,
    output_path="/workspace/output/grpo/",
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
    thinking_as_input_ratio=0,
)
```

## 示例4：SFT 文本数据（Download Dataset）

```python
# workflow: SFT Training (EasyHard)

dataset = DownloadAndCacheDataset(
    id=1,
    dataset_name="pyromind/easyhard-24k",
    cache_dir="/workspace/datasets/pyromind/easyhard-24k",
    download_source="huggingface",
)
model = CloneAndCacheModel(
    id=2,
    model="Qwen/Qwen3-4B",
    target_path="/workspace/models/",
)
dataset_kind = DatasetConfigBuilderTextNode(
    id=3,
    user_prompt_field="question",
    assistant_response_field="answer",
)
dataset_config = DatasetConfigBuilderNode(
    id=4,
    train_data_path=dataset.dataset_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
)
model_config = ModelConfigBuilderNode(
    id=5,
    model_path=model.model_path,
    model_type="auto",
)
lora_config = LoraConfigBuilderNode(id=6, lora_rank=32)
training_config = TrainingConfigBuilderNode(
    id=7,
    learning_rate=1e-5,
    batch_size=4,
    grad_accum=4,
    num_epochs=1,
)
accelerate_config = AccelerateConfigBuilderNode(id=8, zero_stage=0)
sft_train = ModelTrainSFTNode(
    id=9,
    dataset_config=dataset_config.dataset_config,
    model_config=model_config.model_config,
    lora_config=lora_config.lora_config,
    training_config=training_config.training_config,
    accelerate_config=accelerate_config.accelerate_config,
    output_path="/workspace/output/sft/",
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
)
```

## 示例5：SFT messages 数据（Download Dataset）

```python
# workflow: SFT Training (Agentic Tool Calls)

dataset = DownloadAndCacheDataset(
    id=1,
    dataset_name="pyromind/agentic-tool-call-dataset-12k",
    cache_dir="/workspace/datasets/pyromind/agentic-tool-call-dataset-12k",
    download_source="huggingface",
)
model = CloneAndCacheModel(
    id=2,
    model="Qwen/Qwen3-4B",
    target_path="/workspace/models/",
)
dataset_kind = DatasetConfigBuilderMessageNode(id=3, messages_field="messages")
dataset_config = DatasetConfigBuilderNode(
    id=4,
    train_data_path=dataset.dataset_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
)
model_config = ModelConfigBuilderNode(
    id=5,
    model_path=model.model_path,
    model_type="auto",
)
lora_config = LoraConfigBuilderNode(id=6, lora_rank=32)
training_config = TrainingConfigBuilderNode(
    id=7,
    learning_rate=1e-5,
    batch_size=4,
    grad_accum=4,
    num_epochs=1,
)
accelerate_config = AccelerateConfigBuilderNode(id=8, zero_stage=0)
sft_train = ModelTrainSFTNode(
    id=9,
    dataset_config=dataset_config.dataset_config,
    model_config=model_config.model_config,
    lora_config=lora_config.lora_config,
    training_config=training_config.training_config,
    accelerate_config=accelerate_config.accelerate_config,
    output_path="/workspace/output/sft/",
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
)
```

## 示例6：基模 benchmark

```python
# workflow: Base Model Benchmark

dataset = DownloadAndCacheDataset(
    id=1,
    dataset_name="pyromind/easyhard-24k",
    cache_dir="/workspace/datasets/pyromind/easyhard-24k",
    download_source="huggingface",
)
model = CloneAndCacheModel(
    id=2,
    model="Qwen/Qwen3-0.6B",
    target_path="/workspace/models/",
)
dataset_kind = DatasetConfigBuilderTextNode(
    id=3,
    user_prompt_field="question",
    assistant_response_field="answer",
)
dataset_config = DatasetConfigBuilderNode(
    id=4,
    train_data_path=dataset.dataset_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
)
metrics = MetricsConfigBuilderNode(
    id=5,
    entry="compute_gsm8k",
    name="gsm8k",
)
vllm = VLLMInference(
    id=6,
    model_path=model.model_path,
    port=3000,
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
)
bench = ModelEvalApiNode(
    id=7,
    endpoint=vllm.endpoint,
    endpoint_api_key="empty",
    endpoint_model="default",
    output_path="/workspace/outputs/bench",
    dataset_config=dataset_config.dataset_config,
    metrics_config=metrics.metrics_config,
    max_samples=100,
)
```
