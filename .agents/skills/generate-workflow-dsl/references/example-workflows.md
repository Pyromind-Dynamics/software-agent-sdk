# 示例工作流模板

选最接近的示例改参数（数据路径、字段映射、模型、超参），不要从零拼节点。
最常见的"用户 storage 数据训练"场景直接用示例6。

## 示例1：数据集预处理与验证

```python
# workflow: Dataset Processing

dataset = CloneAndCacheDataset(
    id="10",
    dataset="openai/gsm8k",
    target_path="/workspace/datasets/",
)

dataset_kind = DatasetConfigBuilderTextNode(
    id="16",
    user_prompt_field="question",
    assistant_response_field="answer",
)

train_file = PathJoinNode(
    id="11",
    base_path=dataset.dataset_path,
    subpath="main/train-00000-of-00001.parquet",
)

train_jsonl = DatasetToJsonlNode(
    id="13",
    dataset_path=train_file.joined_path,
)

dataset_config = DatasetConfigBuilderNode(
    id="9",
    train_data_path=train_jsonl.jsonl_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
)

dataset_validation = DatasetValidatorNode(
    id="15",
    dataset_config=dataset_config.dataset_config,
    check_reasoning=False,
    limit=0,
    verbose=False,
    preview_html_path="/workspace/datasets/preview.html",
)
```

## 示例2：SFT 训练（平台预置数据集）

```python
# workflow: SFT Training

dataset = CloneAndCacheDataset(
    id="1",
    dataset="pyromind/self-cognition",
    target_path="/workspace/datasets/",
)

model = CloneAndCacheModel(
    id="2",
    model="Qwen/Qwen3-0.6B",
    target_path="/workspace/models/",
)

train_file = PathJoinNode(
    id="3",
    base_path=dataset.dataset_path,
    subpath="self-cognition.jsonl",
)

dataset_kind = DatasetConfigBuilderTextNode(
    id="4",
    user_prompt_field="user_prompt",
    assistant_response_field="gt",
)

dataset_config = DatasetConfigBuilderNode(
    id="5",
    train_data_path=train_file.joined_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
)

model_config = ModelConfigBuilderNode(
    id="6",
    model_path=model.model_path,
    model_type="auto",
)

lora_config = LoraConfigBuilderNode(
    id="7",
    lora_rank=8,
    lora_dropout=0.05,
    target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
)

training_config = TrainingConfigBuilderNode(
    id="8",
    learning_rate=1e-4,
    batch_size=2,
    grad_accum_steps=2,
    num_epochs=1,
    save_steps=500,
    save_total_limit=3,
)

accelerate_config = AccelerateConfigBuilderNode(
    id="9",
    zero_stage=0,
)

sft_train = ModelTrainSFTNode(
    id="10",
    dataset_config=dataset_config.dataset_config,
    model_config=model_config.model_config,
    lora_config=lora_config.lora_config,
    training_config=training_config.training_config,
    accelerate_config=accelerate_config.accelerate_config,
    output_path="/workspace/output/sft/",
)

merge = ModelMergeLoraNode(
    id="11",
    model_path=model.model_path,
    lora_path=sft_train.model_output_path,
    output_path="/workspace/output/merged/",
)
```

## 示例3：GRPO 训练（多模态）

```python
# workflow: GRPO Training

dataset = CloneAndCacheDataset(
    id="1",
    dataset="pyromind/geometry-vqa-vlm-demo",
    target_path="/workspace/datasets/",
)

model = CloneAndCacheModel(
    id="2",
    model="Qwen/Qwen3-VL-4B-Instruct",
    target_path="/workspace/models/",
)

train_file = PathJoinNode(
    id="3",
    base_path=dataset.dataset_path,
    subpath="multimodal-open-r1-test.jsonl",
)

dataset_kind = DatasetConfigBuilderVisionNode(
    id="4",
    user_prompt_field="question",
    assistant_response_field="answer",
    image_field="image_path",
)

dataset_config = DatasetConfigBuilderNode(
    id="5",
    train_data_path=train_file.joined_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
)

model_config = ModelConfigBuilderNode(
    id="6",
    model_path=model.model_path,
    model_type="qwen3vl",
)

lora_config = LoraConfigBuilderNode(
    id="7",
    lora_rank=8,
    lora_dropout=0.05,
    target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
)

training_config = TrainingConfigBuilderNode(
    id="8",
    learning_rate=1e-6,
    batch_size=2,
    grad_accum_steps=2,
    num_epochs=1,
    save_steps=500,
    save_total_limit=3,
)

reward_item_1 = RewardItemBuilderNode(
    id="9",
    entry_function="reward_functions.accuracy_reward",
    name="accuracy",
    weight=1.0,
)

reward_config = RewardConfigBuilderNode(
    id="10",
    reward_item_1=reward_item_1.reward_item,
)

grpo_extra = GRPOTrainingExtraConfigBuilderNode(
    id="11",
    num_generations=4,
    temperature=0.7,
    max_completion_length=200,
    max_prompt_length=20000,
)

accelerate_config = AccelerateConfigBuilderNode(
    id="12",
    zero_stage=0,
)

grpo_train = ModelTrainGRPONode(
    id="13",
    dataset_config=dataset_config.dataset_config,
    model_config=model_config.model_config,
    lora_config=lora_config.lora_config,
    training_config=training_config.training_config,
    accelerate_config=accelerate_config.accelerate_config,
    reward_config=reward_config.reward_config,
    grpo_extra_config=grpo_extra.grpo_extra_config,
    output_path="/workspace/output/grpo/",
)
```

## 示例4：DPO 训练

```python
# workflow: DPO
nb5ddba2 = AccelerateConfigBuilderNode(id=2, zero_stage=1)
n9f22093 = WandbConfigBuilderNode(id=8, wandb_api_key="7c7f22bc2f23cda64c29fd2d78e4112890cee22f", wandb_project="studio_training")
n1a343e9 = CloneAndCacheModel(id=11, model="Qwen/Qwen3-0.6B")
n773f576 = CloneAndCacheDataset(id=13, dataset="pyromind/alpaca-gpt4-llm-demo")
n192c9cf = DatasetExtraConfigBuilderNode(id=15)
n5845bd4 = LoraConfigBuilderNode(id=17)
n566f325 = TrainingConfigBuilderNode(id=18, num_epochs=1, learning_rate=1e-06, logging_steps=1)
n03eae42 = DatasetConfigBuilderTextNode(id=19, system_prompt_field="system_prompt", user_prompt_field="user_prompt")
n009cefc = ModelConfigBuilderNode(id=16, model_path=n1a343e9.model_path)
nb310f52 = PathJoinNode(id=14, base_path=n773f576.dataset_path, subpath="alpaca_gpt4_demo.dpo.jsonl")
na0b732c = DatasetConfigBuilderNode(id=9, train_data_path=nb310f52.joined_path, dataset_kind_config=n03eae42.dataset_kind_config, dataset_extra_config=n192c9cf.dataset_extra_config)
n6b07df6 = ModelTrainDPONode(id=10, output_path="/workspace/models/checkpoints/dpo0510", dataset_config=na0b732c.dataset_config, training_config=n566f325.training_config, model_config=n009cefc.model_config, accelerate_config=nb5ddba2.accelerate_config, lora_config=n5845bd4.lora_config, wandb_config=n9f22093.wandb_config)
```

## 示例5：基模 benchmark 基线评测

部署基模推理服务 → 用（采样后的）数据集 + 指标跑离线评测。训练前先跑一次拿基线分数，
训练后换成合并模型路径再跑一次做对比。

```python
# workflow: Base Model Benchmark
n7cec466 = MetricsConfigBuilderNode(id=9, entry="examples/eval_metrics_common.py:compute_gsm8k", name="gsm8k")
n3a96035 = CloneAndCacheDataset(id=12, dataset="openai/gsm8k")
n3d49b3a = DatasetConfigBuilderTextNode(id=15, user_prompt_field="question", assistant_response_field="answer")
n8e27f71 = CloneAndCacheModel(id=16, model="Qwen/Qwen3-0.6B")
n5e463b4 = PathJoinNode(id=13, base_path=n3a96035.dataset_path, subpath="main/test-00000-of-00001.parquet")
nbcd47fc = VLLMInference(id=7, model_path=n8e27f71.model_path)
n6c59276 = DatasetToJsonlNode(id=14, dataset_path=n5e463b4.joined_path)
n4d04a62 = DatasetConfigBuilderNode(id=5, train_data_path=n6c59276.jsonl_path, dataset_kind_config=n3d49b3a.dataset_kind_config)
n5f3b109 = ModelEvalApiNode(id=6, endpoint=nbcd47fc.endpoint, output_path="/workspace/outputs/bench", dataset_config=n4d04a62.dataset_config, metrics_config=n7cec466.metrics_config, max_samples=100)
```

## 示例6：SFT 训练（用户 storage 数据，messages 格式）

最常见场景：用户把数据传到 storage 并贴了相对路径。先 `preview_dataset` 确认文件名、
格式和字段，再填 PathJoinNode 的 subpath 与字段映射。数据是 prompt/response 独立字段时
把 DatasetConfigBuilderMessageNode 换成 DatasetConfigBuilderTextNode。

```python
# workflow: SFT Training (User Data)

model = CloneAndCacheModel(
    id="1",
    model="Qwen/Qwen3-4B",
    target_path="/workspace/models/",
)

train_file = PathJoinNode(
    id="2",
    base_path="/workspace/",
    subpath="datasets/my_data/train.jsonl",  # 用户贴的相对路径 + preview 确认的文件
)

dataset_kind = DatasetConfigBuilderMessageNode(
    id="3",
    messages_field="messages",               # preview 确认的字段名
)

dataset_config = DatasetConfigBuilderNode(
    id="4",
    train_data_path=train_file.joined_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
)

model_config = ModelConfigBuilderNode(
    id="5",
    model_path=model.model_path,
    model_type="auto",
)

lora_config = LoraConfigBuilderNode(
    id="6",
    lora_rank=32,                             # N=12K、L 中档 → rank 32
    lora_dropout=0.05,
    target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
)

training_config = TrainingConfigBuilderNode(
    id="7",
    learning_rate=1e-5,
    batch_size=4,
    grad_accum_steps=4,
    num_epochs=1,
    save_steps=500,
    save_total_limit=3,
)

accelerate_config = AccelerateConfigBuilderNode(
    id="8",
    zero_stage=0,
)

sft_train = ModelTrainSFTNode(
    id="9",
    dataset_config=dataset_config.dataset_config,
    model_config=model_config.model_config,
    lora_config=lora_config.lora_config,
    training_config=training_config.training_config,
    accelerate_config=accelerate_config.accelerate_config,
    output_path="/workspace/output/sft/",
)

merge = ModelMergeLoraNode(
    id="10",
    model_path=model.model_path,
    lora_path=sft_train.model_output_path,
    output_path="/workspace/output/merged/",
)
```
