# 节点速查与常用参数

节点完整契约（必填项、默认值、枚举）以 `knowledge/nodes/<NodeType>/<NodeType>.md`
为准，本文件只是速查。

## 节点速查表

### 数据与资源

| NodeType | 描述 | 主要输出端口 |
|----------|------|-------------|
| CloneAndCacheDataset | 克隆平台预置数据集 | dataset_path |
| DownloadAndCacheDataset | 下载已清洗的数据集 | dataset_path |
| CloneAndCacheModel | 克隆平台预置基模（枚举见 SKILL.md 第 3 步） | model_path |
| DownloadAndCacheModel | 从 huggingface/modelscope 下载任意开源模型 | model_path |
| PathJoinNode | 拼接 Clone 数据目录，或接入用户上传的 Storage 相对路径 | joined_path |

### 配置构建

| NodeType | 描述 | 主要输出端口 |
|----------|------|-------------|
| DatasetConfigBuilderTextNode | prompt/response 独立字段映射 | dataset_kind_config |
| DatasetConfigBuilderMessageNode | messages 对话格式映射 | dataset_kind_config |
| DatasetConfigBuilderVisionNode | 多模态字段映射 | dataset_kind_config |
| DatasetConfigBuilderNode | 组装完整数据集配置 | dataset_config |
| DatasetExtraConfigBuilderNode | 序列长度、采样上限等 | dataset_extra_config |
| ModelConfigBuilderNode | 模型路径和类型 | model_config |
| LoraConfigBuilderNode | LoRA 配置 | lora_config |
| TrainingConfigBuilderNode | 训练超参数 | training_config |
| AccelerateConfigBuilderNode | 分布式训练配置 | accelerate_config |
| WandbConfigBuilderNode | WandB 实验追踪 | wandb_config |
| RewardItemBuilderNode | 单个奖励项 | reward_item |
| RewardConfigBuilderNode | 组合多个奖励项 | reward_config |
| GRPOTrainingExtraConfigBuilderNode | GRPO 额外配置 | grpo_extra_config |
| MetricsConfigBuilderNode | 内置评估指标（gsm8k/accuracy/bleu/rouge_l） | metrics_config |
| MetricsConfigBuilderCustomNode | 自定义评估指标（entry 为上传的 py:func） | metrics_config |

### 训练与评估执行

| NodeType | 描述 | 主要输出端口 |
|----------|------|-------------|
| ModelTrainSFTNode | SFT 监督微调 | model_output_path |
| ModelTrainDPONode | DPO 偏好优化 | model_output_path |
| ModelTrainGRPONode | GRPO 强化学习 | model_output_path |
| ModelMergeLoraNode | 合并 LoRA 到基模 | merged_model_path |
| VLLMInference | 启动 vLLM 推理服务（OpenAI 兼容） | endpoint |
| ModelEvalApiNode | 调推理端点跑离线 benchmark | benchmark_output_path |
| TestLLMNode | 测试推理端点 | result |
| ContentPreview | 预览文本内容 | content |

组合映射：用户要求“发送测试请求并展示结果”时，使用
`VLLMInference → TestLLMNode → ContentPreview`，由 `ContentPreview` 消费 `result`。

## 常用节点参数

### 数据接入

`CloneAndCacheDataset.dataset` 当前枚举：`Writer/omniact`、`openai/gsm8k`、
`pyromind/alpaca-gpt4-llm-demo`、`pyromind/geometry-vqa-vlm-demo`、
`pyromind/self-cognition`；`target_path` 默认 `/workspace/datasets/`。

`CloneAndCacheModel.model` 当前枚举：`Qwen/Qwen3-0.6B`、`Qwen/Qwen3-1.7B`、
`Qwen/Qwen3-4B`、`Qwen/Qwen3-VL-2B-Instruct`、`Qwen/Qwen3-VL-4B-Instruct`。

```python
# 用户上传的已清洗数据：先 preview，再接入其 Storage 相对文件路径
uploaded_dataset = PathJoinNode(
    id=1,
    base_path="/workspace/",
    subpath="datasets/my_data/train.jsonl",
)

# Clone Dataset：使用上面的平台枚举；其他远程数据集用 DownloadAndCacheDataset
dataset = CloneAndCacheDataset(
    id=2,
    dataset="pyromind/self-cognition",
    target_path="/workspace/datasets/",
)

# Clone 后按该数据集的已知文件结构取训练文件
train_file = PathJoinNode(
    id=3,
    base_path=dataset.dataset_path,
    subpath="self-cognition.jsonl",
)

# Download Dataset：可下载用户指定的远程数据集；这里是已知可直接使用的示例
downloaded_dataset = DownloadAndCacheDataset(
    id=4,
    dataset_name="pyromind/easyhard-24k",
    cache_dir="/workspace/datasets/pyromind/easyhard-24k",
    download_source="huggingface",
)
```

### 字段映射（三选一）

```python
# prompt/response 独立字段
dataset_kind = DatasetConfigBuilderTextNode(
    id=4,
    user_prompt_field="question",
    assistant_response_field="answer",
    # system_prompt_field="system",     # 可选
    # rejected_field="rejected_answer", # DPO 时必填
)

# messages 对话格式
dataset_kind = DatasetConfigBuilderMessageNode(
    id=4,
    messages_field="messages",
    # rejected_field="rejected_messages",  # DPO 时必填
)

# 多模态
dataset_kind = DatasetConfigBuilderVisionNode(
    id=4,
    user_prompt_field="question",
    assistant_response_field="answer",
    image_field="image_path",
)
```

### 数据集与模型配置

```python
dataset_extra = DatasetExtraConfigBuilderNode(
    id=5,
    train_max_samples=0,
    val_max_samples=0,
    sft_collator_entry="train.sft_collator:make_collate_fn",
    dpo_collator_entry="train.dpo_collator:make_collate_fn",
    grpo_collator_entry="train.data.default_vision_grpo_collate:create_grpo_collate_fn",
    max_seq_length=4096,
)

dataset_config = DatasetConfigBuilderNode(
    id=6,
    train_data_path=train_file.joined_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
    dataset_extra_config=dataset_extra.dataset_extra_config,
    # val_data_path=...,          # 可选：验证集
)

model_config = ModelConfigBuilderNode(
    id=7,
    model_path=model.model_path,
    model_type="auto",            # 默认 auto；也可用 qwen3vl、qwen3.5
)
```

### 训练配置

```python
lora_config = LoraConfigBuilderNode(
    id=7,
    lora_rank=8,                  # 按 parameter-decision.md 决策
    lora_dropout=0.05,
    target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
)

training_config = TrainingConfigBuilderNode(
    id=8,
    learning_rate=1e-4,           # 按 parameter-decision.md 决策
    batch_size=2,
    grad_accum=2,
    num_epochs=1,
    save_steps=500,
    save_total_limit=3,
)

accelerate_config = AccelerateConfigBuilderNode(
    id=9,
    zero_stage=0,                 # 单卡 LoRA 固定为 0
)

# 仅启用 WandB 时创建；wandb_api_key 填 Secret 名，不填明文密钥
wandb_config = WandbConfigBuilderNode(
    id=10,
    wandb_api_key="MY_WANDB_KEY",
    wandb_project="studio_training",
    # wandb_name="sft-run",       # 可选
)
```

### 训练执行与后处理

`ModelTrainSFTNode` 必填 `output_path`、`dataset_config`、`training_config`、
`model_config`、`accelerate_config`、`gpu_count`、`gpu_product`；可选 `lora_config`、
`wandb_config`、`thinking_as_input_ratio`。训练和 LoRA 合并节点的 `gpu_product` 只允许
`NVIDIA-H200`、`NVIDIA-H100-80GB-HBM3`。

```python
sft_train = ModelTrainSFTNode(
    id=11,
    dataset_config=dataset_config.dataset_config,
    model_config=model_config.model_config,
    lora_config=lora_config.lora_config,
    training_config=training_config.training_config,
    accelerate_config=accelerate_config.accelerate_config,
    # wandb_config=wandb_config.wandb_config,  # 可选
    thinking_as_input_ratio=0,    # 可选，默认 0
    output_path="/workspace/output/sft/",
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
)

merge = ModelMergeLoraNode(
    id=12,
    model_path=model.model_path,
    lora_path=sft_train.model_output_path,
    output_path="/workspace/output/merged/",
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
    model_type="auto",            # 默认 auto；也可用 qwen3vl、qwen3.5
)
```

### 推理测试与预览

`VLLMInference.gpu_product` 可用 `NVIDIA-H200`、`NVIDIA-H100-80GB-HBM3`、
`NVIDIA-L40S`。用户要求发送测试请求并展示结果时，连接完整链路：

```python
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

### 评测（bench）

```python
infer = VLLMInference(
    id=12,
    model_path=model.model_path,
    port=3000,
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",  # 也可用 NVIDIA-H200、NVIDIA-L40S
)

metrics = MetricsConfigBuilderNode(
    id=13,
    entry="compute_gsm8k",  # 内置指标使用裸函数名枚举
    name="gsm8k",
)
# 自定义指标：先 upload_file_to_pyromind 上传 py 文件，再用
# MetricsConfigBuilderCustomNode(entry="/workspace/script/agent/acc.py:acc_func", name="acc")

evaluate = ModelEvalApiNode(
    id=14,
    endpoint=infer.endpoint,
    endpoint_api_key="empty",
    endpoint_model="default",
    output_path="/workspace/outputs/bench",
    dataset_config=dataset_config.dataset_config,
    metrics_config=metrics.metrics_config,
    max_samples=100,              # 基线评测采样条数，0 表示全量
)
```
