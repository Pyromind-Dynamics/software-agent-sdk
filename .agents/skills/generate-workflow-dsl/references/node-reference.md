# 节点速查与常用参数

节点完整契约（必填项、默认值、枚举）以 `<知识库绝对路径>/nodes/<NodeType>/<NodeType>.md`
为准，本文件只是速查。

## 节点速查表

### 数据与资源

| NodeType | 描述 | 主要输出端口 |
|----------|------|-------------|
| CloneAndCacheDataset | 拉取平台预置数据集 | dataset_path |
| CloneAndCacheModel | 克隆平台预置基模（枚举见 SKILL.md 第 3 步） | model_path |
| DownloadAndCacheModel | 从 huggingface/modelscope 下载任意开源模型 | model_path |
| PathJoinNode | 拼接路径（接入用户 storage 数据的入口） | joined_path |
| DatasetToJsonlNode | HF 数据集目录/parquet 转 JSONL | jsonl_path |
| DatasetValidatorNode | 验证数据集格式 | validation_result |

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
| ContentPreview | 预览文本内容 | — |

## 常用节点参数

### 数据接入

```python
# 平台预置数据集
dataset = CloneAndCacheDataset(
    id="1",
    dataset="openai/gsm8k",             # 数据集标识
    target_path="/workspace/datasets/",
)

# 用户 storage 数据：直接用 PathJoinNode 指向用户贴的相对路径
train_file = PathJoinNode(
    id="2",
    base_path="/workspace/",
    subpath="datasets/my_data/train.jsonl",  # 用户贴的相对路径 + preview 看到的文件名
)

# parquet / HF 目录需先转 JSONL
train_jsonl = DatasetToJsonlNode(id="3", dataset_path=train_file.joined_path)
```

### 字段映射（三选一）

```python
# prompt/response 独立字段
dataset_kind = DatasetConfigBuilderTextNode(
    id="4",
    user_prompt_field="question",
    assistant_response_field="answer",
    # system_prompt_field="system",     # 可选
    # rejected_field="rejected_answer", # DPO 时必填
)

# messages 对话格式
dataset_kind = DatasetConfigBuilderMessageNode(
    id="4",
    messages_field="messages",
    # rejected_field="rejected_messages",  # DPO 时必填
)

# 多模态
dataset_kind = DatasetConfigBuilderVisionNode(
    id="4",
    user_prompt_field="question",
    assistant_response_field="answer",
    image_field="image_path",
)
```

### 数据集与模型配置

```python
dataset_config = DatasetConfigBuilderNode(
    id="5",
    train_data_path=train_jsonl.jsonl_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
    # val_data_path=...,          # 可选：验证集
    # dataset_extra_config=...,   # 可选：max_seq_length 等
)

model_config = ModelConfigBuilderNode(
    id="6",
    model_path=model.model_path,
    model_type="auto",            # "auto" | "qwen3vl" | "qwen3.5"，VL 模型必须显式指定
)
```

### 训练配置

```python
lora_config = LoraConfigBuilderNode(
    id="7",
    lora_rank=8,                  # 按 parameter-decision.md 决策
    lora_dropout=0.05,
    target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
)

training_config = TrainingConfigBuilderNode(
    id="8",
    learning_rate=1e-4,           # 按 parameter-decision.md 决策
    batch_size=2,
    grad_accum_steps=2,
    num_epochs=1,
    save_steps=500,
    save_total_limit=3,
)

accelerate_config = AccelerateConfigBuilderNode(
    id="9",
    zero_stage=0,                 # 单卡 LoRA 固定为 0
)
```

### 训练执行与后处理

```python
sft_train = ModelTrainSFTNode(
    id="10",
    dataset_config=dataset_config.dataset_config,
    model_config=model_config.model_config,
    lora_config=lora_config.lora_config,
    training_config=training_config.training_config,
    accelerate_config=accelerate_config.accelerate_config,
    # wandb_config=...,           # 可选
    output_path="/workspace/output/sft/",
)

merge = ModelMergeLoraNode(
    id="11",
    model_path=model.model_path,
    lora_path=sft_train.model_output_path,
    output_path="/workspace/output/merged/",
)
```

### 评测（bench）

```python
infer = VLLMInference(id="12", model_path=model.model_path)

metrics = MetricsConfigBuilderNode(
    id="13",
    entry="examples/eval_metrics_common.py:compute_gsm8k",  # 枚举见 SKILL.md 第 2 步
    name="gsm8k",
)
# 自定义指标：先 upload_file_to_pyromind 上传 py 文件，再用
# MetricsConfigBuilderCustomNode(entry="/workspace/script/agent/acc.py:acc_func", name="acc")

evaluate = ModelEvalApiNode(
    id="14",
    endpoint=infer.endpoint,
    output_path="/workspace/outputs/bench",
    dataset_config=dataset_config.dataset_config,
    metrics_config=metrics.metrics_config,
    max_samples=100,              # 基线评测采样条数，0 表示全量
)
```
