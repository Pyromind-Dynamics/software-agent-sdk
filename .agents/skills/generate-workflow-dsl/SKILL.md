---
name: generate-workflow-dsl
description: >-
  当用户需要生成 Pyromind 平台上的模型训练工作流时使用此技能。
  先检索知识库了解可用节点和连接模式，然后生成符合 DSL 格式的工作流代码。
triggers:
- 生成工作流
- generate workflow
- 训练工作流
- training workflow
- workflow DSL
---

# 生成 Pyromind 工作流 DSL

## 概述

当用户要求生成模型训练工作流时，你需要：
1. 先用 grep 检索知识库，了解可用的节点类型、参数和连接模式
2. 根据用户需求选择合适的节点组合
3. 按照 DSL 语法格式输出工作流代码

## DSL 语法格式

```python
# workflow: <工作流名称>

variable_name = NodeType(
    id="<唯一节点ID>",
    param1=value1,
    param2=another_variable.output_port,
)
```

### 语法规则

1. 文件开头用注释标明工作流名称：`# workflow: <name>`
2. 每个节点是一个变量赋值：`var = NodeType(id="...", ...)`
3. `id` 参数必须唯一（使用数字字符串）
4. 静态值：字符串 `"value"`、数字 `42`、浮点数 `1e-4`、布尔值 `True`/`False`
5. 节点连接：引用 `变量名.输出端口名` 来连接上游节点的输出
6. 节点定义顺序：被引用的节点必须先定义

## 可用节点速查

### 数据与资源节点

| NodeType | 描述 | 主要输出端口 |
|----------|------|-------------|
| CloneAndCacheDataset | 拉取缓存数据集 | dataset_path |
| CloneAndCacheModel | 拉取缓存模型 | model_path |
| PathJoinNode | 拼接路径 | joined_path |
| DatasetToJsonlNode | 格式转换为JSONL | jsonl_path |
| DatasetValidatorNode | 验证数据集格式 | validation_result |

### 配置构建节点

| NodeType | 描述 | 主要输出端口 |
|----------|------|-------------|
| DatasetConfigBuilderTextNode | 文本任务字段映射 | dataset_kind_config |
| DatasetConfigBuilderVisionNode | 多模态字段映射 | dataset_kind_config |
| DatasetConfigBuilderNode | 构建完整数据集配置 | dataset_config |
| DatasetExtraConfigBuilderNode | 序列长度、采样上限等 | dataset_extra_config |
| ModelConfigBuilderNode | 模型路径和类型 | model_config |
| LoraConfigBuilderNode | LoRA配置 | lora_config |
| TrainingConfigBuilderNode | 训练超参数 | training_config |
| AccelerateConfigBuilderNode | 分布式训练配置 | accelerate_config |
| WandbConfigBuilderNode | WandB实验追踪 | wandb_config |
| RewardItemBuilderNode | 单个奖励项 | reward_item |
| RewardConfigBuilderNode | 组合多个奖励项 | reward_config |
| GRPOTrainingExtraConfigBuilderNode | GRPO额外配置 | grpo_extra_config |

### 训练执行节点

| NodeType | 描述 | 主要输出端口 |
|----------|------|-------------|
| ModelTrainSFTNode | SFT监督微调 | model_output_path |
| ModelTrainDPONode | DPO偏好优化 | model_output_path |
| ModelTrainGRPONode | GRPO强化学习 | model_output_path |

### 推理与评估节点

| NodeType | 描述 | 主要输出端口 |
|----------|------|-------------|
| ModelMergeLoraNode | 合并LoRA到基础模型 | merged_model_path |
| VLLMInference | 启动vLLM推理服务 | endpoint |
| TestLLMNode | 测试推理端点 | result |
| ContentPreview | 预览文本内容 | — |

## 常用节点参数详情

### CloneAndCacheDataset
```python
dataset = CloneAndCacheDataset(
    id="1",
    dataset="openai/gsm8k",           # HuggingFace数据集标识
    target_path="/workspace/datasets/", # 本地缓存目录
)
# 输出: dataset.dataset_path
```

### CloneAndCacheModel
```python
model = CloneAndCacheModel(
    id="2",
    model="Qwen/Qwen3-0.6B",          # 模型标识
    target_path="/workspace/models/",   # 本地缓存目录
)
# 输出: model.model_path
```

### DatasetConfigBuilderTextNode
```python
dataset_kind = DatasetConfigBuilderTextNode(
    id="3",
    user_prompt_field="question",           # 用户输入字段
    assistant_response_field="answer",      # 模型响应字段
    # system_prompt_field="system",         # 可选：系统提示字段
    # rejected_field="rejected_answer",     # 可选：DPO拒绝回答字段
)
# 输出: dataset_kind.dataset_kind_config
```

### DatasetConfigBuilderNode
```python
dataset_config = DatasetConfigBuilderNode(
    id="4",
    train_data_path=train_jsonl.jsonl_path,              # 训练数据路径
    dataset_kind_config=dataset_kind.dataset_kind_config, # 字段映射配置
    # val_data_path=val_jsonl.jsonl_path,                 # 可选：验证集路径
    # dataset_extra_config=extra.dataset_extra_config,    # 可选：额外配置
)
# 输出: dataset_config.dataset_config
```

### ModelConfigBuilderNode
```python
model_config = ModelConfigBuilderNode(
    id="5",
    model_path=model.model_path,  # 模型路径（通常引用CloneAndCacheModel输出）
    model_type="auto",            # 模型类型: "auto", "qwen3vl", "qwen3.5"
)
# 输出: model_config.model_config
```

### LoraConfigBuilderNode
```python
lora_config = LoraConfigBuilderNode(
    id="6",
    lora_rank=8,           # LoRA秩
    lora_dropout=0.05,     # Dropout比率
    target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
)
# 输出: lora_config.lora_config
```

### TrainingConfigBuilderNode
```python
training_config = TrainingConfigBuilderNode(
    id="7",
    learning_rate=1e-4,      # 学习率（GRPO/DPO通常用1e-6）
    batch_size=2,            # 批次大小
    grad_accum_steps=2,      # 梯度累积步数
    num_epochs=1,            # 训练轮次
    save_steps=500,          # 保存间隔
    save_total_limit=3,      # 最大保存数量
)
# 输出: training_config.training_config
```

### ModelTrainSFTNode
```python
sft_train = ModelTrainSFTNode(
    id="8",
    dataset_config=dataset_config.dataset_config,
    model_config=model_config.model_config,
    lora_config=lora_config.lora_config,
    training_config=training_config.training_config,
    accelerate_config=accelerate_config.accelerate_config,
    # wandb_config=wandb.wandb_config,  # 可选
    output_path="/workspace/output/sft/",
)
# 输出: sft_train.model_output_path
```

## 示例工作流

### 示例1：数据集预处理工作流

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

### 示例2：SFT 训练工作流

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

### 示例3：GRPO 训练工作流

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

## 生成工作流的步骤

1. **理解用户需求**：确定训练类型（SFT/DPO/GRPO）、数据集、模型
2. **检索知识库**：用 grep 搜索相关节点文档和连接模式
3. **选择节点组合**：
   - 数据加载：CloneAndCacheDataset → PathJoinNode → DatasetToJsonlNode（如需转换）
   - 字段映射：DatasetConfigBuilderTextNode 或 DatasetConfigBuilderVisionNode
   - 数据集配置：DatasetConfigBuilderNode
   - 模型加载：CloneAndCacheModel → ModelConfigBuilderNode
   - 训练配置：LoraConfigBuilderNode + TrainingConfigBuilderNode + AccelerateConfigBuilderNode
   - 训练执行：ModelTrainSFTNode / ModelTrainDPONode / ModelTrainGRPONode
   - 后处理：ModelMergeLoraNode → VLLMInference → TestLLMNode（可选）
4. **组装 DSL**：按依赖顺序排列节点，确保被引用节点在前
5. **输出结果**：返回完整的 DSL 代码块
