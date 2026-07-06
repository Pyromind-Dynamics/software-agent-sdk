---
name: generate-workflow-dsl
description: >-
  当用户需要生成 Pyromind 平台上的模型训练工作流时使用此技能。
  优先复用技能内置模板，并按需读取相关节点文档，生成符合 DSL 格式的工作流代码。
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
1. 先从本技能内置的示例工作流中选择最接近的模板，不要从零开始搜索
2. 根据用户需求选择合适的节点组合，并按需读取相关节点文档确认参数
3. 按照 DSL 语法格式创建或修改当前工作目录下的 `workflow.py`
4. 创建或修改后正常结束；后端会在本轮 run 结束后自动将当前文件同步给前端

可读取的知识库路径均位于路由提示中的绝对知识库目录下。需要补充检索时，使用 `grep` 传入该绝对路径或其子目录路径：
- 平台基础用法：`<知识库绝对路径>/basic/`
- JupyterLab 与脚本训练：`<知识库绝对路径>/jupyterlab/`
- Python SDK 与脚本训练 API：`<知识库绝对路径>/sdk/`
- Studio 拖拽工作流：`<知识库绝对路径>/studio/`
- 节点 I/O、参数与端口定义：`<知识库绝对路径>/nodes/<NodeType>/<NodeType>.md`
- 外部数据处理样例：`<知识库绝对路径>/dataset_processing_workflow.py`

## 检索与读取策略

优先级：**内置示例模板 → 直接读取相关节点文档 → 必要时宽泛 grep**。

1. 本技能已经包含数据处理、SFT、GRPO 示例。用户要生成 SFT 时，直接以“示例2：SFT 训练工作流”为模板调整数据集、模型和训练参数；用户要生成 GRPO 时，直接以“示例3：GRPO 训练工作流”为模板调整。不要为了确认示例存在而先 grep 完整的 `# workflow: ...` 标题或注释。
2. 节点文档路径是确定的：`<知识库绝对路径>/nodes/<NodeType>/<NodeType>.md`。当已经知道要用哪些节点时，直接用 `file_editor` 查看对应文件，不要先在整个知识库里 grep 节点名来“发现”文件。
3. grep 只用于补充检索，并使用宽泛关键词：`SFT`、`DPO`、`GRPO`、`training`、`dataset`、`reward`，或精确的 `NodeType`。避免使用依赖格式的 pattern，例如完整工作流标题、Markdown 标题锚点、`^###`、`# workflow: ...`。
4. 如果只是生成常见训练工作流，通常只需要读取被选中节点的契约文档；只有用户问平台概念、Studio 操作、SDK 脚本写法，或节点参数仍不明确时，才检索 `basic/`、`studio/`、`sdk/`、`jupyterlab/`。

常见工作流需要优先读取的节点文档：

| 场景 | 相关节点 |
|------|----------|
| 数据处理/验证 | CloneAndCacheDataset、PathJoinNode、DatasetToJsonlNode、DatasetConfigBuilderTextNode、DatasetConfigBuilderVisionNode、DatasetConfigBuilderNode、DatasetValidatorNode |
| SFT | CloneAndCacheDataset、CloneAndCacheModel、PathJoinNode、DatasetConfigBuilderTextNode、DatasetConfigBuilderMessageNode、DatasetConfigBuilderVisionNode、DatasetConfigBuilderNode、ModelConfigBuilderNode、LoraConfigBuilderNode、TrainingConfigBuilderNode、AccelerateConfigBuilderNode、ModelTrainSFTNode、ModelMergeLoraNode |
| DPO | CloneAndCacheDataset、CloneAndCacheModel、PathJoinNode、DatasetConfigBuilderTextNode、DatasetConfigBuilderMessageNode、DatasetConfigBuilderNode、ModelConfigBuilderNode、LoraConfigBuilderNode、TrainingConfigBuilderNode、AccelerateConfigBuilderNode、ModelTrainDPONode |
| GRPO | CloneAndCacheDataset、CloneAndCacheModel、PathJoinNode、DatasetConfigBuilderTextNode、DatasetConfigBuilderVisionNode、DatasetConfigBuilderNode、ModelConfigBuilderNode、LoraConfigBuilderNode、TrainingConfigBuilderNode、AccelerateConfigBuilderNode、RewardItemBuilderNode、RewardConfigBuilderNode、GRPOTrainingExtraConfigBuilderNode、ModelTrainGRPONode |

## DSL 语法格式

`workflow.py` 是**声明式 DSL**，只是借用 Python 语法描述节点及连线，并不是可以在本地直接
运行的 Python 脚本；不要尝试执行它或按 Python 运行时语义推理其行为。测试/验证工作流是否能
真正跑通，需要使用 `debug-workflow` 技能里的 `debug_workflow` 工具（用户说"测试”“test”“调试”
“debug”“试跑”均指此意）。

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
2. **选择模板**：优先复用本技能内置示例。SFT 用示例2，GRPO 用示例3，数据处理/验证用示例1；DPO 可在 SFT 骨架上替换为 DPO 数据字段与 `ModelTrainDPONode`
3. **读取必要节点契约**：根据已选模板和用户需求，直接查看 `<知识库绝对路径>/nodes/<NodeType>/<NodeType>.md`。只读取会实际使用或参数不确定的节点文档，不要用高精度标题 pattern 搜索示例
4. **选择节点组合**：
   - 数据加载：CloneAndCacheDataset → PathJoinNode → DatasetToJsonlNode（如需转换）
   - 字段映射：DatasetConfigBuilderTextNode 或 DatasetConfigBuilderVisionNode
   - 数据集配置：DatasetConfigBuilderNode
   - 模型加载：CloneAndCacheModel → ModelConfigBuilderNode
   - 训练配置：LoraConfigBuilderNode + TrainingConfigBuilderNode + AccelerateConfigBuilderNode
   - 训练执行：ModelTrainSFTNode / ModelTrainDPONode / ModelTrainGRPONode
   - 后处理：ModelMergeLoraNode → VLLMInference → TestLLMNode（可选）
5. **组装 DSL**：按依赖顺序排列节点，确保被引用节点在前
6. **写入文件**：使用 `apply_patch` 将 DSL 写入当前工作目录根路径 `workflow.py`；如果是在修改已有工作流，也只编辑这个固定相对路径。不要手写会话目录的长绝对路径；如必须使用 `file_editor`，也传入 `workflow.py`，由运行时解析到实际文件。不要只说明已经生成，必须实际调用工具创建或修改文件
7. **结束处理**：如果 `workflow.py` 被创建或修改，正常结束；不要调用额外发布工具，后端会自动同步
