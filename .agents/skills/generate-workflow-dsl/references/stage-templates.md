# 阶段模板与组合契约

这里只定义阶段边界、选择条件和产物绑定。仅在本文件未给出所需的具体节点参数或端口时查
`node-reference.md`；训练数值决策查 `parameter-decision.md`。

## 阶段目录

| 阶段 | 核心节点 | 输入 | 输出 |
|---|---|---|---|
| 数据入口 | PathJoin+Load / Clone / Download | 路径或数据标识 | `dataset_path` |
| 数据配置 | Kind/Extra + Dataset Config | 路径、字段映射 | `dataset_config` |
| 模型入口 | Clone/Download + Model Config | 模型标识或模型路径 | `model_config` |
| SFT | ModelTrainSFTNode | dataset/model/training/accelerate | `model_output_path` |
| DPO | ModelTrainDPONode | 偏好数据及通用配置 | `model_output_path` |
| GRPO | ModelTrainGRPONode | prompt、reward 及通用配置 | `model_output_path` |
| Merge | ModelMergeLoraNode | 基模路径、LoRA 输出 | `merged_model_path` |
| 推理 | VLLMInference | 模型路径 | `endpoint` |
| 评测 | Metrics + ModelEvalApiNode | endpoint、评测数据 | `benchmark_output_path` |

## 选择规则

- 明确要求 Benchmark：数据配置 → 基模 → VLLM → Metrics → Eval，不添加训练节点。
- 默认训练：数据配置 → 基模 → SFT；LoRA 且有下游阶段时接 Merge。
- 有 chosen/rejected：把训练阶段替换为 DPO。
- 有可程序化验证目标或 reward：使用 GRPO，并先准备 Reward。
- 训后评测：训练 → Merge（LoRA）→ VLLM → Eval，推理必须吃训练产物。

SFT、DPO、GRPO 的通用输入如下；只按训练类型增加专属配置：

| 训练节点 | 通用输入 | 专属输入 |
|---|---|---|
| ModelTrainSFTNode | dataset/model/training/accelerate/lora | 无 |
| ModelTrainDPONode | dataset/model/training/accelerate/lora | 无 |
| ModelTrainGRPONode | dataset/model/training/accelerate/lora | reward、grpo_extra |

三个训练节点均设置 `output_path`、`gpu_count`、`gpu_product`，并输出
`model_output_path`。不要为相同结构复制完整代码片段。

## 组合契约

### LoRA 合并

```python
merge = ModelMergeLoraNode(
    id="21",
    model_path=base_model.model_path,
    lora_path=sft.model_output_path,
    output_path="/workspace/output/merged/",
    model_type="auto",
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
)
```

### SFT → Merge → GRPO

为合并产物创建新的模型配置；不要复用基模配置，也不要把合并路径写死：

```python
grpo_model_config = ModelConfigBuilderNode(
    id="22",
    model_path=merge.merged_model_path,
    model_type="auto",
)
grpo = ModelTrainGRPONode(
    id="23",
    model_config=grpo_model_config.model_config,
    dataset_config=grpo_dataset_config.dataset_config,
    training_config=grpo_training_config.training_config,
    reward_config=reward_config.reward_config,
    grpo_extra_config=grpo_extra.grpo_extra_config,
    accelerate_config=grpo_accelerate_config.accelerate_config,
    lora_config=grpo_lora_config.lora_config,
    output_path="/workspace/output/grpo/",
    gpu_count=1,
    gpu_product="NVIDIA-H100-80GB-HBM3",
)
```

### Benchmark / Eval

`VLLMInference.model_path` 的绑定按目标选择：

- 基模 Benchmark：`base_model.model_path`。
- LoRA 训后评测：`merge.merged_model_path`。
- Full 训后评测：训练节点的 `model_output_path`。

`ModelEvalApiNode.endpoint` 绑定 `vllm.endpoint`。训练前后比较必须复用同一评测数据切分、
字段映射、Metric 和 `max_samples`。
