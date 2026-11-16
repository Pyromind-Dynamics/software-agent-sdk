# SFT Training 节点

![alt text](/imgs/ModelTrainSFTNode/ModelTrainSFTNode.png)

## 1.1 功能概述

Run SFT training and return output directory.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| output_path | STRING | 是 | 未声明 | — | 持久化训练输出目录，必须使用 `/workspace/...` |
| dataset_config | STRING\|DATASET_CONFIG | 是 | 未声明 | — | 已组装的数据集配置 YAML 字符串 |
| training_config | STRING\|TRAINING_CONFIG | 是 | 未声明 | — | 训练配置 YAML 字符串（`BaseTrainingConfig` 字段） |
| model_config | STRING\|MODEL_CONFIG | 是 | 未声明 | — | 模型配置 YAML 字符串 |
| wandb_config | WANDB_CONFIG | 是 | 未声明 | — | W&B 配置对象 |
| accelerate_config | ACCELERATE_CONFIG | 是 | 未声明 | — | Accelerate 启动配置 YAML |
| gpu_count | INT | 是 | `1` | 最小 1；最大 8 | GPU 数量 |
| gpu_product | STRING | 是 | `"NVIDIA-H100-NVL"` | 可选值：`"NVIDIA-H100-NVL"`、`"NVIDIA-L40S"` | GPU 型号（以部署环境实际可用的显卡为准） |
| lora_config | STRING\|LORA_CONFIG | 否 | 未声明 | — | 来自 Lora Config Builder 的 LoRA 配置 YAML |
| thinking_as_input_ratio | FLOAT | 否 | `0.0` | — | 将思维链内容作为输入的比例 |

> **注意：** GPU 型号与数量以部署环境实际可用的显卡为准。

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| model_output_path | STRING | 训练输出模型目录 |
