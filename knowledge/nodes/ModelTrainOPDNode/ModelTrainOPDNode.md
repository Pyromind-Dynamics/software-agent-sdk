# OPD 训练

![alt text](/imgs/ModelTrainOPDNode/ModelTrainOPDNode.png)

## 1.1 功能概述

运行 OPD（On-Policy Distillation / GKD）训练，需分别提供 student 与 teacher 的 `model_config`。返回训练输出目录。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| output_path | STRING | 是 | 未声明 | — | 持久化训练输出目录，必须使用 `/workspace/...` |
| dataset_config | STRING\|DATASET_CONFIG | 是 | 未声明 | — | 已组装的数据集配置 YAML 字符串 |
| training_config | STRING\|TRAINING_CONFIG | 是 | 未声明 | — | 训练配置 YAML 字符串（`BaseTrainingConfig` 字段） |
| model_config | STRING\|MODEL_CONFIG | 是 | 未声明 | — | 模型配置 YAML 字符串 |
| teacher_model_config | STRING\|TEACHER_MODEL_CONFIG | 是 | 未声明 | — | Teacher 模型配置 YAML 字符串（OPD/GKD） |
| accelerate_config | ACCELERATE_CONFIG | 是 | 未声明 | — | Accelerate 启动配置 YAML |
| gpu_count | INT | 是 | `1` | 最小 1；最大 8 | GPU 数量 |
| gpu_product | STRING | 是 | `"NVIDIA-H100-NVL"` | 可选值：`"NVIDIA-H100-NVL"`、`"NVIDIA-L40S"` | GPU 型号（以部署环境实际可用的显卡为准） |
| lmbda | FLOAT | 否 | `0.5` | — | OPD lambda 混合系数 |
| beta | FLOAT | 否 | `0.5` | — | OPD beta 系数 |
| temperature | FLOAT | 否 | `0.9` | — | 采样温度 |
| max_new_tokens | INT | 否 | `200` | — | Teacher 生成的最大 token 数（OPD） |
| seq_kd | BOOLEAN | 否 | `false` | — | 是否启用序列级知识蒸馏 |
| thinking_as_input_ratio | FLOAT | 否 | `0.0` | — | 将思维链内容作为输入的比例 |
| wandb_config | WANDB_CONFIG | 否 | 未声明 | — | W&B 配置对象 |

> **注意：** GPU 型号与数量以部署环境实际可用的显卡为准。

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| model_output_path | STRING | 训练输出模型目录 |
