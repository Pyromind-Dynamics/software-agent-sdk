# Merge LoRA 节点

![alt text](/imgs/ModelMergeLoraNode/ModelMergeLoraNode.png)

## 1.1 功能概述

将 LoRA adapter 合并到基座模型，输出独立模型与 processor 目录。`model_path` 在当前 schema 中为必填端口，生成工作流时应显式填写或连接基座模型路径；不能因为默认值为空字符串就省略该输入。当前合并节点仅允许单张 GPU。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| lora_path | STRING | 是 | 未声明 | — | Lora path |
| output_path | STRING | 是 | 未声明 | — | 合并模型的持久化输出目录，使用 `/workspace/...` |
| model_path | STRING | 是 | `""` | — | HuggingFace 模型名或本地路径 |
| gpu_count | INT | 是 | `1` | 最小 1；最大 1 | GPU 数量 |
| gpu_product | STRING | 是 | `"NVIDIA-H100-NVL"` | 可选值：`"NVIDIA-H100-NVL"`、`"NVIDIA-L40S"` | GPU 型号（以部署环境实际可用的显卡为准） |
| model_type | STRING | 否 | `"auto"` | 可选值：`"auto"`、`"qwen3vl"`、`"qwen3.5"` | 模型架构类型（`auto`、`qwen3vl`、`qwen3.5`） |

> **注意：** GPU 型号与数量以部署环境实际可用的显卡为准。

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| merged_model_path | STRING | 合并后全量模型路径 |
