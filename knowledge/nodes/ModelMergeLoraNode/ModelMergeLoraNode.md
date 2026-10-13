# Merge LoRA 节点

![alt text](/imgs/ModelMergeLoraNode/ModelMergeLoraNode.png)

## 1.1 功能概述

Merge LoRA adapter into base weights (PEFT merge_and_unload) and save model + processor to a standalone directory. Base model id/path may be omitted if adapter_config.json contains base_model_path_or_path.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| lora_path | STRING | 是 | Lora path |
| output_path | STRING | 是 | 输出目录或文件路径 |
| model_path | STRING | 是 | HuggingFace 模型名或本地路径。默认值：空字符串 |
| model_type | STRING | 否 | 模型架构类型（`auto`、`qwen3vl`、`qwen3.5`）。默认值：`auto` |
| gpu_count | INT | 是 | GPU 数量。默认值：`1` |
| gpu_product | STRING | 是 | GPU 型号（以部署环境实际可用的显卡为准）。选项：`NVIDIA-H200`、`NVIDIA-H100-80GB-HBM3` |

> **注意：** GPU 型号与数量以部署环境实际可用的显卡为准。

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| merged_model_path | STRING | 合并后全量模型路径 |
