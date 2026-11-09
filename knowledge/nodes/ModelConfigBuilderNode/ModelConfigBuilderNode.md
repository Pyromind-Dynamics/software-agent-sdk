# Model Config Builder 节点

![alt text](/imgs/ModelConfigBuilderNode/ModelConfigBuilderNode.png)

## 1.1 功能概述

构建 `model_config` YAML 字符串（模型名称与类型）。LoRA 微调请单独使用 **Lora Config Builder** 节点。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| model_path | STRING | 是 | HuggingFace 模型名或本地路径。默认值：`Qwen/Qwen3-VL-2B-Instruct` |
| model_type | STRING | 否 | 模型架构类型（`auto`、`qwen3vl`、`qwen3.5`）。选项：`auto`, `qwen3vl`, `qwen3.5` |

`model_path` 的默认值不表示该模型存在于 Clone 模板中。模型获取方式以
`CloneAndCacheModel` 的当前环境可用列表为准；不在列表中时使用
`DownloadAndCacheModel`。

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| model_config | STRING\|MODEL_CONFIG | 模型配置 YAML 字符串 |
