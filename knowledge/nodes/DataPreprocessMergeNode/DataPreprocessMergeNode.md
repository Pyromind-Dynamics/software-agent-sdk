# Prompt & Reasoning Merge 节点

![alt text](/imgs/DataPreprocessMergeNode/DataPreprocessMergeNode.png)

## 1.1 功能概述

Merge reasoning/category/prompts into training data.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| metadata_path | STRING | 是 | 未声明 | — | Metadata path |
| output_path | STRING | 是 | 未声明 | — | 持久化输出目录或文件路径，使用 `/workspace/...` |
| reasoning_path | STRING | 是 | 未声明 | — | Reasoning 流水线输出路径 |
| base_prompt | STRING | 否 | `""` | — | Base prompt |
| output_format | STRING | 否 | `"jsonl"` | — | Output format |
| base_prompt_file | STRING | 否 | `""` | — | Base prompt file |
| category_annotations | STRING | 否 | `""` | — | Category annotations |
| prompt_augmentation_path | STRING | 否 | `""` | — | Prompt augmentation path |
| system_prompt | STRING | 否 | `""` | — | System prompt |
| system_prompt_file | STRING | 否 | `""` | — | System prompt file |
| copy_columns | STRING | 否 | `""` | — | Copy columns |
| label_column | STRING | 否 | `"component_type"` | — | Label column |
| gt_column | STRING | 否 | `"text"` | — | Gt column |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| result_output_path | STRING | 流水线输出路径，位于 `/workspace/...` |
