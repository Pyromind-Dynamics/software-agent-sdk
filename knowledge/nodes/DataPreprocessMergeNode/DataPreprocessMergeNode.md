# Prompt & Reasoning Merge 节点

![alt text](/imgs/DataPreprocessMergeNode/DataPreprocessMergeNode.png)

## 1.1 功能概述

Merge reasoning/category/prompts into training data.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| metadata_path | STRING | 是 | Metadata path |
| output_path | STRING | 是 | 输出目录或文件路径 |
| reasoning_path | STRING | 是 | Reasoning 流水线输出路径 |
| base_prompt | STRING | 否 | Base prompt。默认值：空字符串 |
| output_format | STRING | 否 | Output format。默认值：`jsonl` |
| base_prompt_file | STRING | 否 | Base prompt file。默认值：空字符串 |
| category_annotations | STRING | 否 | Category annotations。默认值：空字符串 |
| prompt_augmentation_path | STRING | 否 | Prompt augmentation path。默认值：空字符串 |
| system_prompt | STRING | 否 | System prompt。默认值：空字符串 |
| system_prompt_file | STRING | 否 | System prompt file。默认值：空字符串 |
| copy_columns | STRING | 否 | Copy columns。默认值：空字符串 |
| label_column | STRING | 否 | Label column。默认值：`component_type` |
| gt_column | STRING | 否 | Gt column。默认值：`text` |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| result_output_path | STRING | 流水线输出路径 |
