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

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/DataPreprocessMergeNode/DataPreprocessMergeNode.json`](../../workflow/DataPreprocessMergeNode/DataPreprocessMergeNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/DataPreprocessMergeNode/DataPreprocessMergeNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `dataPreprocessMergeNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
