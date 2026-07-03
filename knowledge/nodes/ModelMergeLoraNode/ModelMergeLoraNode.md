# Merge LoRA 节点

![alt text](/imgs/ModelMergeLoraNode/ModelMergeLoraNode.png)

## 1.1 功能概述

Merge LoRA adapter into base weights (PEFT merge_and_unload) and save model + processor to a standalone directory. Base model id/path may be omitted if adapter_config.json contains base_model_path_or_path.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| lora_path | STRING | 是 | Lora path |
| output_path | STRING | 是 | 输出目录或文件路径 |
| model_path | STRING | 否 | HuggingFace 模型名或本地路径。默认值：空字符串 |
| model_type | STRING | 否 | 模型架构类型（`auto`、`qwen3vl`、`qwen3.5`）。默认值：`auto` |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| merged_model_path | STRING | 合并后全量模型路径 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/ModelMergeLoraNode/ModelMergeLoraNode.json`](../../workflow/ModelMergeLoraNode/ModelMergeLoraNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/ModelMergeLoraNode/ModelMergeLoraNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `modelMergeLoraNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
