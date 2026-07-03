# Chain of Thought Generation 节点

![alt text](/imgs/DataPreprocessReasoningNode/DataPreprocessReasoningNode.png)

## 1.1 功能概述

Run data_preprocess reasoning pipeline and emit output path.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| input_path | STRING | 是 | Input path |
| output_path | STRING | 是 | 输出目录或文件路径 |
| api_base | STRING | 是 | Api base |
| model | STRING | 是 | Model |
| api_key | STRING | 是 | Api key |
| system_prompt | STRING | 否 | System prompt。默认值：空字符串 |
| user_prompt | STRING | 否 | User prompt。默认值：空字符串 |
| user_prompt_column | STRING | 否 | User prompt column。默认值：空字符串 |
| format_lang | STRING | 否 | Format lang。默认值：`en` |
| max_tokens | INT | 否 | 每次 API 调用的最大 token 数。默认值：`1024` |
| thinking | BOOLEAN | 否 | Thinking。默认值：false |
| limit | INT | 否 | 校验预览的最大样本数（0 表示不限制）。默认值：`0` |
| concurrency | INT | 否 | Concurrency。默认值：`1` |
| gt_column | STRING | 否 | Gt column。默认值：`text` |
| num_variations | INT | 否 | Num variations。默认值：`1` |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| reasoning_path | STRING | Reasoning 流水线输出路径 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/DataPreprocessReasoningNode/DataPreprocessReasoningNode.json`](../../workflow/DataPreprocessReasoningNode/DataPreprocessReasoningNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/DataPreprocessReasoningNode/DataPreprocessReasoningNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `dataPreprocessReasoningNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
