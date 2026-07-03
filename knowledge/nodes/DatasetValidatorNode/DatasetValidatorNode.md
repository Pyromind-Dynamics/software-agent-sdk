# Dataset Validator 数据集校验

![alt text](/imgs/DatasetValidatorNode/DatasetValidatorNode.png)

## 1.1 功能概述

校验已组装的 `dataset_config`，生成 dataloader 预览，并输出校验指标与 HTML 预览文件路径。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| dataset_config | STRING | 是 | 已组装的数据集配置 YAML 字符串 |
| check_reasoning | BOOLEAN | 否 | 是否校验数据集中的 reasoning 字段。默认值：false |
| limit | INT | 否 | 校验预览的最大样本数（0 表示不限制）。默认值：`0` |
| verbose | BOOLEAN | 否 | 是否输出详细校验日志。默认值：false |
| preview_html_path | STRING | 否 | HTML 预览输出路径。默认值：`/workspace/datasets/preview.html` |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| metrics_yaml | STRING | 校验指标 YAML 字符串 |
| output_preview_path | STRING | 生成的 HTML 预览文件路径 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/DatasetValidatorNode/DatasetValidatorNode.json`](../../workflow/DatasetValidatorNode/DatasetValidatorNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/DatasetValidatorNode/DatasetValidatorNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `datasetValidatorNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
