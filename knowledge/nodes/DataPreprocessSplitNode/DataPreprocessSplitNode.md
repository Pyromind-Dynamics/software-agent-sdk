# Train-Validation Split 节点

![alt text](/imgs/DataPreprocessSplitNode/DataPreprocessSplitNode.png)

## 1.1 功能概述

Split metadata into train and validation datasets from a shared output path template

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| input_path | STRING | 是 | Input path |
| output_path | STRING | 是 | 输出目录或文件路径 |
| train_ratio | FLOAT | 否 | Train ratio。默认值：`0.8` |
| seed | INT | 否 | 随机种子。默认值：`42` |
| shuffle | BOOLEAN | 否 | Shuffle。默认值：true |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| train_output_path | STRING | 训练集划分输出路径 |
| val_output_path | STRING | 验证集划分输出路径 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/DataPreprocessSplitNode/DataPreprocessSplitNode.json`](../../workflow/DataPreprocessSplitNode/DataPreprocessSplitNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/DataPreprocessSplitNode/DataPreprocessSplitNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `dataPreprocessSplitNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
