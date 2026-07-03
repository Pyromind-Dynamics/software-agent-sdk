# Save Model 节点

![alt text](/imgs/SaveModel/SaveModel.png)  

## 1.1 功能概述

保存训练好的RL模型

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| model | STRING | 是 | 字符串。默认值：- |
| save_path | STRING | 是 | 字符串。默认值：`path/to/save/model` |
| model_name | STRING | 是 | 字符串。默认值：`rl_model` |
| save_format | STRING | 是 | 字符串。默认值：`pkl` Options: pkl, onnx, torchscript, safetensors |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| save_path | STRING | 字符串 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/SaveModel/SaveModel.json`](../../workflow/SaveModel/SaveModel.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/SaveModel/SaveModel.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `saveModel` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
