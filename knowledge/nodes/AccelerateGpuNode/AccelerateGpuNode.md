# Accelerate GPU Node 节点

![alt text](/imgs/AccelerateGpuNode/AccelerateGpuNode.png)

## 1.1 功能概述

GPU node example using accelerate launch mode

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| message | STRING | 是 | Message |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| gpu_info | STRING | GPU 信息字符串 |
| execution_result | STRING | 命令执行结果 |
| execution_output | STRING | 命令 stdout/stderr 输出 |
| message_output | STRING | 状态消息 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/AccelerateGpuNode/AccelerateGpuNode.json`](../../workflow/AccelerateGpuNode/AccelerateGpuNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/AccelerateGpuNode/AccelerateGpuNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `accelerateGpuNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
