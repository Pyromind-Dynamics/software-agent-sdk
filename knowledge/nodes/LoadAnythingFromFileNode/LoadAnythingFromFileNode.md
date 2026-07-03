# Load Anything From File 节点

![alt text](/imgs/LoadAnythingFromFileNode/LoadAnythingFromFileNode.png)

## 1.1 功能概述

Load content from a file path.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| file_path | STRING | 是 | 加载或保存的文件路径 |
| encoding | STRING | 否 | Encoding。默认值：`utf-8` |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| loaded_file_path | STRING | 已加载文件路径 |
| content | * | 已加载文件内容 |
| read_bytes | INT | 读取字节数 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/LoadAnythingFromFileNode/LoadAnythingFromFileNode.json`](../../workflow/LoadAnythingFromFileNode/LoadAnythingFromFileNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/LoadAnythingFromFileNode/LoadAnythingFromFileNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `loadAnythingFromFileNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
