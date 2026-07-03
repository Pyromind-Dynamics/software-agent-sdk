# Save Anything To File 节点

![alt text](/imgs/SaveAnythingToFileNode/SaveAnythingToFileNode.png)

## 1.1 功能概述

Save any provided content to a file path.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| file_path | STRING | 是 | 加载或保存的文件路径 |
| content | * | 否 | 已加载文件内容。默认值：空字符串 |
| append | BOOLEAN | 否 | Append。默认值：false |
| ensure_parent | BOOLEAN | 否 | Ensure parent。默认值：true |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| saved_file_path | STRING | 已保存文件路径 |
| written_bytes | INT | 写入字节数 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/SaveAnythingToFileNode/SaveAnythingToFileNode.json`](../../workflow/SaveAnythingToFileNode/SaveAnythingToFileNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/SaveAnythingToFileNode/SaveAnythingToFileNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `saveAnythingToFileNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
