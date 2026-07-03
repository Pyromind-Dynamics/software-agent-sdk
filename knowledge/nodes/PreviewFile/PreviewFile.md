# File Preview 节点

![alt text](/imgs/PreviewFile/PreviewFile.png)

## 1.1 功能概述

输入存储路径并预览文件。支持的格式包括图片和 TXT 文件。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| file_path | STRING | 是 | 要预览的文件存储路径 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| file_path | STRING | 已预览文件的存储路径 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/PreviewFile/PreviewFile.json`](../../workflow/PreviewFile/PreviewFile.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/PreviewFile/PreviewFile.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `previewFile` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
