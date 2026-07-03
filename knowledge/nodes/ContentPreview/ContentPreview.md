# Content Preview 节点

![alt text](/imgs/ContentPreview/ContentPreview.png)

## 1.1 功能概述

按照指定文件格式展示输入内容。可用于将文本类内容按 `txt`、`json`、`xml` 或 `python` 格式预览。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| content | * | 是 | 要展示的内容 |
| file_format | STRING | 是 | 预览文件格式。默认值：`txt` Options: txt, json, xml, python |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| content | * | 原始预览内容 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/ContentPreview/ContentPreview.json`](../../workflow/ContentPreview/ContentPreview.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/ContentPreview/ContentPreview.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `contentPreview` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
