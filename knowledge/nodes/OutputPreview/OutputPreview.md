# Output Preview 节点

![alt text](/imgs/OutputPreview/OutputPreview.png)

## 1.1 功能概述

用于 workflow 输出预览集成的系统工具节点。当前 SDK 定义未暴露可配置输入或输出。

## 1.2 输入类型

该节点没有输入。

## 1.3 输出类型

该节点没有输出。

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/OutputPreview/OutputPreview.json`](../../workflow/OutputPreview/OutputPreview.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/OutputPreview/OutputPreview.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `outputPreview` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
