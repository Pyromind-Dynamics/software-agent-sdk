# Image Preview 节点

![alt text](/imgs/PreviewImageEx/PreviewImageEx.png)  

## 1.1 功能概述


> **TODO**: 添加功能概述


## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| image_file_path | STRING | 是 | 字符串。默认值：- |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| image_path | STRING | 字符串 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/PreviewImageEx/PreviewImageEx.json`](../../workflow/PreviewImageEx/PreviewImageEx.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/PreviewImageEx/PreviewImageEx.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `previewImageEx` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
