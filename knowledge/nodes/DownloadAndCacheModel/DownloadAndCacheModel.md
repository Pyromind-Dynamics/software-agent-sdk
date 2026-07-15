# Download Model 节点

![alt text](/imgs/DownloadAndCacheModel/DownloadAndCacheModel.png)  

## 1.1 功能概述

下载大模型资源

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述                                                    |
|------|---------|------|-------------------------------------------------------|
| modelname | STRING | 是 | 模型资源路径  默认值 : `Qwen/Qwen2.5-1.5B-Instruct` |
| cache_dir | STRING | 是 | 目标文件夹。  默认值 : `/workspace/models/Qwen/Qwen2.5-1.5B-Instruct` |
| download_source | STRING | 是 | 下载源。默认值：`huggingface` Options: huggingface, modelscope |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| model_path | STRING | 模型引用 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/DownloadAndCacheModel/DownloadAndCacheModel.json`](../../workflow/DownloadAndCacheModel/DownloadAndCacheModel.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/DownloadAndCacheModel/DownloadAndCacheModel.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `downloadAndCacheModel` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |


输出示例：

```json
{
  "task_id": "4808",
  "task_name": "unsaved-workflow",
  "status": "success",
  "error_message": null,
  "nodes": [
    {
      "node_key": "#1",
      "node_id": 18650,
      "node_type": "DownloadAndCacheModel",
      "label": "",
      "start_at": "2026-06-08 11:17:53",
      "end_at": "2026-06-08 11:23:22",
      "duration": "0:05:29",
      "input": {
        "modelname": "Qwen/Qwen2.5-1.5B-Instruct",
        "cache_dir": "/workspace/models/Qwen/Qwen2.5-1.5B-Instruct",
        "download_source": "modelscope"
      },
      "output": {
        "model_path": "/workspace/models/Qwen/Qwen2.5-1.5B-Instruct"
      },
      "raw": {}
    }
  ]
}
```