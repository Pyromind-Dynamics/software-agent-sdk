# Inference (SGlang) 节点

![alt text](/imgs/Inference/Inference.png)

## 1.1 功能概述

此节点根据传入的参数，启动一个推理服务，为后续节点持续提供推理服务，直到工作流完结后结束服务

## 1.2 输入类型

| 参数          | 数据类型 | 必填 | 描述                                                                                                                |
|-------------|---------|----|-------------------------------------------------------------------------------------------------------------------|
| model_path | STRING | 是 | 模型引用。默认值：- |
| environment | ENV | 否 | 环境变量名。默认值：- |
| port | INT | 是 | 推理服务端口号 |
| gpu_count | INT | 是 | 推理服务使用的显卡个数 |
| gpu_product | STRING | 是 | 显卡类型。枚举：NVIDIA-H200、NVIDIA-H100-80GB-HBM3、NVIDIA-L40S |



## 1.3 输出类型

| 参数 | 数据类型 | 描述    |
|------|---------|-------|
| endpoint | STRING | 推理服务的endpoint |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/Inference/Inference.json`](../../workflow/Inference/Inference.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
export PYROMIND_CLUSTER=<your-cluster>
python -m pyromind_sdk.test_run_workflow_cli workflow/Inference/Inference.json --pretty
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `inference` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |





输出示例:

```json
{
  "task_id": "4805",
  "task_name": "start-inference-sglang",
  "status": "success",
  "error_message": null,
  "nodes": [
    {
      "node_key": "#1",
      "node_id": 18647,
      "node_type": "Inference",
      "label": "Inference (SGlang)",
      "start_at": "2026-06-08 11:05:03",
      "end_at": "2026-06-08 11:05:13",
      "duration": "0:00:10",
      "input": {
        "model_path": "",
        "port": 3000,
        "gpu_count": 1,
        "gpu_product": "NVIDIA-L40S",
        "environment": ""
      },
      "output": null
    },
    {
      "node_key": "#2",
      "node_id": 18646,
      "node_type": "LoadModel",
      "label": "Load Model",
      "start_at": "2026-06-08 11:04:43",
      "end_at": "2026-06-08 11:04:56",
      "duration": "0:00:13",
      "input": {
        "model_path": "/workspace/models/Qwen/Qwen2.5-1.5B-Instruct/"
      },
      "output": {
        "model": "/workspace/models/Qwen/Qwen2.5-1.5B-Instruct/"
      },
      "raw": {}
    }
  ]
}
```
