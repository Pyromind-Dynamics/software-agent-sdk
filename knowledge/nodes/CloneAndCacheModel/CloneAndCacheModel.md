# Clone Model 节点

![alt text](/imgs/CloneAndCacheModel/CloneAndCacheModel.png)  

## 1.1 功能概述

从 pyromind 环境中的预置模板快速克隆模型到工作区缓存。模型已存储在环境中，克隆速度极快。克隆后的模型可被下游训练或推理节点使用。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| model | STRING | 是 | 模型标识。从 pyromind 环境中可用的模型模板动态列表中选取（例如 `Qwen/Qwen3-0.6B`、`Qwen/Qwen3-8B`）。列表在运行时从模板目录加载。 |
| target_path | STRING | 是 | 工作区中克隆模型的目标目录。默认值：`/workspace/models/` Options: /workspace/models/ |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| model_path | STRING | 克隆到工作区缓存的模型引用，可传递给下游接受模型输入的训练或推理节点。 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/CloneAndCacheModel/CloneAndCacheModel.json`](../../workflow/CloneAndCacheModel/CloneAndCacheModel.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/CloneAndCacheModel/CloneAndCacheModel.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `cloneAndCacheModel` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |

```bash
python -m pyromind_sdk.test_run_workflow_cli workflow/CloneAndCacheModel/CloneAndCacheModel.json --pretty
```

输出示例：

```json
{
  "task_id": "4879",
  "task_name": "clonemodel",
  "status": "success",
  "error_message": null,
  "nodes": [
    {
      "node_key": "#1",
      "node_id": 18889,
      "node_type": "CloneAndCacheModel",
      "label": "",
      "start_at": "2026-06-09 03:32:14",
      "end_at": "2026-06-09 03:32:23",
      "duration": "0:00:09",
      "input": {
        "model": "Qwen/Qwen3-1.7B",
        "target_path": "/workspace/models/"
      },
      "output": {
        "model_path": "/workspace/models/Qwen/Qwen3-1.7B"
      },
      "raw": {}
    }
  ]
}
```
