# Load Model 节点

![alt text](/imgs/LoadModel/LoadModel.png)

## 1.1 功能概述

加载基础模型用于 RL 训练。该节点从指定路径加载预训练模型，并将模型引用传递给下游节点（如训练、推理等）。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| model_path | STRING | 是 | 模型路径，指向预训练模型所在的目录或文件。默认值：`path/to/model` |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| model | STRING | 加载的模型引用，供下游节点使用 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/LoadModel/LoadModel.json`](../../workflow/LoadModel/LoadModel.json)。

示例配置：
```json
{
  "id": "c0641c5b-0f9e-4c9c-9c81-7799043e7606",
  "name": "LoadModel",
  "nodes": [
    {
      "id": "1",
      "type": "default",
      "position": {
        "x": 629.769820971867,
        "y": 565.9079283887468
      },
      "data": {
        "display_name": "Load Model",
        "nodeType": "LoadModel",
        "nodeDefinition": {
          "input": {
            "required": {
              "model_path": [
                "STRING",
                {
                  "default": "path/to/model"
                }
              ]
            }
          },
          "output": [
            "MODEL"
          ],
          "output_is_list": [
            false
          ],
          "output_name": [
            "model"
          ],
          "category": "models",
          "output_node": false,
          "experimental": false,
          "deprecated": false,
          "python_module": "test_nodes",
          "id": 146,
          "node_type": "system",
          "created_at": "2026-04-15T02:02:42.587170+00:00",
          "updated_at": "2026-04-15T02:02:42.587170+00:00",
          "version": 1,
          "description": "加载基础模型用于RL训练",
          "is_enabled": true,
          "metanode_deleted": false
        },
        "config": {
          "model_path": "/workspace/models/Qwen/Qwen3-8B/"
        },
        "definitionMissing": false,
        "disabled": false
      },
      "measured": {
        "width": 300,
        "height": 112
      },
      "properties": {},
      "style": {
        "width": 300
      }
    }
  ],
  "edges": [],
  "viewport": {
    "x": -203.38823529411764,
    "y": -96.1552941164703,
    "zoom": 0.92
  },
  "timestamp": "2026-06-09T07:49:57.491Z"
}
```

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/LoadModel/LoadModel.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `LoadModel` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |

```bash
python -m pyromind_sdk.test_run_workflow_cli workflow/LoadModel/LoadModel.json --pretty
```

输出示例：

```json
{
  "task_id": "4907",
  "task_name": "loadmodel",
  "status": "success",
  "error_message": null,
  "nodes": [
    {
      "node_key": "#1",
      "node_id": 18959,
      "node_type": "LoadModel",
      "label": "",
      "start_at": "2026-06-09 08:25:16",
      "end_at": "2026-06-09 08:25:28",
      "duration": "0:00:12",
      "input": {
        "model_path": "/workspace/models/Qwen/Qwen3-8B/"
      },
      "output": {
        "model": "/workspace/models/Qwen/Qwen3-8B/"
      },
      "raw": {}
    }
  ]
}
```
