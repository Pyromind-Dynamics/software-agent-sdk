# Join Path 节点

![alt text](/imgs/PathJoinNode/PathJoinNode.png)  

## 1.1 功能概述

路径拼接工具节点：将 `base_path` 与 `subpath` 拼接为完整的路径字符串，供下游节点使用。
`base_path` 与 `subpath` 可来自上游节点输出，也可在节点配置中直接填写
- **典型场景**：将工作区根目录（如 `/workspace/`）与相对子目录（如 `models`、`datasets/gsm8k`）组合，生成模型、数据集或输出文件的完整路径

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| base_path | STRING | 是 | 基础路径（目录前缀）。默认值：`""`。示例：`/workspace/` |
| subpath | STRING | 是 | 待拼接的子路径（相对路径或目录名）。默认值：`""`。示例：`models` |
| environment | ENV | 否 | 可选的环境变量配置。默认值：- |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| joined_path | STRING | 拼接后的完整路径。例如 `base_path=/workspace/`、`subpath=models` 时输出 `/workspace/models` |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/PathJoinNode/PathJoinNode.json`](../../workflow/PathJoinNode/PathJoinNode.json)。

示例节点配置：

```json
{
  "base_path": "/workspace/",
  "subpath": "models",
  "environment": ""
}
```

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/PathJoinNode/PathJoinNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `pathJoinNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |

```bash
python -m pyromind_sdk.test_run_workflow_cli workflow/PathJoinNode/PathJoinNode.json --pretty
```

输出示例：

```json
{
  "task_name": "unsaved-workflow",
  "status": "success",
  "nodes": [
    {
      "node_key": "#2",
      "node_id": 18563,
      "node_type": "PathJoinNode",
      "label": "Join Path",
      "start_at": "2026-06-08 03:21:17",
      "end_at": "2026-06-08 03:21:27",
      "duration": "0:00:10",
      "input": {
        "base_path": "/workspace/",
        "subpath": "models",
        "environment": ""
      },
      "output": {
        "joined_path": "/workspace/models"
      },
      "raw": {}
    }
  ]
}
```