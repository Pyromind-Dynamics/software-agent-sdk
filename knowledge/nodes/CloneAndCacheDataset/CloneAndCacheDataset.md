# Clone Dataset 节点

![alt text](/imgs/CloneAndCacheDataset/CloneAndCacheDataset.png)  

## 1.1 功能概述

克隆数据集

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| dataset | STRING | 是 | 要克隆的官方预置数据集标识，格式为 组织名/数据集名（如 openai/gsm8k）。需从平台支持的预设列表中选择，可选项如下: OpenGVLab/ScaleCUA-Data, Writer/omniact, agibot-world/AgiBotWorld-Alpha-327, agibot-world/AgiBotWorld-Alpha-327-Extract, agibot-world/AgiBotWorld-Alpha-CtrlWorld-327, agibot-world/AgiBotWorld-Alpha-Lerobot-327, agibot-world/AgiBotWorld-Alpha-Openpi-327, cadene/droid_1.0.1, gui-360/gui-excel, gui-360/processed_data, henryhe0123/PC-Agent-E, openai/gsm8k, ritzzai/GUI-R1, xlangai/aguvis-stage1, xlangai/aguvis-stage2, zonghanHZH/UGround-V1-8k |
| target_path | STRING | 是 | 克隆后的数据集存放目录。默认值：`/workspace/datasets/` Options: /workspace/datasets/ |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| dataset_path | STRING | 克隆完成后的数据集路径 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/CloneAndCacheDataset/CloneAndCacheDataset.json`](../../workflow/CloneAndCacheDataset/CloneAndCacheDataset.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/CloneAndCacheDataset/CloneAndCacheDataset.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `cloneAndCacheDataset` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |


输出示例：

```json
{
  "task_id": "4794",
  "task_name": "cloneandcachedataset",
  "status": "success",
  "error_message": null,
  "nodes": [
    {
      "node_key": "#2",
      "node_id": 18630,
      "node_type": "CloneAndCacheDataset",
      "label": "",
      "start_at": "2026-06-08 09:17:33",
      "end_at": "2026-06-08 09:17:39",
      "duration": "0:00:06",
      "input": {
        "dataset": "openai/gsm8k",
        "target_path": "/workspace/datasets/"
      },
      "output": {
        "dataset_path": "/workspace/datasets/openai/gsm8k"
      },
      "raw": {}
    }
  ]
}
```