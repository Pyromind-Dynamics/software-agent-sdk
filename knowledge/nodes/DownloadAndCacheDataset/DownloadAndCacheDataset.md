# Download Dataset 节点

![alt text](/imgs/DownloadAndCacheDataset/DownloadAndCacheDataset.png)  

## 1.1 功能概述

从远程仓库下载指定数据集，缓存到本地目录，并输出数据集路径供下游节点使用。

- **数据源选择**：支持 `huggingface`、`modelscope` 两种下载源
- **本地缓存**：将数据集下载到 `cache_dir` 指定的目录；若目录不存在会自动创建
- **路径输出**：返回缓存后的 `dataset_path`，供 LoadDataset、数据预处理或训练节点使用

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| dataset_name | STRING | 是 | 数据集名称或仓库 ID。默认值：`openai/gsm8k`。示例：`openai/gsm8k` |
| cache_dir | PATH | 是 | 本地缓存目录。默认值：`/workspace/datasets/openai/gsm8k` |
| download_source | STRING | 是 | 下载源。默认值：`huggingface`。可选：`huggingface`、`modelscope` |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| dataset_path | STRING | 下载完成后数据集的本地路径。例如 `cache_dir=/workspace/datasets/openai/gsm8k` 时，输出 `/workspace/datasets/openai/gsm8k` |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/DownloadAndCacheDataset/DownloadAndCacheDataset.json`](../../workflow/DownloadAndCacheDataset/DownloadAndCacheDataset.json)。

示例节点配置：

```json
{
  "dataset_name": "openai/gsm8k",
  "cache_dir": "/workspace/datasets/openai/gsm8k",
  "download_source": "huggingface"
}
```

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/DownloadAndCacheDataset/DownloadAndCacheDataset.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `downloadAndCacheDataset` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |

```bash
python -m pyromind_sdk.test_run_workflow_cli workflow/DownloadAndCacheDataset/DownloadAndCacheDataset.json --pretty
```

输出示例：

```json
{
  "task_name": "unsaved-workflow",
  "status": "success",
  "nodes": [
    {
      "node_type": "DownloadAndCacheDataset",
      "label": "Download Dataset",
      "input": {
        "dataset_name": "openai/gsm8k",
        "cache_dir": "/workspace/datasets/openai/gsm8k",
        "download_source": "huggingface"
      },
      "output": {
        "dataset_path": "/workspace/datasets/openai/gsm8k"
      }
    }
  ]
}
```
