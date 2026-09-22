# Download File 节点

![alt text](/imgs/DownloadURLNode/DownloadURLNode.png)

## 1.1 功能概述

从 URL 下载文件到本地工作区路径。节点支持分片并行 Range 下载与断点续传，严格校验 HTTP 206 部分内容响应，并将分片合并为单个输出文件。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| url | STRING | 是 | 远程文件 URL。 |
| output_path | STRING | 是 | 合并后文件写入的本地路径。 |
| expected_sha256 | STRING | 否 | 可选的预期 SHA-256 校验值。设置后会对下载文件进行校验。 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| path | STRING | 下载完成后的本地文件路径。 |

## 1.4 Workflow JSON 定义

单节点 workflow 示例见 [workflow/DownloadURLNode/DownloadURLNode.json](../../workflow/DownloadURLNode/DownloadURLNode.json)。

示例节点配置：

```json
{
  "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/config.json",
  "output_path": "/workspace/downloads/config.json",
  "expected_sha256": ""
}
```

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/DownloadURLNode/DownloadURLNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `DownloadURLNode example` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |

```bash
python -m pyromind_sdk.test_run_workflow_cli workflow/DownloadURLNode/DownloadURLNode.json --pretty
```
