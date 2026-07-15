# LoadImage 节点

![alt text](/imgs/LoadImage/LoadImage.png) 

## 1.1 功能概述

从 `/workspace/image_cache/` 目录下读取所选的图片，并输出路径。

支持的图片格式：`.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.svg`

## 1.2 输入类型

| 参数  | 数据类型      | 必填 | 描述 |
|-------|---------------|----|------|
| image | STRING | 是 | 从 `/workspace/image_cache/` 中动态读取的图片文件列表。用户通过下拉菜单选择所需图片。支持 `image_upload` 扩展（允许用户上传新图片）。默认值：列表中第一个图片路径。 |
## 1.3 输出类型

| 参数            | 数据类型 | 描述 |
|-----------------|----------|------|
| image_file_path | STRING | 图片在工作区中的完整路径，格式为 `/workspace/{所选图片路径}`。供下游节点（如 RL 训练、Rollout 等）读取图片内容使用。当前无路径存在性校验。 |
## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/LoadImage/LoadImage.json`](../../workflow/LoadImage/LoadImage.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/LoadImage/LoadImage.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `LoadImageTest` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |

```bash
python -m pyromind_sdk.test_run_workflow_cli workflow/LoadImage/LoadImage.json --pretty
```

输出示例：

```json
{
  "task_id": "4340",
  "task_name": "LoadImageTest",
  "status": "success",
  "nodes": [
    {
      "node_key": "#1",
      "node_id": 17729,
      "node_type": "LoadImage",
      "label": "Load Image",
      "start_at": "2026-06-04 03:48:51",
      "end_at": "2026-06-04 03:49:03",
      "duration": "0:00:12",
      "input": {
        "image": "image_cache/1.png"
      },
      "output": {
        "image_file_path": "/workspace/image_cache/1.png"
      },
      "raw": {}
    }
  ]
}
```
