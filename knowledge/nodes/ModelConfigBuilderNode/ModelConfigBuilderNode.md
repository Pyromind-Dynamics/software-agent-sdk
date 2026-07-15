# Model Config Builder 节点

![alt text](/imgs/ModelConfigBuilderNode/ModelConfigBuilderNode.png)

## 1.1 功能概述

构建 `model_config` YAML 字符串（模型名称与类型）。LoRA 微调请单独使用 **Lora Config Builder** 节点。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| model_path | STRING | 否 | HuggingFace 模型名或本地路径。默认值：`Qwen/Qwen3-VL-2B-Instruct` |
| model_type | STRING | 否 | 模型架构类型（`auto`、`qwen3vl`、`qwen3.5`）。选项：`auto`, `qwen3vl`, `qwen3.5` |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| model_config | STRING\|MODEL_CONFIG | 模型配置 YAML 字符串 |
## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/ModelConfigBuilderNode/ModelConfigBuilderNode.json`](../../workflow/ModelConfigBuilderNode/ModelConfigBuilderNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/ModelConfigBuilderNode/ModelConfigBuilderNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `modelConfigBuilderNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
