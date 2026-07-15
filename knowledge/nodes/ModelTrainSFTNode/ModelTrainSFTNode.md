# SFT Training 节点

![alt text](/imgs/ModelTrainSFTNode/ModelTrainSFTNode.png)

## 1.1 功能概述

Run SFT training and return output directory.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| output_path | STRING | 是 | 输出目录或文件路径 |
| dataset_config | STRING\|DATASET_CONFIG | 是 | 已组装的数据集配置 YAML 字符串 |
| training_config | STRING\|TRAINING_CONFIG | 是 | 训练配置 YAML 字符串（`BaseTrainingConfig` 字段） |
| model_config | STRING\|MODEL_CONFIG | 是 | 模型配置 YAML 字符串 |
| lora_config | STRING\|LORA_CONFIG | 否 | 来自 Lora Config Builder 的 LoRA 配置 YAML |
| wandb_config | WANDB_CONFIG | 否 | W&B 配置对象 |
| thinking_as_input_ratio | FLOAT | 否 | 将思维链内容作为输入的比例。默认值：`0.0` |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| model_output_path | STRING | 训练输出模型目录 |
## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/ModelTrainSFTNode/ModelTrainSFTNode.json`](../../workflow/ModelTrainSFTNode/ModelTrainSFTNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/ModelTrainSFTNode/ModelTrainSFTNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `modelTrainSFTNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
