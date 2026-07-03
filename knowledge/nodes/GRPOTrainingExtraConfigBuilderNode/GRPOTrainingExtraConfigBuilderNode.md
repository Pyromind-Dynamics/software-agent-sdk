# GRPO Training Extra Config Builder 节点

![alt text](/imgs/GRPOTrainingExtraConfigBuilderNode/GRPOTrainingExtraConfigBuilderNode.png)

## 1.1 功能概述

Build grpo_extra_config YAML (GRPO-only training fields; use with shared training_config + ModelTrainGRPONode).

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| max_steps | INT | 否 | 最大训练步数（GRPO extra）。默认值：`200` |
| num_generations | INT | 否 | 每个 prompt 的生成数（GRPO extra）。默认值：`8` |
| max_prompt_length | INT | 否 | 最大 prompt 长度（GRPO extra）。默认值：`2048` |
| max_completion_length | INT | 否 | 最大 completion 长度（GRPO extra）。默认值：`2048` |
| temperature | FLOAT | 否 | 采样温度。默认值：`0.7` |
| enable_chord | BOOLEAN | 否 | 是否启用 CHORD 算法（GRPO extra）。默认值：false |
| enable_hint | BOOLEAN | 否 | 是否启用 hint 模式（GRPO extra）。默认值：false |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| grpo_extra_config | STRING | GRPO 专用额外训练配置 YAML |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/GRPOTrainingExtraConfigBuilderNode/GRPOTrainingExtraConfigBuilderNode.json`](../../workflow/GRPOTrainingExtraConfigBuilderNode/GRPOTrainingExtraConfigBuilderNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/GRPOTrainingExtraConfigBuilderNode/GRPOTrainingExtraConfigBuilderNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `gRPOTrainingExtraConfigBuilderNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
