# Lora Config Builder 节点

![alt text](/imgs/LoraConfigBuilderNode/LoraConfigBuilderNode.png)

## 1.1 功能概述

构建 `lora_config` YAML 字符串（rank、dropout、target modules 等）。通过训练节点的可选 `lora_config` 输入传入。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| lora_rank | INT | 否 | LoRA rank。默认值：`8` |
| lora_dropout | FLOAT | 否 | LoRA dropout。默认值：`0.05` |
| target_modules | STRING | 否 | LoRA 目标模块（逗号分隔）。默认值：`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| exclude_modules | STRING | 否 | LoRA 排除模块（逗号分隔）。默认值：空字符串 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| lora_config | STRING | 来自 Lora Config Builder 的 LoRA 配置 YAML |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/LoraConfigBuilderNode/LoraConfigBuilderNode.json`](../../workflow/LoraConfigBuilderNode/LoraConfigBuilderNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/LoraConfigBuilderNode/LoraConfigBuilderNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `loraConfigBuilderNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
