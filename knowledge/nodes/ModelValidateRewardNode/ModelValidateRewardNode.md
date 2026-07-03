# Validate Reward 节点

![alt text](/imgs/ModelValidateRewardNode/ModelValidateRewardNode.png)

## 1.1 功能概述

Validate reward composition and score aggregation.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| reward_config | STRING | 是 | 来自 Reward Config Builder 的奖励配置 YAML |
| completions | STRING | 否 | 用于奖励校验的可选 completions JSON。默认值：空字符串 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| status | STRING | 校验状态字符串 |
| scores | STRING | 校验得分的 YAML/JSON 字符串 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/ModelValidateRewardNode/ModelValidateRewardNode.json`](../../workflow/ModelValidateRewardNode/ModelValidateRewardNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/ModelValidateRewardNode/ModelValidateRewardNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `modelValidateRewardNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
