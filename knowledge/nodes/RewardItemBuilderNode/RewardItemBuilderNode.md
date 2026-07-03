# Reward Item Builder（系统入口）

![alt text](/imgs/RewardItemBuilderNode/RewardItemBuilderNode.png)

## 1.1 功能概述

使用内置奖励函数入口构建单条 reward item YAML。非空 `kwargs` 时启用 factory 模式。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| entry | STRING | 是 | Python 入口路径（`module.py:function_name`）。选项：`examples/geometry_vqa/reward.py:geometry_vqa_thinking_reward`, `examples/geometry_vqa/reward.py:geometry_vqa_answer_reward` |
| name | STRING | 是 | 指标或 reward item 名称 |
| kwargs | STRING | 否 | 关键字参数字典 YAML（非空时启用 factory 模式）。默认值：空字符串 |
| weight | FLOAT | 否 | 组合得分中的权重系数。默认值：`1.0` |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| reward_item | STRING | 单条 reward item YAML 字符串 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/RewardItemBuilderNode/RewardItemBuilderNode.json`](../../workflow/RewardItemBuilderNode/RewardItemBuilderNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/RewardItemBuilderNode/RewardItemBuilderNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `rewardItemBuilderNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
