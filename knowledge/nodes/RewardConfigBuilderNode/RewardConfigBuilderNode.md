# Reward Config Builder 节点

![alt text](/imgs/RewardConfigBuilderNode/RewardConfigBuilderNode.png)

## 1.1 功能概述

Build reward config YAML from up to five reward item YAML strings and normalize flag.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| reward_item_1 | STRING | 否 | Reward item YAML #1。默认值：空字符串 |
| reward_item_2 | STRING | 否 | Reward item YAML #2。默认值：空字符串 |
| reward_item_3 | STRING | 否 | Reward item YAML #3。默认值：空字符串 |
| reward_item_4 | STRING | 否 | Reward item YAML #4。默认值：空字符串 |
| reward_item_5 | STRING | 否 | Reward item YAML #5。默认值：空字符串 |
| normalize | BOOLEAN | 否 | 是否对组合奖励得分归一化。默认值：false |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| reward_config | STRING | 来自 Reward Config Builder 的奖励配置 YAML |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/RewardConfigBuilderNode/RewardConfigBuilderNode.json`](../../workflow/RewardConfigBuilderNode/RewardConfigBuilderNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/RewardConfigBuilderNode/RewardConfigBuilderNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `rewardConfigBuilderNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
