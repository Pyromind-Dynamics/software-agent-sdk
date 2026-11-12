# Reward Config Builder 节点

![alt text](/imgs/RewardConfigBuilderNode/RewardConfigBuilderNode.png)

## 1.1 功能概述

Build reward config YAML from up to five reward item YAML strings and normalize flag.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| reward_item_1 | STRING\|REWARD_ITEM | 否 | `""` | — | Reward item YAML #1 |
| reward_item_2 | STRING\|REWARD_ITEM | 否 | `""` | — | Reward item YAML #2 |
| reward_item_3 | STRING\|REWARD_ITEM | 否 | `""` | — | Reward item YAML #3 |
| reward_item_4 | STRING\|REWARD_ITEM | 否 | `""` | — | Reward item YAML #4 |
| reward_item_5 | STRING\|REWARD_ITEM | 否 | `""` | — | Reward item YAML #5 |
| normalize | BOOLEAN | 否 | `false` | — | 是否对组合奖励得分归一化 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| reward_config | STRING\|REWARD_CONFIG | 来自 Reward Config Builder 的奖励配置 YAML |
