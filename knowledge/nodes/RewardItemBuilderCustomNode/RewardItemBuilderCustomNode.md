# Reward Item Builder（自定义入口）

![alt text](/imgs/RewardItemBuilderCustomNode/RewardItemBuilderCustomNode.png)

## 1.1 功能概述

从自定义 Python 入口路径构建单条 reward item YAML。非空 `kwargs` 时启用 factory 模式。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| entry | STRING | 是 | `""` | — | Python 入口路径（`module.py:function_name`） |
| name | STRING | 是 | `""` | — | 指标或 reward item 名称 |
| kwargs | STRING | 否 | `""` | — | 关键字参数字典 YAML（非空时启用 factory 模式） |
| weight | FLOAT | 否 | `1.0` | — | 组合得分中的权重系数 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| reward_item | STRING\|REWARD_ITEM | 单条 reward item YAML 字符串 |
