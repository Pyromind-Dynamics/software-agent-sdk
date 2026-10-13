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
| reward_item | STRING\|REWARD_ITEM | 单条 reward item YAML 字符串 |
