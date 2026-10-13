# Validate Reward 节点

![alt text](/imgs/ModelValidateRewardNode/ModelValidateRewardNode.png)

## 1.1 功能概述

Validate reward composition and score aggregation.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| reward_config | STRING\|REWARD_CONFIG | 是 | 来自 Reward Config Builder 的奖励配置 YAML |
| completions | STRING | 是 | 用于奖励校验的 completions JSON（与 ground-truth 数据对比） |
| ground_truths | STRING | 是 | 用于校验 completions 的 ground-truth 数据。默认值：空字符串 |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| status | STRING | 校验状态字符串 |
| scores | STRING | 校验得分的 YAML/JSON 字符串 |
