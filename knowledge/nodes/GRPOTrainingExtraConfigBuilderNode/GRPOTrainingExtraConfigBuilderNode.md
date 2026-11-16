# GRPO Training Extra Config Builder 节点

![alt text](/imgs/GRPOTrainingExtraConfigBuilderNode/GRPOTrainingExtraConfigBuilderNode.png)

## 1.1 功能概述

Build grpo_extra_config YAML (GRPO-only training fields; use with shared training_config + ModelTrainGRPONode).

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| max_steps | INT | 否 | `200` | — | 最大训练步数（GRPO extra） |
| num_generations | INT | 否 | `8` | — | 每个 prompt 的生成数（GRPO extra） |
| max_prompt_length | INT | 否 | `2048` | — | 最大 prompt 长度（GRPO extra） |
| max_completion_length | INT | 否 | `2048` | — | 最大 completion 长度（GRPO extra） |
| temperature | FLOAT | 否 | `0.7` | — | 采样温度 |
| enable_chord | BOOLEAN | 否 | `false` | — | 是否启用 CHORD 算法（GRPO extra） |
| enable_hint | BOOLEAN | 否 | `false` | — | 是否启用 hint 模式（GRPO extra） |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| grpo_extra_config | STRING\|GRPO_EXTRA_CONFIG | GRPO 专用额外训练配置 YAML |
