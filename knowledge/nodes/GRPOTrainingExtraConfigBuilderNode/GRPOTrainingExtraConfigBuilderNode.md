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
| grpo_extra_config | STRING\|GRPO_EXTRA_CONFIG | GRPO 专用额外训练配置 YAML |
