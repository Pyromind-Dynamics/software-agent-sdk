# Training Config Builder 节点

![alt text](/imgs/TrainingConfigBuilderNode/TrainingConfigBuilderNode.png)

## 1.1 功能概述

将 SFT/GRPO/DPO 共享训练超参数组装为 `training_config` YAML 字符串，供下游训练节点使用。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| num_epochs | FLOAT | 否 | `2` | — | 训练 epoch 数 |
| batch_size | INT | 否 | `2` | — | 每设备 batch size |
| grad_accum | INT | 否 | `2` | — | 梯度累积步数 |
| learning_rate | FLOAT | 否 | `0.0001` | — | 优化器学习率 |
| lr_scheduler_type | STRING | 否 | `"constant"` | 可选值：`"linear"`、`"cosine"`、`"cosine_with_restarts"`、`"polynomial"`、`"constant"`、`"constant_with_warmup"` | 学习率调度器类型 |
| logging_steps | INT | 否 | `5` | — | 每 N 步记录日志 |
| save_steps | INT | 否 | `500` | — | 每 N 步保存 checkpoint |
| save_total_limit | INT | 否 | `3` | — | 保留 checkpoint 的最大数量 |
| eval_steps | INT | 否 | `500` | — | 每 N 步执行评估 |
| seed | INT | 否 | `42` | — | 随机种子 |
| resume_from_checkpoint | STRING | 否 | `""` | — | 恢复训练的 checkpoint 路径 |
| max_grad_norm | FLOAT | 否 | `2` | — | 梯度裁剪最大范数 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| training_config | STRING\|TRAINING_CONFIG | 训练配置 YAML 字符串（`BaseTrainingConfig` 字段） |
