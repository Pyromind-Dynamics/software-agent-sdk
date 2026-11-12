# Validate GRPO Dataloader 节点

![alt text](/imgs/ModelValidateGRPODataloaderNode/ModelValidateGRPODataloaderNode.png)

## 1.1 功能概述

Validate GRPO dataloader configuration, reward config, and rollout behavior.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| dataset_config | STRING\|DATASET_CONFIG | 是 | 未声明 | — | 已组装的数据集配置 YAML 字符串 |
| rollout_size | INT | 否 | `3` | — | Rollout size |
| check_reasoning | BOOLEAN | 否 | `false` | — | 是否校验数据集中的 reasoning 字段 |
| limit | INT | 否 | `0` | — | 校验预览的最大样本数（0 表示不限制） |
| verbose | BOOLEAN | 否 | `false` | — | 是否输出详细校验日志 |
| preview_html_path | STRING | 否 | `""` | — | HTML 预览输出路径 |
| reward_config | STRING\|REWARD_CONFIG | 否 | `""` | — | 来自 Reward Config Builder 的奖励配置 YAML |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| metrics_json | STRING | 校验指标 JSON 字符串 |
| output_preview_path | STRING | 生成的 HTML 预览文件路径 |
