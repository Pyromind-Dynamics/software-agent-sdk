# Wandb Config Builder 节点

![alt text](/imgs/WandbConfigBuilderNode/WandbConfigBuilderNode.png)

## 1.1 功能概述

Build wandb_config YAML string for W&B training runtime env.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| wandb_api_key | STRING | 是 | W&B API Key |
| wandb_project | STRING | 是 | W&B 项目名称 |
| wandb_name | STRING | 否 | W&B Run 名称。默认值：空字符串 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| wandb_config | WANDB_CONFIG | W&B 配置对象 |
