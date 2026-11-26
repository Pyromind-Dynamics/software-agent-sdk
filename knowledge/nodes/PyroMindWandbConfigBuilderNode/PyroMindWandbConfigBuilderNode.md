# PyroMindWandbConfigBuilderNode

![alt text](/imgs/WandbConfigBuilderNode/WandbConfigBuilderNode.png)

## 1.1 功能概述

生成训练节点使用的 `wandb_config`。新建工作流的 DSL 构造器和 XYFlow 的 `data.nodeType` 均使用 `PyroMindWandbConfigBuilderNode`。

此节点对应上海 WORB 实例。查询训练 run、loss 曲线和报告时，使用实例地址 [上海 WORB](https://worb-cn-east-1.pyromind-asia.cn/)。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| wandb_api_key | STRING | 是 | `""` | — | W&B API Key |
| wandb_project | STRING | 是 | `""` | — | W&B 项目名称 |
| wandb_name | STRING | 否 | `""` | — | W&B Run 名称 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| wandb_config | WANDB_CONFIG | W&B 配置对象 |
