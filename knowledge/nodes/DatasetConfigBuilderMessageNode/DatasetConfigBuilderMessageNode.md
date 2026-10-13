# Dataset Kind Config Builder（Messages）

![alt text](/imgs/DatasetConfigBuilderMessageNode/DatasetConfigBuilderMessageNode.png)

## 1.1 功能概述

构建对话/消息格式数据集的 `dataset_kind_config` YAML 片段。输出连接至 `DatasetConfigBuilderNode.dataset_kind_config`。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| messages_field | STRING | 是 | messages 字段名 |
| rejected_field | STRING | 否 | 拒绝回复 / rejected messages 字段名。默认值：`rejected_messages` |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| dataset_kind_config | STRING\|DATASET_KIND_CONFIG | 来自 Kind Config Builder 节点的 dataset kind 配置 YAML |
