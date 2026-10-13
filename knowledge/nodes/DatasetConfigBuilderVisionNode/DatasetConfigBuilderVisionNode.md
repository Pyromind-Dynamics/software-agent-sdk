# Dataset Kind Config Builder（Vision）

![alt text](/imgs/DatasetConfigBuilderVisionNode/DatasetConfigBuilderVisionNode.png)

## 1.1 功能概述

构建视觉数据集（图片 + 文本字段）的 `dataset_kind_config` YAML 片段。输出连接至 `DatasetConfigBuilderNode.dataset_kind_config`。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| system_prompt_field | STRING | 否 | system prompt 字段名。默认值：空字符串 |
| user_prompt_field | STRING | 否 | user prompt 字段名。默认值：空字符串 |
| image_field | STRING | 是 | 图片字段名。默认值：空字符串 |
| assistant_response_field | STRING | 否 | 助手回复 / ground-truth 字段名。默认值：`gt` |
| rejected_field | STRING | 否 | 拒绝回复 / rejected messages 字段名。默认值：`rejected_answer` |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| dataset_kind_config | STRING\|DATASET_KIND_CONFIG | 来自 Kind Config Builder 节点的 dataset kind 配置 YAML |
