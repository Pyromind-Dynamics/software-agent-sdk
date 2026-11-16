# Lora Config Builder 节点

![alt text](/imgs/LoraConfigBuilderNode/LoraConfigBuilderNode.png)

## 1.1 功能概述

构建 `lora_config` YAML 字符串（rank、dropout、target modules 等）。通过训练节点的可选 `lora_config` 输入传入。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| lora_rank | INT | 否 | `8` | — | LoRA rank |
| lora_dropout | FLOAT | 否 | `0.05` | — | LoRA dropout |
| target_modules | STRING | 否 | `"q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"` | — | LoRA 目标模块（逗号分隔） |
| exclude_modules | STRING | 否 | `""` | — | LoRA 排除模块（逗号分隔） |
| lora_adapter | STRING | 否 | `""` | — | LoRA adapter 配置值；schema 未说明加载行为，使用前查节点运行说明 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| lora_config | STRING\|LORA_CONFIG | 来自 Lora Config Builder 的 LoRA 配置 YAML |
