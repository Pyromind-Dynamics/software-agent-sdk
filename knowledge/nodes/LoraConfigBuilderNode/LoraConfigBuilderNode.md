# Lora Config Builder 节点

![alt text](/imgs/LoraConfigBuilderNode/LoraConfigBuilderNode.png)

## 1.1 功能概述

构建 `lora_config` YAML 字符串（rank、dropout、target modules 等）。通过训练节点的可选 `lora_config` 输入传入。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| lora_rank | INT | 否 | LoRA rank。默认值：`8` |
| lora_dropout | FLOAT | 否 | LoRA dropout。默认值：`0.05` |
| target_modules | STRING | 否 | LoRA 目标模块（逗号分隔）。默认值：`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| exclude_modules | STRING | 否 | LoRA 排除模块（逗号分隔）。默认值：空字符串 |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| lora_config | STRING\|LORA_CONFIG | 来自 Lora Config Builder 的 LoRA 配置 YAML |
