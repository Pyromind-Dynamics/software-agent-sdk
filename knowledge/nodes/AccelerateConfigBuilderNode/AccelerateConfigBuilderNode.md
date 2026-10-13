# Accelerate Config Builder 节点

![alt text](/imgs/AccelerateConfigBuilderNode/AccelerateConfigBuilderNode.png)

## 1.1 功能概述

Build accelerate_config YAML string for accelerate launch.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| zero_stage | INT | 否 | DeepSpeed ZeRO stage。默认值：`2` |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| accelerate_config | ACCELERATE_CONFIG | Accelerate 启动配置 YAML |
