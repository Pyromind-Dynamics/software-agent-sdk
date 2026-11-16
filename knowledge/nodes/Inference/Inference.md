# Inference (SGlang) 节点

![alt text](/imgs/Inference/Inference.png)

## 1.1 功能概述

此节点根据传入的参数，启动一个推理服务，为后续节点持续提供推理服务，直到工作流完结后结束服务

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| model_path | STRING | 是 | 未声明 | — | 模型引用 |
| port | INT | 是 | `3000` | 最小 1；最大 8000 | 推理服务端口号 |
| gpu_count | INT | 是 | `1` | 最小 1；最大 8 | 推理服务使用的显卡个数 |
| gpu_product | STRING | 是 | `"NVIDIA-H100-NVL"` | 可选值：`"NVIDIA-H100-NVL"`、`"NVIDIA-L40S"` | 显卡类型 |
| environment | ENV | 否 | 未声明 | — | 环境变量名 |
| startup_probe_max_duration | INT | 否 | `1200` | 最小 30；最大 86400 | 推理服务启动探针最长等待时间，单位秒 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| endpoint | STRING | 推理服务的endpoint |
