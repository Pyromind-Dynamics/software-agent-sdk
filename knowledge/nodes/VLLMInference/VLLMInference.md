# Inference (VLLM) 节点

![alt text](/imgs/VLLMInference/VLLMInference.png)

## 1.1 功能概述

使用 vLLM 运行推理，提供 OpenAI 兼容的 API 端点。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| model_path | STRING | 是 | 未声明 | — | 模型路径引用 |
| port | INT | 是 | `3000` | 最小 1；最大 8000 | 推理服务端口号 |
| gpu_count | INT | 是 | `1` | 最小 1；最大 8 | 推理服务使用的显卡数量 |
| gpu_product | STRING | 是 | `"NVIDIA-H100-NVL"` | 可选值：`"NVIDIA-H100-NVL"`、`"NVIDIA-L40S"` | 显卡类型 |
| environment | ENV | 否 | 未声明 | — | 环境变量 |
| max_model_len | INT | 否 | 未声明 | — | 最大模型上下文长度 |
| startup_probe_max_duration | INT | 否 | `1200` | 最小 30；最大 86400 | 推理服务启动探针最长等待时间，单位秒 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| endpoint | STRING | 推理服务的端点地址 |
