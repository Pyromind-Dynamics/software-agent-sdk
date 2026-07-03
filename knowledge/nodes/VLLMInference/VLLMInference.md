# Inference (VLLM) 节点

![alt text](/imgs/VLLMInference/VLLMInference.png)

## 1.1 功能概述

使用 vLLM 运行推理，提供 OpenAI 兼容的 API 端点。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| model_path | STRING | 是 | 模型路径引用 |
| port | INT | 是 | 推理服务端口号。默认值：`3000` |
| gpu_count | INT | 是 | 推理服务使用的显卡数量。默认值：`1` |
| gpu_product | STRING | 是 | 显卡类型。默认值：`NVIDIA-H100-NVL` Options: NVIDIA-H100-NVL, NVIDIA-L40S, NVIDIA-H200, NVIDIA-B200, NVIDIA-H100-80GB-HBM3 |
| environment | ENV | 否 | 环境变量 |
| max_model_len | INT | 否 | 最大模型上下文长度 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| endpoint | STRING | 推理服务的端点地址 |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/VLLMInference/VLLMInference.json`](../../workflow/VLLMInference/VLLMInference.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
export PYROMIND_CLUSTER=<your-cluster>
```
