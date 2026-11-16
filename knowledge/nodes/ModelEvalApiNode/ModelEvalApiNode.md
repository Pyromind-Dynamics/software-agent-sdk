# Benchmark By API 离线评测

![alt text](/imgs/ModelEvalApiNode/ModelEvalApiNode.png)

## 1.1 功能概述

通过 OpenAI 兼容 Inference API 进行离线基准评测。加载数据集、批量调用端点、计算指标并输出评测报告。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| endpoint | STRING | 是 | `"http://127.0.0.1:3000/v1"` | — | OpenAI 兼容推理 API 地址 |
| endpoint_api_key | STRING | 是 | `"empty"` | — | 推理 API Key |
| endpoint_model | STRING | 是 | `"default"` | — | 推理端点提供的模型名称 |
| output_path | STRING | 是 | 未声明 | — | 持久化评测输出目录，使用 `/workspace/...` |
| dataset_config | STRING\|DATASET_CONFIG | 是 | 未声明 | — | 已组装的数据集配置 YAML 字符串 |
| metrics_config | STRING\|METRICS_CONFIG | 是 | 未声明 | — | Metrics config |
| max_samples | INT | 否 | `0` | — | 最大评测样本数（0 表示全部） |
| max_tokens | INT | 否 | `256` | — | 每次 API 调用的最大 token 数 |
| temperature | FLOAT | 否 | `0.01` | — | 采样温度 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| benchmark_output_path | STRING | 基准评测报告输出路径，位于 `/workspace/...` |
