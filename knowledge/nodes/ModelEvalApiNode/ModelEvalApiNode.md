# Benchmark By API 离线评测

![alt text](/imgs/ModelEvalApiNode/ModelEvalApiNode.png)

## 1.1 功能概述

通过 OpenAI 兼容 Inference API 进行离线基准评测。加载数据集、批量调用端点、计算指标并输出评测报告。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| endpoint | STRING | 是 | OpenAI 兼容推理 API 地址 |
| endpoint_api_key | STRING | 是 | 推理 API Key。默认值：`empty` |
| endpoint_model | STRING | 是 | 推理端点提供的模型名称。默认值：`default` |
| output_path | STRING | 是 | 输出目录或文件路径 |
| dataset_config | STRING\|DATASET_CONFIG | 是 | 已组装的数据集配置 YAML 字符串 |
| metrics_config | STRING\|METRICS_CONFIG | 是 | Metrics config |
| max_samples | INT | 否 | 最大评测样本数（0 表示全部）。默认值：`0` |
| max_tokens | INT | 否 | 每次 API 调用的最大 token 数。默认值：`256` |
| temperature | FLOAT | 否 | 采样温度。默认值：`0.01` |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| benchmark_output_path | STRING | 基准评测报告输出路径 |
