# Metrics Config Builder（自定义入口）

![alt text](/imgs/MetricsConfigBuilderCustomNode/MetricsConfigBuilderCustomNode.png)

## 1.1 功能概述

从自定义 Python 入口路径构建 metrics 配置 YAML（例如 `/workspace/nodes/custom_entry/custom_metrics.py:compute_gsm8k`）。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| entry | STRING | 是 | Python 入口路径（`module.py:function_name`） |
| name | STRING | 是 | 指标或 reward item 名称 |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| metrics_config | STRING\|METRICS_CONFIG | 来自 Metrics Config Builder 的指标配置 YAML |
