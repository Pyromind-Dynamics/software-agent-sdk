# Metrics Config Builder（系统入口）

![alt text](/imgs/MetricsConfigBuilderNode/MetricsConfigBuilderNode.png)

## 1.1 功能概述

使用内置评估入口（GSM8K、accuracy、BLEU、ROUGE-L）构建单条 metrics 配置 YAML。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| entry | STRING | 是 | `"compute_gsm8k"` | 可选值：`"compute_gsm8k"`、`"compute_accuracy"`、`"compute_bleu"`、`"compute_rouge_l"` | 平台内置指标名称；自定义脚本使用 MetricsConfigBuilderCustomNode |
| name | STRING | 是 | `""` | — | 指标或 reward item 名称 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| metrics_config | STRING\|METRICS_CONFIG | 来自 Metrics Config Builder 的指标配置 YAML |
