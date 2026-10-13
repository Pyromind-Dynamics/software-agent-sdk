# Metrics Config Builder（系统入口）

![alt text](/imgs/MetricsConfigBuilderNode/MetricsConfigBuilderNode.png)

## 1.1 功能概述

使用内置评估入口（GSM8K、accuracy、BLEU、ROUGE-L）构建单条 metrics 配置 YAML。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| entry | STRING | 是 | Python 入口路径（`module.py:function_name`）。选项：`examples/eval_metrics_common.py:compute_gsm8k`, `examples/eval_metrics_common.py:compute_accuracy`, `examples/eval_metrics_common.py:compute_bleu`, `examples/eval_metrics_common.py:compute_rouge_l` |
| name | STRING | 是 | 指标或 reward item 名称 |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| metrics_config | STRING\|METRICS_CONFIG | 来自 Metrics Config Builder 的指标配置 YAML |
