# Dataset Validator 数据集校验

![alt text](/imgs/DatasetValidatorNode/DatasetValidatorNode.png)

## 1.1 功能概述

校验已组装的 `dataset_config`，生成 dataloader 预览，并输出校验指标与 HTML 预览文件路径。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| dataset_config | STRING\|DATASET_CONFIG | 是 | 已组装的数据集配置 YAML 字符串 |
| check_reasoning | BOOLEAN | 否 | 是否校验数据集中的 reasoning 字段。默认值：false |
| limit | INT | 否 | 校验预览的最大样本数（0 表示不限制）。默认值：`0` |
| verbose | BOOLEAN | 否 | 是否输出详细校验日志。默认值：false |
| preview_html_path | STRING | 否 | HTML 预览输出路径。默认值：`/workspace/datasets/preview.html` |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| metrics_yaml | STRING | 校验指标 YAML 字符串 |
| output_preview_path | STRING | 生成的 HTML 预览文件路径 |
