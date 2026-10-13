# Dataset Config Builder 节点

![alt text](/imgs/DatasetConfigBuilderNode/DatasetConfigBuilderNode.png)

## 1.1 功能概述

构建 SFT/GRPO/DPO 训练的基础 `dataset_config` YAML 字符串。需配合 **Dataset Kind Config Builder**（Vision / Text / Messages）和 **Dataset Extra Config Builder** 节点，分别提供 `dataset_kind_config` 与 `dataset_extra_config` 输入。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| train_data_path | STRING | 是 | 训练数据路径 |
| val_data_path | STRING | 否 | 验证数据路径。默认值：空字符串 |
| dataset_kind_config | STRING\|DATASET_KIND_CONFIG | 否 | 来自 Kind Config Builder 节点的 dataset kind 配置 YAML。默认值：空字符串 |
| dataset_extra_config | STRING\|DATASET_EXTRA_CONFIG | 否 | 来自 Dataset Extra Config Builder 节点的额外配置 YAML。默认值：空字符串 |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| dataset_config | STRING\|DATASET_CONFIG | 已组装的数据集配置 YAML 字符串 |
