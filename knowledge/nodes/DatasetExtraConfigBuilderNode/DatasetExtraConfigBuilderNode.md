# Dataset Extra Config Builder 节点

![alt text](/imgs/DatasetExtraConfigBuilderNode/DatasetExtraConfigBuilderNode.png)

## 1.1 功能概述

构建 `dataset_extra_config` YAML 片段（collator 入口、最大样本数、最大序列长度等）。输出连接至 `DatasetConfigBuilderNode.dataset_extra_config`。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| train_max_samples | INT | 否 | `0` | — | 最大训练样本数（0 表示全部） |
| val_max_samples | INT | 否 | `0` | — | 最大验证样本数（0 表示全部） |
| sft_collator_entry | STRING | 否 | `"train.sft_collator:make_collate_fn"` | — | SFT collator 入口（`module:function`） |
| dpo_collator_entry | STRING | 否 | `"train.dpo_collator:make_collate_fn"` | — | DPO collator 入口（`module:function`） |
| grpo_collator_entry | STRING | 否 | `"train.data.default_vision_grpo_collate:create_grpo_collate_fn"` | — | GRPO collator 入口（`module:function`） |
| max_seq_length | INT | 否 | `4096` | — | 最大序列长度 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| dataset_extra_config | STRING\|DATASET_EXTRA_CONFIG | 来自 Dataset Extra Config Builder 节点的额外配置 YAML |
