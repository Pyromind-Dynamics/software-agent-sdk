# Train-Validation Split 节点

![alt text](/imgs/DataPreprocessSplitNode/DataPreprocessSplitNode.png)

## 1.1 功能概述

Split metadata into train and validation datasets from a shared output path template

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| input_path | STRING | 是 | 未声明 | — | Input path |
| output_path | STRING | 是 | 未声明 | — | 持久化输出目录或文件路径，使用 `/workspace/...` |
| train_ratio | FLOAT | 否 | `0.8` | — | Train ratio |
| seed | INT | 否 | `42` | — | 随机种子 |
| shuffle | BOOLEAN | 否 | `true` | — | Shuffle |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| train_output_path | STRING | 训练集划分输出路径，位于 `/workspace/...` |
| val_output_path | STRING | 验证集划分输出路径，位于 `/workspace/...` |
