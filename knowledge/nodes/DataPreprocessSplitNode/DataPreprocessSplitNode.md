# Train-Validation Split 节点

![alt text](/imgs/DataPreprocessSplitNode/DataPreprocessSplitNode.png)

## 1.1 功能概述

Split metadata into train and validation datasets from a shared output path template

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| input_path | STRING | 是 | Input path |
| output_path | STRING | 是 | 输出目录或文件路径 |
| train_ratio | FLOAT | 否 | Train ratio。默认值：`0.8` |
| seed | INT | 否 | 随机种子。默认值：`42` |
| shuffle | BOOLEAN | 否 | Shuffle。默认值：true |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| train_output_path | STRING | 训练集划分输出路径 |
| val_output_path | STRING | 验证集划分输出路径 |
