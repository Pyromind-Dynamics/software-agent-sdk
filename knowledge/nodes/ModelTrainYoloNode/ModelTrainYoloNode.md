# YOLO Training 节点

## 1.1 功能概述

使用数据集、训练、模型和 YOLO 专用配置训练 YOLO 模型。节点支持原生单视图 3 通道训练，也支持通过 Early Fusion 将多视图输入融合为 3 通道后训练。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| output_dir | STRING | 是 | 未声明 | — | 训练 run 的输出目录；权重写入其下的 `weights/` |
| dataset_config | STRING\|DATASET_CONFIG | 是 | 未声明 | — | 数据集配置；YOLO 至少使用其中的训练 JSONL 路径，可选验证 JSONL 路径 |
| training_config | STRING\|TRAINING_CONFIG | 是 | 未声明 | — | 共用训练配置；YOLO 使用 epoch、batch size、学习率、随机种子和断点续训配置 |
| model_config | STRING\|MODEL_CONFIG | 是 | 未声明 | — | 模型配置；`model_path` 必须指向已存在的本地 `.pt` 权重 |
| gpu_count | INT | 是 | `1` | 最小 1；最大 8 | GPU 数量 |
| gpu_product | STRING | 是 | `"NVIDIA-H100-NVL"` | 可选值：`"NVIDIA-H100-NVL"`、`"NVIDIA-L40S"` | GPU 型号（以部署环境实际可用的显卡为准） |
| yolo_extra_config | STRING | 否 | `""` | — | YOLO 专用训练配置，来自 `YoloTrainingExtraConfigBuilderNode` |
| wandb_config | WANDB_CONFIG | 否 | `""` | — | W&B 配置；不接则不上报 W&B |
| early_fusion_config | STRING | 否 | `""` | — | 多视图 Early Fusion 配置；单视图训练不要连接 |

> **注意：** GPU 型号与数量以部署环境实际可用的资源为准。

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| model_output_dir | STRING | 训练 run 目录，与输入的 `output_dir` 相同；最佳权重位于其下的 `weights/best.pt` |
| early_fusion | BOOLEAN | 本次训练是否启用了 Early Fusion |

## 1.4 数据和配置要求

- `dataset_config.train_data_path` 必须指向单个训练 JSONL 文件；`val_data_path` 为空时跳过验证。
- 检测任务使用 `views` 和 `shapes[].xywhn`；分割任务使用至少 3 个归一化点组成的 `shapes[].points`；分类任务在样本行上使用 `label` 或 `class`。
- `task` 必须与 JSONL 标注和模型权重类型一致，可选 `detect`、`segment` 或 `classify`。
- 共用的 `TrainingConfigBuilderNode` 中，`grad_accum`、学习率调度、日志、保存、验证步数和最大梯度范数不会映射到 YOLO 训练。
- 预训练权重只通过 `model_config.model_path` 提供；YOLO 专用配置不保存权重路径。

## 1.5 接线说明

单视图数据使用原生 3 通道训练，不连接 `early_fusion_config`。多视图数据需要连接 `EarlyFusionConfigBuilderNode`，并设置 `data_channels = 3 × views 数量`，例如双视图为 6。

训练完成后，可将 `model_output_dir` 直接连接到 `ModelInferYoloNode.model_weight_path`；也可以通过 `PathJoinNode` 拼接一次 `weights/best.pt` 后连接。不要对已经指向 `.pt` 文件的路径再次拼接权重子路径。
