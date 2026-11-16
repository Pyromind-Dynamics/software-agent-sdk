# Inference (YOLO) 节点

## 1.1 功能概述

使用 YOLO 权重对与训练、验证集格式一致的测试 JSONL 执行推理，输出逐样本预测、汇总页面和可选的可视化图片。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| model_weight_path | STRING | 是 | 未声明 | — | `.pt` 权重文件或训练 run 目录；目录会依次查找 `weights/best.pt`、`weights/last.pt` 和 `best.pt` |
| test_data_path | STRING | 是 | 未声明 | — | 单个测试 JSONL 文件，格式和视图数应与训练数据一致 |
| gpu_count | INT | 是 | `1` | 最小 1；最大 8 | GPU 数量 |
| gpu_product | STRING | 是 | `"NVIDIA-H100-NVL"` | 可选值：`"NVIDIA-H100-NVL"`、`"NVIDIA-L40S"` | GPU 型号（以部署环境实际可用的显卡为准） |
| output_dir | STRING | 否 | `""` | — | 结果目录；为空时使用测试 JSONL 所在目录 |
| yolo_extra_config | STRING | 否 | `""` | — | YOLO 配置；推理主要使用其中的 `imgsz` 和 `task` |
| batch_size | INT | 否 | `8` | — | 推理批大小 |
| conf | FLOAT | 否 | `0.5` | — | 置信度阈值 |
| save_viz | BOOLEAN | 否 | `false` | — | 是否生成 `viz/*.jpg` 可视化结果 |

> **注意：** GPU 型号与数量以部署环境实际可用的资源为准。

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| result_output_dir | STRING | 推理结果目录 |

结果目录包含：

- `preds.jsonl`：逐样本预测结果，包括样本标识、视图、预测数量、最高置信度和检测结果。
- `summary.html`：推理结果汇总。
- `viz/`：仅在 `save_viz=true` 时生成。

## 1.4 接线与 Early Fusion

`model_weight_path` 可以直接接收 `ModelTrainYoloNode.model_output_dir`，也可以接收通过 `PathJoinNode` 拼出的 `weights/best.pt`。

推理节点不接收 `early_fusion_config`。它会根据测试 JSONL 第一条数据的 `views` 数量计算输入通道数，并与 checkpoint 中的融合通道核对。因此，单视图权重应使用单视图测试 JSONL，多视图权重应使用相同视图数的测试 JSONL；不一致时会报错。

如需复用训练时的分辨率和任务类型，可以将同一个 `YoloTrainingExtraConfigBuilderNode.yolo_extra_config` 同时连接到训练和推理节点。
