# Early Fusion Config Builder 节点

## 1.1 功能概述

为 YOLO 多视图训练生成 Early Fusion 配置，将 N 个 RGB 视图组成的 `3N` 通道输入融合为模型骨干网络使用的 3 通道输入。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| data_channels | INT | 否 | `6` | — | 输入通道数，应为 `3 × views 数量`；双视图为 6，三视图为 9 |
| model_channels | INT | 否 | `3` | — | 融合后输入 YOLO 骨干网络的通道数，通常保持 3 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| early_fusion_config | STRING | Early Fusion 配置 YAML |

## 1.4 使用说明

- 仅连接到 `ModelTrainYoloNode.early_fusion_config`，推理节点不需要此配置。
- 单视图数据不要连接本节点，直接使用原生 3 通道训练。
- 多视图 JSONL 中每条样本的 `views` 数量应一致，并与 `data_channels` 对应。例如双视图数据使用 `data_channels=6`。
- 推理时，`ModelInferYoloNode` 会从测试 JSONL 的视图数推断通道，并与 checkpoint 中保存的融合配置核对。
