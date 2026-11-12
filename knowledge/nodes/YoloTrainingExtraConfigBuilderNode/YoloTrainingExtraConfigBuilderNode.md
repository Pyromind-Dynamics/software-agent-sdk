# YOLO Training Extra Config Builder 节点

## 1.1 功能概述

生成 YOLO 专用配置。该配置可连接到 `ModelTrainYoloNode`，也可连接到 `ModelInferYoloNode`；推理只使用其中的图像尺寸和任务类型。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| imgsz | INT | 否 | `960` | — | 训练和推理图像尺寸；显存不足时可调低为 640 或 320 |
| patience | INT | 否 | `20` | — | 提前停止的耐心 epoch 数；设为 0 表示不提前停止，推理不使用 |
| freeze | STRING | 否 | `"15"` | — | 冻结骨干网络前 N 层，也可填写逗号分隔的层索引；推理不使用 |
| workers | INT | 否 | `8` | — | DataLoader worker 数，推理不使用 |
| exist_ok | BOOLEAN | 否 | `true` | — | 是否允许训练输出目录已存在，推理不使用 |
| task | STRING | 否 | `"detect"` | 可选值：`"detect"`、`"segment"`、`"classify"` | YOLO 任务类型，必须与数据标注和权重一致 |
| pretrained | BOOLEAN | 否 | `true` | — | 是否使用 `model_config.model_path` 指定的权重初始化；推理不使用 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| yolo_extra_config | STRING | YOLO 专用配置 YAML |

## 1.4 使用说明

- `freeze="15"` 表示冻结模型序列中索引 0 至 14 的前 15 层；设为 `"0"` 可进行全网微调。
- Early Fusion 会根据实际层数限制冻结数量，融合层本身不在该冻结范围内。
- 预训练权重路径不在本节点配置，应通过 `ModelConfigBuilderNode.model_path` 提供。
- `task=detect` 使用检测权重，`task=segment` 使用分割权重，`task=classify` 使用分类权重，不能只修改任务值而不更换相应数据和权重。
