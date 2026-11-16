# Clone Dataset 节点

![alt text](/imgs/CloneAndCacheDataset/CloneAndCacheDataset.png)  

## 1.1 功能概述

克隆数据集

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| dataset | STRING | 是 | 未声明 | 候选：`"OpenGVLab/ScaleCUA-Data"`、`"Writer/omniact"`、`"agibot-world/AgiBotWorld-Alpha-327"`、`"agibot-world/AgiBotWorld-Alpha-327-Extract"`、`"agibot-world/AgiBotWorld-Alpha-CtrlWorld-327"`、`"agibot-world/AgiBotWorld-Alpha-Lerobot-327"`、`"agibot-world/AgiBotWorld-Alpha-Openpi-327"`、`"cadene/droid_1.0.1"`、`"gui-360/gui-excel"`、`"gui-360/processed_data"`、`"henryhe0123/PC-Agent-E"`、`"openai/gsm8k"`、`"pyromind/alpaca-gpt4-llm-demo"`、`"pyromind/geometry-vqa-vlm-demo"`、`"pyromind/self-cognition"`、`"ritzzai/GUI-R1"`、`"xlangai/aguvis-stage1"`、`"xlangai/aguvis-stage2"`、`"zonghanHZH/UGround-V1-8k"` | 平台预置数据集标识，按当前环境返回的候选列表选择 |
| target_path | STRING | 是 | `"/workspace/datasets/"` | — | 克隆后的数据集存放目录 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| dataset_path | STRING | 克隆完成后的数据集路径 |
