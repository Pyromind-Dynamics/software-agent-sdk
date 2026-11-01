---
name: avi-baseline-evaluation
description: AVI 图片质检基模评测的业务口径与证据要求。
---

# AVI 基线评测

Init 的基模 bench 使用 inference-evaluation 的真实推理、评分和报告能力，
不要求先建立或完善 Graph。核查实际模型、评测集、GT、媒体顺序、预处理和生成参数，
保存可复用配置。模型目录不等于单个 Storage 文件，不要求人为改成索引文件。

按真实标签检查分类正确性、各类 Precision/Recall/F1、混淆矩阵、漏检和误放行；
有定位 GT 才算同坐标系框指标。业务正确性优先于格式项，不把加权综合分当成准确率。
无业务阈值时展示结果与差距，不推断上线标准。小样或部分失败说明数量与覆盖率，
不能展示为完整 baseline。向 Manager 交付结果、证据、局限和建议。
