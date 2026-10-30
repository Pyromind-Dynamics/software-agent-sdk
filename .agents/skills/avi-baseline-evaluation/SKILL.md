---
name: avi-baseline-evaluation
description: AVI 图片质检实验的基模评测口径与证据要求。
---

# AVI 基线评测

Eval 为已确认计划中的目标任务准备基模 bench；AVI 计划使用 Inference 和 Evaluator
阶段表达推理与评分工作，阶段不等于 Agent。
固定评测集、GT、媒体、预处理、模型版本、生成参数和评分规则，供后续同口径比较。

根据实际标签定义检查分类正确性、各类 Precision/Recall/F1、混淆矩阵、漏检和误放行；
有定位 GT 时再使用相同坐标系的框指标。未提供业务阈值时展示结果与差距，不推断上线标准。
小样或部分失败必须说明实际数量与覆盖率，不能标成完整 baseline。
本角色不清洗数据、不训练、不部署服务、不切流；结果交给 Manager 判断后续计划。
