# EchoMind Project 执行记录与交付

在 `public_data/workflowPLAN.json` 对应的计划任务上记录 `business_status`
（pending、in_progress、blocked、completed 或 failed）、`decision` 和 `next_action`。
`attempt` 从 1 开始。提交后填写真实的 `platform_task_id`。有回执但归属尚待确认时保留回执。
每个 `verified_outputs` 键使用计划声明的输出名，值为已核验的绝对 `/workspace/...` Storage 路径。

Dataset 回放使用已审阅的准备、标注和导出脚本；核对实际训练 reader 能读取交付格式，
多模态 SFT 还需检查 messages 字段与媒体顺序。Train 交付实际选定的 checkpoint 或
合并模型，并把同一路径传给获批的 Eval。组装报告时复用已有评测预测。

## 计划声明的报告

Main Agent 在当前任务和 attempt 的 evidence_paths 中交接原始结果、指标或清单的绝对路径，在 verified_outputs 中记录已经核实的非报告产物。执行结束后，Runtime 调用对应阶段的报告子 Agent；它读取这些证据，分析、格式化、校验并提交主报告，Runtime 保存版本回执。纯分析任务交接复用的上游证据路径。
