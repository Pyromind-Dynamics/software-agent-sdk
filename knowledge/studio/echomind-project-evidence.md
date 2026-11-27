# EchoMind Project 执行记录与交付

在 `public_data/workflowPLAN.json` 对应的计划任务上记录 `business_status`
（pending、in_progress、blocked、completed 或 failed）、`decision` 和 `next_action`。
`attempt` 从 1 开始。提交后填写真实的 `platform_task_id`。有回执但归属尚待确认时保留回执。
每个 `verified_outputs` 键使用计划声明的输出名，值为已核验的绝对 `/workspace/...` Storage 路径。

Dataset 回放使用已审阅的准备、标注和导出脚本；核对实际训练 reader 能读取交付格式，
多模态 SFT 还需检查 messages 字段与媒体顺序。Train 交付实际选定的 checkpoint 或
合并模型，并把同一路径传给获批的 Eval。组装报告时复用已有评测预测。

## 计划声明的报告

为 kind=report 的输出在 Storage 保存 JSON，并将路径登记在对应输出名下。
summary 写出结论和证据来源。复用历史结果时也从已核验的原始指标生成同一格式。

- Init、Train、Eval 提供非空 metrics 数组，每项含 label、数值 value 和 unit。计数用 count，分数按实际量纲填写，0–1 的比例用 ratio，已乘 100 的百分数用 percent。缺失值不补零。
- Dataset 提供非空 files 数组；每项记录 name、绝对 path、实际 sample_count，以及可验证的 size_bytes 和 distributions（维度到标签计数）。布尔验证结果放在 checks，metrics 可选。
- 通过 product_versions 保存版本时，记录返回的 report_version_id。读取冻结文本报告用 product_versions read 和实际 artifact_id。声明的交付物由 Runtime 核验并关联到对应尝试。
