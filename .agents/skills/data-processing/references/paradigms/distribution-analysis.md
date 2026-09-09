# 分布分析

目标是从源数据得到可审计的标签分布和 Gap 草案，不借用 eval 结论。

## 路由

- 每条样本已有可确定映射的标签：生成纯 Python 统计脚本，全量统计，模型调用
  必须为零。
- 只有标签定义、样本未标注，等同无标签。
- 无标签：preview 真实 schema（文本最多 10 条，图片沿用工具上限），生成带
  稳定 ID、`single|multi` 和判定边界的 taxonomy；先试标 3 条并等待确认。
- 无法确定样本边界或无法排除验证集时停止，请用户指定。

## 执行契约

Taxonomy 确认后，`N <= 200` 全量标注；否则按 `sample_id`、固定 seed 做稳定
哈希抽样，严格取 200 个唯一训练样本。文本选 DataFlow PromptedGenerator /
FormatStrPromptedGenerator；图像或混合数据选 managed image runtime。模型输出
必须符合严格 JSON Schema，解析失败、超时、unknown 写入 `failures.jsonl`。

用 `run_dataset_analysis` 做本地 3 条小样；确认后才调用
`submit_dataset_analysis`。只可选 `dataflow_text`、`dataflow_vision`、
`avi_pcb_cpu`，不得传 pip 依赖。

精确报告写 count/ratio；抽样报告同时写样本 count/ratio、总体数量估算和 95%
Wilson 区间，并标记 `sampled_estimate`。多标签比例允许总和超过 100%。
需要断点续跑时，spec 必须写入按文件内容（目录按相对路径排序）计算的 SHA-256
`source_fingerprint`；输入变化后必须新建 run，不能复用旧抽样/检查点。

Agent 基于报告和业务上下文起草 `gap_plan.json`；工具不硬编码“低频必补”。
抽样时 `current_count` 是样本实测，`estimated_total_count` 才是总体估计。
用户确认目标数量后将 Gap 状态改为 `approved`。

核心产物：`taxonomy.json`、`analysis_spec.json`、`labels.jsonl`、
`distribution_report.json`、`gap_plan.json`、`progress.json`、
`failures.jsonl`、`validation.json`、`report.json`、`report.html`。
