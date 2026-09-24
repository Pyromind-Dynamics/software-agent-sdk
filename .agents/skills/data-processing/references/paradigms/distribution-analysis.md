# 分布分析

目标是从源数据得到可审计的标签分布和 Gap 草案，不借用 eval 结论。
涉及共享领域知识（如 PCB AVI/AOI）时，先按对应 case 文档给出的运行时知识库逻辑
路径读取领域参考，不要相对本 skill 目录拼接。

## 路由

- 每条样本已有可确定映射的标签：生成纯 Python 统计脚本，全量统计，模型调用
  必须为零。
- 只有标签定义、样本未标注，等同无标签。
- 无标签：按真实 schema 取少量样例行（文本最多 10 条，图片按需取样），生成带
  稳定 ID、`single|multi` 和判定边界的 taxonomy；先试标 3 条并等待确认。
- 无法确定样本边界或无法排除验证集时停止，请用户指定。

## 执行契约

先探查 schema：直接读 `storage/<input_path>`。已有标签且源文件在本地可处理范围内
时，生成纯 Python Pipeline，并以 `model_profile=none` 调用 `df_run_pipeline`
对 `storage/` 源路径做全量聚合；超出本地可处理范围时，先对 sample 的逻辑小样验证
同一脚本，再用 `df_submit_pipeline` 对 Storage 源路径全量执行。

Taxonomy 确认后，`N <= 1000` 全量标注；否则按 `sample_id`、固定 seed 做稳定
哈希抽样，严格取 1000 个唯一训练样本。文本选 DataFlow PromptedGenerator /
FormatStrPromptedGenerator；图像或混合数据选 managed image runtime。模型输出
必须符合严格 JSON Schema，解析失败、超时、unknown 写入 `failures.jsonl`。

无标签的 3 条试标输入由直读 `storage/` 提供，不由 `df_run_pipeline` 截断。
Taxonomy 确认后，业务抽样中的全部记录都应进入打标。
文本用 `model_profile=text`，图像用 `vision`；工具不接受 requirements。

精确报告写 count/ratio；抽样报告同时写样本 count/ratio、总体数量估算和 95%
Wilson 区间，并标记 `sampled_estimate`。多标签比例允许总和超过 100%。
需要断点续跑时，spec 必须写入按文件内容（目录按相对路径排序）计算的 SHA-256
`source_fingerprint`；输入变化后必须新建 run，不能复用旧抽样/检查点。

Agent 基于报告和业务上下文起草 `gap_plan.json`；工具不硬编码“低频必补”。
抽样时 `current_count` 是样本实测，`estimated_total_count` 才是总体估计。
用户确认目标数量后将 Gap 状态改为 `approved`。

核心产物：`processed.jsonl`、`taxonomy.json`、`analysis_spec.json`、`labels.jsonl`、
`distribution_report.json`、`gap_plan.json`、`progress.json`、
`failures.jsonl`、`validation.json`、`report.json`、`report.html`。
