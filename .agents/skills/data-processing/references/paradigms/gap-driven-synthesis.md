# Gap 驱动合成

只从已确认的 `gap_plan.json` 生成 `augmentation_plan.json`。计划须声明每个
Gap、策略 ID、数量、宿主筛选、seed、单宿主最大复用次数和输出 schema。

## 选型

- 普通文本：PromptedGenerator / FormatStrPromptedGenerator。
- SFT：CondorGenerator → CondorRefiner → AlpagasusFilter。
- 推理、代码、多轮、Function Call：选 DataFlow 1.0.10 对应 Generator/Filter。
- 图片输入生成问答、描述或回答：managed image runtime。
- AVI/PCB 新像素：仅 `avi_pcb.*` 插件；`model_profile=none` 的普通 Python
  Pipeline 调用 staged `avi_pcb_runtime.execute_plan`，不从兄弟仓库导入。
- 其他图片新像素：报告 `missing synthesis strategy`，不猜绘图规则。

所有模型调用保留 LoggingLLMServing 的分批、墙钟截止、失败账本和断点状态。
AVI/PCB 策略必须在一次调用内同时生成图片、diff、bbox/类别与血缘；校验图片
有变化、bbox 合法、标签一致、diff 非空，同 seed 可复现。验证/测试/eval 样本
不能作为宿主；split 不明确时停止。

先用 `preview_dataset(mode="sample", n=3)` 准备输入，并用
`df_run_pipeline` 执行小样。文本展示记录，图片展示
before/after/diff/annotation 复核图。用户明确确认后才调用
`df_submit_pipeline`。运行中用 `df_check_progress`，介入时用
`df_stop_task`。执行器不截断输入；AVI 小样数量由小样输入和派生的小样计划控制。

独立交付包包含：`augmentation_plan.json`、`processed.jsonl`、`assets/`、
`provenance.jsonl`、`validation.json`、`progress.json`、`failures.jsonl`、
`report.json`、`report.html`。不改源数据、不合并父数据集、不创建版本记录。
