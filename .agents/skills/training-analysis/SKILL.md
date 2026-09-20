---
name: training-analysis
description: >-
  使用 training_analysis 业务工具分析 Pyromind 训练任务的指标、稳定性和
  优化方向。用户提到训练效果、loss 异常、NaN、发散、过拟合、训练对比或
  超参调优时使用；task_id 是定位训练 run 的必要输入。
---

# 训练分析

通过 `training_analysis` 完成训练 run 的定位、指标探查、诊断和报告。认证、
数据源连接和临时文件由服务端管理，不向用户索取，也不在回答中展示。

## 固定流程

1. **探查**：调用 `training_analysis(operation="probe", task_id="...")`，确认
   目标 run、可用指标键和配置键。
2. **分析**：调用 `training_analysis(operation="analyze", task_id="...")`。[train.jsonl](../../../../../Downloads/sft_aoi/train.jsonl)
   用 `metric` 指定主指标，需要同时查看多个指标时用 `keys`（最多 20 个）。
3. **报告**：需要可复用的 Markdown 结论时调用
   `training_analysis(operation="report", task_id="...")`。可选的
   `output_path` 必须位于 `public_data/training-analysis/`；未指定时使用默认
   报告路径，并在结果中返回 `report_path`。

`run_url` 只能作为已有 `task_id` 的定位提示，不能替代 task_id。一次分析只
   针对一个 task；对比多次训练时分别分析每个 task，再比较指标、配置、状态和
   诊断结果。先探查再选择指标，避免请求无关的大量 history。

## 结论与后续

说明指标趋势、NaN/尖峰、发散、收敛和过拟合信号，并把结论与配置变化对应起来。
报告出现数据异常或调优收益不稳定时，建议继续检查训练数据质量（空值、重复、
格式、标签分布和异常样本）。需要补数据时，将训练证据、数据来源和待验证的
假设交给 [data-processing](../data-processing/SKILL.md)：先检查源数据并确认
Gap，再进入 [合成范式](../data-processing/references/paradigms/gap-driven-synthesis.md)
选择合成方法、试跑和复核。
loss 只能提示问题，不能证明某个类别缺样，也不能直接据此决定扩增数量。
训练参数调整仍按单变量探针进行；用户确认后交给工作流生成能力，并在新
task 完成后重新分析。

训练分析取数只使用 `training_analysis`，不调用 terminal 或直接运行本 skill
下的脚本，不手工传递认证信息、保存临时认证文件或直接访问 W&B。
交接后的数据处理执行方式以 data-processing 的工具与 SOP 为准。
