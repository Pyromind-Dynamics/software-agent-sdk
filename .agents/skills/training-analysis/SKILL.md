---
name: training-analysis
description: >-
  使用 training_analysis 发现训练 checkpoint/eval 产物，分析指标、稳定性和
  优化方向。用户提到训练效果、loss 异常、NaN、发散、过拟合、训练对比或
  超参调优时使用；task_id 是定位训练 run 的必要输入。
---

# 训练分析

通过 `training_analysis` 完成训练 run 的定位、指标探查、诊断和报告。认证、
数据源连接和临时文件由服务端管理，不向用户索取，也不在回答中展示。

## 训练产物

训练完成后用 `operation="artifacts"` 读取实际任务的 output_path，发现所有
checkpoint-n 与 eval_step_n.jsonl；不依赖 W&B。多个训练节点时用
`training_node_id` 明确目标。返回文件指纹、样例、字段和模型完整性；不能将
不同 step 的 eval 与 checkpoint 配对。路径未解析或文件不可访问时按缺口处理。

已有 eval 交给 [inference-evaluation](../inference-evaluation/SKILL.md) 的
已有预测评分入口：读取真实样例后生成一份评分脚本，计算所有 step，并复用基线预测。
结合实际训练 step 的 W&B 序列解释选择；log_step 不等于 checkpoint step，未记录值不等于 NaN。

## 指标分析

1. **探查**：调用 `training_analysis(operation="probe", task_id="...")`，确认
   目标 run、可用指标键和配置键。
2. **分析**：调用 `training_analysis(operation="analyze", task_id="...")`。
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
执行训练的完成回调还需按原任务衔接 [inference-evaluation](../inference-evaluation/SKILL.md)：
用户未排除评测时，先复用训练 eval 产物选择 checkpoint，已有基线按同口径复用评分。仅 train + val 不新增模型推理；独立 test 在验证集选模后
使用所选 checkpoint 推理评分，不参与选模；LoRA 合并所选 step。
人工复核先完成分布分析，再导入现有标注组件创建独立任务，不依赖完整评测报告。
已有可比基线直接复用。只分析已有 run 或明确不评测时不新增任务；缺少验证集、GT 或可用产物时
说明缺口，不以训练集或训练 loss 宣称效果通过。W&B 取数失败不阻止已有模型产物的效果评测。
训练参数调整仍按单变量探针进行；用户确认后交给工作流生成能力，并在新
task 完成后重新分析。

训练分析取数只使用 `training_analysis`，不调用 terminal 或直接运行本 skill
下的脚本，不手工传递认证信息、保存临时认证文件或直接访问 W&B。
交接后的数据处理执行方式以 data-processing 的工具与 SOP 为准。
