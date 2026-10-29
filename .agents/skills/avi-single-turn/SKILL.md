---
name: avi-single-turn
description: 为 AVI 图片质检设计实验计划，覆盖基模评测、数据处理、训练与训练后评测，支持只启用部分阶段。
---

# AVI 实验计划

本步只提供计划管理。计划、待确认提案和历史版本通过实验计划工具读取；持久化由 Runtime 负责。

## 参考流程

默认阶段图：`Init → Data → Train → Eval`。另保留 Inference、Tracer 两个未启用的占位阶段（`placeholder: true, enabled: false`）。

| 阶段 | 计划中的职责 | 典型输入与预期产物 |
|---|---|---|
| Init | 使用基模跑 bench，建立比较基线 | 基模、固定评测集 → baseline 指标与报告 |
| Data | 检查数据，设计清洗或预打标方案 | 原始数据 → 供训练使用的数据版本 |
| Train | 配置并训练模型 | 数据版本、基模 → checkpoint 与训练报告 |
| Eval | 按相同口径评测训练结果并与 baseline 比较 | checkpoint、固定评测集、baseline → 比较结论与改进建议 |

Init 是图中第一个独立阶段，后续连接 Data；后续实现中可复用 Eval 子 Agent 的评测能力。阶段表示业务工作，任务表示阶段内的一项活，阶段不等于 Agent。

用户可以只做评测或启用其他阶段组合。以用户目标调整阶段启用情况和连线，无需收集未启用阶段的训练参数。Inference、Tracer 当前仍保持占位。运行中的跳过、重试、回退和输入失效处理属于后续能力。

## 计划内容

明确实验目标、已有模型和数据、评价口径、预期结果与约束。未知的资源或执行参数写入 `unresolved`，不编造 URI，也不因为暂缺训练参数而阻止保存评测计划。

- `resources` 登记所需外部资源：`id, name, uri, description`；尚未确定的资源也登记占位项，`uri: null`，缺失信息记入 `unresolved`，任务输入仍通过 `resource_id` 引用，不能只填名称。固定评测集用于后续 baseline 与训练后评测的同口径比较。
- `stages` 使用稳定的局部 ID，填写 `name, description, enabled, placeholder`。
- `tasks` 填写稳定局部 ID、`stage_id`、自由名称、说明、输入、预期输出和完成条件声明。推荐任务名如“基模评测”“检查标注完整性”“小样训练验证”，无固定业务 task_type。
- 外部输入写作 `{name, resource_id}`；上游预期产物写作 `{name, task_id, output_name}`，对应上游 `outputs` 中的名称。产物尚未生成也可用于规划。
- `outputs` 使用 `{name, description}`；`completion_conditions` 使用 `{kind, description}` 描述完成依据，如报告产出、外部任务终态。本步只保存声明，不判断完成。
- `edges` 使用阶段 ID 的 `{source, target}`，表示阶段先后引导。任务输入声明表达具体参数来源；图和输入依赖都不能形成环。

首次用 `create_experiment_plan` 生成完整草稿。修改前用 `get_experiment_plan` 获取最新版本；用 `revise_experiment_plan` 提交完整新计划、`expected_revision` 和修改摘要。存在待确认草稿时同时提交它的 `source_proposal_id`。未改动的阶段、任务 ID 保持稳定。

前端确认后才创建项目或新版本。草稿可展示图，不能宣称已经执行任何阶段，也不能把用户在聊天中说“确认”当作数据库提交成功。
