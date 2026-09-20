---
name: data-processing
description: >-
  Pyromind 数据处理统一入口。用于格式转换、DataFlow 内容处理、从源数据分析
  标签分布、PCB AVI/AOI 图片预打标、按已确认 Gap 合成文本或图片，以及数据自带镜像中的
  环境编排处理。统一执行预览、小样、人工确认、异步全量和报告校验。
license: MIT
---

# 数据处理

数据处理任务统一从本 skill 进入：先按路由表选定处理范式并读取对应 playbook
（范式内按 case 路由表读取场景 case 文档），再按通用 SOP 执行。storage 数据
先用 `preview_dataset` 探查；需要执行本地 Pipeline 时再用 `sample` 或
`materialize` 模式将明确范围的数据放入会话工作区。

领域参考（如 PCB AVI/AOI）不内联在本文件：由对应 case 文档按需引入，并在那里
给出运行时知识库逻辑路径的读取方式。本地试跑只用于验证链路是否打通；正式批量
与正规化执行统一在平台侧用 `df_submit_pipeline` 完成。

## 范式路由（先做这一步）

| 需求特征 | 处理范式 | playbook |
|---|---|---|
| 格式转换/字段映射/简单过滤，无 LLM、无跨行操作 | format-conversion | references/paradigms/format-conversion/playbook.md |
| 内容级清洗/抽样/生成/评分，用 DataFlow 算子或 LLM | llm-pipeline | references/paradigms/llm-pipeline/playbook.md |
| PCB AVI/AOI 真点/假点预判、问题分类及区域定位 | llm-pipeline（PCB 预打标，默认 structured JSONL） | references/paradigms/llm-pipeline/playbook.md → cases/pcb-inspection.md |
| 从源数据统计/推断标签分布并提出 Gap，不依赖 eval | distribution-analysis | references/paradigms/distribution-analysis.md |
| 按已确认 Gap 合成文本或图片数据 | gap-driven-synthesis | references/paradigms/gap-driven-synthesis.md |
| 单条数据要在特定环境（数据自带镜像）执行复杂流程（跑命令/测试/判定），需多阶段编排 | environment-processing | references/paradigms/environment-processing/playbook.md |
| S2/LeRobot 机器人数据的多模态对齐、静止帧清理、批量转换与校验 | environment-processing（embodied case） | references/paradigms/environment-processing/cases/embodied-data-cleaning.md |
| 编排任务无匹配 ProcessingProfile（一次性/探索性） | environment-processing（模式 B） | 同上 |

负向边界：仅创建/管理单个沙箱容器 → sandbox；训练效果/loss 分析 →
training-analysis；生成训练工作流 → generate-workflow-dsl；工作流调试 →
debug-workflow。sandbox 不得用来搬运或组装 Pipeline 的 Storage 输入。路由不确定
时 AskUserQuestion，不要猜。

## 通用 SOP（所有场景共享的控制面骨架）

1. **探查**：preview_dataset 先行；storage 路径本地不可见，不要先在本地找。
   目录列表超 100 条会被截断——用 `path_filter` 子串精确定位条目，不要反复
   翻页重预览；每个数据集的结构确认一次完成（列表 + schema + 样例）。
2. **选型**：按上表读取范式 playbook；范式内按 case 路由表只读取当前场景
   相关 reference；领域参考由 case 文档按需引入。
3. **本地执行**：Agent 按真实 schema 写 Python Pipeline。`df_run_pipeline`
   完整处理传入的本地输入，不负责抽样；清洗/合成小样由 preview 选择的输入
   和计划控制，精确分布统计可处理完整 materialize 文件。
   依赖第三方库的探索性检查也通过该工具执行；通常不传 `python`，由工具使用
   服务端配置的解释器（`DATAFLOW_PYTHON` 或服务端自身 Python）。terminal 的
   Python、依赖和沙箱权限与之不同，不能用终端导入失败判定 Pipeline 不可用。
   环境缺失以工具实际预检/执行结果为准，再处理运行环境配置。
4. **门禁**：Taxonomy 与合成小样通过后必须获得用户明确确认才提交后续全量；
   已有标签的确定性全量统计不设人工门禁，且模型调用必须为零。
5. **全量**：需要平台执行时统一用 `df_submit_pipeline`；提交前先按
   llm-pipeline playbook 的「平台全量输入」确认 `input_path` 已指向 Storage 输入，
   工作区文件用 `upload_file_to_pyromind` 落位。DataFlow/EDP 平台任务（含具身
   case 的逐 episode 执行）等待终态回调，运行中用 df_check_progress 观察。
   需介入平台任务时先 df_stop_task 停任务。
6. **分诊**：回调后先看 report.json / validation / verdicts，按失败分类决定
   resume 还是新 run；终止/失败不自动重提交，交用户决策。
   report.failures 非空时（failures.jsonl：无响应/截止/解析失败的记录），
   把其中的 input 行重组为子集输入补跑同一 pipeline 并合并产出，单条挂起
   不再阻塞整轮（墙钟截止见 dataflow-common.md）。子集文件须与原 source
   放在**同一目录** —— input 行保留的是相对图片路径，只有同目录才能解析
   回原图（绝对路径会被 pipeline 以"必须是相对 POSIX 路径"拒绝）。
7. **交付**：展示产物与统计；场景内的产物契约与失败分类学以 playbook 为准。

一期边界：分布分析不读取 eval/rubric/badcase；不注册 Dataset Version 或
SQLite 元数据；不处理音视频。图片合成可复用、组合或编写策略；适用条件、
业务规则和验证依据写入合成计划，详见 gap-driven-synthesis，不默认沿用
其他场景的像素或类别规则。
