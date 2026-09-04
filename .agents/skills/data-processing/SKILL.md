---
name: data-processing
description: >-
  Pyromind 数据处理统一入口，含三种处理范式（按执行基底划分，每种范式下
  细分场景 case）：格式转换/字段映射/简单过滤（format-conversion，无 LLM）；
  DataFlow 抽样/清洗/生成/评分/格式化（llm-pipeline，覆盖 SFT、DPO、推理、
  代码、RAG、多轮对话、Function Call、质量评估、Text2SQL、多模态标注等
  case）；编排式场景数据处理（environment-processing：单条数据的处理
  需在数据自带镜像的特定环境中执行复杂流程，编排"渲染分片→逐条
  执行→聚合"多阶段链路，覆盖 tmax 终端任务验证和具身机器人数据清洗
  等编排 case）。
  统一 SOP：预览探查→范式选型→小样试跑→用户确认→平台全量→回调分诊→交付。
  仅创建/管理单个沙箱容器用 sandbox；训练效果分析用
  training-analysis；生成训练/评测工作流用 generate-workflow-dsl。
triggers:
  - 数据清洗
  - 数据准备
  - 数据处理
  - 格式转换
  - 字段映射
  - tmax
  - 可用性验证
  - 可行性验证
  - 环境数据处理
  - 环境编排
  - 具身智能数据清洗
  - LeRobot
  - S2机器人数据
license: MIT
---

# 数据处理

数据处理任务统一从本 skill 进入：先按路由表选定处理范式并读取对应 playbook
（范式内按 case 路由表读取场景 case 文档），再按通用 SOP 执行。storage 数据
只能用 `preview_dataset` 查看，不得本地下载或用本地文件工具读取。

## 范式路由（先做这一步）

| 需求特征 | 处理范式 | playbook |
|---|---|---|
| 格式转换/字段映射/简单过滤，无 LLM、无跨行操作 | format-conversion | references/paradigms/format-conversion/playbook.md |
| 内容级清洗/抽样/生成/评分，用 DataFlow 算子或 LLM | llm-pipeline | references/paradigms/llm-pipeline/playbook.md |
| 单条数据要在特定环境（数据自带镜像）执行复杂流程（跑命令/测试/判定），需多阶段编排 | environment-processing | references/paradigms/environment-processing/playbook.md |
| S2/LeRobot 机器人数据的多模态对齐、静止帧清理、批量转换与校验 | environment-processing（embodied case） | references/paradigms/environment-processing/cases/embodied-data-cleaning.md |
| 编排任务无匹配 ProcessingProfile（一次性/探索性） | environment-processing（模式 B） | 同上 |

负向边界：仅创建/管理单个沙箱容器 → sandbox；训练效果/loss 分析 →
training-analysis；生成训练工作流 → generate-workflow-dsl；工作流调试 →
debug-workflow。路由不确定时 AskUserQuestion，不要猜。

## 通用 SOP（所有场景共享的控制面骨架）

1. **探查**：preview_dataset 先行；storage 路径本地不可见，不要先在本地找。
2. **选型**：按上表读取范式 playbook；范式内按 case 路由表只读取当前场景
   相关 reference。
3. **小样**：按 playbook 规定的试跑形态先小样（limit=3 / sample / smoke 片），
   迭代过程不向用户展示，只展示符合预期的结果。
4. **门禁**：小样通过后必须获得用户明确确认才提交全量；禁止 agent 默认全量。
5. **全量**：按场景 case 的执行约定提交。DataFlow/EDP 平台任务等待终态回调，
   运行中用 df_check_progress 观察；具身 Sandbox case 按其 PID、日志和
   report.json 约定轮询。需介入平台任务时先 df_stop_task 停任务。
6. **分诊**：回调后先看 report.json / validation / verdicts，按失败分类决定
   resume 还是新 run；终止/失败不自动重提交，交用户决策。
7. **交付**：展示产物与统计；场景内的产物契约与失败分类学以 playbook 为准。
