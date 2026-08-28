---
name: data-processing
description: >-
  Pyromind 数据处理统一入口，含三种处理范式（按执行基底划分，每种范式下
  细分场景 case）：格式转换/字段映射/简单过滤（format-conversion，无 LLM）；
  DataFlow 抽样/清洗/生成/评分/格式化（llm-pipeline，覆盖 SFT、DPO、推理、
  代码、RAG、多轮对话、Function Call、质量评估、Text2SQL、多模态标注等
  case）；依赖真实运行环境的处理（environment-processing，按数据自带
  镜像起沙箱执行任务并判定，当前以 tmax 可用性验证为代表 case）。
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
| 逐条依赖真实运行环境（docker 镜像、工具链、判定/执行） | environment-processing | references/paradigms/environment-processing/playbook.md |
| 环境依赖任务但无匹配 ProcessingProfile | environment-processing（模式 B） | 同上 |

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
5. **全量**：平台提交后等待 Kafka 终态回调；运行中用 df_check_progress 观察；
   需介入时先 df_stop_task 停任务，再修改再提交。
6. **分诊**：回调后先看 report.json / validation / verdicts，按失败分类决定
   resume 还是新 run；终止/失败不自动重提交，交用户决策。
7. **交付**：展示产物与统计；场景内的产物契约与失败分类学以 playbook 为准。
