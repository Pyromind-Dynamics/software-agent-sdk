---
name: experiment-evaluation
description: Prepare reproducible evaluation proposals from a confirmed experiment task, with explicit input evidence and human execution approval.
---

# 实验评测

使用任务包中的 run_id；阶段名称和任务名称不决定工具行为。
通过 read_experiment_resource 读取任务声明的模型版本清单、数据、标签或评价要求，保留返回的 SHA-256。
需要进一步检查输入时可用 read；只有声明的输入才可进入执行提案，不能把读到的其他文件当作已批准输入。

提交 submit_experiment_evaluation 的内容：

- run_id：任务包给出的执行身份。
- configuration：真实 input_path、model_path、model_version、dataset_config、evaluation_config、生成参数及计算资源。
- input_fingerprints：以资源 ID 为键，填写读取工具返回的真实 SHA-256。
- expected_artifacts：predictions.jsonl、metrics.json、report.json、evaluation_report.html。
- conditions：external_task_succeeded、artifacts_verified、agent_result_valid，每项使用 kind 字段。

dataset_config 使用 reference_field 和 user_prompt_field 或 messages_field，并按真实数据确定媒体字段。
评价采用可确定执行的 rubric：精确匹配、字段判等、JSON 合法性或框 IoU。模型产生预测，固定代码评分。
已知版本与来源须能核验。只有可变模型目录时，说明需要版本或模型文件清单，不伪造版本标识。
资源未知、文件不可读或评价口径不清时，明确报告缺口。用户确认计划不代表授权运行评测。
本入口仅生成执行提案，不调用 df_submit_evaluation 或平台创建接口。
