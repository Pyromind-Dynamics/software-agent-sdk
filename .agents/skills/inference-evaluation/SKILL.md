---
name: inference-evaluation
description: 分析带 ground truth 的测试集，制定确定性 rubric，通过 Pyromind Storage 模型推理和 CPU 评测生成指标与 HTML 报告。
---

# 模型推理评测

当前 Agent 负责理解任务和制定 rubric；被测模型只生成 prediction，固定的评测脚本负责评分。

## 输入与评价标准

使用 `preview_dataset` 确认测试集 JSONL、输入字段、GT、媒体路径和代表性样本。
媒体相对路径基于 JSONL 所在目录；不要重复设置数据集目录为 `media_base_dir`。
读取 [输入契约](references/input-contract.md) 和 [Rubric 编写](references/rubric-authoring.md)，
在当前工作区写出 `public_data/inference_dataset_config.json` 和
`public_data/inference_evaluation_config.json`。

选择少量互不重复的评价维度，每项包含 name、criterion、weight、required 和 evaluator。
关键业务正确性高于格式要求；核心项可设 required，失败时整体不通过。
只评价能从数据和输出要求中确定的内容。首版不使用语义 Judge 或被测模型自评。

## 执行

只讨论方案或评价标准时，完成配置与说明即可。用户要求实际评测或生成报告时调用
`df_submit_pipeline`，传入 `input_path` 和结构化 `inference` 参数，参见
[两节点执行契约](references/command-node-pipeline.md)。

工具负责预检、冻结上传脚本和配置、生成并校验工作流、同步画布和正式提交：
`VLLMInference.endpoint → CustomCommandCPUNode.param`。
无需在本机运行 GPU 推理，不手工上传或修改内置脚本，不传 script_path、model_profile、
output_schema 或 convert_format。不使用 workflow_debug 或 run_workflow。
不接受已有外部 endpoint、自定义节点类型或其他拓扑。

## 结果与恢复

提交返回 task_id、run_id 和 output_dir。使用 `df_check_progress` 查询进度；用户要求停止时
使用 `df_stop_task` 停止整个工作流。平台回调只代表任务终态，不代表业务指标达标。

回调后通过 `preview_dataset` 读取 output_dir 下的 report.json 和 metrics.json，核对样本数、
成功 prediction 数、完成评分数、通过率及执行错误。通过 `df_check_progress` 的 artifact_urls 获取报告访问地址后交付
HTML 链接；不要把容器路径当成用户可打开的地址，也不要编造签名下载链接。
若工具未返回可访问 URL，交付 Storage 位置及汇总，明确报告需从 Storage 打开。

模式 `resume` 使用原 run_id，input_path 不变，可省略 inference 复用冻结配置。
已有 prediction 不重复请求，仅补发缺失 prediction；业务不通过不等于需要重新推理。
模型引用、数据内容、配置或评测脚本变化必须新建 full 运行。模型路径必须指向不可变版本。
活跃任务需先停止，等待停止成功后再续跑；提交结果不确定时不要盲目重复提交。
