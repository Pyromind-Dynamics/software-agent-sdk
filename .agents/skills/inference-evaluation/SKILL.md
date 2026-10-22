---
name: inference-evaluation
description: 模型效果评测的统一入口，包括基模 Benchmark、训后评测、GT 对比和评测报告。分析测试集并制定确定性 rubric，使用 Storage、中心仓或外部仓模型，通过平台推理和 CPU 评测生成指标与 HTML 报告。
---

# 模型推理评测

当前 Agent 负责理解任务和制定 rubric；被测模型只生成 prediction，固定的评测脚本负责评分。
业务效果评测统一走本 Skill，包括训练前后的独立评测；不通过 generate-workflow-dsl
另建 Metric/Eval 评测链。训练工作流负责训练及模型产物准备；已有预测评分不需要加载或合并模型。
训练后按数据集角色处理：只有 train + val 时，复用已有 eval_step_n.jsonl 评分选模，
不额外启动模型推理。存在独立 test 时，先用 val 选择 checkpoint，再对该模型执行 test
推理和同口径评分；test 不参与选模。缺少已有预测时报告缺口，不自动补跑 GPU。

## 已有预测与训练产物

有 eval_step_n.jsonl 或已有基线 prediction 时，先读真实样例并明确业务判分含义，
按 [已有预测评分契约](references/saved-predictions.md) 生成本轮解析、评分脚本和说明，
通过 `df_submit_pipeline(evaluation=...)` 在 CPU 评分所有 step。此入口计算评分，可生成报告，不初始化人工标注，也不调用模型。评分规则变化时复用原始预测，新建评分运行。
训练集与验证集标签分布随报告展示；优先查看所选 checkpoint 的 badcase。
`iteration_context.json` 汇总目标、数据、step 对比、选模和人工反馈，供下一轮判断；
原因假设由 Agent 结合证据提出，不套用固定的补数据或重训规则。
独立 test 使用下面的推理入口，传 inference.selection_run_id 绑定验证集选模批次；
Runtime 校验所选 checkpoint、test 来源和独立性，CPU 使用该批次冻结的 Python 评分脚本。
parse_record 应支持 context.source=test，标识必须与 dataset_config 一致。LoRA 合并使用所选 step。

## 输入与评价标准

用 `read`/`terminal` 直读 `storage/` 下的测试集 JSONL，确认输入字段、GT、
媒体路径和代表性样本。媒体相对路径基于 JSONL 所在目录；不要重复设置数据集
目录为 `media_base_dir`。
读取 [输入契约](references/input-contract.md) 和 [Rubric 编写](references/rubric-authoring.md)，
在当前工作区写出 `public_data/inference_dataset_config.json` 和
`public_data/inference_evaluation_config.json`。

选择少量互不重复的评价维度，每项包含 name、criterion、weight、required 和 evaluator。
关键业务正确性高于格式要求；核心项可设 required，失败时整体不通过。
只评价能从数据和输出要求中确定的内容。首版不使用语义 Judge 或被测模型自评。
分类任务配置类别字段判等；要求结构化输出时检查 JSON；有框 GT 时按同坐标系配置 IoU。
读取 report.json / metrics.json 的 rubric_metrics：字符串类别有混淆矩阵、各类 Precision/Recall/F1
和 Macro-F1，框有总体 Precision/Recall/F1；关注业务关键类别的漏检、错放行和格式率、延迟。
未给出业务验收阈值时报告对照结果与差距，不将默认 rubric 通过阈值宣称为上线标准。

## 执行

只讨论方案或评价标准时，完成配置与说明即可。用户要求实际评测或生成报告时调用
`df_submit_pipeline`，传入 `input_path` 和结构化 `inference` 参数，参见
[执行契约](references/command-node-pipeline.md)。

GPU 按任务实际需求选择：结合模型规模与精度、上下文长度、媒体分辨率和并发估算显存，
优先满足需求的最小卡型、最少卡数，并考虑平台资源可用性和排队情况。
在 inference 中明确填写 gpu_product、gpu_count；执行契约中的单张 L40S 仅为示例。
不要默认使用 H200；仅在较小卡无法满足显存或必要性能要求，或用户明确指定时选用，
并简述选卡依据。样本量大不直接等于需要大卡，可通过分批、降低并发控制资源需求。

模型来源只写在 evaluation_config_path 指向的 JSON 的 model_source 中，不是工具参数。
storage 仅用于 Storage 中已有的模型文件，inference.model_path 为其 Storage 绝对路径。
Storage 中没有目标模型文件时，使用 clone（Clone Model）或 download（Download Model）节点准备模型。
clone 时该字段为中心仓模型标识；download 时为外部仓模型名称，
download_source 支持 huggingface（默认）和 modelscope。
用户指定来源时遵从；未指定来源的模型标识先用 clone，由平台实时校验中心仓可用性。
中心仓不支持时说明原因并给出 download 选择，不静默切换来源，不维护固定模型名单。

工具生成 public_data/workflow_canvas/workflow.py，预检并冻结脚本和配置，校验后正式提交。
storage 使用 Inference → CPU；clone/download 在 Inference 前增加对应模型准备节点，
其 model_path 输出绑定 Inference.model_path，再连接 Inference.endpoint → CPU.param。
不要手写评测 workflow.py 或通过另一工具重复提交；工具生成并提交同一份最终工作流。
无需在本机运行 GPU 推理，不手工上传或修改内置脚本，不传 script_path、model_profile、
output_schema 或 convert_format。不使用 workflow_debug 或 run_workflow。
所选 checkpoint 使用 model_source.type=checkpoint；full 直接推理，lora 合并准确 adapter 和基模，
参见已有预测评分契约。不接受已有外部 endpoint 或自定义节点类型。

## 结果与恢复

提交返回 task_id、run_id 和 output_dir。使用 `df_check_progress` 查询进度；用户要求停止时
使用 `df_stop_task` 停止整个工作流。平台回调只代表任务终态，不代表业务指标达标。

回调后直读 output_dir 下的 report.json 和 metrics.json，核对样本数、
成功 prediction 数、完成评分数、通过率及执行错误。评测报告的**首选交付链接**是
`df_check_progress` 返回的 `evaluation_report_url`（稳定链接，浏览器携带有效 Storage 凭据打开、链接不过期、
独立复核任务支持页内打标自动保存）：在最终回答中以 Markdown 链接 `[查看完整评测报告](稳定链接)` 展示，
原样保留完整 URL。`artifact_urls` 里的签名预览 URL 只是 artifact 级只读兜底
（7 天过期、无打标），仅在稳定链接缺失时附带。
不要把容器路径当成用户可打开的地址，也不要编造任何下载链接。
若两者都未返回，交付 Storage 位置及汇总，明确报告暂不可在线打开。
报告入口不要求本服务登录或 JWT 验签；Storage 拒绝访问时应检查当前浏览器的 Storage 凭据。

模式 `resume` 使用原 run_id，input_path 不变，可省略 inference 复用冻结配置。
已有 prediction 不重复请求，仅补发缺失 prediction；业务不通过不等于需要重新推理。
模型来源、模型引用、数据内容、配置或评测脚本变化必须新建 full 运行。
Storage 模型使用不可变版本；clone/download 恢复仍执行同一准备节点，由平台处理缓存，
只保证来源声明和配置一致，不承诺远端同名模型内容不变。
活跃任务需先停止，等待停止成功后再续跑；提交结果不确定时不要盲目重复提交。

## 人工打标与独立报告

已有 badcase 的人工打标交给 [label-studio](../label-studio/SKILL.md)，通过
`label_studio_project` 创建标注项目，保留样本、GT、预测和评分来源。
打标无需启动分布分析、重新评分或远端工作流。

用户明确需要全量分布统计或单独生成报告时，按
[独立产物契约](references/review-tasks.md) 使用 `df_submit_pipeline(products=...)`。
profile 统计数据分布，report 渲染已有结果，均使用 CPU，不调用模型。
评测报告内的独立复核运行仅用于用户明确选择该报告界面的场景，不能替代 Label Studio 打标。
rubric 变化后新建评分版本，保留旧结果和人工标注。
