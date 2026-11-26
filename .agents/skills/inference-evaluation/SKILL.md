---
name: inference-evaluation
description: 模型效果评测的统一入口，包括基模 Benchmark、训后评测、GT 对比和评测报告。分析测试集并制定确定性 rubric，使用 Storage、中心仓或外部仓模型，通过平台推理和 CPU 评测生成指标与 HTML 报告。
---

# 模型推理评测

当前 Agent 负责理解任务和制定 rubric；被测模型只生成 prediction，固定的评测脚本负责评分。
业务效果评测统一走本 Skill，包括训练前后的独立评测；不通过 generate-workflow-dsl
另建 Metric/Eval 评测链。训练工作流负责训练及模型产物准备；已有预测评分不需要加载或合并模型。
以计划中的评测对象为准。评测训练产出的模型时，从上游已核验的输出读取目标模型路径，用
`operation="inference"` 在计划指定的数据集上生成逐样本预测并评分。已合并的模型用
`model_source.type="storage"`。即使只有 train 和 val，也在指定的 val 上执行这项评测。
训练期间的 eval_step 文件用于已有预测评分或验证集选模。复用前核对模型版本和样本范围；验证集更换或扩大时，使用目标模型在指定验证集上推理。独立 test 在验证集选模后推理和评分。

平台执行先读 [SDK 执行与回调](../workflow-execution/references/sdk-execution.md)。
恢复历史运行且当前动态清单仍列出 `df_submit_evaluation` 时，沿用该入口恢复原运行；
新 SDK run 使用下面的普通准备函数，不重建或改写旧运行归属。
已启用 SDK 时，在 terminal 用 `pyromind_common.platform.evaluation_core.EvaluationOptions`
校验本文的业务参数，再用 `pyromind_common.platform.evaluation.stage_evaluation`
冻结输入和 workflow.py。调用保留的 `validate_workflow_dsl`，用工作区相对的
`staged.workflow_path` 作为 `dsl_path`、`staged.name` 作为 `name`；取得
工具 content 中的 `validated_workflow_path` 后，用 `Path` 读取已保存的实际图，
通过普通 `submit_workflow` 函数直接调用 SDK 创建一次。
出现 `artifact_unavailable` 时，即使 valid=true 也停止 SDK 续投，不重构或重写图绕过。
默认准备阶段不发 HTTP；`prepare_evaluation` 只供本来就有独立校验登录身份的调用方，
不能假定 SDK Key 能替代校验路由的 Cookie。
轻量准备函数已部署到 SDK 解释器，不需要导入旧 Tool 模型或 Host 包。
SDK 的 `studio.create()` 不负责推理配置、评分、预测复用或报告准备；
不要用普通工作流提交函数跳过评测准备。未启用 SDK 的会话按当前实际清单选择入口。

## 已有预测与训练产物

有 eval_step_n.jsonl 或已有基线 prediction 时，先读真实样例并明确业务判分含义，
按 [已有预测评分契约](references/saved-predictions.md) 生成本轮解析、评分脚本和说明，
通过评测的 `operation="score"` 在 CPU 评分所有 step。此操作计算评分，可生成报告，不初始化人工标注，也不调用模型。评分规则变化时复用原始预测，新建评分运行。
任务允许本地 CPU 评分时，按同一参考中的「本地 CPU 评分交付」直接生成、校验并发布结构化 JSON，再回报 JSON 和报告路径。
训练集与验证集标签分布随报告展示；优先查看所选 checkpoint 的 badcase。
`iteration_context.json` 汇总目标、数据、step 对比、选模和人工反馈，供下一轮判断；
原因假设由 Agent 结合证据提出，不套用固定的补数据或重训规则。
独立 test 使用下面的推理入口，传 selection_run_id 绑定验证集选模批次；
同时把已授权 Project 的冻结选择证据作为 `ValidationSelection` 传给准备函数的
`selection`。函数校验精确 checkpoint、test 来源及与 train/validation 的路径和摘要差异，
CPU 使用验证时冻结的评分脚本或 Reward。不能仅凭自报 task_id 证明来源或执行权限。
parse_record 应支持 context.source=test，标识必须与 dataset_config 一致。LoRA 合并使用所选 step。

## 输入与评价标准

用 `read`/`terminal` 直读 `/workspace` 下的测试集 JSONL，确认输入字段、GT、
媒体路径和代表性样本。媒体相对路径基于 JSONL 所在目录；不要重复设置数据集
目录为 `media_base_dir`。
读取 [输入字段与媒体映射](references/input-contract.md) 和
[GT 解析与 Rubric 配置](references/rubric-authoring.md)，
在当前工作区写出 `public_data/inference_dataset_config.json` 和
`public_data/inference_evaluation_config.json`。
写完评价配置后，按 Rubric 文档中所选 evaluator 的字段白名单逐项检查；
不同 evaluator 的字段不得混用，检查通过后再提交。

选择少量互不重复的评价维度，每项包含 name、criterion、weight、required 和 evaluator。
关键业务正确性高于格式要求；核心项可设 required，失败时整体不通过。
只评价能从数据和输出要求中确定的内容。首版不使用语义 Judge 或被测模型自评。
分类任务配置类别字段判等；要求结构化输出时检查 JSON；有框 GT 时按同坐标系配置 IoU。
读取 report.json / metrics.json 的 rubric_metrics：字符串类别有混淆矩阵、各类 Precision/Recall/F1
和 Macro-F1，框有总体 Precision/Recall/F1；关注业务关键类别的漏检、错放行和格式率、延迟。
未给出业务验收阈值时报告对照结果与差距，不将默认 rubric 通过阈值宣称为上线标准。

## 执行

只讨论方案或评价标准时，完成配置与说明即可。用户要求实际评测或生成报告时，使用
`operation="inference"`，提供 `input_path`、`model_path`、
`dataset_config_path`、`evaluation_config_path`，参见
[执行契约](references/command-node-pipeline.md)。

GPU 按任务实际需求选择：结合模型规模与精度、上下文长度、媒体分辨率和并发估算显存，
优先满足需求的最小卡型、最少卡数，并考虑平台资源可用性和排队情况。
明确填写 gpu_product、gpu_count；执行契约中的单张 L40S 仅为示例。
不要默认使用 H200；仅在较小卡无法满足显存或必要性能要求，或用户明确指定时选用，
并简述选卡依据。样本量大不直接等于需要大卡，可通过分批、降低并发控制资源需求。

模型来源只写在 evaluation_config_path 指向的 JSON 的 model_source 中，不是提交参数。
storage 仅用于 Storage 中已有的模型文件，model_path 为其 Storage 绝对路径。
Storage 中没有目标模型文件时，使用 clone（Clone Model）或 download（Download Model）节点准备模型。
clone 时该字段为中心仓模型标识；download 时为外部仓模型名称，
download_source 支持 huggingface（默认）和 modelscope。
用户指定来源时遵从；未指定来源的模型标识先用 clone，由平台实时校验中心仓可用性。
中心仓不支持时说明原因并给出 download 选择，不静默切换来源，不维护固定模型名单。

`stage_evaluation` 冻结运行目录内的 workflow.py、脚本和配置，返回 `StagedEvaluation`。
保留的校验 Tool 将服务端实际图保存为文件并返回路径；需要同步画布时使用同一份冻结 DSL，
提交时读取返回路径下的实际图。
storage 使用 Inference → CPU；clone/download 在 Inference 前增加对应模型准备节点，
其 model_path 输出绑定 Inference.model_path，再连接 Inference.endpoint → CPU.param。
不要手写评测 workflow.py 或通过另一入口重复提交；准备流程生成的最终校验图只提交一次。
无需在本机运行 GPU 推理；按 staging 契约发布冻结文件，不修改内置脚本，不传 script_path、model_profile、
output_schema 或 convert_format。不另提交工作流调试或正式工作流场景。
所选 checkpoint 使用 model_source.type=checkpoint；full 直接推理，lora 合并准确 adapter 和基模，
参见已有预测评分契约。不接受已有外部 endpoint 或自定义节点类型。

## 结果与恢复

平台 SDK 回执提供 task_id；调用方预先确定 run_id 和 staging.storage，并分别保存。
SDK 提交使用当前执行上下文的 out_id，客户端设 `max_retries=0`。
Project 场景将实际 task_id 绑定到对应 task/attempt；绑定失败只补关联，不再创建平台任务。
远端从实际 `output_dir` 挂载目录读取进度，按
[挂载进度参考](../data-processing/references/storage-progress.md) 调用普通 `read_progress`；
本机没有挂载时，只使用当前清单仍列出的进度入口。
用户要求停止整个工作流时，已启用 SDK 的会话按
[停止参考](../data-processing/references/sdk-stop-task.md) 调用 SDK；其他会话使用
`df_stop_task`。平台回调只代表任务终态，不代表业务指标达标。

回调后直读 output_dir 下的 report.json 和 metrics.json，核对样本数、
成功 prediction 数、完成评分数、通过率及执行错误。评测报告的**首选交付链接**是
产品记录已登记、与当前授权运行匹配的稳定报告入口（浏览器仍需有效 Storage 凭据）。
已有入口可作为 `stable_report_url` 交给 `read_progress`，从 `evaluation_report_url`
原样取得；在最终回答中用 `[查看完整评测报告](稳定链接)` 展示。仅有 UUID、目录名或
HTML 文件不证明报告 API 已登记，不能自行拼接稳定 URL 或承诺页内打标自动保存。
缺少已确认入口时，对读取结果 `artifact_paths` 中实际存在的 HTML 调用 `get_storage_url`，
原样交付其返回 URL，并说明这是只读预览；需要人工标注时使用 `label_studio_project`。
不要把容器路径当成用户可打开的地址，也不要编造任何下载链接。
若两类链接都不可用，交付 Storage 位置及汇总，明确报告暂不可在线打开。
报告入口不要求本服务登录或 JWT 验签；Storage 拒绝访问时应检查当前浏览器的 Storage 凭据。

`operation="resume"` 配合原 `resume_run_id`，input_path 不变，不再传模型或配置参数，复用冻结配置。
独立 test 从原 staging 读取 validation-selection.json；不重新选模或改写冻结选择证据。
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
[独立产物契约](references/review-tasks.md) 使用 `operation="profile"` 或 `"report"`。
profile 统计数据分布，report 渲染已有结果，均使用 CPU，不调用模型。
评测报告内的独立复核运行仅用于用户明确选择该报告界面的场景，不能替代 Label Studio 打标。
rubric 变化后新建评分版本，保留旧结果和人工标注。

评估已登记的训练 checkpoint 时，提交原 checkpoint 路径与 `model_source.type="checkpoint"`。
LoRA 同时填写实际基模，准备函数在评测图内安排合并。已知 Project 产物版本时可传
`source_checkpoint_version_id`；此字段本身不证明归属。按 Project 元数据入口记录真实任务、
父 checkpoint、报告和指标；普通准备函数不自动创建这些业务记录。
