# 数据飞轮实现说明

训练产物分析直接复用 eval_step_n.jsonl，由 Agent 根据真实样本生成评分脚本，提交 CPU 批次。Runtime 依据冻结的业务规则选择同 step、完整且可比的 checkpoint，并登记逐 step 报告与下一轮证据。评分、报告生成、人工复核分别执行。

只有 train + validation 时不额外推理；rubric 变化仅创建新 CPU 评分运行。有独立 test 时，先冻结验证集选出的 checkpoint，再在完整 test 上推理并使用相同脚本评分；test 不参与选模。

## 层与依赖

- `pyromind-runtime/domain/training_evaluation.py`：通用候选、指标目标、约束、评分结果与选模结果。
- `application/checkpoint_selection.py`：约束优先、主/次指标、较早 step 的确定性排序。W&B 只作为诊断证据。
- `application/training_evaluations.py`：校验批次指纹、关联子报告、生成选模与迭代上下文，通过 Product Store 和 EvaluationStorage 持久化。
- `domain/review.py` 与 `application/review_tasks.py`：独立数据分布和复核任务模型、来源校验、人工反馈上下文更新；不执行评分脚本、不触发推理或重新渲染报告。
- `application/test_evaluation.py`：验证独立 test 来源和验证集选模结果，冻结相同业务评分脚本，强制使用准确的所选权重与基模。
- `ConversationRuntime`：完成回调、会话恢复、报告读取和人工反馈写入后的协调；版本系统登记所选模型和合并模型父来源。
- `openhands-tools`：训练产物访问与 CPU 工作流打包。评分代码只在 CPU worker 子进程/平台容器运行，不由 HTTP 或 Product Runtime 加载执行。
- 两个 Adapter 消费同一可选 `SessionSpec.product_instructions`，沿用统一工具和模型。前端继续使用现有报告/标注 API。

依赖方向不变。External Task Registry 只登记真实批次任务；各 step 是批次的 PipelineRun 子记录，不伪造平台任务。

## 接口与产物

`training_analysis(operation="artifacts", task_id=..., training_node_id=...)` 返回实际工作流快照中的训练目录、配置和 checkpoint/eval 发现结果，不依赖 W&B。多训练节点必须明确目标；解析失败返回缺口。

`df_submit_pipeline(evaluation={source_manifest_path, scoring_spec_path, scoring_script_path}, input_path=...)` 冻结清单、业务说明与脚本，生成一个 CPU 节点。脚本入口为 `parse_record` 和 `score_case`；详细契约见 `.agents/skills/inference-evaluation/references/saved-predictions.md`。

批次产物：

- `batch_result.json`：全部 step 的指标、完整性诊断、比较组和模型完整性。
- `checkpoint_selection.json`：候选排名、排除原因、验收状态、所选模型和遥测诊断。
- `iteration_context.json`：业务口径、数据版本/分布、本轮及历史引用、人工复核覆盖率与结论、预算和证据缺口。
- `evaluation_report.html`：总览及逐 step 稳定报告链接。
- `steps/n/`：独立评测 ID、逐条结果、指标及可选只读 HTML；不初始化人工标注。

已知样本的空预测保留在分母中计失败。坏行、缺 GT、重复 ID、验证集不完整或生成配置不能确认的 step 不参与选模。不同比较组不混排，所有 step 共享冻结脚本。验收约束未指定时 `meets_target=null`，有约束但不达标时为 `false`。

后续确需推理时，`model_source={"type":"checkpoint","kind":"lora","base_model":"/models/base"}` 与准确的 `inference.model_path=/.../checkpoint-n` 生成 Merge → Inference → CPU；完整权重使用 kind=full。LoRA 基模先通过已有模型准备能力解析到 Storage，显式 checkpoint 不会被项目默认模型替换。

最终 test 提交指定 `inference.selection_run_id`，Runtime 要求输入与冻结 test 一致、与 train/validation 路径及内容指纹不同，且模型与所选 checkpoint 完全一致。推理完成后使用验证集评分脚本，而非另设三分类判等。失败请求仍保留业务 GT，计入通过率和分类召回率的分母。文件指纹校验不等价于证明 train/test 没有近似重复，数据来源仍需提供可信切分。

## 独立人工复核

`df_submit_pipeline(products={operation, manifest_path, extraction_script_path?}, input_path=...)` 复用提交、任务登记和 CPU 执行链路，支持：

- `profile`：用确定性提取脚本全量统计 train/validation 分布。各数据集独立诊断；训练集统计失败不丢弃完整验证集的选模证据，但 profile 不完整时禁止开放复核。
- `review`：从已有评分结果或外部 badcase 清单导入，要求数据版本、类别映射与 ready profile 匹配。冻结 badcase、预填输入/GT/预测/失败原因/分布，复用已有标注库、标签定义和组件。
- `report`：只读已有结果，独立渲染现有 HTML，不初始化人工标注、不要求分布完成。

评分可设置 `render_reports=false`。渲染失败不会使已完成评分无效；复核任务的数据初始化先于渲染，恢复渲染不重置已提交标注。

新 rubric 创建新评分运行和新 review_id，旧任务与人工标注保留，不继承或展示在新任务中。外部清单只有 badcase，不能推导全数据集通过率、Recall 或 F1，界面明确隐藏这些整体统计。

新增 `/review-tasks/{review_id}` 与其 report/annotations/cases/media/exports 路由，复用现有 Runtime、鉴权和不可变标注提交协议。旧 `/evaluation-runs` 路由及历史数据继续可读；CPU 执行任务对应实际平台任务，不为报告伪造任务。契约示例见 `.agents/skills/inference-evaluation/references/review-tasks.md`。

## 兼容与恢复

`PipelineRun.result`、`SessionSpec.product_instructions` 均为新增带默认值字段；旧任务、报告和标注记录仍按原协议读取。评分运行使用现有状态存储和会话隔离。重复完成处理幂等，子 run ID 从批次 ID 和 step 确定；输入变化必须新建运行，旧人工标注不迁移到新评分版本。恢复重新执行 CPU 评分，不请求模型，已提交的人工标注版本保留。

报告 API 继续由请求携带的 Storage 凭据授权；不将 Cookie、Authorization 或 Token 写入持久化状态。服务重启期间未能读取 Storage 的批次，在会话恢复或打开总览报告时再次协调。

## 验证

本次回归：630 passed，包含完整 `tests/pyromind_runtime`，以及评分、推理评测、CPU 提交、训练产物、进度、报告和标注 API。现有 FastAPI/Starlette 测试有两条弃用警告，无测试失败。相关代码执行 pre-commit，包括 Ruff、Pyright 和依赖边界检查。

覆盖 Runtime、架构边界、选模、CPU 提交、报告/标注 API、稀疏 W&B、实际训练 step、SFT/GRPO 产物、多训练节点和上游路径、混合预测结构、非法/空预测、坏行/重复 ID、未保存最佳 eval、缺权重、LoRA 基模和版本绑定。

真实 Agent 场景 `b09_saved_prediction_flywheel.py` 检查真实 LLM 生成脚本、全部 step、正确选模和零额外 GPU 提交；`b10_independent_badcase_review.py` 进一步检查 rubric 变化后重算、分布前置、独立新任务和 badcase 变化。隔离平台回放运行生产 CPU worker，其 transport 有单独 smoke test。真实 LLM 运行需要 `LLM_API_KEY`、`LLM_BASE_URL` 和测试模型配置。当前环境未配置这两个凭据，未执行真实 LLM 回归，也未向生产平台提交训练/推理任务。

```bash
uv run pytest tests/pyromind_runtime
uv run pytest tests/skills/test_training_evaluation.py tests/skills/test_inference_evaluation.py tests/tools/training_analysis/test_training_analysis.py tests/tools/data_preparation/test_inference_submit.py tests/agent_server/test_evaluation_runs_api.py
python tests/integration/run_infer.py --llm-config '{"model":"YOUR_TEST_MODEL"}' --eval-ids b09_saved_prediction_flywheel
python tests/integration/run_infer.py --llm-config '{"model":"YOUR_TEST_MODEL"}' --eval-ids b10_independent_badcase_review
```
