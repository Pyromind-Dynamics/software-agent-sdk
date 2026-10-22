# 已有预测评分契约

输入来自 `training_analysis(operation="artifacts", task_id=...)`。读取各 eval
的样例、训练/验证集样例和执行配置后生成以下三个工作区文件；原始文件只读。

## 提交

```python
df_submit_pipeline(
    input_path="/models/training-output",
    evaluation={
        "source_manifest_path": "public_data/source_manifest.json",
        "scoring_spec_path": "public_data/scoring_spec.json",
        "scoring_script_path": "public_data/score_eval.py",
    },
)
```

与 `inference`、`products`、清洗脚本参数互斥。工具冻结脚本、说明、清单和运行时，生成
CustomCommandCPUNode 工作流；全量处理，无 GPU、无模型调用。
恢复使用 `mode="resume"` 和原 `resume_run_id`，可以省略 evaluation；同一冻结
输入恢复，不修改旧评分口径。脚本、源文件或配置变化需要新建 full run。

## 来源清单

```json
{
  "task_id": "1051",
  "training_node_id": "train",
  "output_path": "/models/training-output",
  "evaluations": [
    {"step": 200, "path": "/models/training-output/eval_step_200.jsonl", "sha256": "文件内容SHA256"}
  ],
  "checkpoints": [
    {"step": 200, "path": "/models/training-output/checkpoint-200", "base_model": "实际基模引用"}
  ],
  "datasets": {
    "train": {"path": "/datasets/train.jsonl", "sha256": "文件内容SHA256"},
    "validation": {"path": "/datasets/val.jsonl", "sha256": "文件内容SHA256"}
  },
  "generation": {"temperature": 0, "max_tokens": 1024},
  "telemetry": {},
  "training": {},
  "prior_actions": [],
  "budget": null
}
```

generation 填实际执行参数，不照抄示例。每个 eval 可以提供自己的 generation。
schema_version=1 时文件名与 step 必须一致；schema_version=2 支持其他已有预测文件名，
仍用唯一非负 step 标识条目。dataset_role 默认为 validation，test/train 永不参与选模；已有基线预测可在该条声明 `role="baseline"`，使用独立 step
标识且不提供 checkpoint。SHA256 对原始文件内容计算。
缺少验证集对应关系或生成配置时可以计算指标，但该结果不参与选模。
telemetry 放 training_analysis 的诊断及带 step 的指标，不能放凭据。

## 评分说明与脚本

scoring_spec 包含非空 `business_rule`、`selection` 和 `validation_examples`。
分类归并可用 `label_mapping` 显式记录原始标签到业务类别的映射，供分布统计使用；
脚本应使用同一映射。原始类别混淆矩阵需要 score_case 返回 `original_prediction_label`。
selection 格式：

```json
{
  "primary": {"metric": "pass_rate", "direction": "max"},
  "secondary": [{"metric": "classification.per_class.fail.recall", "direction": "max"}],
  "constraints": [{"metric": "classification.per_class.fail.recall", "operator": "ge", "value": 0.95}]
}
```

约束只填写用户确认的业务要求；未指定主指标时默认 pass_rate，未指定约束时
meets_target 为 null。指标相同取较早 step。不同可比组不混排；需要明确比较组时
selection.comparison_group 填计算出的组标识并新建评分运行。

`score_eval.py` 提供两个函数：

- `parse_record(raw_record, context)`：context.source 为 train、validation、test 或 evaluation，
  evaluation 另有 step。返回 `id`（或可验证的稳定 `input_key`）、`reference`、
  `prediction`、`original_label`，可选 `category`、`sample`、`media`。
  train/validation 没有 prediction 是正常情况。各来源的 reference 必须归一到
  同一 GT 表示，保留原始语义，不为业务归并改写 GT。
- `score_case(case, scoring_spec)`：返回布尔 `passed`，可选 `overall_score`（0..1）、
  `evidence`、`failure_reason`、`metrics`、`details`。分类任务同时返回业务口径的
  `reference_label` 和 `prediction_label`，以计算混淆矩阵和 P/R/F1。
  原始标签统计来自 original_label，业务类别统计来自 reference_label。

脚本处理真实字段结构，不从文件顺序生成跨 step ID。已知样本的非法/空预测应返回
passed=false、prediction_label=null；脚本异常会记录为评分故障，不能当作有效选模证据。
不在评分函数中访问模型、修改源文件或加载另一个评分脚本。

validation_examples 是 `{"record": 原始样例, "passed": 预期布尔值}` 的列表，至少
包含应通过、应失败的案例，并覆盖业务映射与无效预测。先在沙箱用代表性样本验证
生成脚本；CPU 全量任务还会自动运行这些对照案例。所有 step 共用该冻结脚本。

## 结果

批次目录提供 `batch_result.json`、`checkpoint_selection.json`、
`iteration_context.json` 和总览报告。每个 `steps/n/` 目录有独立 run_id、
metrics、逐条预测/评分和可选 HTML 报告，沿用稳定报告入口。清单 render_reports=false
可仅评分；报告故障不丢弃评分结果。评分不创建人工标注；已有 badcase 打标使用
[label-studio](../../label-studio/SKILL.md) 的 `label_studio_project`，无需额外提交 profile 或 review 作业。
总览区分最佳 eval 与最佳已保存 checkpoint；不存在同 step 模型时不取邻近 checkpoint。
LoRA 模型引用包含 adapter 与基模，后续推理合并该 step，不使用训练末尾的默认 merge。

## 所选 checkpoint 需要后续推理时

使用现有 inference 入口，`inference.model_path` 填所选 checkpoint-n 的准确 Storage 路径，
其评测配置的 `model_source` 写 `{"type":"checkpoint","kind":"full"}`。
LoRA 写 `{"type":"checkpoint","kind":"lora","base_model":"/models/实际基模"}`；
基模必须已准备到 Storage，远端模型标识先通过已有 clone/download 能力解析到实际文件。
工具生成所选 LoRA 的 Merge → Inference → CPU，合并输出位于本次推理 output_dir/merged-model，
版本登记记录父 checkpoint。该显式引用不会被项目当前默认模型替换。
这一步只用于缺少目标预测、独立测试集或生成配置变化的评测，不为全 step 评分触发。

仅 train + val 时，不额外推理；rubric 变化只重新 CPU 评分。存在独立 test 时，先完成
验证集选模，再在 inference 中传 selection_run_id=选模批次 run_id，input_path 使用该批次
清单 datasets.test.path，model_path 与 model_source 精确对应选中 checkpoint。
测试集文件必须与 train/validation 不同且指纹不同；应由数据来源确认独立划分，路径和指纹
检查不能证明无近重复或业务泄漏。test 不参与选模，不把 test 的表现用于重新挑选 checkpoint。
