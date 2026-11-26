# 独立分布分析、复核与报告契约

已有 badcase 人工打标使用 [label-studio](../../label-studio/SKILL.md) 的
`label_studio_project`，无需本契约的 profile 或 review 作业。
本契约用于用户明确要求远端分布统计、报告生成或评测报告内的复核界面。

这些远端产物各自使用单个 CPU 作业。遵循
[SDK 执行与回调](../../workflow-execution/references/sdk-execution.md)，用
`pyromind_common.platform.evaluation_core.EvaluationOptions` 校验下述业务参数，
再由 `pyromind_common.platform.evaluation.stage_evaluation` 冻结清单、准备脚本与工作流，
该步骤不发 HTTP。按[执行契约](command-node-pipeline.md)，用 `staged.workflow_path` 的
工作区相对路径和 `staged.name` 调用保留的 `validate_workflow_dsl`；取得
工具 content 中的 `validated_workflow_path`，用 `Path` 读取已保存的实际图，
再由普通 `submit_workflow` 函数调用 SDK 创建一次。
出现 `artifact_unavailable` 时即使 valid=true 也停止，不重构或重写图绕过。
轻量准备函数已部署，不调用旧评测提交 Tool，也不假定 SDK Key 能代替校验 Cookie。
以下示例直接填写 operation、input_path 和下述字段，
不使用 products 外层对象，也不直接作为 SDK 请求体。
SDK 创建使用当前上下文的 out_id 和 `max_retries=0`；提交结果未确认时不重投。
平台 task_id 与业务 run_id、output_dir 分别保存；Project 按 task/attempt 关联真实回执。
复核 run_id 同时是独立 review_id。远端按
[挂载进度参考](../../data-processing/references/storage-progress.md) 读取已知运行目录；
只有产品记录已登记并确认与当前授权运行匹配的报告入口，才作为稳定链接交付。
这些入口均不调用模型；review/report 不执行用户评分脚本。

## 数据分布

```json
{
  "operation": "profile",
  "input_path": "/datasets/train.jsonl",
  "manifest_path": "public_data/profile.json",
  "extraction_script_path": "public_data/extract_dataset.py"
}
```

profile.json：

```json
{
  "datasets": {
    "train": {"path": "/datasets/train.jsonl", "sha256": "文件内容SHA256"},
    "validation": {"path": "/datasets/val.jsonl", "sha256": "文件内容SHA256"}
  },
  "label_mapping": {"pass": "pass", "defect": "pass", "fail": "fail"}
}
```

可加入独立 test。extract_dataset.py 定义 `parse_record(raw, context)`，遵守已有预测
解析契约；只提取 id/reference/original_label 等，数据集没有 prediction 是正常情况。
统计完整文件，输出 dataset_profile.json；status=ready 才能开放复核。
已有评分批次的 dataset_profile.json 也可复用，无需重新提交 profile。

## 复核任务

```json
{"operation": "review", "input_path": "/scores/steps/100/predictions.jsonl", "manifest_path": "public_data/review.json"}
```

review.json 的评分来源示例：

```json
{
  "title": "本轮 badcase 复核",
  "source": {
    "kind": "scoring",
    "path": "/scores/steps/100/predictions.jsonl",
    "sha256": "逐条结果文件SHA256",
    "scoring_run_id": "该step的run_id",
    "scoring_spec_sha256": "run_manifest中的evaluation_config_sha256"
  },
  "scoring_manifest": {
    "path": "/scores/steps/100/run_manifest.json", "sha256": "文件内容SHA256"
  },
  "metrics": {"path": "/scores/steps/100/metrics.json", "sha256": "文件内容SHA256"},
  "profile": {"path": "/scores/dataset_profile.json", "sha256": "文件内容SHA256"},
  "datasets": {
    "train": {"path": "/datasets/train.jsonl", "sha256": "文件内容SHA256"},
    "validation": {"path": "/datasets/val.jsonl", "sha256": "文件内容SHA256"}
  },
  "label_mapping": {"pass": "pass", "defect": "pass", "fail": "fail"}
}
```

datasets、label_mapping 必须与所选 profile 完全一致；不可引用其他数据版本的分布。
同一文件的内容指纹由实际内容计算。metrics 可省略；缺少完整评分指标时不展示整体质量率。
可传 report_spec 复用现有 annotation_schema 标签定义；未指定使用内置标签和组件。
只为需要人工复核的来源创建任务，通常是所选 checkpoint，不为每个 step 自动创建任务。

外部已有 badcase 清单把 source.kind 改为 import，去掉 scoring_run_id、
scoring_spec_sha256、scoring_manifest 和 metrics。清单每行包含稳定 id、input 或 media、
failure_reason，可选 reference、prediction、original_label。字段不同用 field_mapping：

```json
{"id": "sample_id", "input": "prompt", "reference": "gt", "prediction": "answer",
 "failure_reason": "problem", "original_label": "label"}
```

未提供的映射使用同名字段；未知类别显示未知，不猜测。导入清单只有 badcase，不能计算
整个验证集的通过率、召回率或 F1。输入、GT、预测、失败原因和分布自动预填，标注组件复用。

任务生成 review_task.json、独立 run_manifest.json、冻结 predictions.jsonl、
annotation r0 文件和现有标注界面。已登记运行的稳定入口是 `/review-tasks/{review_id}/report`；
annotations、cases、exports 接口复用现有保存及分页协议。评分源的 run_id 不等于 review_id。
SDK 创建任务和生成 HTML 本身不完成报告 API 登记，不能仅凭 run_id 拼接可编辑链接。
没有已确认稳定入口时，通过 `get_storage_url` 交付真实 HTML 的只读预览；如需人工标注，
使用保留的 `label_studio_project`，不承诺预览页具备标注自动保存能力。

评分规则变化：新建评分运行，再创建新复核任务。旧任务和标注保留，不展示、不继承。
恢复使用 operation="resume"、不变的 input_path 和原 resume_run_id，不传 manifest 等配置；新来源/规则/分布必须新建任务。
渲染失败时评分和已初始化的复核数据仍保留；恢复原任务可重试渲染，不会重置已保存标注。

## 单独生成报告

```json
{"operation": "report", "input_path": "/scores/steps/100/predictions.jsonl", "manifest_path": "public_data/report.json"}
```

manifest 使用与复核相同的 source、scoring_manifest、可选 metrics 和 report_spec，
无需 profile，不初始化人工标注。此入口只读取和渲染已有结果。
新报告只读，人工编辑从独立复核任务进入。历史评测报告及历史标注链接继续可用。
